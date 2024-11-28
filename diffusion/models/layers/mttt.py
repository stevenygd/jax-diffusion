"""
Adapted from https://github.com/stevenygd/mttt/blob/main/inner_loop.py
"""
from typing import Any, Optional
import flax
import jax
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import field


class TTTEncoder(nn.Module):
  inp_dim: int
  hid_dim: int = 256
  out_dim: int = 256
  n_layers: int = 2
  layer_norm: bool = True 
  residual: bool = True
  residual_bf: bool = False # position of the residual
  ln_eps: float = 1e-6
  bias_init: str = "zeros"
  kernel_init: str = "zeros"
  use_bias: bool = False
  
  def _kernel_init_(self):
    return {
      "xavier": nn.initializers.xavier_uniform(),
      "zeros": nn.initializers.zeros_init(),
    }[self.kernel_init]
  
  def _bias_init_(self):
    return {
      "zeros": nn.initializers.zeros_init() 
    }[self.bias_init]

  @nn.compact
  def __call__(self, x):
    x_orig = x
    for i in range(self.n_layers - 1):
      x = nn.Dense(
        self.hid_dim, 
        use_bias=self.use_bias,
        kernel_init=self._kernel_init_(),
        bias_init=self._bias_init_(),
        name="inner_enc_Dense_%d" % i
      )(x)
      x = nn.gelu(x)
    x = nn.Dense(
      self.out_dim, name="inner_enc_Dense_out",
      use_bias=self.use_bias,
      kernel_init=self._kernel_init_(),
      bias_init=self._bias_init_(),
    )(x)
    # This layernorm is useful as in: https://github.com/karan-dalal/
    # MTTT-LLM-EasyLM/blob/ae9491da9340e08e72af2a8b5b3f06bb5b448b2e/EasyLM/
    # models/llama/ttt_inner_loop.py#L815
    x_res = x_orig
    if self.residual and self.inp_dim != self.out_dim:
      x_res = nn.Dense(
        self.out_dim,
        use_bias=self.use_bias,
        kernel_init=self._kernel_init_(),
        bias_init=self._bias_init_(),
        name="inner_enc_residual",
      )(x_orig)
    if self.residual and self.residual_bf: 
      x = x_res + x
    if self.layer_norm: 
      x = nn.LayerNorm(epsilon=self.ln_eps)(x)
    if self.residual and not self.residual_bf: 
      x = x_res + x
    return x
 
  
class MTTTMultiHeadSelfAttention(nn.Module):
  """
    My implementation of MTTT multi-head self attention layer. Multiple Heads.
    Implemented via vmap.
  """
  head_dim: int                 # Dimension for Q, K, and V.
  num_heads: int                # number of heads.
  enc_dim: Optional[int] = None # Encoder: hidden dimension. Default=head_dim.
  out_dim: Optional[int] = None # Encoder output dimension. Default=head_dim.
  
  enc_layers: int = 1           # Encoder: number of layers.
  ln_eps: float = 1e-5          # LayerNorm epsilon
  lr: float = 1.0               # Inner loop learning rate.
  learnable_lr: bool = False    # Learning learning rate
  learnable_init: bool = False  # Whether initialization is learnable
  n_iters: int = 1              # Total of inner loop iterations 
  n_epoch: int = 1              # Number of epochs (i.e. times seeing the ctx)
  shuffle: bool = True          # Shuffle both input/output data of inner loop.
  shuffle_separate: bool = False# Shuffle the mapping between input&output data.
  shuffle_per_head: bool = False# Different head has different shuffle
  enc_ln: bool =False           # Whether use Layernorm in encoder.
  enc_bias: bool = False        # Whether use bias in encoder.
  update_enc_ln: bool = False   # Whether update encoder LN with inner loss.
  inp_ln: bool = False          # Whether use Layernorm in input of inner opt.
  trg_ln: bool = False          # Whether use Layernorm in target of inner opt.
  out_ln: bool = False          # Whether use Layernorm in output of inner opt.
  enc_residual: bool = False    # Whether use encoder residual layer.
  enc_residual_bf: bool = False # Whether use encoder residual apply bf LN.
  enc_use_bias: bool = False    # Encoder using bias
  enc_bias_init: str = 'zeros'  # Encoder bias initialization type
  enc_kernel_init:str= 'zeros'  # Encoder kernel initialization type
  
  def setup(self):
    # Set-up inner loop encoder
    encoder = TTTEncoder(
      inp_dim=self.head_dim,
      out_dim=self.out_dim if self.out_dim is not None else self.head_dim, 
      hid_dim=self.enc_dim if self.enc_dim is not None else self.head_dim, 
      n_layers=self.enc_layers,
      layer_norm=self.enc_ln,
      residual=self.enc_residual,
      residual_bf=self.enc_residual_bf,
      ln_eps=self.ln_eps,
      use_bias=self.enc_use_bias,
      bias_init=self.enc_bias_init,
      kernel_init=self.enc_kernel_init,
    )
    def _enc_init_(init_rng):
      params = [
        encoder.init(
          jax.random.fold_in(init_rng, i), jnp.ones((1, self.head_dim)))
        for i in range(self.num_heads)]
      return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *params)
    self.enc_init_params = self.param("enc_init_params", init_fn=_enc_init_)
    # Vmap through different heads
    self.enc_forward = jax.vmap(encoder.apply, in_axes=(0, 0))
    
    # Learnable lr
    ilr = jnp.ones((self.n_iters, self.num_heads)) * self.lr
    if self.learnable_lr:
      self.ilr = self.param("ilr", init_fn=lambda _: ilr)
    else:
      self.ilr = ilr
    
    # Set-up inner loop loss and stepping 
    def inner_loss(params, data):
      """[params] : {key:(#heads, ...)}, 
         [data]: (#heads, mini_batch_size, D_x+D_y)"""
      x = data[..., :self.head_dim]
      y = data[..., self.head_dim:]
      y_ = self.enc_forward(params, x)
      return 0.5 * jnp.square(y - y_).mean() * self.num_heads
    
    def inner_step(params, ilr_data):
      """ [state]  {key: (#heads, ...)}
          [data]   (#heads, mini-batch-size, D)  """
      ilr, data = ilr_data
      # [loss_val] is a scalar
      loss_val, grads = jax.value_and_grad(inner_loss)(params, data)
      # TODO: controling the update of LN stats.
      params_ = jax.tree_util.tree_map(
        lambda p, g: p - ilr.reshape(-1, *[1] * len(g.shape[1:])) * g,
        params, grads)
      loss_val_af = inner_loss(params_, data)
      return params_, {"bf": loss_val, "af": loss_val_af}
    self.inner_step = inner_step
   
    # NOTE: these are legacy LNs, doesn't help much, will delete later 
    self.inp_layer_norm = nn.LayerNorm(epsilon=self.ln_eps) \
      if self.inp_ln else lambda x: x
    self.trg_layer_norm = nn.LayerNorm(epsilon=self.ln_eps) \
      if self.trg_ln else lambda x: x
    self.out_layer_norm = nn.LayerNorm(epsilon=self.ln_eps) \
      if self.out_ln else lambda x: x
      
    # Transformation to match the encoder dimension
    if self.out_dim is not None and self.out_dim != self.head_dim:
      self.trg_proj = nn.Dense(self.out_dim)
    else:
      self.trg_proj = lambda x: x
    
  def __call__(self, q, k, v, training: bool, return_aux: bool):
    """
      [q]: (B, #heads, N, D)
      [k]: (B, #heads, N, D)
      [v]: (B, #heads, N, D)
    """
    assert len(q.shape) == len(k.shape) == len(v.shape)
    assert self.num_heads == q.shape[1] == k.shape[1] == v.shape[1]
    B, H, N = q.shape[0], q.shape[1], q.shape[2]
    train_inp = k
    train_trg = v
    test_inp = q
    k_norm = jnp.square(k).mean()**0.5
    q_norm = jnp.square(q).mean()**0.5
    v_norm = jnp.square(v).mean()**0.5
    train_inp = self.inp_layer_norm(train_inp)
    train_trg = self.trg_layer_norm(train_trg)
    train_trg = self.trg_proj(train_trg) # (B, #heads, N, enc.out_dim)
    train_inp_norm = jnp.square(train_inp).mean()**0.5
    train_trg_norm = jnp.square(train_trg).mean()**0.5
    
    # (B, #heads, N, D) -> (B, #heads, N, 2*D)
    data_chunks = jnp.concatenate([train_inp, train_trg], axis=-1)
    # (B, #heads, N, D) -> (B, #heads, ttl_N, 2*D)
    data_chunks = jnp.repeat(data_chunks, repeats=self.n_epoch, axis=2)
    if self.shuffle:
      if self.shuffle_per_head:
        rng_s = self.make_rng("mt3")
        def _shuffle_fn_(data_):
          """[data_] shape=(ttl_N, 2D)"""
          rng_s_i = jax.random.fold_in(rng_s, jax.lax.axis_index("bshd"))
          data_ = jax.random.permutation(rng_s_i, data_, axis=0)
          return data_
        data_chunks = jax.vmap(_shuffle_fn_, axis_name="bshd")(
          data_chunks.reshape(B * H, *data_chunks.shape[-2:]))
        data_chunks = data_chunks.reshape(B, H, *data_chunks.shape[-2:])
      else:
        data_chunks = jax.random.permutation(
          self.make_rng("mt3"), data_chunks, axis=2)
    # Select a subset that's divisible by the total iterations.
    max_N = (data_chunks.shape[2] // self.n_iters) * self.n_iters
    data_chunks = data_chunks[:, :, :max_N, :]
    # (B, #heads, ttl_N, 2D)  -> [(B, #heads, bs, 2D)] -> (B, #iters, #heads, bs, 2D)
    data_chunks = jnp.stack(jnp.split(data_chunks, self.n_iters, axis=2), axis=1)
    
    def _inner_loop_(data_chunk_batch):
      # Inner loop, input=[qk_chunk_batch] (#iters, #heads, N', 2D)
      # {...key: (#heads, param_shape)...}
      # Scan over the iterations.
      init_params = self.enc_init_params
      if not self.learnable_init:
        init_params = jax.lax.stop_gradient(init_params)
      return jax.lax.scan(
        self.inner_step, 
        init_params,
        (self.ilr, data_chunk_batch))
    # Input:  (B, #iters, #heads, N//#iters, 2D)
    # Output: {key: (#heads, ...)}
    final_params, loss_lst = jax.vmap(_inner_loop_)(data_chunks)
    
    # Outter loop
    # {key:(B, #heads,...)}, (B, #heads, N, D) -> (B, #heads, N, D_enc)
    # VMAP over batches, [enc_forward] will vmap over #heads
    test_inp = self.inp_layer_norm(test_inp)
    test_inp_norm = jnp.square(test_inp).mean() ** 0.5
    test_out = jax.vmap(
      self.enc_forward, in_axes=(0, 0))(final_params, test_inp)
    # (B, #heads, N, D_enc) -> (B, N, #heads, D_enc) -> (B, N, D)
    test_out = test_out.transpose((0, 2, 1, 3)).reshape(B, N, -1)
    test_out = self.out_layer_norm(test_out)
    test_out_norm = jnp.square(test_out).mean() ** 0.5
    if return_aux:
      return test_out, {
        "inner_losses": loss_lst,
        "norms": {
          "k_norm": k_norm,
          "q_norm": q_norm,
          "v_norm": v_norm,
          "train_inp_norm": train_inp_norm,
          "train_trg_norm": train_trg_norm,
          "test_inp_norm": test_inp_norm,
          "test_out_norm": test_out_norm,
        }
      }
    return test_out


if __name__ == "__main__":
  import lovely_jax
  from time import time
  lovely_jax.monkey_patch()
  rng = jax.random.PRNGKey(1)
  
  # with jax.default_device(jax.devices("cpu")[0]):
  D = 6 * 64
  num_heads = 6
  head_dim = D // num_heads
  layer = MTTTMultiHeadSelfAttention(
    head_dim=head_dim,
    num_heads=num_heads,
    n_iters=1,
    n_epoch=1,
    lr=1.0
  )
  B, N = 2, 10_000
  rng, spl = jax.random.split(rng)
  q = k = v = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * 10
  rng, spl = jax.random.split(rng)
  params = layer.init(spl, q, k, v, training=True, return_aux=True)
  print("Params:", params)
  
  # @jax.jit
  # def forward(rng, q, k, v):
  #   return layer.apply(
  #     params, q, k, v, training=True, return_aux=False, rngs={"mt3": rng})
  
  # s = time()
  # rng, spl = jax.random.split(rng)
  # out = forward(spl, q, k, v)
  # print("Jitting:", time() - s)
  
  # s = time()
  # rng, spl = jax.random.split(rng)
  # out = forward(spl, q, k, v)
  # print("Jitted :", time() - s)
  
  # import timeit 
  # rng, spl = jax.random.split(rng)
  # print(timeit.timeit(lambda: forward(spl, q, k, v), number=100))
  
  # Now comparing between linear attention
  from attention import LinearAttention
  linattn = LinearAttention(
    dim=D, num_heads=num_heads, elu=False, normalizer="constant")
  rng, spl = jax.random.split(rng)
  params_la = linattn.init(
    spl, jnp.ones((B, N, D)), training=False, return_aux=False)
  
  rng, spl = jax.random.split(rng)
  params_la_2 = linattn.init(
    spl, jnp.ones((B, N, D)), training=False, return_aux=False)
  
  def err(x, y):
    return (jnp.abs(x-y) / jnp.abs(y)).mean()

  mult = 100 
  rng, spl = jax.random.split(rng) 
  q = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
  rng, spl = jax.random.split(rng) 
  k = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
  rng, spl = jax.random.split(rng) 
  v = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
  out_la = linattn.apply(params_la, q, k, v, method=linattn._linear_attention_)
  print(out_la)
  out_la_2 = linattn.apply(params_la, q, k, v, method=linattn._linear_attention_)
  print(out_la_2)
  print("init-diff", err(out_la_2, out_la))
  
  rng, spl = jax.random.split(rng)
  out_mt3 = layer.apply(
    params, q, k, v, training=False, return_aux=False, rngs={"mt3": spl})
  print(out_mt3)
  print("diff", err(out_mt3, out_la))
  
  rng, spl = jax.random.split(rng)
  out_mt3_2 = layer.apply(
    params, q, k, v, training=False, return_aux=False, rngs={"mt3": spl})
  print(out_mt3)
  print("diff-btw", err(out_mt3_2, out_mt3))
  print("diff", err(out_mt3_2, out_la))

  print("=" * 80) 
  print("Gradient") 
  print("=" * 80) 
  
  def make_qkv(rng):
    rng, spl = jax.random.split(rng) 
    q = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
    rng, spl = jax.random.split(rng) 
    k = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
    rng, spl = jax.random.split(rng) 
    v = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * mult
    return q, k, v
  
  def loss_mt3(params, rng):
    q, k, v = make_qkv(rng)
    out = layer.apply(
      params, q, k, v, training=False, return_aux=False, rngs={"mt3": spl})
    return jnp.square(out - 1).mean()
  rng, spl = jax.random.split(rng) 
  grads_mt3 = jax.grad(loss_mt3)(params, spl)
  print(grads_mt3)
  
  def loss_la(params, rng):
    q, k, v = make_qkv(rng)
    out = linattn.apply(params_la, q, k, v, method=linattn._linear_attention_)
    return jnp.square(out).mean()
  rng, spl = jax.random.split(rng) 
  grads_la = jax.grad(loss_la)(params_la, spl)
  print(grads_la)