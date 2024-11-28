"""
Adapted from https://github.com/stevenygd/mttt/blob/main/inner_loop.py
"""
from typing import Any, Optional
import flax
import jax
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import field
from diffusion.models.layers.mttt import TTTEncoder


class ARMTTTMultiHeadSelfAttention(nn.Module):
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
  enc_ln: bool =False           # Whether use Layernorm in encoder.
  enc_bias: bool = False        # Whether use bias in encoder.
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
    def inner_loss(params, x, y):
      """[params] : {key:(#heads, ...)}, 
         [x]: (#heads, mini_batch_size, D_x)
         [y]: (#heads, mini_batch_size, D_y)
      """
      y_ = self.enc_forward(params, x)
      return 0.5 * jnp.square(y - y_).mean() * self.num_heads
    
    def inner_step(params, ilr_data):
      """ [state]  {key: (#heads, ...)}
          [data]   (#heads, mini-batch-size, D)  """
      ilr, data = ilr_data
      x_tr = data[..., :self.head_dim]
      y_tr = data[..., self.head_dim:-self.head_dim]
      x_te = data[..., -self.head_dim:]
      loss_val, grads = jax.value_and_grad(inner_loss)(params, x_tr, y_tr)
      # TODO: controling the update of LN stats.
      params_ = jax.tree_util.tree_map(
        lambda p, g: p - ilr.reshape(-1, *[1] * len(g.shape[1:])) * g,
        params, grads)
      # [loss_val] is a scalar
      loss_val_af = inner_loss(params_, x_tr, y_tr)
      out = self.enc_forward(params_, x_te)
      return params_, (out, {"bf": loss_val, "af": loss_val_af})
    self.inner_step = inner_step
   
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
    train_inp_norm = jnp.square(train_inp).mean()**0.5
    train_trg_norm = jnp.square(train_trg).mean()**0.5
    test_inp_norm = jnp.square(test_inp).mean() ** 0.5
    
    # (B, #heads, N, D) -> (B, #heads, N, 2*D)
    data_chunks = jnp.concatenate([train_inp, train_trg, test_inp], axis=-1)
    # (B, #heads, N, D) -> (B, #heads, ttl_N, 2*D)
    data_chunks = jnp.repeat(data_chunks, repeats=self.n_epoch, axis=2)
    if self.shuffle:
      rng_s = self.make_rng("mt3")
      def _shuffle_fn_(data_):
        """[data_] shape=(ttl_N, 2D)"""
        rng_s_i = jax.random.fold_in(rng_s, jax.lax.axis_index("bshd"))
        data_ = jax.random.permutation(rng_s_i, data_, axis=0)
        return data_
      data_chunks = jax.vmap(_shuffle_fn_, axis_name="bshd")(
        data_chunks.reshape(B * H, *data_chunks.shape[-2:]))
      data_chunks = data_chunks.reshape(B, H, *data_chunks.shape[-2:])
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
        self.inner_step, init_params, (self.ilr, data_chunk_batch))
    # Input:  (B, #iters, #heads, N//#iters, 2D)
    # Output: {key: (#heads, ...)}
    _, (test_out, loss_lst) = jax.vmap(_inner_loop_)(data_chunks)
    
    # (B, #iters, #heads, N // #iters, D_enc) -> (B,  #iters, N // #iters, #heads, D_enc)
    # (B,  #iters, N // #iters, #heads, D_enc) -> (B, N, #heads x D)
    test_out = test_out.transpose((0, 1, 3, 2, 4)).reshape(B, N, -1)
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
  

class BiDirARMTTTMultiHeadSelfAttention(nn.Module):
  """Bidirectional AR MTTT Multi-Head Self-Attention."""
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
  enc_ln: bool =False           # Whether use Layernorm in encoder.
  enc_bias: bool = False        # Whether use bias in encoder.
  enc_residual: bool = False    # Whether use encoder residual layer.
  enc_residual_bf: bool = False # Whether use encoder residual apply bf LN.
  enc_use_bias: bool = False    # Encoder using bias
  enc_bias_init: str = 'zeros'  # Encoder bias initialization type
  enc_kernel_init:str= 'zeros'  # Encoder kernel initialization type
  
  def setup(self):
    self.forward_layer = ARMTTTMultiHeadSelfAttention(
      head_dim=self.head_dim,
      num_heads=self.num_heads,
      enc_dim=self.enc_dim,
      out_dim=self.out_dim,
      enc_layers=self.enc_layers,
      ln_eps=self.ln_eps,
      lr=self.lr,
      learnable_lr=self.learnable_lr,
      learnable_init=self.learnable_init,
      n_iters=self.n_iters,
      n_epoch=self.n_epoch,
      shuffle=self.shuffle,
      enc_ln=self.enc_ln,
      enc_bias=self.enc_bias,
      enc_residual=self.enc_residual,
      enc_residual_bf=self.enc_residual_bf,
      enc_use_bias=self.enc_use_bias,
      enc_bias_init=self.enc_bias_init,
      enc_kernel_init=self.enc_kernel_init
    )
    self.backward_layer = ARMTTTMultiHeadSelfAttention(
      head_dim=self.head_dim,
      num_heads=self.num_heads,
      enc_dim=self.enc_dim,
      out_dim=self.out_dim,
      enc_layers=self.enc_layers,
      ln_eps=self.ln_eps,
      lr=self.lr,
      learnable_lr=self.learnable_lr,
      learnable_init=self.learnable_init,
      n_iters=self.n_iters,
      n_epoch=self.n_epoch,
      shuffle=self.shuffle,
      enc_ln=self.enc_ln,
      enc_bias=self.enc_bias,
      enc_residual=self.enc_residual,
      enc_residual_bf=self.enc_residual_bf,
      enc_use_bias=self.enc_use_bias,
      enc_bias_init=self.enc_bias_init,
      enc_kernel_init=self.enc_kernel_init
    )
   
  def __call__(self, q, k, v, training: bool, return_aux: bool):
    """
      [q]: (B, #heads, N, D)
      [k]: (B, #heads, N, D)
      [v]: (B, #heads, N, D)
    """
    forward_aux, backward_aux = {}, {}
    forward_out = self.forward_layer(q, k, v, training, return_aux)
    if return_aux:
      forward_out, forward_aux = forward_out
      
    q_back = jnp.flip(q, axis=2)
    k_back = jnp.flip(k, axis=2)
    v_back = jnp.flip(v, axis=2)
    backward_out = self.backward_layer(
      q_back, k_back, v_back, training, return_aux)
    if return_aux:
      backward_out, backward_aux = backward_out
    test_out = jnp.multiply(forward_out, jnp.flip(backward_out, axis=1))
    if return_aux:
      return test_out, {
        "forward": {
          "out": forward_out,
          "aux": forward_aux
        },
        "backward": {
          "out": backward_out,
          "aux": backward_aux,
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
  # layer = ARMTTTMultiHeadSelfAttention(
  layer = BiDirARMTTTMultiHeadSelfAttention(
    head_dim=head_dim,
    num_heads=num_heads,
    n_iters=10,
    n_epoch=1,
    lr=1.0
  )
  B, N = 2, 10_000
  rng, spl = jax.random.split(rng)
  q = k = v = jax.random.normal(spl, shape=(B, num_heads, N, head_dim)) * 10
  rng, spl = jax.random.split(rng)
  params = layer.init(spl, q, k, v, training=True, return_aux=True)
  print("Params:", params)