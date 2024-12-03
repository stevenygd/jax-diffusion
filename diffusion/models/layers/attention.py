################################################################################
# Adapted from timms
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
################################################################################
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import field
from typing import Dict, Any 
from omegaconf import OmegaConf
from diffusion.models.utils import Identity
from diffusion.models.utils import precision_str_to_type
from diffusion.models.s4d import S4D 
from diffusion.models.ttt import TTTLinear, TTTLinearBase, TTTMLP, TTTMLPBase


class JaxAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
   
    @nn.compact 
    def __call__(self, x, training: bool, return_aux: bool):
        outs = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            normalize_qk=self.qk_norm,
            use_bias=self.qkv_bias,
            bias_init=nn.initializers.zeros_init(),
            out_bias_init=nn.initializers.zeros_init(),
            deterministic=not training,
            dropout_rate=self.attn_drop
        )(x)
        if return_aux:
            return outs, {}
        return outs


class LinearAttention(nn.Module):
    """Linear Attention Layer (https://arxiv.org/abs/2006.16236)"""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    # Linear Attention parameters
    elu: bool = True                    # TODO check default params
    normalizer: str = "adaptive"        # TODO: check default
    scale_q: bool = False 
    scale_qkv: float = 1.

    def setup(self):
        super().__init__()
        # TODO: this is input [dim]
        assert self.dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(
            self.dim * 3, use_bias=self.qkv_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init()
        )
        # NOTE: Pytorch layernorm is eps=1e-5
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5) \
            if self.qk_norm else Identity 
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5) \
            if self.qk_norm else Identity 
        self.attn_dropout = nn.Dropout(self.attn_drop)
        self.proj = nn.Dense(
            self.dim,
            # TODO: check pytorch version whether use bias here?
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init()
        )
        self.proj_dropout = nn.Dropout(self.proj_drop)
        
    def _linear_attention_(self, q, k, v):
        """"Args:
            [q] (B, H, N, D)
            [k] (B, H, N, D)
            [v] (B, H, N, D)
        """
        if self.elu:
            q = jax.nn.elu(q) + 1.
            k = jax.nn.elu(k) + 1.

        k_v = jnp.einsum("...ki,...kj->...ij", k, v)
        numerator = jnp.einsum("...ik,...kj->...ij", q, k_v)

        if self.normalizer == 'adaptive':
            sum_k = k.sum(axis=-2, keepdims=True)
            denominator = jnp.einsum("...ik,...jk->...ij", q, sum_k)
        elif self.normalizer == 'constant':
            denominator = v.shape[-2] * v.shape[-1]  # normalizer = N * d / H
        else:
            raise NotImplementedError(
                "Linear Attention Normalizer %s Not Implemented." 
                % (self.config.normalizer))

        y = numerator / denominator
        y = jnp.einsum("...hnd->...nhd", y)
        y = y.reshape(*y.shape[:2], -1)
        return y

    def __call__(self, x, training: bool, return_aux: bool):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose((2, 0, 3, 1, 4))  #shape=(3, B, #heads, N, head_dim)
        qkv = self.scale_qkv * qkv
        qkv = qkv.reshape(3, B, self.num_heads, N, self.head_dim)  
        # Q, K, V will have shape: (B, #heads, N, head_dim)
        q, k, v = qkv # shape=(B, #heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        # NOTE: linear attention doesn't have square
        if self.scale_q:
            q = q * self.scale
        x = self._linear_attention_(q, k, v)
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        if return_aux:
            return x, {}
        return x
   
    
class S4DAttention(nn.Module):
    # Hidden dimension
    dim: int
    
    # Placeholder
    num_heads: int
    qkv_bias: bool = False 
   
    # SSM configs
    d_expansion: int = 1
    # d_state: int = 64
    # rnn_dim=128 outside, and d_state=self.rnn_dim
    d_state: int = 128
    dropout_rate: float = 0.0
    transposed: bool = False
    kernel_args: Dict[str, Any] = field(default_factory=dict)
    
    # Computation precision
    dtype: str = "float32"

    def setup(self):
        self.h = self.dim
        self.i = self.h * self.d_expansion

        self.pre_bdense = nn.Dense(
            self.i, 
            dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
        )
        self.pre_fdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
        ) 
        self.post_bdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("ssm/post_dense/inp", "ssm/post_dense/out")),
            bias_init=nn.initializers.zeros_init()
        ) 
        self.post_fdense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("ssm/post_dense/inp", "ssm/post_dense/out")),
            bias_init=nn.initializers.zeros_init()
        )
        self.post_dense = nn.Dense(
            self.i, dtype=precision_str_to_type(self.dtype),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
        )

        self.bssm = S4D(d_model=self.i, **self.kernel_args)
        self.fssm = S4D(d_model=self.i, **self.kernel_args)

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        # self.activation = nn.gelu

    def activation(self, x):
        return nn.gelu(x.astype(jnp.float32))

    def __call__(self, u, training: bool = False, return_aux: bool = False):
        forward_u = self.activation(self.pre_fdense(u))
        backward_u = self.activation(self.pre_bdense(jnp.flip(u, axis=1)))

        forward_u, _ = self.fssm(forward_u)
        backward_u, _ = self.bssm(backward_u)

        forward_u = self.post_fdense(forward_u)
        backward_u = jnp.flip(self.post_bdense(backward_u), axis=1)

        y = jnp.multiply(forward_u, backward_u)
        y = self.post_dense(y)

        y = self.dropout(y, deterministic=not training)

        if return_aux:
            return y, {}

        return y
    
    
class TTTAttention(nn.Module):
    """Interface with DiT attention layer."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    mttt_type: str = "linear"
    mttt_kwargs: Any = None
    separate_qkv: bool = False
    grad_checkpoint_qkv: bool = False
    grad_checkpoint_out: bool = False
    ttt_out_norm: bool = False # whether add normalization after ttt (bf mul)
    proj_norm: bool = False # whether add normalization before projection
    mult_norm: bool = True  # whether add normalization before multiplication
    apply_gate: bool = True # whether apply gating after ttt
    learn_token_idx: bool = True # whether learn the token idx
    sigmoid_learnable_token_idx: bool = False # whether apply sigmoid to learnable token idx
    use_ar_mask: bool = True # whether AR within mini-batch
    
    # Mixed precision
    qkv_dtype: str = "float32"
    qkv_ptype: str = "float32"
    out_dtype: str = "float32"
    out_ptype: str = "float32"

    def ttt(self, ttt_layer, x, q, k, v, pos_ids):
        q_, k_, v_, eta, inp_stat = self.get_ttt_inputs(
            ttt_layer, x, q, k, v, position_ids=pos_ids) 
        z, ttt_aux = ttt_layer.ttt(q_, k_, v_, eta)
        if self.mult_norm:
            z = ttt_layer.post_norm(z)
        if self.ttt_out_norm:
            z = ttt_layer.apply_gate(x, z)
        return z, {
            "input_stats": inp_stat,
            "output_stats": ttt_aux
        }
       
    def get_ttt_inputs(self, ttt, batch, XQ, XK, XV, position_ids):
        B, N, F = batch.shape
        n_mini_batch = N // ttt.mini_batch_size
        X = batch.reshape(B, *ttt.seq_shape, ttt.width)

        if ttt.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, :: ttt.mini_batch_size, ...].reshape(
                B, n_mini_batch, ttt.num_heads, ttt.head_dim
            )
            XK_last_in_mini_batch = XK[:, :: ttt.mini_batch_size, ...].reshape(
                B, n_mini_batch, ttt.num_heads, ttt.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(
                B, 1, ttt.num_heads, ttt.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = (
                (ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        XQ = nn.with_logical_constraint(XQ, ("B", "N", "D"))
        XK = nn.with_logical_constraint(XK, ("B", "N", "D"))
        XV = nn.with_logical_constraint(XV, ("B", "N", "D"))

        XQ = ttt._split_heads(XQ)
        XK = ttt._split_heads(XK)
        XV = ttt._split_heads(XV)
        
        if position_ids is not None:
            freqs_cis = jnp.take(
                ttt.freqs_cis, position_ids % ttt.mini_batch_size, axis=0)
            XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=ttt.dtype)

        XQ = ttt._split_mini_batches(XQ)
        XK = ttt._split_mini_batches(XK)
        XV = ttt._split_mini_batches(XV)

        eta = ttt.get_eta(X)

        return (XQ, XK, XV, eta, {
            "ssl_tgt_last_in_mini_batch_from_mean_mse": ssl_tgt_last_in_mini_batch_from_mean_mse,
            "eta": eta.mean(),
            "eta_per_step": eta.mean(axis=(0, 1, -2, -1)),
        })
    
    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.mttt_kwargs.hidden_size = self.dim
        self.mttt_kwargs.num_attention_heads = self.num_heads
        match self.mttt_type:
            case "linear":
                ttt_class = TTTLinear
            case "linear_base":
                ttt_class = TTTLinearBase
            case "mlp":
                ttt_class = TTTMLP
            case "mlp_base":
                ttt_class = TTTMLPBase
            case _:
                raise NotImplementedError(
                    f"TTT type {self.mttt_type} not implemented.")
       
        common_args = {
            "width": self.dim, "num_heads": self.num_heads, 
            "config": self.mttt_kwargs, "dtype": jnp.float32, 
            "set_up_qkvo": False,
            "sigmoid_learnable_token_idx": self.sigmoid_learnable_token_idx,
            "learn_token_idx": self.learn_token_idx,
            "use_ar_mask": self.use_ar_mask
        } 
        self.ttt_forward = ttt_class(**common_args)
        self.ttt_backward = ttt_class(**common_args)
      
        qkv_dense = nn.Dense 
        if self.grad_checkpoint_qkv: 
            qkv_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.qkv = qkv_dense(
            self.dim * (6 if self.separate_qkv else 3), 
            use_bias=self.qkv_bias,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
             bias_init=nn.with_logical_partitioning(
                 nn.initializers.zeros_init(), ("qkv/bias")),
            dtype=precision_str_to_type(self.qkv_dtype),
            param_dtype=precision_str_to_type(self.qkv_ptype),
        )
        
        out_dense = nn.Dense
        if self.grad_checkpoint_out:
            out_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.proj = out_dense(
            self.dim,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.out_dtype),
            param_dtype=precision_str_to_type(self.out_ptype),
        )
        if self.proj_drop > 0.:
            self.proj_dropout = nn.Dropout(self.proj_drop) 
        else:
            self.proj_dropout = lambda x, **kwargs: x 
        if self.proj_norm:
            self.proj_ln = nn.LayerNorm()
        else:
            self.proj_ln = lambda x: x
        
        self.head_dim = self.dim // self.num_heads
        self.mttt_kwargs.hidden_size = self.dim
        self.mttt_kwargs.num_attention_heads = self.num_heads
        match self.mttt_type:
            case "linear":
                ttt_class = TTTLinear
            case "linear_base":
                ttt_class = TTTLinearBase
            case "mlp":
                ttt_class = TTTMLP
            case "mlp_base":
                ttt_class = TTTMLPBase
            case _:
                raise NotImplementedError(
                    f"TTT type {self.mttt_type} not implemented.")
       
        common_args = {
            "width": self.dim * 2, "num_heads": self.num_heads * 2, 
            "config": self.mttt_kwargs, "dtype": jnp.float32, 
            "set_up_qkvo": False,
            "sigmoid_learnable_token_idx": self.sigmoid_learnable_token_idx,
            "learn_token_idx": self.learn_token_idx,
            "use_ar_mask": self.use_ar_mask
        } 
        self.ttt_layer = ttt_class(**common_args)
      
        qkv_dense = nn.Dense 
        if self.grad_checkpoint_qkv: 
            qkv_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.qkv = qkv_dense(
            self.dim * (6 if self.separate_qkv else 3), 
            use_bias=self.qkv_bias,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), 
                ("qkv/inp", "qkv/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.qkv_dtype),
            param_dtype=precision_str_to_type(self.qkv_ptype),
        )
        
        out_dense = nn.Dense
        if self.grad_checkpoint_out:
            out_dense = nn.remat(
                nn.Dense, policy=jax.checkpoint_policies.nothing_saveable)
        self.proj = out_dense(
            self.dim,
            # kernel_init=nn.initializers.xavier_uniform(),
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.xavier_uniform(), ("proj/inp", "proj/out")),
            bias_init=nn.initializers.zeros_init(),
            dtype=precision_str_to_type(self.out_dtype),
            param_dtype=precision_str_to_type(self.out_ptype),
        )
        if self.proj_drop > 0.:
            self.proj_dropout = nn.Dropout(self.proj_drop) 
        else:
            self.proj_dropout = lambda x, **kwargs: x 
        if self.proj_norm:
            self.proj_ln = nn.LayerNorm()
        else:
            self.proj_ln = lambda x: x
        
    def __call__(self, x, training: bool, return_aux: bool = False):
        """[x] B, N, C"""
        B, N, C = x.shape
        
        # Transform K, Q, V
        x = x.astype(precision_str_to_type(self.qkv_dtype)) 
        qkv = self.qkv(x)
        qkv = qkv.astype(jnp.float32) # the output of qkv will cast to flaot32
        if self.separate_qkv:
            qkv = qkv.reshape(B, N, 6, -1)
            # shape=(B, N, 6, dim) -> (6, B, N, dim) -> 6 x (B, N, dim)
            q_f, k_f, v_f, q_b, k_b, v_b = qkv.transpose((2, 0, 1, 3)) 
        else:
            qkv = qkv.reshape(B, N, 3, -1)
            # shape=(B, N, 3, dim) -> (3, B, N, dim) -> (B, N, dim)
            q_f, k_f, v_f = qkv.transpose((2, 0, 1, 3)) 
            q_b, k_b, v_b = q_f, k_f, v_f
         
        position_ids = jnp.arange(N)[None, :]
        x_flip = jnp.flip(x, axis=-2)
        xx = jnp.concat([x, x_flip], axis=-1)
        
        q_flip = jnp.flip(q_b, axis=-2)
        qq = jnp.concat([q_f, q_flip], axis=-1)
        
        k_flip = jnp.flip(k_b, axis=-2)
        kk = jnp.concat([k_f, k_flip], axis=-1)
        
        v_flip = jnp.flip(v_b, axis=-2)
        vv = jnp.concat([v_f, v_flip], axis=-1)
        
        x, aux = self.ttt(self.ttt_layer, xx, qq, kk, vv, position_ids)
        
        # Combine forward and backward 
        xf, xb = jnp.split(x, 2, axis=-1)
        x = jnp.multiply(xf, jnp.flip(xb, axis=-2))
        
        # Output layer 
        x = self.proj_ln(x)
        x = x.astype(precision_str_to_type(self.out_dtype))
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        x = x.astype(jnp.float32) # next layer will take float32
        if return_aux:
            return x, {"forward": aux, "backward": aux}
        return x
    
    

if __name__ == "__main__":
    import lovely_jax
    lovely_jax.monkey_patch()
    
    # with jax.default_device(jax.devices("cpu")[0]):
    with jax.default_device(jax.devices("gpu")[0]):
        rng = jax.random.PRNGKey(1)
        B = 1
        N = 128 
        D = 6 * 64
        H = 64
        scale_q = False 
        qkv_bias = True 
        mttt_type = "base"
        # mttt_type = "ar"
        # mttt_type = "bdar"
    
        LA = LinearAttention(
            dim=D, num_heads=H, qkv_bias=qkv_bias, 
            # scale_qkv=jnp.sqrt(2),
            elu=False, normalizer="constant", scale_q=scale_q)
        rng, spl = jax.random.split(rng)
        la_params = LA.init(spl, jnp.ones((B, N, D)), training=True, return_aux=False)
        print("LA", la_params)

        MT3 = MTTTAttention(
            dim=D, num_heads=H, qkv_bias=qkv_bias, 
            mttt_type=mttt_type,
            scale_q=scale_q)
        rng, spl2 = jax.random.split(rng)
        mt3_params = MT3.init(
            {"mt3": spl2, "params": spl}, 
            jnp.ones((B, N, D)), training=True, return_aux=False,
        )
        print("MT3", mt3_params)
        
        
        # TTT Layer
        # NOTE: inner_itr==3 doesn't work
        # python train.py --config config_patch.py \
        #         --config.inner.TTT.inner_encoder=mlp_1 \
        #         --config.inner.TTT.inner_encoder_init=zero \
        #         --config.inner.TTT.inner_encoder_bias=False \
        #         --config.inner.TTT.decoder_LN=False \
        #         --config.inner.TTT.train_init=False \
        ttt_cfg = OmegaConf.create({
                "inner_encoder": "mlp_1", 
                "inner_encoder_bias": False, 
                "inner_encoder_init": "zero", 
                "inner_lr": (1.0,),
                "inner_itr": 1, 
                "SGD": False, 
                "decoder_LN": False
        })
        vit3 = ViTTTAttention(dim=D, num_heads=H, ttt_cfg=ttt_cfg)
        vit3_params = vit3.init(
            {"params": spl}, jnp.ones((B, N, D)), training=True, 
            return_aux=False)
        print("ViT3", vit3_params)
        
        vit3_la = ViTTTLinearAttention(
            dim=D, num_heads=H, qkv_bias=qkv_bias,
            elu=False, normalizer="constant", scale_q=scale_q)
        vit3_la_params = vit3_la.init(
            {"params": spl}, jnp.ones((B, N, D)), training=True, 
            return_aux=False)
        print("ViT3LA", vit3_la_params)
        
        print("=" * 80) 
        print("Parameter matching")
        print("=" * 80) 
        def err(x, y, eps=1e-8):
            return (jnp.abs(x - y) / (jnp.abs(y) + eps)).mean()
            # return jnp.abs(x - y).max(), jnp.abs(x - y).mean(), jnp.abs(x - y).min()
            # return jnp.abs(x - y).mean()
            
        def compute_diff(dict_a, dict_b):
            return {
                k: (compute_diff(dict_a[k], dict_b[k]) 
                    if isinstance(dict_a[k], dict) 
                    else err(dict_a[k], dict_b[k]))
                for k in dict_b.keys()
            }
        print(compute_diff(mt3_params, la_params))
        
        def make_data(rng) :
            return jax.random.normal(rng, (B, N, D))
        
        print("=" * 80) 
        print("Forward pass matching")
        print("=" * 80) 
        rng, spl = jax.random.split(rng)
        x = make_data(spl)
        la_out = LA.apply(la_params, x, training=True, return_aux=False)
        print("Out", la_out)
        la_out2 = LA.apply(la_params, x, training=True, return_aux=False)
        print("BETWEEN", err(la_out2, la_out))
        
        print('-' * 20, "MT3", '-' * 20)
        rng, spl = jax.random.split(rng)
        mt3_out, mt3_aux = MT3.apply(
            mt3_params, x, training=True, return_aux=True,
            rngs={"mt3": spl}
        ) 
        print("Out", mt3_out)
        print("MT3", err(mt3_out, la_out))
        
        rng, spl = jax.random.split(rng)
        mt3_out_2 = MT3.apply(
            mt3_params, x, training=True, return_aux=False,
            rngs={"mt3": spl}
        )
        print("BTM", err(mt3_out, mt3_out_2))
        print("MT3", err(mt3_out_2, la_out))
        
        print('-' * 20, "VIT3", '-' * 20)
        vit3_out = vit3.apply(
            vit3_params, x, training=True, return_aux=False,
            rngs={"idx": spl}
        )
        print("OUT", vit3_out)
        print("VT3", err(vit3_out, la_out))
        
        print('-' * 20, "VIT3-LA", '-' * 20)
        vit3_la_out = vit3_la.apply(
            vit3_la_params, x, training=True, return_aux=False,
            rngs={"idx": spl}
        )
        print("OUT", vit3_la_out)
        print("VT3LA", err(vit3_la_out, la_out))
        print("VT3LA v.s. VT3", err(vit3_la_out, vit3_out))
        
        print("=" * 80) 
        print("Cross parameter testing")
        print("=" * 80) 
        rng, spl = jax.random.split(rng)
        la_out_3 = LA.apply(mt3_params, x, training=True, return_aux=False)
        print("LA-MT params v.s. LA-LA params", err(la_out_3, la_out))
        print("LA-MT params v.s. MT-MT params", err(la_out_3, mt3_out_2))
        
        # print("=" * 80) 
        # print("Gradient") 
        # print("=" * 80) 
        # def loss_la(params, x):
        #     out = LA.apply(params, x, training=False, return_aux=False)
        #     return jnp.square(out).mean()
        # grads_la = jax.grad(loss_la)(la_params, x)
        # print("LA :\n", grads_la["params"])
        
        # def loss_mt3(params, x):
        #     out = MT3.apply(
        #         params, x, training=False, return_aux=False, rngs={"mt3": spl})
        #     return jnp.square(out).mean()
        # grads_mt3 = jax.grad(loss_mt3)(mt3_params, x)
        # qkv_params = {
        #     k: v for k, v in grads_mt3["params"].items() 
        #     if k in grads_la["params"].keys()}
        # print("MT3 KQV:\n", qkv_params)
        # print(
        #     "MT3 KQV error:\n\t", 
        #     jax.tree_util.tree_map(err, qkv_params, grads_la["params"])
        # )
        
        # print("MT3 attn:\n", 
        #     {k: v for k, v in grads_mt3["params"].items() 
        #     if k not in grads_la["params"].keys()})
        