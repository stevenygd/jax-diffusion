################################################################################
# Adapted from timms
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
################################################################################
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from dataclasses import field
from typing import Optional, Any 
from omegaconf import OmegaConf
from diffusion.models.layers.mttt import MTTTMultiHeadSelfAttention
from diffusion.models.layers.mttt_ar import ARMTTTMultiHeadSelfAttention, BiDirARMTTTMultiHeadSelfAttention
from diffusion.models.layers.vittt import TTTLayer, LinearAttention as ViT3LinearAttentionLayer
from diffusion.models.layers.attention_diffuser import FlaxAttention
from diffusion.models.utils import precision_str_to_type


identity = lambda x, **kwargs: x


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


# For better reproduction, this is direct Torch Translate.
# NOTE: this has not fused attention.
# NOTE: I use DiT's initialization for the DenseLayer (bias zero init)
class TorchAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    # norm_layer: nn.Module = nn.LayerNorm,

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
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5) if self.qk_norm else identity 
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5) if self.qk_norm else identity 
        self.attn_dropout = nn.Dropout(self.attn_drop)
        self.proj = nn.Dense(
            self.dim,
            # TODO: check pytorch version whether use bias here?
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init()
        )
        self.proj_dropout = nn.Dropout(self.proj_drop)

    def __call__(self, x, training: bool, return_aux: bool):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose((2, 0, 3, 1, 4))  # shape=(3, B, #heads, N, head_dim)
        # Torch: unbined the first dimension
        # q, k, v = qkv.unbind(0)
        # Q, K, V will have shape: (B, #heads, N, head_dim)
        q, k, v = qkv # shape=(B, #heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        # TORCH: attn = q @ k.transpose(-2, -1)
        attn = q @ k.transpose((0, 1, 3, 2))
        # TORCH: attn = attn.softmax(dim=-1)
        attn = nn.softmax(attn, axis=-1)  # shape=(B, #heads, N, N)
        attn = self.attn_dropout(attn, deterministic=not training)
        # (B, #heads, N, N) @ (B, #heads, N, head_dim) -> (B, #heads, N, head_dim)
        x = attn @ v  
        # TORCH: x = x.transpose(1, 2).reshape(B, N, C)
        x = x.transpose((0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        if return_aux:
            return x, {}
        return x


class DiffuserAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    
    # Specific to Diffuser Attention
    use_mem_efficient_attn: bool = False
    dtype: str = "float32"
   
    @nn.compact 
    def __call__(self, x, training: bool, return_aux: bool):
        outs = FlaxAttention(
            query_dim=self.dim,
            heads=self.num_heads,
            dim_head=self.dim // self.num_heads, 
            dropout=self.proj_drop,
            use_memory_efficient_attention=self.use_mem_efficient_attn,
            split_head_dim=False,
            dtype=precision_str_to_type(self.dtype)
        )(x, deterministic=not training)
        if return_aux:
            return outs, {}
        return outs


   
class MTTTAttention(nn.Module):
    """ Attention wrapper with MTTT backend. """
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    scale_q: bool = False 
    scale_qkv: float = 1.
    
    # MTTT config
    mttt_type: str = "base" # options: base, ar
    mttt_kwargs: Optional[dict] = field(default_factory=dict)

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
        self.q_norm = nn.LayerNorm(epsilon=1e-5) \
            if self.qk_norm else identity 
        self.k_norm = nn.LayerNorm(epsilon=1e-5) \
            if self.qk_norm else identity 
        
        match self.mttt_type:
            case "bdar": 
                self.attn = BiDirARMTTTMultiHeadSelfAttention(
                    head_dim=self.head_dim,
                    num_heads=self.num_heads,
                    **self.mttt_kwargs
                )
            case "ar": 
                self.attn = ARMTTTMultiHeadSelfAttention(
                    head_dim=self.head_dim,
                    num_heads=self.num_heads,
                    **self.mttt_kwargs
                )
            case "base": 
                self.attn = MTTTMultiHeadSelfAttention(
                    head_dim=self.head_dim,
                    num_heads=self.num_heads,
                    **self.mttt_kwargs
                )
            case _: 
                raise NotImplemented
            
        self.proj = nn.Dense(
            self.dim,
            # TODO: check pytorch version whether use bias here?
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init()
        )
        self.proj_dropout = nn.Dropout(self.proj_drop)

    def __call__(self, x, training: bool, return_aux: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose((2, 0, 3, 1, 4))  # shape=(3, B, #heads, N, head_dim)
        qkv = qkv * self.scale_qkv
        # shape=(3, B, #heads, N, head_dim)
        qkv = qkv.reshape(3, B, self.num_heads, N, self.head_dim)  
        # Q, K, V will have shape: (B, #heads, N, head_dim)
        q, k, v = qkv # shape=(B, #heads, N, head_dim)
        
        # NOTE: these will be added inside MT3 module
        q, k = self.q_norm(q), self.k_norm(k)
        if self.scale_q:
            q = q * self.scale
        
        x = self.attn(q, k, v, training, return_aux) 
        if return_aux:
            x, attn_aux = x
            
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=not training)
        
        if return_aux:
            return x, attn_aux
        return x

    
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
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5) if self.qk_norm else identity 
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5) if self.qk_norm else identity 
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
        qkv = qkv.transpose((2, 0, 3, 1, 4))  # shape=(3, B, #heads, N, head_dim)
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
    
    
class ViTTTAttention(nn.Module):
    """ViTTT Attention Layer https://github.com/stevenygd/mttt/tree/main"""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    ttt_cfg: Optional[Any] = field(default_factory=dict)
    # TTT Parameters

    def setup(self):
        super().__init__()
        assert self.dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.ttt = TTTLayer(
            self.dim, num_heads=self.num_heads, 
            config=self.ttt_cfg)
        
    def __call__(self, x, training: bool, return_aux: bool):
        x, loss_lst = self.ttt(x)
        if return_aux:
            data = {"inner_losses": {"bf": loss_lst, "af": loss_lst}}
            return x, data
        return x
    
    
class ViTTTLinearAttention(nn.Module):
    """ViTTT Linear Attention Layer https://github.com/stevenygd/mttt/tree/main
       NOTE: params just matching the Linear Attention Layer above!
    """
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
        assert self.dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        assert self.qkv_bias, "QKV Bias is not implemented for Linear Attention"
        self.lin_attn = ViT3LinearAttentionLayer(
            self.dim, num_heads=self.num_heads, 
            config=OmegaConf.create({
                "elu": self.elu, 
                "normalizer": self.normalizer, 
            }))
        
    def __call__(self, x, training: bool, return_aux: bool):
        x = self.lin_attn(x)
        if return_aux:
            return x, {} 
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
        