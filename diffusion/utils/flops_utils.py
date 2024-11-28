import jax.numpy as jnp
import flax.linen as nn
from diffusion.models.layers.attention import TorchAttention, LinearAttention, JaxAttention
from diffusion.models.layers.ffn import Mlp


def bmm_flops(x_shape, y_shape, backward=False, unit=1):
    bs, M, K = x_shape[:-2], x_shape[-2], x_shape[-1]
    bs_, K_, N = y_shape[:-2], y_shape[-2], y_shape[-1]
    assert list(bs) == list(bs_), f"{bs}, {bs_}"
    assert K == K_, f"{K}, {K_}"
    flops = float(jnp.prod(bs)) * M * N * (2 * K - 1) / unit
    if backward:
        flops *= 3
    out_shape = jnp.array([*bs, M, N]) 
    return out_shape, flops

def softmax_flops(x_shape, axis=-1, backward=False, unit=1):
    # NOTE: very proximate
    return x_shape, float(jnp.prod(x_shape)) * 3 / unit

def swish_flops(x_shape, backward=False, unit=1):
    return float(jnp.prod(x_shape)) * 5 / unit


def dense_flops(shape, layer: nn.Dense, backward=False, unit=1):
    assert isinstance(layer, nn.Dense), f"{type(layer)}"
    """(...., n) @ (...., n, m) -> (..., m)"""
    bs_cnt = jnp.prod(shape[:-1])
    n = shape[-1] 
    m = layer.features
    flops_per_pnt = float(m) * (2 * n - 1) / unit
    if layer is None or layer.use_bias:
        flops_per_pnt += float(m) / unit
        
    flops = bs_cnt * flops_per_pnt
    if backward:
        flops *= 3
    out_shape = jnp.array([*shape[:-1], m], dtype=jnp.float32)
    return out_shape, flops
 
def groupnorm_flops(shape, layer: nn.GroupNorm, backward=False, unit=1):
    # Number of groups
    G = layer.num_groups
    B, H, W, C = shape

    mean_flops = (jnp.prod(shape) + B*G ) / unit
    std_flops = (jnp.prod(shape) * 2 + B*G*2) / unit
    norm_flops = jnp.prod(shape) * 2 / unit
    flops = mean_flops + std_flops + norm_flops
    if layer.use_bias:
        flops += jnp.prod(shape) / unit
    if layer.use_scale:
        flops += jnp.prod(shape) / unit

    if backward:
        flops *= 3
    return shape, flops      
 
def ln_flops(shape, layer: nn.LayerNorm, backward=False, unit=1):
    # sum x / cnt
    mean_flops = (jnp.prod(shape) + shape[0]) / unit
    # sqrt{sum (x-mu)**2 / cnt}
    std_flops = (jnp.prod(shape) * 2 + shape[0] * 2) / unit
    norm_flops = jnp.prod(shape) * 2 / unit
    flops = mean_flops + std_flops + norm_flops
    if layer is None or layer.use_bias:
        flops += jnp.prod(shape) / unit
    if layer is None or layer.use_scale:
        flops += jnp.prod(shape) / unit
    
    if backward:
        flops *= 3 
    return shape, flops


def self_attn_flops(shape, attn: TorchAttention, backward=False, unit=1):
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}
  
    B, N, C = shape

    _, flops_ = dense_flops(shape, attn.qkv, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_

    assert not attn.qk_norm, "Not supported."
    q_shape = jnp.array([B, attn.num_heads, N, attn.head_dim])
    kt_shape = jnp.array([B, attn.num_heads, attn.head_dim, N])
    
    qkt_shape, flops_ = bmm_flops(
        q_shape, kt_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    
    attn_shape, flops_ = softmax_flops(
        qkt_shape, axis=-1, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    
    v_shape = jnp.array([B, attn.num_heads, N, attn.head_dim])
    (B, H_, N_, D_), flops_ = bmm_flops(
        attn_shape, v_shape, backward=backward, unit=unit)
    flops += flops_
    assert N == N_ and D_ == attn.head_dim and H_ == attn.num_heads, "Shape check"
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    x_shape, flops_ = dense_flops(
        jnp.array([B, N, C]), attn.proj, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_
    
    return x_shape, flops, block_attn_flops_dict

def jax_attn_flops(shape, attn: JaxAttention, backward=False, unit=1):
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}
  
    B, N, C = shape
    # print(shape)

    qkv = nn.Dense(
        features=attn.dim * 3, 
        use_bias=attn.qkv_bias,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros
    )
    _, flops_ = dense_flops(shape, qkv, backward=backward, unit=unit) #
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_

    head_dim = attn.dim // attn.num_heads
    assert not attn.qk_norm, "Not supported."
    q_shape = jnp.array([B, attn.num_heads, N, head_dim]) 
    kt_shape = jnp.array([B, attn.num_heads, head_dim, N]) 
    
    qkt_shape, flops_ = bmm_flops(
        q_shape, kt_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    
    attn_shape, flops_ = softmax_flops(
        qkt_shape, axis=-1, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    
    v_shape = jnp.array([B, attn.num_heads, N, head_dim]) 
    (B, H_, N_, D_), flops_ = bmm_flops(
        attn_shape, v_shape, backward=backward, unit=unit)
    flops += flops_
    assert N == N_ and D_ == head_dim and H_ == attn.num_heads, "Shape check"
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    proj = nn.Dense(
        attn.dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros_init()
        )

    x_shape, flops_ = dense_flops(
        jnp.array([B, N, C]), proj, backward=backward, unit=unit) #
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_

    return x_shape, flops, block_attn_flops_dict

def linear_attn_flops(shape, attn: LinearAttention, backward=False, unit=1):
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}

    B, N, C = shape
    _, flops_ = dense_flops(shape, attn.qkv, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_
    assert not attn.qk_norm, "Not supported."
    q_shape = jnp.array([B, attn.num_heads, N, attn.head_dim])
    k_shape = jnp.array([B, attn.num_heads, N, attn.head_dim])
    kt_shape = jnp.array([B, attn.num_heads, attn.head_dim, N])
    v_shape = jnp.array([B, attn.num_heads, N, attn.head_dim])
    
    # Original code: k_v = jnp.einsum("...ki,...kj->...ij", k, v)
    kv_shape, flops_ = bmm_flops(kt_shape, v_shape, backward=backward, unit=unit) 
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    # Original code: numerator = jnp.einsum("...ik,...kj->...ij", q, k_v)
    _, flops_ = bmm_flops(q_shape, kv_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    if attn.normalizer == 'adaptive':
        # ORIGINAL CODE: sum_k = k.sum(axis=-2, keepdims=True)
        flops += float(jnp.prod(k_shape)) / unit
        block_attn_flops_dict['DiTBlockAttn-attnFlops'] += float(jnp.prod(k_shape)) / unit
         # ORIGINAL CODE: denominator = jnp.einsum("...ik,...jk->...ij", q, sum_k)
        _, flops_ = bmm_flops(q_shape, kt_shape, backward=backward, unit=unit)
        flops += flops_
        block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    elif attn.normalizer == "constant":
        pass
    else:
        raise NotImplemented
    # dividing 
    flops += float(jnp.prod(kv_shape)) / unit
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += float(jnp.prod(kv_shape)) / unit
    
    x_shape, flops_ = dense_flops(
        jnp.array([B, N, C]), attn.proj, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_
    
    return x_shape, flops


def mlp_flops(x_shape, layer: Mlp, backward=False, unit=1):
    flops = 0
    x_shape, flops_ = dense_flops(
        x_shape, layer.fc1, backward=backward, unit=unit)
    flops += flops_
    
    # TODO(guandao): activation, this takes more!
    flops += (jnp.prod(x_shape) * 3 if backward else jnp.prod(x_shape)) / unit
    
    x_shape, flops_ = dense_flops(
        x_shape, layer.fc2, backward=backward, unit=unit)
    flops += flops_
    return x_shape, flops


def conv2d_flops(x_shape, layer: nn.Conv, backward=False, unit=1):
    assert isinstance(layer, nn.Conv), f"{type(layer)}"

    B, H, W, C = x_shape
    C_out = layer.features
    k1, k2 = layer.kernel_size
    stride1, stride2 = layer.strides

    if type(layer.padding) == tuple:
        pad1, pad2 = layer.padding
        H_out = (H - k1 + pad1[0]+pad1[1]) // stride1 + 1
        W_out = (W - k2 + pad2[0]+pad2[1]) // stride2 + 1
    elif layer.padding == 'VALID':
        H_out = jnp.floor((H - k1) / stride1) +1
        W_out = jnp.floor((W - k2) / stride2) +1
    elif layer.padding == 'SAME':
        H_out = jnp.ceil(H / stride1)
        W_out = jnp.ceil(W / stride2)
    else:
        raise ValueError(f"padding for {layer.padding} not implemented")

    ###
    # flops_per_pnt = C_out * (2*C*k1*k2 -1) / unit
    # if layer.use_bias:
    #     flops_per_pnt += C_out / unit
    # flops = B * H_out * W_out * flops_per_pnt
    # out_shape = jnp.array([B, H_out, W_out, C_out], dtype=jnp.float32)
    ###

    out_shape, flops = bmm_flops(x_shape = jnp.array([B, H_out, W_out, C_out, k1*k2*C]),\
                                y_shape = jnp.array([B, H_out, W_out, k1*k2*C, 1]),\
                                backward = backward, unit=unit)
    
    if layer.use_bias:
        bias_flops = B * H_out * W_out * C_out / unit
        if backward:
            bias_flops *= 3
        flops += bias_flops
        
    out_shape = out_shape[:-1]

    return out_shape, flops
