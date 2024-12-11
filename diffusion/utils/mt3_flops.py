from diffusion.models.layers.attention import TTTAttention
# from diffusion.models.layers.mttt import MTTTMultiHeadSelfAttention, TTTEncoder 
# from diffusion.models.layers.ttt import TTTLMBiDirAttention, TTTBase
# from diffusion.models.layers.ttt import MTTTMultiHeadSelfAttention, TTTEncoder 
from diffusion.models.layers.ttt import TTTBase
from diffusion.utils.flops_utils import *


def mt3_attn_flops(
    shape, layer: TTTAttention, backward=False, unit=1):
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}

    B, N, C = shape
    # print(shape)
    _, flops_ = dense_flops(shape, layer.qkv, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_
    assert not layer.qk_norm, "Not supported."
    
    # MT3 attention layer
    _, flops_ = mt3_flops(shape, layer.attn, backward=backward, unit=unit)
    if layer.mttt_type in ['base', 'ar']:    
        flops += flops_
        block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    elif layer.mttt_type == 'bdar':
        flops += 2*flops_
        block_attn_flops_dict['DiTBlockAttn-attnFlops'] += 2*flops_
    else:
        raise ValueError(f"Undefined mttt_type: {layer.mttt_type}. Should be in ['base', 'ar', 'bdar']")
    
    # Final layer 
    x_shape, flops_ = dense_flops(
        jnp.array([B, N, C]), layer.proj, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_
    
    return x_shape, flops, block_attn_flops_dict


def mt3_flops(shape, layer: MTTTMultiHeadSelfAttention, backward=False, unit=1):
    flops = 0
    B, N = shape[:2]
    # Computer inner loop updates
    # total_iters = layer.n_epoch * layer.n_iters
    total_iters = layer.n_iters
    mini_bs = float(N * layer.n_epoch / layer.n_iters)
    train_data_shape_per_head = jnp.array([
        B, total_iters, mini_bs, layer.head_dim])
    # Backward=True to compute updating rule including SGD flops
    _, flops_ = mt3_enc_flops(
        train_data_shape_per_head, layer, backward=True, unit=unit)
    # TODO: does this need to multiply for backward of backward?
    flops += flops_ * layer.num_heads * (3 if backward else 1)
    
    # Compute testing view
    test_data_shape_per_head = jnp.array([B, layer.num_heads, N, layer.head_dim])
    # NOTE: backward = False to compute encoder layer forward
    shape, flops_ = mt3_enc_flops(
        test_data_shape_per_head, layer, backward=False, unit=unit)
    flops += flops_ * layer.num_heads * (3 if backward else 1)
    return shape, flops
    

def mt3_enc_flops(shape, layer: MTTTMultiHeadSelfAttention,
                  backward=False, unit=1):
    bs = shape[:-2]
    flops = 0.
    params = 0. # count parameter to compute SGD flops
    out_dim = layer.out_dim if layer.out_dim is not None else layer.head_dim
    hid_dim = layer.enc_dim if layer.enc_dim is not None else layer.head_dim
    for _ in range(layer.enc_layers- 1):
        inp_dim = shape[-1]
        dense = nn.Dense(hid_dim, use_bias=layer.enc_use_bias)
        shape, flops_ = dense_flops(shape, dense, backward=backward, unit=unit) 
        flops += flops_
        params += hid_dim * (inp_dim + (1 if dense.use_bias else 0))
        
        # TODO: Add activation flops (still not accurate)
        flops += jnp.prod(shape) * (3 if backward else 1) / unit
        
    inp_dim = shape[-1]
    dense = nn.Dense(out_dim, use_bias=layer.enc_use_bias)
    shape, flops_ = dense_flops(shape, dense, backward=backward, unit=unit) 
    flops += flops_
    params += out_dim * (inp_dim + (1 if dense.use_bias else 0))
    
    if layer.enc_ln:
        shape, flops_ = ln_flops(shape, None, backward=backward, unit=unit) 
        flops += flops_
        params += shape[-1] * 2 
        
    if layer.enc_residual:
        flops += jnp.prod(shape) * (3 if backward else 1) / unit
       
    # If including the backward pass 
    if backward:
        # Add loss compute ((x-y)**2).mean(), ~ 3 flops per unit
        flops += jnp.prod(shape) * (3 if backward else 1) * 3 / unit
    
        # Add SGD params
        flops += params * jnp.prod(bs) / unit
    return shape, flops


def mt3_lm_bdar_flops(x_shape, layer: TTTLMBiDirAttention, backward=False, unit=1):
    assert isinstance(layer, TTTLMBiDirAttention), f"{type(layer)}"
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}

    B, N, C = x_shape

    _, flops_ = dense_flops(
        x_shape, layer.qkv, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_
    
    if hasattr(layer, 'ttt_forward') and hasattr(layer, 'ttt_backward'):
        tttlayer_ = layer.ttt_forward
        num_heads = tttlayer_.num_heads
    elif hasattr(layer, 'ttt_layer'):
        tttlayer_ = layer.ttt_layer
        num_heads = tttlayer_.num_heads // 2
    else:    
        raise ValueError(f"{layer.__class__} does not have ttt_forward/backward or ttt_layer.")

    attempts = 3
    for attempt in range(attempts):
        try:
            n_mini_batch = tttlayer_.n_mini_batch
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                raise e
    mini_bs = tttlayer_.mini_batch_size
    head_dim = tttlayer_.head_dim
    tttlayer = MTTTMultiHeadSelfAttention(
        head_dim = head_dim,
        num_heads = num_heads,
        n_iters = n_mini_batch,
        enc_layers=2,
        enc_dim=head_dim * 4,
    )
    dense_layer = nn.Dense(features=1, use_bias=True) # for eta
    ##  Forward
    # eta flops: eta = ttt.get_eta(X)
    _, flops_ = dense_flops(jnp.array([B, n_mini_batch, mini_bs, C]), dense_layer, backward=backward, unit=unit)
    flops_ = num_heads * flops_
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_
    
    _, flops_ = mt3_flops(x_shape, tttlayer, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## Backward
    _, flops_ = dense_flops(jnp.array([B, n_mini_batch, mini_bs, C]), dense_layer, backward=backward, unit=unit)
    flops_ = num_heads * flops_
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    _, flops_ = mt3_flops(x_shape, tttlayer, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    ## combine forward and backward (multiplication)
    flops_ = jnp.prod(x_shape) *(3 if backward else 1) / unit
    flops += flops_ 
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    x_shape, flops_ = dense_flops(x_shape, layer.proj, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_

    return x_shape, flops, block_attn_flops_dict
