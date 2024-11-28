import jax.numpy as jnp
from diffusion.models.layers.patch_embed import PatchEmbed, SimpleDiffusionEmbedder, SimpleDiffusionResBlock, SimpleDiffusionUnpacify
from diffusion.utils.flops_utils import *

# identical to x_emb_flops at dit_flops.py
def patchemb_flops(x_shape, layer: PatchEmbed, backward=False, unit=1):
    ps = layer.patch_size
    B, C, H, W = x_shape
    H, W = H // ps, W // ps
    D = layer.embed_dim
    flops = (B * H * W) * (C * D * ps ** 2) / unit
    if backward:
        flops *= 3
    x_shape = jnp.array([B, H * W, D], dtype=jnp.float32)
    return x_shape, flops

def simpdiff_emb_flops(x_shape, c_shape, layer: SimpleDiffusionEmbedder, backward=False, unit=1):
    assert isinstance(layer, SimpleDiffusionEmbedder)
    flops = 0 
    B, C, H, W = x_shape
    D = layer.embed_dim # 384

    num_res_blocks = layer.num_res_blocks # (2, 2, 2)
    channel_multiplier = layer.channel_multiplier # (1, 2, 4)
    base_channel = layer.base_channel # 32
    n_downs = int(jnp.log(float(layer.patch_size)) / jnp.log(2.)) # 2

    h_shape = jnp.array([B, W, H, C], dtype=jnp.float32) # (256, 256, 256, 3)
    
    if n_downs > 0:
        h_conv = nn.Conv(base_channel * channel_multiplier[0],
                         (3,3), (1,1), 'SAME')
        h_shape, flops_ = conv2d_flops(h_shape, h_conv, backward=backward, unit=unit)
        flops += flops_
    # print(flops_, flops, h_shape) -> h_shape (256,256,256,32)

    for i in range(n_downs):
        num_blocks = num_res_blocks[i]
        channel_mul = channel_multiplier[i]
        for _ in range(num_blocks):
            simpdiff_resblock = SimpleDiffusionResBlock(base_channel * channel_mul)
            h_shape, flops_ = simpdiff_resblock_flops(h_shape, c_shape, simpdiff_resblock, backward=backward, unit=unit)
            flops += flops_
            
        # Average polling
        h_shape = jnp.array([
            h_shape[0], h_shape[1] // 2, h_shape[2] // 2, h_shape[3]
        ], dtype=jnp.float32)
        flops += jnp.prod(h_shape) * (3 if  backward else 1) / unit
        
        # Final layer
        if i <  n_downs - 1:
            h_dense = nn.Dense(base_channel * channel_multiplier[i+1])
            h_shape, flops_ = dense_flops(h_shape, layer=h_dense, backward=backward, unit=unit)
            flops += flops_
        # print(flops_, flops, h_shape)
    # h_shape (256,64,64,128)

    h_patch_embed = PatchEmbed(patch_size=1, embed_dim = D)
    h_shape = jnp.array([h_shape[0], h_shape[3], h_shape[1], h_shape[2]], dtype=jnp.float32)
    h_shape, flops_ = patchemb_flops(h_shape, h_patch_embed, backward=backward, unit=unit)
    flops += flops_

    return h_shape, flops

def simpdiff_resblock_flops(h_shape, c_shape, layer: SimpleDiffusionResBlock, backward=False, unit=1):
    assert isinstance(layer, SimpleDiffusionResBlock)
    flops = 0
    
    h_shape, flops_ = ln_flops(h_shape, layer=None, backward=backward, unit=unit)
    flops += flops_
    
    flops += swish_flops(h_shape, backward=backward, unit=unit) 
    
    h_conv = nn.Conv(layer.dim, (3,3), (1,1), 'SAME')
    h_shape, flops_ = conv2d_flops(h_shape, h_conv, backward=backward, unit=unit)
    flops += flops_

    h_dense = nn.Dense(layer.dim*2)
    _, flops_ = dense_flops(c_shape, layer=h_dense, backward=backward, unit=unit)
    flops += flops_

    h_shape, flops_ = ln_flops(h_shape, layer=None, backward=backward, unit=unit)
    flops += flops_

    flops += swish_flops(h_shape, backward=backward, unit=unit)

    h_shape, flops_ = conv2d_flops(h_shape, h_conv, backward=backward, unit=unit)
    flops += flops_

    flops += jnp.prod(h_shape)*(3 if backward else 1) / unit

    return h_shape, flops

def simpdiff_unpatchify_flops(x_shape, c_shape, layer: SimpleDiffusionUnpacify, backward=False, unit=1):
    assert isinstance(layer, SimpleDiffusionUnpacify)
    flops = 0
    B, T, D = x_shape
    n_ups = int(jnp.log(float(layer.patch_size))/jnp.log(2.))

    x_shape = jnp.array([B, T**0.5, T**0.5, D], dtype=jnp.float32)

    for i in range(n_ups-1, -1, -1):
        num_blocks = layer.num_res_blocks[i]
        channel_mul = layer.channel_multiplier[i]
        
        # upsample
        x_shape = jnp.array([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]], dtype=jnp.float32)
        
        dense_layer = nn.Dense(layer.base_channel * channel_mul)
        x_shape, flops_ = dense_flops(x_shape, layer=dense_layer, backward=backward, unit=unit)
        flops += flops_

        for _ in range(num_blocks):
            simpdiff_resblock = SimpleDiffusionResBlock(layer.base_channel * channel_mul)
            x_shape, flops_ = simpdiff_resblock_flops(x_shape, c_shape, simpdiff_resblock, backward=backward, unit=unit)
            flops += flops_
    
    if layer.last_conv:
        conv_layer = nn.Conv(layer.out_channels, (3,3), (1,1), 'SAME')
        x_shape, flops_ = conv2d_flops(x_shape, conv_layer, backward=backward, unit=unit)
        flops += flops_
    else:
        dense_layer = nn.Dense(layer.out_channels)
        x_shape, flops_ = dense_flops(x_shape, layer=dense_layer, backward=backward, unit=unit)
        flops += flops_
    
    x_shape = jnp.array([x_shape[0], x_shape[3], x_shape[1], x_shape[2]], dtype=jnp.float32)

    return x_shape, flops