import jax.numpy as jnp
import flax.linen as nn
from diffusion.models.layers.patch_embed import (ConvNetEmbed, FlaxEncoder)
from diffusion.utils.flops_utils import *
from diffusion.utils.vae_flax_flops import *

def convnet_emb_flops(
    x_shape, c_shape, layer: ConvNetEmbed, backward=False, unit=1):
    assert isinstance(layer, ConvNetEmbed), f"{type(layer)}"
    flops = 0

    B, C, H, W = x_shape
    x_shape = jnp.array([B, H, W, C])
    
    x_shape, flops_ = flax_enc_flops(x_shape, layer.encoder, backward=backward, unit=unit)
    flops += flops_

    B, H, W, C = x_shape
    assert C == layer.embed_dim, f"{C}, {layer.embed_dim}"
    x_shape = jnp.array([B, H*W, layer.embed_dim])
    return x_shape, flops

def flax_enc_flops(
    x_shape, layer: FlaxEncoder, backward=False, unit=1):
    assert isinstance(layer, FlaxEncoder), f"{type(layer)}"
    flops = 0

    # in
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_in, backward=backward, unit=unit)
    flops += flops_

    # downsampling
    for i, down_block in enumerate(layer.down_blocks):
        x_shape, flops_ = flax_down_enc_block_flops(x_shape, down_block, backward=backward, unit=unit)
        flops += flops_

    # end - norm_out, swish, conv_out
    x_shape, flops_ = groupnorm_flops(x_shape, layer=layer.conv_norm_out, backward=backward, unit=unit)
    flops += flops_
    flops += swish_flops(x_shape, backward=backward, unit=unit)
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_out, backward=backward, unit=unit)
    flops += flops_

    return x_shape, flops
