import jax.numpy as jnp
import flax.linen as nn
from diffusers.models.vae_flax import (
    FlaxResnetBlock2D, FlaxDownEncoderBlock2D, FlaxDownsample2D,
    FlaxUNetMidBlock2D, FlaxUpDecoderBlock2D, FlaxUpsample2D, 
    FlaxAttentionBlock
)
from diffusion.utils.flops_utils import *

def flax_resnet_block_flops(
    x_shape, layer: FlaxResnetBlock2D, backward=False, unit=1):
    assert isinstance(layer, FlaxResnetBlock2D), f"{type(layer)}"
    flops = 0

    # FLOPs for norm1, swish, and conv1
    x_shape, flops_ = groupnorm_flops(x_shape, layer=layer.norm1, backward=backward, unit=unit)
    flops += flops_

    flops += swish_flops(x_shape, backward=backward, unit=unit)
    
    x_shape, conv1_flops = conv2d_flops(x_shape, layer=layer.conv1, backward=backward, unit=unit)
    flops += conv1_flops

    # FLOPs for norm2, swish, and conv2
    x_shape, flops_ = groupnorm_flops(x_shape, layer=layer.norm2, backward=backward, unit=unit)
    flops += flops_
    flops += swish_flops(x_shape, backward=backward, unit=unit)
    x_shape, conv2_flops = conv2d_flops(x_shape, layer=layer.conv2, backward=backward, unit=unit)
    flops += conv2_flops

    # FLOPs for shortcut if applicable
    if layer.conv_shortcut is not None:
        x_shortcut_shape, shortcut_flops = conv2d_flops(x_shape, layer=layer.conv_shortcut, backward=backward, unit=unit)
        flops += shortcut_flops
        assert (x_shape == x_shortcut_shape).all(), f"{x_shape} {x_shortcut_shape}"
    
    flops += float(jnp.prod(x_shape)) / unit
    if backward:
        flops += 2*float(jnp.prod(x_shape)) / unit

    return x_shape, flops

def flax_unet_midblock2d_flops(
    x_shape, layer: FlaxUNetMidBlock2D, backward=False, unit=1):
    assert isinstance(layer, FlaxUNetMidBlock2D), f"{type(layer)}"
    flops = 0

    x_shape, flops_ = flax_resnet_block_flops(x_shape, layer=layer.resnets[0], backward=backward, unit=unit)
    for i in range(layer.num_layers):
        x_shape, flops_ = flax_resnet_block_flops(x_shape, layer=layer.resnets[i+1], backward=backward, unit=unit)
        flops += flops_
        x_shape, flops_ = flax_attn_block_flops(x_shape, layer=layer.attentions[i], backward=backward, unit=unit)
        flops += flops_

    return x_shape, flops

def flax_attn_block_flops(hidden_shape, layer: FlaxAttentionBlock, backward=False, unit=1, breakdown=False):
    assert isinstance(layer, FlaxAttentionBlock), f"{type(layer)}"
    flops = 0
    block_attn_flops_dict = {'DiTBlockAttn-kqvFlops': 0, 'DiTBlockAttn-attnFlops': 0, 'DiTBlockAttn-outFlops': 0}
  
    # batch, height, width, channels = hidden_states.shape
    B, H, W, C = hidden_shape
    # hidden_states = self.group_norm(hidden_states)
    hidden_shape, flops_ = groupnorm_flops(hidden_shape, layer=layer.group_norm, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_

    # hidden_states = hidden_states.reshape((batch, height * width, channels))
    hidden_shape = jnp.array([B, H*W, C], dtype=jnp.float32)
    
    # query = self.query(hidden_states)
    _, flops_ = dense_flops(hidden_shape, layer=layer.query, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_
    # key = self.key(hidden_states)
    _, flops_ = dense_flops(hidden_shape, layer=layer.key, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_
    # value = self.value(hidden_states)
    _, flops_ = dense_flops(hidden_shape, layer=layer.value, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-kqvFlops'] += flops_

    # query = self.transpose_for_scores(query)
    q_shape = jnp.array([B, layer.num_heads, H*W, C/layer.num_heads], dtype=jnp.float32)
    # key = self.transpose_for_scores(key)
    k_shape = jnp.array([B, layer.num_heads, C/layer.num_heads, H*W], dtype=jnp.float32)
    # value = self.transpose_for_scores(value)
    v_shape = jnp.array([B, layer.num_heads, H*W, C/layer.num_heads], dtype=jnp.float32)

    # scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads)) - ignored
    # attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
    attn_weight_shape, flops_ = bmm_flops(q_shape, k_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    # attn_weights = nn.softmax(attn_weights, axis=-1)
    attn_weight_shape, flops_ = softmax_flops(attn_weight_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    # hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
    hidden_shape, flops_ = bmm_flops(attn_weight_shape, v_shape, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-attnFlops'] += flops_

    # hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
    # new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
    # hidden_states = hidden_states.reshape(new_hidden_states_shape)
    hidden_shape = jnp.array([hidden_shape[0], hidden_shape[2], layer.channels], dtype=jnp.float32)

    # hidden_states = self.proj_attn(hidden_states)
    hiden_shape, flops_ = dense_flops(hidden_shape, layer=layer.proj_attn, backward=backward, unit=unit)
    flops += flops_
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_

    # hidden_states = hidden_states.reshape((batch, height, width, channels))
    hidden_shape = jnp.array([B, H, W, C], dtype=jnp.float32)

    # hidden_states = hidden_states + residual
    flops += float(jnp.prod(hidden_shape)) / unit
    if backward:
        flops += 2*float(jnp.prod(hidden_shape)) / unit
    block_attn_flops_dict['DiTBlockAttn-outFlops'] += flops_

    if breakdown:
        return hidden_shape, flops, block_attn_flops_dict
    return hidden_shape, flops

def flax_up_dec_block_flops(x_shape, layer: FlaxUpDecoderBlock2D, backward=False, unit=1):
    assert isinstance(layer, FlaxUpDecoderBlock2D), f"{type(layer)}"
    flops = 0

    for i, resnet in enumerate(layer.resnets):
        x_shape, flops_ = flax_resnet_block_flops(x_shape, layer=resnet, backward=backward, unit=unit)
        flops += flops_

    if layer.add_upsample:
        x_shape, flops_ = flax_upsample_flops(x_shape, layer=layer.upsamplers_0, backward=backward, unit=unit)
        flops += flops_

    return x_shape, flops

def flax_down_enc_block_flops(
    x_shape, layer: FlaxDownEncoderBlock2D, backward=False, unit=1):
    assert isinstance(layer, FlaxDownEncoderBlock2D), f"{type(layer)}"
    flops = 0

    for i, resnet in enumerate(layer.resnets):
        x_shape, flops_ = flax_resnet_block_flops(x_shape, layer=resnet, backward=backward, unit=unit)
        flops += flops_

    if layer.add_downsample:
        x_shape, flops_ = flax_downsample_flops(x_shape, layer=layer.downsamplers_0, backward=backward, unit=unit)
        flops += flops_

    return x_shape, flops

def flax_upsample_flops(x_shape, layer: FlaxUpsample2D, backward=False, unit=1):
    assert isinstance(layer, FlaxUpsample2D), f"{type(layer)}"
    flops = 0

    # hidden_states = jax.image.resize(hidden_states, shape=(batch, height * 2, width * 2, channels),method="nearest")
    B, H, W, C = x_shape
    x_shape = jnp.array([B, H*2, W*2, C])
    
    # hidden_states = self.conv(hidden_states)
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv, backward=backward, unit=unit)
    flops += flops_

    return x_shape, flops

def flax_downsample_flops(
    x_shape, layer: nn.Conv, backward=False, unit=1):
    assert isinstance(layer, FlaxDownsample2D), f"{type(layer)}"
    flops = 0

    x_shape = jnp.array([x_shape[0], x_shape[1]+1, x_shape[2]+1, x_shape[3]])
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv, backward=backward, unit=unit)
    flops += flops_
    return x_shape, flops
