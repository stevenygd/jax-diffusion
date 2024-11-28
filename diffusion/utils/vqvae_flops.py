import jax
import jax.numpy as jnp
from diffusers.models import FlaxAutoencoderKL
from diffusers.models.vae_flax import FlaxEncoder, FlaxDecoder
from utils.flops_utils import *
from utils.vae_flax_flops import *

def compute_vqvae_flops(shape, vae:FlaxAutoencoderKL, phase, unit=1):
    assert isinstance(vae, FlaxAutoencoderKL), f"{type(vae)}"
    assert phase in ["train", "data_prep", "inference"], f"{phase}"

    flops = 0
    if phase == "train":
        flops_, posterior_shape = vqvae_encode_flops(shape, vae, backward=True, unit=unit)
        flops += flops_
        print(f"\t+ {flops_:.4f} encoder FLOPs per iter (in: {shape} - out: {posterior_shape})")

        hidden_states_shape = posterior_shape
        hidden_states_shape = posterior_shape.at[-1].set(posterior_shape[-1] // 2)

        flops_, sample_shape = vqvae_decode_flops(hidden_states_shape, vae, backward=True, unit=unit)
        flops += flops_
        print(f"\t+ {flops_:.4f} decoder FLOPs per iter (in: {hidden_states_shape} - out: {sample_shape})")
        assert (sample_shape == shape).all(), f"{sample_shape} != {shape}, input and output shape should be the same"

    elif phase == "data_prep":
        flops, posterior_shape = vqvae_encode_flops(shape, vae, backward=False, unit=unit)
        print(f"\t+ {flops:.4f} encoder FLOPs per iter (in: {shape} - out: {posterior_shape})")
    
    else:
        flops, hidden_states_shape = vqvae_decode_flops(shape, vae, backward=False, unit=unit)
    
    return flops

def vqvae_encode_flops(x_shape, vae: FlaxAutoencoderKL, backward=False, unit=1):
    flops = 0

    # shape change after jnp.transpose(x, (0, 2, 3, 1))
    x_shape = jnp.array([x_shape[0], x_shape[2], x_shape[3], x_shape[1]], dtype=jnp.float32)

    # hidden_states = self.encoder(sample, deterministic=deterministic)
    x_shape, flops_ = flax_enc_flops(x_shape, layer=vae.encoder, backward=backward, unit=unit)
    flops += flops_

    # moments = self.quant_conv(hidden_states)
    x_shape, flops_ = conv2d_flops(x_shape, layer=vae.quant_conv, backward=backward, unit=unit)
    flops += flops_

    # posterior = FlaxDiagonalGaussianDistribution(moments)
    flops += 2*jnp.prod(x_shape) / unit
    if backward:
        flops += 2*jnp.prod(x_shape) / unit
    #FlaxAutoencoderKLOutput(latent_dist=posterior)
    return flops, x_shape

def vqvae_decode_flops(x_shape, vae: FlaxAutoencoderKL, backward=False, unit=1):
    flops = 0

    if x_shape[-1] != vae.config.latent_channels:
        # shape change after jnp.transpose(latents, (0, 2, 3, 1))
        x_shape = jnp.array([x_shape[0], x_shape[2], x_shape[3], x_shape[1]], dtype=jnp.float32)
    
    # hidden_states = self.post_quant_conv(latents)
    x_shape, flops_ = conv2d_flops(x_shape, layer=vae.post_quant_conv, backward=backward, unit=unit)
    flops += flops_

    # hidden_states = self.decoder(hidden_states, deterministic=deterministic)
    x_shape, flops_ = flax_dec_flops(x_shape, layer=vae.decoder, backward=backward, unit=unit)
    flops += flops_

    # shape change after jnp.transpose(hidden_states, (0, 3, 1, 2))
    x_shape = jnp.array([x_shape[0], x_shape[3], x_shape[1], x_shape[2]], dtype=jnp.float32)

    # FlaxDecoderOutput(sample=hidden_states)
    return flops, x_shape

def flax_enc_flops(x_shape, layer: FlaxEncoder, backward=False, unit=1):
    assert isinstance(layer, FlaxEncoder), f"{type(layer)}"
    flops = 0

    # in
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_in, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_in {flops_:.4f} ({x_shape})")

    # downsampling
    for i, down_block in enumerate(layer.down_blocks):
        x_shape, flops_ = flax_down_enc_block_flops(x_shape, down_block, backward=backward, unit=unit)
        flops += flops_
        # print(f"down_block {i} (in={down_block.in_channels}, out={down_block.out_channels}): {flops_:.4f} ({x_shape})")

    # middle
    x_shape, flops_ = flax_unet_midblock2d_flops(x_shape, layer=layer.mid_block, backward=backward, unit=unit)
    flops += flops_
    # print(f"mid_block: {flops_:.4f} ({x_shape})")

    # end - norm_out, swish, conv_out
    x_shape, flops_ = groupnorm_flops(x_shape, layer=layer.conv_norm_out, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_norm_out: {flops_:.4f} ({x_shape})")

    flops += swish_flops(x_shape, backward=backward, unit=unit)
    # print(f"swish: {swish_flops(x_shape, backward=backward, unit=unit):.4f}")

    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_out, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_out: {flops_:.4f} ({x_shape})")
    # print(f"-> flax_enc_flops: {flops:.4f}\n\n")

    return x_shape, flops

def flax_dec_flops(x_shape, layer: FlaxDecoder, backward=False, unit=1):
    assert isinstance(layer, FlaxDecoder), f"{type(layer)}"
    flops = 0

    # z to block_in
    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_in, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_in {flops_:.4f} ({x_shape})")

    # middle
    x_shape, flops_ = flax_unet_midblock2d_flops(x_shape, layer=layer.mid_block, backward=backward, unit=unit)
    flops += flops_
    # print(f"mid_block: {flops_:.4f} ({x_shape})")

    # upsampling
    for i, up_block in enumerate(layer.up_blocks):
        x_shape, flops_ = flax_up_dec_block_flops(x_shape, up_block, backward=backward, unit=unit)
        flops += flops_
        # print(f"up_block {i} (in={up_block.in_channels}, out={up_block.out_channels}): {flops_:.4f} ({x_shape})")
    
    # end - norm_out, swish, conv_out
    x_shape, flops_ = groupnorm_flops(x_shape, layer=layer.conv_norm_out, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_norm_out: {flops_:.4f} ({x_shape})")

    flops += swish_flops(x_shape, backward=backward, unit=unit)
    # print(f"swish: {swish_flops(x_shape, backward=backward, unit=unit):.4f}")

    x_shape, flops_ = conv2d_flops(x_shape, layer=layer.conv_out, backward=backward, unit=unit)
    flops += flops_
    # print(f"conv_out: {flops_:.4f} ({x_shape})")
    # print(f"-> flax_dec_flops: {flops:.4f}\n\n")

    return x_shape, flops

