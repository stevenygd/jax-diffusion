from diffusers.models import FlaxAutoencoderKL
from utils.vqvae_flops import compute_vqvae_flops
import jax
from jax import numpy as jnp
from utils.flops_utils import *

def compare_with_torch_attention():
    from models.layers.attention import TorchAttention
    from diffusers.models.vae_flax import FlaxAttentionBlock
    from utils.vae_flax_flops import flax_attn_block_flops
    import jax

    unit = 1e9
    unit_name = "GFlops"

    B = 64
    num_heads = 4
    C = 512 
    head_dim = C // num_heads # 128

    ### FlaxAttentionBlock ###
    x_shape = jnp.array([B,32,32,C])
    flax_attn = FlaxAttentionBlock(
        channels = C,
        num_head_channels = head_dim,
    )
    flax_attn = flax_attn.bind(
        flax_attn.init(jax.random.PRNGKey(0), jnp.ones(x_shape))
        )
    out_shape, flops, block_attn_flops_dict = flax_attn_block_flops(
        x_shape, flax_attn, backward=False, unit=unit, breakdown=True)
    print(f"flax_attn: {out_shape.astype(jnp.int_)}, {flops} {unit_name}")
    print('\t', {k:round(v.item(),4) for k,v in block_attn_flops_dict.items()}, '\n\n')

    ### TorchAttention ###
    x_shape = jnp.array([B,32*32,C])
    torch_attn = TorchAttention(
        dim = C,
        num_heads = num_heads,
        qkv_bias = True
    )
    torch_attn = torch_attn.bind(
        torch_attn.init(jax.random.PRNGKey(0), jnp.ones(x_shape), training=False, return_aux=False)
        )
    out_shape, flops, block_attn_flops_dict = self_attn_flops(x_shape, torch_attn, backward=False, unit=unit)
    print(f"torch_attn: {out_shape.astype(jnp.int_)}, {flops} {unit_name}")
    print('\t', {k:round(v.item(),4) for k,v in block_attn_flops_dict.items()}, '\n')
    
def compare_dense():
    import flax.linen as nn
    x_shape = jnp.array([1,32,32,512])
    dense3 = nn.Dense(features = 512 * 3, use_bias=True)
    out_shape, flops = dense_flops(x_shape, dense3, backward=False, unit=1)
    print(flops)
    import pdb; pdb.set_trace()
    dense = nn.Dense( features = 512,use_bias = True)
    out_shape, flops = dense_flops(x_shape, dense, backward=False, unit=1)
    print(flops)
    import pdb; pdb.set_trace()

def check_shapes():
    from diffusers.models.vae_flax import FlaxEncoder, FlaxDecoder
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        # This works
        "runwayml/stable-diffusion-v1-5",
        revision="flax",
        subfolder="vae",
        dtype=jnp.float32,
        from_pt=False
    )
    vae = vae.bind({'params':vae_params})
    x_shape = jnp.array([12, 256, 256, 3])
    x = jnp.ones(x_shape)
    hidden_states = vae.encoder(x)
    import pdb; pdb.set_trace()

    moments = vae.quant_conv(hidden_states)
    import pdb; pdb.set_trace()

    x_shape = jnp.array([12, 3, 256, 256])
    x = jnp.ones(x_shape)
    import pdb; pdb.set_trace()
    posterior = vae.encode(x, deterministic=True)
    hidden_states = posterior.latent_dist.mode()
    import pdb; pdb.set_trace()


def main():
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        # This works
        "runwayml/stable-diffusion-v1-5",
        revision="flax",
        subfolder="vae",
        dtype=jnp.float32,
        from_pt=False
    )
    vae = vae.bind({'params':vae_params})
    unit = 1e12
    unit_name = "TFlops"
    
    x_shape = jnp.array([12, 3, 256, 256])
    num_iters = 1_600_000
    print(f"train FLOPs (input size: {x_shape}):")
    train_flops = compute_vqvae_flops(
        x_shape, vae, "train", unit=unit)
    print(f"\t= {train_flops:.4f} {unit_name} x {num_iters/1e6} M iters = {train_flops*num_iters/1e6:.3f} M TFlops")

    x_shape = jnp.array([1, 3, 256, 256])
    num_images = 1_200_000
    print(f"\ndata_prep FLOPs (input size: {x_shape}):")
    data_prep_flops = compute_vqvae_flops(
        x_shape, vae, "data_prep", unit=unit)
    print(f"\t= {data_prep_flops:.4f} {unit_name} x {num_images/1e6} M images = {data_prep_flops*num_images/1e6:.3f} M TFlops")

    print(f"\n== Total FLOPs: {train_flops*num_iters/1e6 + data_prep_flops*num_images/1e6:.3f} M TFlops")

if __name__ == "__main__":
    # compare_with_torch_attention()
    # check_shapes()
    main()