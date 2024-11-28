# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: 
https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import jax
import hydra
import jax.numpy as jnp
from main import create_train_state
from omegaconf import DictConfig, OmegaConf
from utils import dit_flops 
from pprint import pprint

def multi():
    return jax.device_count() > jax.local_device_count()

def is_main():
    return jax.process_index() == 0

def compute_flops_per_iter(args, backward, unit):
    """Run flop computing. """
    rng = jax.random.PRNGKey(args.global_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
   
    use_latent = args.get("use_latent", True)
    if hasattr(args, "image_size"): latent_size = args.image_size // 8 if use_latent else args.image_size
    else: _, _, _, latent_size, _ = args["data_shape"]["x"]

    if hasattr(args, "latent_dim"): latent_dim = args.latent_dim
    else: _, _, latent_dim, _, _ = args["data_shape"]["x"]
    data = (
        jnp.ones((1, latent_dim, latent_size, latent_size)),   # x
        jnp.ones((1,), dtype=jnp.int32)                        # y
    )
    rng, spl = jax.random.split(rng)
    ema_state, model = create_train_state(spl, data, args, return_model=True)

    x_shape = jnp.array(
        [args.global_batch_size, latent_dim, latent_size, latent_size], 
        dtype=jnp.float32) 
    t_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)
    y_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)

    unit, unit_name = 1e12, "TFlops"
    flops_per_iter, block_flops_dict = dit_flops.dit_flops(
        x_shape, y_shape, t_shape, model.bind(ema_state.ema_params), 
        backward=backward, unit=unit) 

    return flops_per_iter, block_flops_dict

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig):
    """Run flop computing. """

    assert args.inference.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.inference.cfg_scale > 1.0

    unit, unit_name = 1e12, "TFlops"
    
    # args.model.dit_block_kwargs = {'mlp_type':'mlp'}
    inf_flops_per_iter, block_flops_dict = compute_flops_per_iter(args, backward=False, unit=unit)

    print(f"Forward")
    print(f"\ttotal flops per iteration: {inf_flops_per_iter} {unit_name}")
    # print(f"\tblock flops dict: {block_flops_dict}")
    print(f"\tblock flops dict: ")
    pprint(block_flops_dict)
    # print("Backward", train_flops_per_iter, unit_name)
    # print(train_flops_per_iter, unit_name, inf_flops_per_iter, unit_name)
    

if __name__ == "__main__":
    main()
