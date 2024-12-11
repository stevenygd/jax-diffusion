# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP/DDI.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: 
https://github.com/openai/guided-diffusion/tree/main/evaluations

This file samples only from a single checkpoint. For multi-ckpt sampling, see sample_ddp_jax.py.
"""
import os
import jax
import glob
import hydra
import flax
import orbax
from flax.training import orbax_utils
import shutil
import numpy as np
from PIL import Image
import flax.jax_utils
import os.path as osp
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial
import orbax.checkpoint as ocp
from diffusion.evaluation import adm_eval
from diffusion.losses import create_diffusion
from diffusers.models import FlaxAutoencoderKL
from omegaconf import DictConfig, OmegaConf
# from diffusion.utils import dit_flops 
from diffusion.utils import train_utils
import wandb

def create_train_state_no_mesh(rng, data, args, logging=None, return_model=False):
    # model_name, latent_size, num_classes, latent_dim):

    model=train_utils.create_model(args)
    
    x, y = data
    # NOTE: this is taking the data of one device
    x = jnp.array(x).reshape(-1, *x.shape[-3:])
    y = jnp.array(y).reshape(-1)
    t = jnp.ones(shape=y.shape, dtype=jnp.float32)
    rng, spl = jax.random.split(rng)
    spl1, spl2, spl3, spl4 = jax.random.split(spl, 4)
    params = model.init(
        {'params': spl1, 'dropout': spl2, "label_emb": spl3, "mt3": spl4},
        x, t, y, training=True)
    tx = train_utils.make_opt(args)
    ema_state = train_utils.EMATrainState.create(
        apply_fn=model.apply, params=params, tx=tx, ema_decay=args.ema_decay)
    if return_model:
        return ema_state, model
    return ema_state


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    flst = glob.glob(f"{sample_dir}/*.png")
    pbar = tqdm(total=num, desc="Building .npz file from samples", unit="sample")
    for i in range(len(flst)):
        try:
            sample_pil = Image.open(flst[i])
        except Exception as e:
            print(f"Error loading {flst[i]}: {e}")
            os.remove(flst[i])
            continue
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
        pbar.update(1)
        if pbar.n == num:
            break
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def create_sample_dir(args, ckpt_step):
    # Create folder to save samples:
    model_string_name = args.model.name.replace("/", "-")
    folder_name = f"{model_string_name}-"\
                  f"size-{args.image_size}-" \
                  f"cfg-{args.inference.cfg_scale}-" \
                  f"seed-{args.global_seed}-" \
                  f"step-{args.inference.num_sampling_steps}-" \
                  f"nsmp-{args.inference.num_fid_samples}"
    sample_folder_dir = osp.join(
        args.resume, args.inference.sample_dir, str(ckpt_step), folder_name)
    return sample_folder_dir


def sample_checkpoint(args, rng, ema_state, checkpoint_manager, ckpt_step, 
                      p_sample_latents, flops_per_iter=None, flops_unit_name=None):
    
    pid = jax.process_index()
    
    sample_folder_dir = create_sample_dir(args, ckpt_step)
    if args.inference.remove_existing_sample_dir:
        shutil.rmtree(sample_folder_dir, ignore_errors=True)
    
    output_metrics_path = f"{sample_folder_dir}-metrics.npy"
    metrics = None 
    if osp.isfile(output_metrics_path):
        print(f"Metric file exist {output_metrics_path}")
        try:
            metrics = np.load(output_metrics_path, allow_pickle=True).item()
            print("Metrics", metrics)
        except Exception as e:
            print("Exception in loading metrics", e)
            os.remove(output_metrics_path)
            print(f"Deleted: {output_metrics_path}")
    else:
        print(f"Metric file doesn't exist {output_metrics_path}")
    
    output_npz_path = f"{sample_folder_dir}.npz"
    if metrics is None and not osp.isfile(output_npz_path):
        print(f"metrics is None: {metrics is None}, output_npz_path exists: {osp.isfile(output_npz_path)} => sampling")
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        
        # Resume model
        abstract_state = {"state": ema_state, 'step': -1}
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.local_devices(), ('x',)),
            jax.sharding.PartitionSpec(),
        )
        restore_args = jax.tree_util.tree_map(
            lambda _: ocp.ArrayRestoreArgs(sharding=sharding), abstract_state)
        ckpt = checkpoint_manager.restore(
            ckpt_step, args=ocp.args.PyTreeRestore(
                abstract_state, restore_args=restore_args
            )
        )
        train_step = int(ckpt['step'])
        print("Restored")
        params = ckpt["state"].ema_params
        if not isinstance(params, flax.core.FrozenDict):
            params = flax.core.FrozenDict(params)
        params = flax.jax_utils.replicate(params)
        print("Obtained params")

        # Find out how many images are there
        total_devices = jax.device_count() 
        num_exist_imgs = len(glob.glob(f"{sample_folder_dir}/*.png"))
        print(f"Total number of images already exists: {num_exist_imgs}")

        sample_per_iter = total_devices*args.inference.per_proc_batch_size
        total_iter = int(np.ceil(args.inference.num_fid_samples / sample_per_iter))
        done_iter = num_exist_imgs // sample_per_iter

        print(f"Starting iteration {done_iter}/{total_iter}")
        for i in tqdm(range(done_iter, total_iter), unit="iter", desc="Sampling"):
            rng_i = jax.random.fold_in(rng, i)
            rng_i = jax.random.split(rng_i, jax.local_device_count())
            samples = p_sample_latents(rng_i, params)
            samples = np.array(samples)
            samples = samples.reshape(-1, *samples.shape[-3:])
            for j in tqdm(range(samples.shape[0]), leave=False):
                sample = np.array(samples[j]).astype(jnp.uint8)
                sample_save_path = f"{sample_folder_dir}/pid{pid}-{i*sample_per_iter + j:06d}.png"
                Image.fromarray(sample).save(sample_save_path)

        if is_main():
            print("Extract npz")
            output_npz_path = create_npz_from_sample_folder(
                sample_folder_dir, args.inference.num_fid_samples)
            print("Done with npz.")
    
    if is_main() and metrics is None:
        assert osp.isfile(args.inference.ref_batch), f"Ref batch {args.inference.ref_batch} does not exist."
        print("reference batch", args.inference.ref_batch)
        print("output_npz_path", output_npz_path)
        metrics = adm_eval._run_eval_(
            ref_batch = args.inference.ref_batch,
            sample_batch =output_npz_path, 
            batch_size=args.inference.get("adm_eval_batch_size", 1024)
        )
        if args.inference.num_sampling_steps == 256: # the default value
            key = f"eval-{args.inference.num_fid_samples}"
        else:
            key = f"eval-{args.inference.num_fid_samples}-{args.inference.num_sampling_steps}"

        metrics = {
            f"{key}/eval_steps": train_step,
            **{f"{key}/{k}": float(v) for k, v in metrics.items()}
        }
        if flops_per_iter is not None:
            metrics[f"{key}/train_flops_{flops_unit_name}"] = float(train_step * flops_per_iter)
        
        print("Metrics", metrics)
        np.save(output_metrics_path, metrics)
        print("Save", output_metrics_path)
        
    assert metrics is not None
    metrics = {
        "/".join([n.strip() for n in k.split("/")]): v
        for k, v in metrics.items()
    }
    metrics["ckpt_step"] = ckpt_step
    print("=" * 80)
    print("Metrics", metrics)
    print("=" * 80)
    print(f"[{pid}] Done.")

    return metrics


def multi():
    return jax.device_count() > jax.local_device_count()


def is_main():
    return jax.process_index() == 0


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args: DictConfig):
    """Run sampling. """
    if args.multi_process:
        jax.distributed.initialize()
    print("Devices", jax.devices())
    # setup_tpu_metrics()
    rng = jax.random.PRNGKey(args.global_seed)
    rng = jax.random.fold_in(rng, jax.process_index())

    assert str(args.image_size) in args.inference.ref_batch, f"Ref batch {args.inference.ref_batch} doesn't match image size {args.image_size}"
    
    # wandb
    expr_name = args.wandb.expr_name
    if (not args.wandb.log_on_main) or is_main():
        try:
            wandb.init(
                entity=args.wandb.entity,       # or args.wandb.entity
                project=args.wandb.project,     # or args.wandb.project
                name=f"{args.inference.name}sampling-{expr_name}",
                group=f"group-{expr_name}",
                config=OmegaConf.to_object(args),
                resume=True
            )
        except:
            id_ = wandb.util.generate_id()
            print("Unable to resume, generate new ID", id_)
            wandb.init(
                entity=args.wandb.entity,       # or args.wandb.entity
                project=args.wandb.project,     # or args.wandb.project
                name=f"{args.inference.name}sampling-{expr_name}",
                group=f"group-{expr_name}",
                config=OmegaConf.to_object(args),
                id=id_
            )

    # Resume configuration
    assert args.resume is not None
    experiment_dir = args.resume
    # Stores saved model checkpoints
    checkpoint_dir = osp.join(f"{experiment_dir}", "checkpoints")
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."

    if args.resume_step > 0:
        assert os.path.isdir(os.path.join(checkpoint_dir, f'{args.resume_step}')), \
            f"Desired step does not exist. Please check if there is a checkpoint file under {os.path.join(checkpoint_dir, f'{args.resume_step}')}"
        resume_steps = [int(args.resume_step)]
    else:
        resume_steps = [int(ckpt) for ckpt in os.listdir(checkpoint_dir) if ckpt.isdigit()]
        resume_steps = sorted(resume_steps, reverse=True)
        # resume_steps = [ckpt for ckpt in resume_steps if ckpt%25000==0] + [ckpt for ckpt in resume_steps if ckpt%25000!=0]

    print("Resume steps for this run:", resume_steps)

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
    ema_state, model = create_train_state_no_mesh(spl, data, args, return_model=True)

    # x_shape = jnp.array(
    #     [args.global_batch_size, latent_dim, latent_size, latent_size], 
    #     dtype=jnp.float32) 
    # t_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)
    # y_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)

    # unit, unit_name = 1e12, "TFlops"
    # flops_per_iter, _ = dit_flops.dit_flops(
    #     x_shape, y_shape, t_shape, model.bind(ema_state.ema_params), 
    #     backward=True, unit=unit) 
    # print(f'flops_per_iter: {flops_per_iter} {unit_name}')
    if not hasattr(args, "loss" ):
        diffusion = create_diffusion(timestep_respacing=str(args.inference.num_sampling_steps))
    else:
        diffusion = create_diffusion(**args.loss, num_samplesteps=args.inference.num_sampling_steps)
    if use_latent:
        vae_dir = args.vae_dir
        vae_config = np.load(os.path.join(vae_dir, 'config.npy'), allow_pickle=True).item()
        vae_params = np.load(os.path.join(vae_dir, 'params.npy'), allow_pickle=True).item()
        vae = FlaxAutoencoderKL.from_config(vae_config)

    assert args.inference.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.inference.cfg_scale > 1.0
    
    # Setup function 
    @jax.jit 
    def sample_latents(rng, params):
        rng = jax.random.fold_in(rng, jax.process_index())
        
        # Sample inputs:
        n = args.inference.per_proc_batch_size
        rng, spl1, spl2 = jax.random.split(rng, 3)
        z = jax.random.normal(
            spl1, shape=(n, latent_dim, latent_size, latent_size))
        y = jax.random.randint(
            spl2, minval=0, maxval=args.num_classes, shape=(n,))
        if using_cfg:
            z = jnp.concat([z, z], axis=0)
            y_null = jnp.array([1000] * n)
            y = jnp.concat([y, y_null], axis=0)
            rng, spl, spl2, spl3 = jax.random.split(rng, 4)
            model_kwargs = dict(
                y=y, cfg_scale=args.inference.cfg_scale, training=False,
                # For MTTT, still needs dropout + mt3
                rngs={"dropout": spl, "mt3": spl2, "label_emb": spl3} 
            )
            sample_fn = partial(
                ema_state.train_state.apply_fn, 
                params, method="forward_with_cfg"
            )
        else:
            rng, spl, spl2, spl3 = jax.random.split(rng, 4)
            model_kwargs = dict(
                y=y, training=False,
                rngs={"dropout": spl, "mt3": spl2, "label_emb": spl3})
            sample_fn = partial(
                ema_state.train_state.apply_fn, 
                params 
            )

        if args.inference.name == "ddim":
            samples = diffusion.ddim_sample_loop(
                rng, sample_fn, z.shape, 
                noise=z, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False)
        elif args.inference.name == "rectflow":
            samples = diffusion.p_sample_loop(
                rng, sample_fn, z.shape, 
                noise=z, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False)
        else:
            samples = diffusion.p_sample_loop(
                rng, sample_fn, z.shape, 
                noise=z, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False)
                
        if using_cfg:
            # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples, _ = jnp.split(samples, 2, axis=0)

        if use_latent:
            samples = vae.apply(
                {"params": vae_params}, samples / 0.18215, method=vae.decode
            ).sample
            samples = jnp.clip(
                127.5 * samples + 128.0, min=0, max=255
            ).transpose((0, 2, 3, 1))
        else:
            # Samples will be between [-1, 1]
            samples = jnp.clip(
                255 * (samples + 1.) * 0.5, min=0, max=255
            ).transpose((0, 2, 3, 1))
            
        return samples
    p_sample_latents = jax.pmap(sample_latents, axis_name="i")

    print("Resume checkpoint manager:", checkpoint_dir)
    options = orbax.checkpoint.CheckpointManagerOptions()
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)

    for ckpt_step in resume_steps:
        print("Processing ", ckpt_step)
            
        rngi = jax.random.fold_in(rng, ckpt_step)
        metrics = sample_checkpoint(
            args=args, rng=rngi, 
            ema_state=ema_state,
            checkpoint_manager=checkpoint_manager, 
            ckpt_step=ckpt_step,
            p_sample_latents=p_sample_latents,
            # flops_per_iter=flops_per_iter,
            # flops_unit_name=unit_name
        )
        if (not args.wandb.log_on_main) or is_main():
            wandb.log(metrics)
            print("Logged metrics to wandb")
        

if __name__ == "__main__":
    main()