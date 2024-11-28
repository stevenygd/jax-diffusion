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
import time
from datetime import datetime
import pytz
import hydra
import flax
import orbax
import shutil
import tempfile
import numpy as np
from PIL import Image
import flax.jax_utils
import os.path as osp
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial
import orbax.checkpoint as ocp
from evaluation import adm_eval
from main import create_train_state
from diffusion_jax import create_diffusion
from diffusers.models import FlaxAutoencoderKL
from omegaconf import DictConfig, OmegaConf
from utils import dit_flops 
import threading


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    flst = glob.glob(f"{sample_dir}/*.png")
    num = min(len(flst), num)
    n = num
    for i in tqdm(range(n), desc="Building .npz file from samples"):
        try:
            sample_pil = Image.open(flst[i])
        except Exception as e:
            print(f"Error loading {flst[i]}: {e}")
            os.remove(flst[i])
            n +=1
            continue
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
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
    if args.inference.get("mode", "ddpm") == "ddim":
        folder_name = f"{folder_name}-ddim"
    elif args.inference.get("mode", "ddpm") == "rectflow":
        folder_name = f"{folder_name}-rectflow"
    sample_folder_dir = osp.join(
        args.resume, args.inference.sample_dir, str(ckpt_step), folder_name)
    return sample_folder_dir


def sample_checkpoint(args, rng, ema_state, checkpoint_manager, ckpt_step, 
                      p_sample_latents, flops_per_iter, flops_unit_name):
    
    pid = jax.process_index()
    
    sample_folder_dir = create_sample_dir(args, ckpt_step)
    if args.inference.remove_existing_sample_dir:
        shutil.rmtree(sample_folder_dir, ignore_errors=True)
    
    output_metrics_path = f"{sample_folder_dir}-metrics.npy"
    metrics = None 
    start = None
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
        total_iter = int(np.ceil(
            (args.inference.num_fid_samples)/ 
             sample_per_iter
        ))

        if num_exist_imgs == 0: start = time.time()

        process_file_path = f"{sample_folder_dir}_process.npy"
        if not osp.isfile(process_file_path):
            process_dict = dict()
            np.save(process_file_path, process_dict)
                
        while True:
            review = False
            num_exist_imgs = len(glob.glob(f"{sample_folder_dir}/*.png"))
            if num_exist_imgs >= args.inference.num_fid_samples:
                print(f"already sampled {num_exist_imgs} images, quit sampling.")
                if osp.isfile(output_npz_path):
                    print(f"{output_npz_path} file exists, quit this run.")
                    return False
                break

            process_dict = np.load(process_file_path, allow_pickle=True).item()
            for k in range(total_iter):
                if k not in process_dict:
                    utc_now = datetime.now(pytz.utc)
                    pdt_now = utc_now.astimezone(pytz.timezone('US/Pacific'))
                    process_dict[k] = [os.getpid(), pdt_now.strftime('%Y-%m-%d %H:%M:%S')]
                    np.save(process_file_path, process_dict)
                    i = k
                    print(f"> Process {process_dict[i][0]} started sampling {i}th batch({i*sample_per_iter}~{(i+1)*sample_per_iter-1}) at {process_dict[i][1]}")
                    break
                if k == total_iter-1:
                    print(f"Sampling for all k is in process/completed according to {process_file_path}. Needs to review.")
                    review = True
            
            if review:
                process_dict = np.load(process_file_path, allow_pickle=True).item()
                for k in range(total_iter):
                    if len(process_dict[k]) == 2:
                        started_time = datetime.strptime(process_dict[k][1], '%Y-%m-%d %H:%M:%S')
                        started_time = pytz.timezone('US/Pacific').localize(started_time)
                        current_time = datetime.now(pytz.timezone('US/Pacific'))
                        diff = (current_time - started_time).total_seconds()/60
                        if diff > 40:
                            print(f"Sampling for {k}th batch({k*sample_per_iter}~{(k+1)*sample_per_iter-1}) is taking too long. Restarting.")
                            process_dict.pop(k)
                            np.save(process_file_path, process_dict)
                            break
                if (len(process_dict)==total_iter) and (k == total_iter-1):
                    print(f"Review finished. Sampling is done for all batches.")
                    return False
                continue
            
            rng_i = jax.random.fold_in(rng, i)
            rng_i = jax.random.split(rng_i, jax.local_device_count())
            samples = p_sample_latents(rng_i, params)
            samples = np.array(samples)
            samples = samples.reshape(-1, *samples.shape[-3:])
            for j in tqdm(range(samples.shape[0]), leave=False):
                sample = np.array(samples[j]).astype(jnp.uint8)
                sample_save_path = f"{sample_folder_dir}/pid{pid}-{i*sample_per_iter + j:06d}.png"
                # assert not os.path.exists(sample_save_path)
                Image.fromarray(sample).save(sample_save_path)
            
            process_dict = np.load(process_file_path, allow_pickle=True).item()
            if os.path.exists(f"{sample_folder_dir}/pid{pid}-{i*sample_per_iter:06d}.png") and \
                os.path.exists(f"{sample_folder_dir}/pid{pid}-{(i+1)*sample_per_iter-1:06d}.png"):
                utc_now = datetime.now(pytz.utc)
                pdt_now = utc_now.astimezone(pytz.timezone('US/Pacific'))
                process_dict[i].append(pdt_now.strftime('%Y-%m-%d %H:%M:%S'))
                np.save(process_file_path, process_dict)
                print(f"Process {os.getpid()} finished sampling {i}th batch({i*sample_per_iter}~{(i+1)*sample_per_iter-1})")
            else:
                process_dict.pop(i)
                np.save(process_file_path, process_dict)
                return False

        if is_main():
            print("Extract npz")
            output_npz_path = create_npz_from_sample_folder(
                sample_folder_dir, args.inference.num_fid_samples)
            print("Done with npz.")
    
    # TODO: does ADM support multi-host?    
    if is_main() and metrics is None:
        assert osp.isfile(args.inference.ref_batch), f"Ref batch {args.inference.ref_batch} does not exist."
        print("reference batch", args.inference.ref_batch)
        print("output_npz_path", output_npz_path)
        metrics = adm_eval._run_eval_(
            ref_batch = args.inference.ref_batch,
            sample_batch =output_npz_path, 
            batch_size=args.inference.get("adm_eval_batch_size", 64)
        )
        if args.inference.num_sampling_steps == 256: # the default value
            key = f"eval-{args.inference.num_fid_samples}"
        else:
            # NOTE: the " " at the end is annoiying
            key = f"eval-{args.inference.num_fid_samples}-{args.inference.num_sampling_steps}"

        # try :
        #     train_step = train_step
        # except UnboundLocalError:
        if True:
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
        metrics = {
            f"{key}/eval_steps": train_step,
            f"{key}/train_flops_{flops_unit_name}": float(train_step * flops_per_iter),
            **{f"{key}/{k}": float(v) for k, v in metrics.items()}
        }
        
        if start is not None: metrics["sampling_time"] = time.time() - start
        else: metrics["sampling_time"] = 0
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
    return True


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

    assert args.resume_step > 0, f"Something wrong in lauch eval script, resume_step = {args.resume_step}"
    assert str(args.image_size) in args.inference.ref_batch, f"Ref batch {args.inference.ref_batch} doesn't match image size {args.image_size}"
    
    # Resume configuration
    assert args.resume is not None
    experiment_dir = args.resume
    # Stores saved model checkpoints
    checkpoint_dir = osp.join(f"{experiment_dir}", "checkpoints") 
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
    assert os.path.isdir(os.path.join(checkpoint_dir, f'{args.resume_step}')), \
        f"Desired step does not exist. Please check if there is a checkpoint file under {os.path.join(checkpoint_dir, f'{args.resume_step}')}"

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
    flops_per_iter, _ = dit_flops.dit_flops(
        x_shape, y_shape, t_shape, model.bind(ema_state.ema_params), 
        backward=True, unit=unit) 
    print(f'flops_per_iter: {flops_per_iter} {unit_name}')
    if not hasattr(args, "loss" ):
        diffusion = create_diffusion(timestep_respacing=str(args.inference.num_sampling_steps))
    else:
        diffusion = create_diffusion(**args.loss, num_samplesteps=args.inference.num_sampling_steps)
    if use_latent:
        # vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        #     # This works
        #     "runwayml/stable-diffusion-v1-5",
        #     revision="flax",
        #     subfolder="vae",
        #     dtype=jnp.float32,
        #     from_pt=False,
        #     # DOESN'T WORK: f"stabilityai/sd-vae-ft-mse",
        #     # DOESN'T WORK: f"stabilityai/sd-vae-ft-ema",
        # )
        vae_dir = os.path.join(os.sep.join(args.checkpoint_dir.split(os.sep)[:5]), 'vae')
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
        # sample_fn = lambda *arg, **kwargs: model.apply(params, *args, **kwargs)
        # Sample images:
        if args.inference.get("mode", "ddpm") == "ddim":
            samples = diffusion.ddim_sample_loop(
                rng, sample_fn, z.shape, 
                noise=z, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False)
        elif args.inference.get("mode", "ddpm") == "rectflow":
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

    ckpt_step = args.resume_step
    print("Processing ", ckpt_step)
        
    rngi = jax.random.fold_in(rng, ckpt_step)
    # try:
    status = sample_checkpoint(
        args=args, rng=rngi, 
        ema_state=ema_state,
        checkpoint_manager=checkpoint_manager, 
        ckpt_step=ckpt_step,
        p_sample_latents=p_sample_latents,
        flops_per_iter=flops_per_iter,
        flops_unit_name=unit_name
    )
    assert status, "Sampling failed."
        

if __name__ == "__main__":
    main()
