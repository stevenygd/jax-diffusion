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
import json
import flax
import orbax
import shutil
import numpy as np
from PIL import Image
import flax.jax_utils
import os.path as osp
from tqdm import tqdm
import jax.numpy as jnp
from flax import jax_utils
from diffusion.utils import dit_flops 
from diffusion.utils import train_utils
from diffusion.utils import sharding 
from datetime import datetime
from functools import partial
from jax.lib import xla_bridge
import orbax.checkpoint as ocp
from diffusion.evaluation import adm_eval
from diffusion.models import dit
from diffusion.losses import create_diffusion
from diffusers.models import FlaxAutoencoderKL
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
import flax.linen as nn

from diffusion.configs_mlc import CFG
from absl import app
from absl import flags
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS
flags.DEFINE_string("resume_config", None, "Resume config.")
flags.DEFINE_string("vaedir", None, "Directory to store the VAE.")
flags.DEFINE_boolean("multi_process", True, "Multi process training.")
flags.DEFINE_boolean("tpu", True, "TPU training.")
flags.DEFINE_boolean("gpu", False, "GPU training.")
flags.DEFINE_string("server_address", None, "Coordinator address.")
flags.DEFINE_integer("num_nodes", 1, "Number of processes.")
flags.DEFINE_integer("node_rank", 0, "Process index.")
flags.DEFINE_boolean(
    "evaluate_single_ckpt", False, "Whether evaluate a single checkpoint.")
flags.DEFINE_string("flops_unit", "TFlops", "Flop units.")
flags.DEFINE_string(
    "ref_batch", 
    "/mnt/disks/data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz", 
    "Location of the reference batch.")
flags.DEFINE_integer("resume_step", -1, "Which step to resume.")
flags.DEFINE_integer("batch_size", 32, "Evaluation batch size.")
flags.DEFINE_integer("adm_eval_batch_size", 64, "Evaluation batch size for FID.")
flags.DEFINE_integer("num_fid_samples", 10_000, "Number of FID samples.")
flags.DEFINE_float("cfg_scale", 1.0, "CFG scale.")


UNITS = {
    "PFlops": 1e15,
    "TFlops": 1e12,
    "GFlops": 1e9,
    "MFlops": 1e6,
}


def create_train_state(rng, data, args, logging=None, return_model=False):
    # model_name, latent_size, num_classes, latent_dim):
    use_latent = args.get("use_latent", True)
    if use_latent:
        latent_size = args.image_size // 8
    else:
        latent_size = args.image_size
    # Create model:
    model = dit.Model(
        input_size=latent_size,
        num_classes=args.num_classes, 
        in_channels=args.latent_dim,
        **args.model)
    
    # if logging: logging.info("Model:", repr(model))
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


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_npz_from_sample_folder(sample_dir, output_dir=None, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    if output_dir == None:
        output_dir = sample_dir
    samples = []
    flst = glob.glob(f"{sample_dir}/*.png")
    num = min(len(flst), num)
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(flst[i])
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def get_fid_samples(args):
    if FLAGS.num_fid_samples is not None:
        fid_samples = FLAGS.num_fid_samples
    else:
        fid_samples = args.inference.num_fid_samples
    return int(fid_samples)

def get_cfg_scale(args):
    if FLAGS.num_fid_samples is not None:
        cfg_scale = FLAGS.cfg_scale
    else:
        cfg_scale = args.inference.cfg_scale
    return cfg_scale 


def get_batch_size(args):
    if FLAGS.batch_size is not None:
        n = FLAGS.batch_size
    else:
        n = args.inference.per_proc_batch_size
    return n

def create_sample_dir(args, ckpt_step):
    # Create folder to save samples:
    model_string_name = args.model.name.replace("/", "-")
    folder_name = f"{model_string_name}-"\
                  f"size-{args.image_size}-" \
                  f"cfg-{get_cfg_scale(args)}-" \
                  f"seed-{args.global_seed}-" \
                  f"step-{args.inference.num_sampling_steps}-" \
                  f"nsmp-{get_fid_samples(args)}"
    sample_folder_dir = osp.join(
        args.resume, args.inference.sample_dir, str(ckpt_step), folder_name)
    return sample_folder_dir


def sample_checkpoint(args, mesh, rng, ema_state, checkpoint_manager, ckpt_step, 
                      p_sample_latents, flops_per_iter, flops_unit_name, 
                      evaluate_single_ckpt, reeval_metrics=False, writer=None):
    
    pid = jax.process_index()
    
    sample_folder_dir = create_sample_dir(args, ckpt_step)
    if args.inference.remove_existing_sample_dir:
        shutil.rmtree(sample_folder_dir, ignore_errors=True)
    
    output_metrics_path = f"{sample_folder_dir}-metrics.npy"
    metrics = None 
    start = None
    # TODO: this is not thread safe
    if reeval_metrics and osp.isfile(output_metrics_path):
        os.remove(output_metrics_path)
        print(f"Deleted: {output_metrics_path}")
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
    
    # Check if the reference batch exists before proceeding
    assert str(args.image_size) in args.inference.ref_batch, f"Ref batch {args.inference.ref_batch} doesn't match image size {args.image_size}"
    ref_batch = args.inference.ref_batch
    if not osp.isfile(ref_batch):
        ref_batch = FLAGS.ref_batch
    assert osp.isfile(ref_batch), f"Ref batch {ref_batch} does not exist."

    # Resume model
    print("Resuming...")
    ema_state, train_step = train_utils.restore_checkpoint(
        checkpoint_manager, ema_state, 
        logger=None, resume_step=ckpt_step)
    print("Checkpoint restored...")
    params = ema_state.ema_params
    # if not isinstance(params, flax.core.FrozenDict):
    #     params = flax.core.FrozenDict(params)
    print("Obtained params...")

    output_npz_path = f"{sample_folder_dir}.npz"
    if metrics is None and not osp.isfile(output_npz_path):
        print(f"metrics is None: {metrics is None}, output_npz_path exists: {osp.isfile(output_npz_path)}")
        print(f"{sample_folder_dir}.npz exists, skip.")
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

        # Find out how many images are there
        num_exist_imgs = len(glob.glob(f"{sample_folder_dir}/*.png"))
        print(f"Total number of images already exists: {num_exist_imgs}")
        if num_exist_imgs == 0: start = time.time()
        total_samples = get_fid_samples(args) - num_exist_imgs
        print(f"Total number of samples that will be sampled: {total_samples}")
        pbar = range(total_samples)
        if is_main():
            pbar = tqdm(pbar, total=total_samples, desc="Sampling") 
        total = 0 
        while total < get_fid_samples(args):
            rng_i = jax.random.fold_in(rng, jax.process_index())
            rng_i = jax.random.fold_in(rng, total)
            # Return (#local devices, #batch_size, R, R, 3)
            # rng_i = jax.asarray(jax.random.split(
            #     rng_i, mesh.devices.shape[0] * mesh.devices.shape[1]))
            samples = p_sample_latents(rng_i, params)
            # samples = jax.device_get(samples)
            sample_shape = samples.shape[-3:]
            samples = jax.experimental.multihost_utils.process_allgather(samples)
            samples = samples.reshape(-1, *sample_shape)
            print(f"{total:04d}:{total_samples:04d}", samples.shape)

            # Save samples to disk as individual .png files
            for i in tqdm(range(samples.shape[0]), leave=False):
                if i % (pid + 1) == 0:
                    sample = np.array(samples[i]).astype(jnp.uint8)
                    Image.fromarray(sample).save(
                        # f"{tmp_dir}/pid{pid}-{total:06d}.png")
                        f"{sample_folder_dir}/pid{pid}-{total:06d}.png")
                if is_main():
                    pbar.update(1)
                total += 1
                
        if is_main():
            num_exist_imgs = len(glob.glob(f"{sample_folder_dir}/*.png"))
            assert num_exist_imgs >= get_fid_samples(args), f"Number of images {num_exist_imgs} is less than {get_fid_samples(args)}"
            print("Extract npz")
            output_npz_path = create_npz_from_sample_folder(
                sample_folder_dir, get_fid_samples(args))
            print("Done with npz.")
        
    
    if is_main() and metrics is None:
        metrics = adm_eval._run_eval_(
            ref_batch, output_npz_path, batch_size=FLAGS.adm_eval_batch_size)
        num_fid_samples = get_fid_samples(args)
        if args.inference.num_sampling_steps == 256: # the default value
            key = f"eval-{num_fid_samples}"
        else:
            # NOTE: the " " at the end is annoiying
            key = f"eval-{num_fid_samples}-{args.inference.num_sampling_steps}"
        
        train_flops_per_iter = flops_per_iter[0] if isinstance(flops_per_iter, tuple) else flops_per_iter
        metrics = {
            f"{key}/eval_steps": int(train_step),
            f"{key}/train_flops_{flops_unit_name}": float(train_step * train_flops_per_iter),
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
        metrics["ckpt_step"] = int(ckpt_step)
        metrics["train_step"] = int(train_step)
        train_flops_per_iter = flops_per_iter[0] if isinstance(flops_per_iter, tuple) else flops_per_iter
        metrics[f"train_flops_{flops_unit_name}"] = float(train_step * train_flops_per_iter)
        print("=" * 80)
        print("Metrics", metrics)
        print("=" * 80)
        if writer is not None:
            print("Logging")
            flattend_metrics = flatten_dict(metrics)
            for k, v in flattend_metrics.items():
                writer.add_scalar(k, v, train_step)
            print("Logging done")
        print(f"[{pid}] Done.")
    return True


def multi():
    return jax.device_count() > jax.local_device_count()


def is_main():
    return jax.process_index() == 0


def main(_):
    if FLAGS.multi_process:
        if FLAGS.tpu:
            jax.distributed.initialize()
        elif FLAGS.gpu:
            jax.distributed.initialize(
                # TODO: add these arguments
                coordinator_address=FLAGS.server_address,
                num_processes=FLAGS.num_nodes,
                process_id=FLAGS.node_rank)
    
    assert FLAGS.resume_config is not None, \
        "Both config and resume config is None"
    with open(FLAGS.resume_config, "r") as f:
        config_dict = json.load(f)
    args = CFG(**config_dict)
        
    # Check GPU
    jax_platform = xla_bridge.get_backend().platform.lower()
    assert jax_platform in ["tpu", "gpu"], f"Jax not using GPU:{jax_platform}"

    # Initialize random seed
    rng = jax.random.PRNGKey(int(args.global_seed))
    rng = jax.random.fold_in(rng, jax.process_index())

    # Setup an experiment folder:
    expr_name = args.expr_name
    experiment_dir = args.experiment_dir
    checkpoint_dir = args.checkpoint_dir
    workdir = args.work_dir
    writer = SummaryWriter(log_dir=osp.join(workdir, "logs", expr_name))
   
    # Logging directory 
    logger_dir = osp.join(experiment_dir, "logs")
    logger = train_utils.create_logger(logger_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    local_device_count = jax.local_device_count()
    total_device_count = jax.device_count()
    logger.info(
        f"Devices: {jax.devices()[:4]}... "
        f"{local_device_count}/{total_device_count} devices.")
    print(
        f"[pid={jax.process_index()}] "
        f"Devices: {jax.devices()[:4]}... "
        f"{local_device_count}/{total_device_count} devices.")

    # Setup checkpoint, each device has its own checkpoint
    options = ocp.CheckpointManagerOptions(
        max_to_keep=args.max_ckpt_keep, 
        create=True, enable_async_checkpointing=True)
    orbax_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options=options)
    
    mesh = sharding.get_mesh(args)
    with mesh:
        print("Shading mesh:", mesh)

        # Set-up data
        use_latent = args.use_latent
        if hasattr(args, "image_size"): 
            if use_latent:
                latent_size = args.image_size // 8 
            else:
                latent_size = args.image_size
        else: 
            _, _, _, latent_size, _ = args.data_shape.x
        if hasattr(args, "latent_dim"): 
            latent_dim = args.latent_dim
        else: 
            _, _, latent_dim, _, _ = args.data_shape.x
        data = (
            # x
            jnp.ones((
                FLAGS.batch_size, latent_dim, latent_size, latent_size)),
            # y
            jnp.ones((FLAGS.batch_size,), dtype=jnp.int32)
        )
        
        x_partition, y_partition = sharding.get_data_partition_rules()
        x_sharding = NamedSharding(mesh, x_partition)
        y_sharding = NamedSharding(mesh, y_partition)

        # Create train state
        rng, spl = jax.random.split(rng)
        print("Model init rng:", spl)
        ema_state, ema_state_sharding, model = train_utils.create_train_state(
            args, mesh, spl, data[0], data[1])
        
        # Computing flops
        unit, unit_name = UNITS[FLAGS.flops_unit], FLAGS.flops_unit
        x_shape = jnp.array(
            [args.global_batch_size, latent_dim, latent_size, latent_size], 
            dtype=jnp.float32) 
        t_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)
        y_shape = jnp.array([args.global_batch_size], dtype=jnp.float32)
        flops_per_iter, _ = dit_flops.dit_flops(
            x_shape, y_shape, t_shape, model.bind(ema_state.ema_params), 
            backward=True, unit=unit) 
        print(f'Flops_per_iter: {flops_per_iter} {unit_name}')

        if not hasattr(args, "loss" ):
            diffusion = create_diffusion(timestep_respacing=str(args.inference.num_sampling_steps))
        else:
            diffusion = create_diffusion(**args.loss, num_samplesteps=args.inference.num_sampling_steps)
        if use_latent:
            vae_dir = FLAGS.vaedir
            if vae_dir is None:
                vae_dir = os.path.join(os.sep.join(args.checkpoint_dir.split(os.sep)[:5]), 'vae')
            vae_config = np.load(os.path.join(vae_dir, 'config.npy'), allow_pickle=True).item()
            vae_params = np.load(os.path.join(vae_dir, 'params.npy'), allow_pickle=True).item()
            vae = FlaxAutoencoderKL.from_config(vae_config)

        cfg_scale = get_cfg_scale(args)
        assert cfg_scale >= 1.0, f"In almost all cases, cfg_scale be >= 1.0 {cfg_scale}"
        using_cfg = cfg_scale > 1.0
        
        # Setup function 
        pid = jax.process_index()
        def sample_latents(rng, params):
            # rng = jax.random.fold_in(rng, pid)
            # rng = jax.random.fold_in(rng, jax.lax.axis_index("dp"))
            # rng = jax.random.fold_in(rng, jax.lax.axis_index("fsdp"))
            # Sample inputs:
            n = get_batch_size(args)
            print("=" * 8)
            print(rng)
            print("=" * 8)
            rng, spl1, spl2 = jax.random.split(rng, 3)
            z = jax.random.normal(
                spl1, shape=(n, latent_dim, latent_size, latent_size))
            z = jax.lax.with_sharding_constraint(z, x_sharding)
            y = jax.random.randint(
                spl2, minval=0, maxval=args.num_classes, shape=(n,))
            y = jax.lax.with_sharding_constraint(y, y_sharding)
            if using_cfg:
                z = jnp.concat([z, z], axis=0)
                y_null = jnp.array([1000] * n)
                y = jnp.concat([y, y_null], axis=0)
                rng, spl, spl2, spl3 = jax.random.split(rng, 4)
                model_kwargs = dict(
                    y=y, cfg_scale=cfg_scale, training=False,
                    # For MTTT, still needs dropout + mt3
                    rngs={"dropout": spl, "mt3": spl2, "label_emb": spl3} 
                )
                sample_fn = partial(
                    model.apply, params, method="forward_with_cfg")
            else:
                rng, spl, spl2, spl3 = jax.random.split(rng, 4)
                model_kwargs = dict(
                    y=y, training=False,
                    rngs={"dropout": spl, "mt3": spl2, "label_emb": spl3})
                sample_fn = partial(model.apply, params)
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
        
        p_sample_latents = jax.jit(
            sample_latents, 
            in_shardings=(
                # rng, split and shard across devices
                # NamedSharding(mesh, PS(("dp", "fsdp"))),  
                NamedSharding(mesh, PS(None)),  
                # state sharding
                ema_state_sharding.ema_params   
            ),
            out_shardings=x_sharding,
        )
        
        success = set() # ckpt steps
        while True:
            # Setup checkpoint manager
            print("Resume checkpoint manager:", checkpoint_dir)
            options = orbax.checkpoint.CheckpointManagerOptions()
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                checkpoint_dir, orbax_checkpointer, options)
            ckpt_steps = sorted(checkpoint_manager.all_steps(), reverse=True)
            print("All checkpoints:", ckpt_steps)

            if FLAGS.evaluate_single_ckpt:
                assert FLAGS.resume_step in ckpt_steps, \
                    f"Desired checkpoint step does not exist. Please check if there is a checkpoint file under {checkpoint_dir}"
                ckpt_steps = [FLAGS.resume_step]

            print("Desired checkpoint for this run:", ckpt_steps)

            if len(ckpt_steps) == 0:
                print("Sleeping...(%d mins)" % (args.inference.sleep_interval / 60.))
                time.sleep(args.inference.sleep_interval)
                continue
            processed = [
                ckpt_step for ckpt_step in ckpt_steps if ckpt_step in success]
            print("Processed: ", processed)
            todos = [
                ckpt_step for ckpt_step in ckpt_steps if ckpt_step not in success]
            if len(todos) == 0:
                # print("All checkpoint processed. Finished")
                print(f"Checkpoint {ckpt_steps} processed. Finished")
                break
            ckpt_step = todos[0]
            print("Processing ", ckpt_step)
                
            rngi = jax.random.fold_in(rng, ckpt_step)
            # try:
            status = sample_checkpoint(
                args=args, mesh=mesh, rng=rngi, 
                ema_state=ema_state,
                checkpoint_manager=checkpoint_manager, 
                ckpt_step=ckpt_step,
                p_sample_latents=p_sample_latents,
                flops_per_iter=flops_per_iter,
                flops_unit_name=unit_name,
                evaluate_single_ckpt=FLAGS.evaluate_single_ckpt,
                reeval_metrics=args.inference.get("reeval_metrics", False),
                writer=writer
            )
            if status: success.add(ckpt_step)
            del checkpoint_manager
            del orbax_checkpointer 
            print("Sleeping...(%d mins)" % (args.inference.sleep_interval / 60.))
            time.sleep(args.inference.sleep_interval)
            print("Wake up...")

if __name__ == '__main__':
    app.run(main)