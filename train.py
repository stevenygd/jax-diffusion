# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint 
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import orbax
import wandb
import yaml
import os
import os.path as osp
import jax.numpy as jnp
from jax.lib import xla_bridge
from typing import Any
import flax.linen as nn
import orbax.checkpoint as ocp
import importlib
import time
from diffusion.utils import avg_meter
from diffusion.utils import vis_ttt
from diffusion.utils import train_utils
from diffusion.utils import log_utils
from diffusion.utils import sharding 
import tensorflow as tf 
# Sharding
import logging
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")


if "WANDB_TEAM" in os.environ:
    WANDB_TEAM = os.environ["WANDB_TEAM"]
else:
    WANDB_TEAM = "xnf"

if "WANDB_PROJECT" in os.environ:
    WANDB_PROJECT = os.environ["WANDB_PROJECT"]
else:
    WANDB_PROJECT = "mt3"
    
################################################################################
#                                Helper Functions                              #
################################################################################

def makedirs(dirname, main_only=False, **kwargs):
    if not main_only or jax.process_index() == 0:
        if dirname.startswith("gs://"):
            return
        try:
            os.makedirs(dirname, **kwargs)
        except:
            assert os.path.isdir(dirname), \
                f"pid:{jax.process_index()} dname={dirname}" 
    # TODO: this is still not threadsafe!
    orbax.checkpoint.multihost.sync_global_processes(f"mkdir-{dirname}")


def is_main():
    return jax.process_index() == 0
    
    
@hydra.main(version_base=None, 
            config_path="diffusion/configs", 
            config_name="config")
def main(args: DictConfig):
    """
    Trains a new DiT model.
    """
    # If it's multi GPU we will initialize distributed
    if args.multi_process:
        jax.distributed.initialize()
        
    # Check GPU
    jax_platform = xla_bridge.get_backend().platform.lower()
    assert jax_platform in ["tpu", "gpu"], f"Not using GPU/TPU:{jax_platform}"
    
    # Setup wandb
    local_device_count = jax.local_device_count()
    total_device_count = jax.device_count()
    
    # Syncrhonize runtime
    time_sec = jnp.ones(
        (local_device_count,), dtype=jnp.int32) * int(time.time())
    time_sec = jax.pmap(lambda x: jax.lax.pmin(x, 'i'), axis_name='i')(time_sec)
    time_sec = int(time_sec[0])
    run_time = datetime.fromtimestamp(time_sec).strftime('%Y-%b-%d-%H-%M-%S')
    model_str = args.model.name.replace("/", "-")
    pid = jax.process_index()
    if args.resume:
        expr_name = args.wandb.expr_name
        project_name = args.wandb.project
    else:
        expr_name = f"{args.expr_name}-{model_str}-{run_time}"
        project_name = WANDB_PROJECT
    if (not args.wandb.log_on_main) or is_main():
        # NOTE even resuming will create a new wandb log
        run = wandb.init(
            entity=WANDB_TEAM,
            project=project_name,
            name=f"training-{expr_name}-pid={pid}",
            group=f"group-{expr_name}",
            config=OmegaConf.to_object(args),
            dir=args.wandb_dir
        )
        args.wandb.id = run.id 
        args.wandb.entity = WANDB_TEAM
        args.wandb.project = project_name 
        args.wandb.expr_name = expr_name

    # Setup an experiment folder:
    experiment_dir = osp.join(args.results_dir, expr_name)
    makedirs(args.results_dir, main_only=True, mode=0o777, exist_ok=True)
    makedirs(experiment_dir, main_only=True, mode=0o777, exist_ok=True)
    checkpoint_dir = osp.join(experiment_dir, "checkpoints")
    makedirs(checkpoint_dir, main_only=True, mode=0o777, exist_ok=True)
    profile_dir = osp.join(experiment_dir, "profile")
    makedirs(profile_dir, main_only=True, mode=0o777, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir 
    args.profile_dir = profile_dir 
    args.experiment_dir = experiment_dir
    profile_dir = osp.join(profile_dir, str(pid))
        
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory created at {experiment_dir}")
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
        create=True,
        enable_async_checkpointing=True,
    )
    orbax_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, 
        orbax_checkpointer, options=options)

    # Create device mesh
    mesh = sharding.get_mesh(args)
    with mesh:
        print("Shading mesh:", mesh)
        
        # Setup data:
        # Local batch size is the dataloader for ONE process 
        dataloader_lib = importlib.import_module(
            f"diffusion.datasets.{args.data_loader}")
        # Initialize random seed
        rng = jax.random.PRNGKey(int(args.global_seed))
        rng, spl = jax.random.split(rng)
        print("Dataloader rng", spl)
        local_batch_size = args.global_batch_size
        _, loader = dataloader_lib.make_data_loader(
            args, mesh, local_batch_size=local_batch_size, rng=spl)
    
        x_partition, y_partition = sharding.get_data_partition_rules()
        x_sharding = NamedSharding(mesh, x_partition)
        y_sharding = NamedSharding(mesh, y_partition)
        
        ########################################################################
        # Create train state
        ########################################################################
        spl, rng = jax.random.split(rng)
        x, y = next(loader)
        
        # Create get model parameter shape 
        model = train_utils.create_model(args)
        tx = train_utils.make_opt(args)
        def make_ema_state(rng, x, y):
            params = train_utils.init_model(rng, (x, y), model)
            ema_state = train_utils.EMATrainState.create(
                apply_fn=None, params=params, tx=tx, 
                ema_decay=args.ema_decay)
            return ema_state
        
        # Obtaining the ema_state shapes
        abstract_ema_state = jax.eval_shape(make_ema_state, spl, x, y)
        logical_spec_ema_state = nn.get_partition_spec(abstract_ema_state)
        ps_rule = sharding.get_logical_partition_rules()
        ema_state_sharding = nn.logical_to_mesh_sharding(
            logical_spec_ema_state, mesh, ps_rule)

        make_ema_state_sharded = jax.jit(
            make_ema_state,  
            in_shardings=(
                NamedSharding(mesh, PS()),  # rng
                x_sharding, y_sharding),    # x, y sharding
            out_shardings=ema_state_sharding,
            # check_rep=False
        )
        ema_state = make_ema_state_sharded(spl, x, y)
       
        # Resume checkpoint
        start_iter = 0
        if args.resume:
            ema_state, start_iter = train_utils.restore_checkpoint(
                checkpoint_manager, ema_state, logger=logger, 
                resume_step=args.get("resume_step", None))
        param_count = sum(
            x.size for x in
            jax.tree_util.tree_leaves(ema_state.train_state.params))
        logger.info(f"DiT Parameters: {param_count:,}")
        if is_main():
            print(f"Model Parameters: {param_count:,}")

        # Make stepping function
        logger.info(f"=== START: Make training step ===")
        train_step_fn = train_utils.make_train_step(args, model)
        train_step_fn = jax.jit(
            train_step_fn, 
            in_shardings=(
                ema_state_sharding,             # state sharding
                x_sharding, y_sharding,         # data sharding
                NamedSharding(mesh, PS(None))   # rng, split and shard across devices
            ),
            out_shardings=(
                ema_state_sharding,         # state sharding
                NamedSharding(mesh, PS()),  # aux, will aggregate inside 
            ), 
            donate_argnums=(0,)
        )
        logger.info(f"===  END : Make training step ===")

        # NOTE: this has to be outside or dataloader are not the same across
        # Example datlaoader for sampling
        x, y = next(iter(loader))
        x, y = jnp.array(x), jnp.array(y)
        rng, spl = jax.random.split(rng)
        if not args.resume and is_main():
            logger.info(f"=== START: Saving config ===")
            # If it's not resuming, then we will save the config file
            cfg_obj = OmegaConf.to_object(args)
            cfg_obj["resume"] = experiment_dir
            cfg_obj["data_shape"] = {
                "x": list(x.shape),
                "y": list(y.shape),
            }
            with open(
                osp.join(experiment_dir, "config.yaml"), "w") as yml_out_file:
                yaml.dump(cfg_obj, yml_out_file)
            logger.info(f"=== START: Saving config ===")

        # Variables for monitoring/logging purposes:
        loading_time_meter = avg_meter.AverageMeter("loading-time")
        logging_time_meter = avg_meter.AverageMeter("logging-time")
        stepping_time_meter = avg_meter.AverageMeter("stepping-time")
        checkpoint_time_meter = avg_meter.AverageMeter("checkpoint-time")
        ttl_train_time_start = time.time()

        train_steps = start_iter
        total_iters = args.total_iters
        log_every = args.log_every
        ckpt_every = args.ckpt_every
        # Split out a line of rng for reproducing training performance
        train_rng, rng = jax.random.split(rng)
        train_rng = jax.random.fold_in(train_rng, jax.process_index())
        logger.info(f"Training for {total_iters} iterations...")
        
        # with jax.profiler.trace(profile_dir):
        if args.get("profile", False):
            jax.profiler.start_trace(profile_dir)
        for train_steps in range(start_iter, total_iters):
            
            # Book keeping
            data_start_time = time.time()
            spls, train_rng = jax.random.split(train_rng)
            x, y = next(loader) # (#devices, local-bs, ...) if multi, or (bs, ...)
            loading_time_meter.update(time.time() - data_start_time)
        
            # Train step, this can be async 
            step_start_time = time.time()
            ema_state, aux = train_step_fn(ema_state, x, y, spls)
            stepping_time_meter.update(time.time() - step_start_time)
            
            # Log loss values:
            if (train_steps % log_every == 0 or 
                train_steps < 10 or
                (train_steps < 100 and train_steps % 10 == 0)) and is_main():
                
                # Add the blocking time
                step_start_time = time.time()
                aux = jax.device_get(aux)
                loss_val = aux["loss_val"]
                gnorm = aux["grad_norm"]
                diff_loss = aux["loss_dict"]["loss"]
                stepping_time_meter.add_to_sum(time.time() - step_start_time)
                
                logging_start_time = time.time()
                
                # Measure training speed:
                steps_per_sec = (
                    (train_steps - start_iter) / (time.time() - ttl_train_time_start))
                logger.info(
                    f"(step={train_steps:07d})"
                    # f" loss={loss_val:.4f} diff={diff_loss:.4f}"
                    f" steps/s={steps_per_sec:.2f}"
                    f" time(load,train,log)=({loading_time_meter.avg:.4f},"
                    f"{stepping_time_meter.avg:.4f},"
                    f"{logging_time_meter.avg:.4f})")

                # Plot inner loss if it exists
                if (not args.wandb.log_on_main) or is_main():
                    if args.get("log_ttt", False):
                        vis_ttt.vis_ttt_lm(aux) 
                        vis_ttt.vis_ttt_orig(aux)
                    if args.get("log_dit_stats", True):
                        log_utils.log_dit_stats(aux) 
                    if args.get("log_loss_per_t", False):
                        log_utils.log_loss_per_time(aux)
    
                    # Log wandb
                    wandb.log({
                        "train_steps": train_steps,
                        "orig_train_steps": train_steps,
                        "loss": loss_val.mean(),
                        "gnorm": gnorm.mean(),
                        "diff_loss": diff_loss.mean(),
                        # Time profiling
                        "logging_time": logging_time_meter.avg,
                        "stepping_time": stepping_time_meter.avg,
                        "loading_time": loading_time_meter.avg,
                    })
                    
                logging_time_meter.update(time.time() - logging_start_time)

            # Save model checkpoint:
            if ckpt_every > 0 and train_steps % ckpt_every == 0 and (
                train_steps > 0 or args.get("test_checkpoint", False)):
                ckpt_start_time = time.time()
                train_utils.save_checkpoint(
                    train_steps, checkpoint_manager, ema_state, logger=logger)
                checkpoint_time_meter.update(time.time() - ckpt_start_time)

        # Save the final checkpoint
        if ckpt_every > 0:
            logger.info("Start final checkpoint!")
            ckpt_start_time = time.time()
            train_utils.save_checkpoint(
                train_steps, checkpoint_manager, ema_state, logger=logger)
            checkpoint_time_meter.update(time.time() - ckpt_start_time)
            logger.info("Finished final checkpoint!")

        logger.info("Done!")
        if (not args.wandb.log_on_main) or is_main():
            wandb.finish()

        if args.get("profile", False):
            jax.profiler.stop_trace()

if __name__ == "__main__":
    main()