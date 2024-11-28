# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint 
from datetime import datetime
from diffusion.utils import avg_meter
import jax
import os
import json
import os.path as osp
import jax.numpy as jnp
from jax.lib import xla_bridge
from jax.lib import xla_bridge
from typing import Any
import flax.linen as nn
import orbax.checkpoint as ocp
import importlib
import time
import orbax.checkpoint as ocp
import tensorflow as tf 
# Sharding
# from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from diffusion.utils import train_utils
from diffusion.utils import sharding 
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
from diffusion.configs_mlc import CFG, concretize
from absl import app
from absl import flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter


CONFIG = config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string(
    "resume_config", None, "Json file to resume the experiment from.")
flags.DEFINE_boolean("resume", False, "Resume the experiment.")
flags.DEFINE_integer("resume_step", -1, "Resume step.")
flags.DEFINE_boolean("multi_process", True, "Multi process training.")
flags.DEFINE_boolean("tpu", True, "TPU training.")
flags.DEFINE_boolean("gpu", False, "GPU training.")
flags.DEFINE_string("server_address", None, "Coordinator address.")
flags.DEFINE_string("datadir", "data/", "Data directory.")
flags.DEFINE_integer("num_nodes", 1, "Number of processes.")
flags.DEFINE_integer("node_rank", 0, "Process index.")
flags.DEFINE_boolean("test_checkpoint", False, "Test checkpoint.")

# Sharding
flags.DEFINE_integer("dp_dim", -1, "DP")
flags.DEFINE_integer("fsdp_dim", 1, "FSDP.")
flags.DEFINE_integer("tp_dim", 1, "TP.")


def main(argv):
    if FLAGS.multi_process:
        if FLAGS.tpu:
            jax.distributed.initialize()
        elif FLAGS.gpu:
            jax.distributed.initialize(
                # TODO: add these arguments
                coordinator_address=FLAGS.server_address,
                num_processes=FLAGS.num_nodes,
                process_id=FLAGS.node_rank)
            
    args = FLAGS.config
    if args is None or FLAGS.resume:
        assert FLAGS.resume_config is not None, \
            "Both config and resume config is None"
        with open(FLAGS.resume_config, "r") as f:
            config_dict = json.load(f)
        args = CFG(**config_dict)
        
    args.dp_dim = FLAGS.dp_dim
    args.fsdp_dim = FLAGS.fsdp_dim
    args.tp_dim = FLAGS.tp_dim
        
    # Check GPU
    jax_platform = xla_bridge.get_backend().platform.lower()
    assert jax_platform in ["tpu", "gpu"], f"Jax not using GPU:{jax_platform}"
    
    local_device_count = jax.local_device_count()
    total_device_count = jax.device_count()
    time_sec = jnp.ones(
        (local_device_count,), dtype=jnp.int32) * int(time.time())
    time_sec = jax.pmap(lambda x: jax.lax.pmin(x, 'i'), axis_name='i')(time_sec)
    time_sec = int(time_sec[0])
    run_time = datetime.fromtimestamp(time_sec).strftime('%Y-%b-%d-%H-%M-%S')
    model_str = args.model.name.replace("/", "-")
    if FLAGS.resume:
        expr_name = args.expr_name
    else:
        expr_name = f"{args.expr_name}-{model_str}-{run_time}"

    # Initialize tensorbaord writer
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = args.work_dir
    else:
        args.work_dir = workdir
    if train_utils.is_main():
        writer = SummaryWriter(log_dir=osp.join(workdir, "logs", expr_name))
    train_utils.makedirs(workdir, main_only=True, mode=0o777, exist_ok=True)

    # Initialize random seed
    rng = jax.random.PRNGKey(int(args.global_seed))
    pid = jax.process_index()

    # Setup an experiment folder:
    experiment_dir = osp.join(workdir, "checkpoints", expr_name)
    train_utils.makedirs(
        workdir, main_only=True, mode=0o777, exist_ok=True)
    train_utils.makedirs(
        experiment_dir, main_only=True, mode=0o777, exist_ok=True)
    checkpoint_dir = osp.abspath(osp.join(experiment_dir, "ckpt"))
    train_utils.makedirs(
        checkpoint_dir, main_only=True, mode=0o777, exist_ok=True)
    profile_dir = osp.abspath(osp.join(workdir, "logs", expr_name, "profile"))
    train_utils.makedirs(
        checkpoint_dir, main_only=True, mode=0o777, exist_ok=True)
    profile_dir = osp.join(profile_dir, str(pid))
        
    logger_dir = osp.join(experiment_dir, "logs")
    train_utils.makedirs(logger_dir, mode=0o777, exist_ok=True)
    logger = train_utils.create_logger(logger_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(
        f"Devices: {jax.devices()[:4]}... "
        f"{local_device_count}/{total_device_count} devices.")

    # Setup checkpoint, each device has its own checkpoint
    options = ocp.CheckpointManagerOptions(
        max_to_keep=args.max_ckpt_keep, 
        create=True, enable_async_checkpointing=True)
    orbax_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options=options)

    # Create device mesh
    mesh = sharding.get_mesh(args)
    with mesh:
        logger.info("Shading mesh: %s" % mesh)
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
        ema_state, ema_state_sharding, model = train_utils.create_train_state(
            args, mesh, spl, x, y)
        
        # Resume checkpoint
        start_iter = 0
        if FLAGS.resume:
            logger.info("Resuming checkpoint")
            ema_state, start_iter = train_utils.restore_checkpoint(
                checkpoint_manager, ema_state, logger=logger, 
                resume_step=FLAGS.resume_step)
        param_count = sum(
            x.size for x in
            jax.tree_util.tree_leaves(ema_state.train_state.params))
        logger.info(f"DiT Parameters: {param_count:,}")

        # Make stepping function
        logger.info(f"=== START: Make training step ===")
        train_step_fn = train_utils.make_train_step(args, model)
        train_step_fn = jax.jit(
            train_step_fn, 
            in_shardings=(
                ema_state_sharding,             # state sharding
                x_sharding, y_sharding,         # data sharding
                # NamedSharding(mesh, PS("dp"))   # rng, split and shard across devices
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
        rng, spl = jax.random.split(rng)
        
        # Save configuration:
        if not FLAGS.resume and train_utils.is_main():
            logger.info(f"=== START: Saving config ===")
            config = concretize(args)
            json_out_file = os.path.join(experiment_dir, "config.json")
            out_json_dict = json.loads(config.to_json())
            out_json_dict["resume"] = experiment_dir
            out_json_dict["data_shape"] = {
                "x": list(x.shape),
                "y": list(y.shape),
            }
            out_json_dict["workdir"] = workdir 
            out_json_dict["checkpoint_dir"] = checkpoint_dir 
            out_json_dict["experiment_dir"] = experiment_dir
            out_json_dict["expr_name"] = expr_name
            with open(json_out_file, "w") as fout:
                json.dump(out_json_dict, fout)
            logger.info("Complete Config: %s" % json_out_file)
            logger.info(f"=== END: Saving config ===")

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
        
        if args.get("profile", True):
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
                (train_steps < 100 and train_steps % 10 == 0)
                ):
                # Add the blocking time
                step_start_time = time.time()
                aux = jax.device_get(aux)
                loss_val = aux["loss_val"]
                gnorm = aux["grad_norm"]
                moe_loss = aux["loss_dict"]["moe_loss"]
                diff_loss = aux["loss_dict"]["loss"]
                stepping_time_meter.add_to_sum(time.time() - step_start_time)
                
                logging_start_time = time.time()
                
                # Measure training speed:
                steps_per_sec = (
                    (train_steps - start_iter) / (time.time() - ttl_train_time_start))

                logger.info(
                    f"(step={train_steps:07d})"
                    f" loss={loss_val:.4f} diff={diff_loss:.4f}"
                    f" moe={moe_loss:.4f} gnorm={gnorm:.4f}"
                    f" steps/s={steps_per_sec:.2f}"
                    f" time(load,train,log)=({loading_time_meter.avg:.4f},"
                    f"{stepping_time_meter.avg:.4f},"
                    f"{logging_time_meter.avg:.4f})")

                # # Plot inner loss if it exists
                if train_utils.is_main():
                    for k, v in {
                        "loss": loss_val,
                        "diff_loss": diff_loss,
                        "moe_loss": moe_loss,
                        "grad_norm": gnorm,
                        "steps_per_sec": steps_per_sec,
                        "loading_time": loading_time_meter.avg,
                        "stepping_time": stepping_time_meter.avg,
                        "logging_time": logging_time_meter.avg,
                    }.items():
                        writer.add_scalar(f"train/{k}", v, train_steps)
                        
                logging_time_meter.update(time.time() - logging_start_time)

            # Save model checkpoint:
            if ckpt_every > 0 and train_steps % ckpt_every == 0 and (
                train_steps > 0 or FLAGS.test_checkpoint):
                ckpt_start_time = time.time()
                logger.info(f"[{train_steps}] Wrap-up previous checkpoint")
                checkpoint_manager.wait_until_finished()
                train_utils.save_checkpoint(
                    train_steps, checkpoint_manager, ema_state, logger=logger)
                checkpoint_time_meter.update(time.time() - ckpt_start_time)

        # Save the final checkpoint
        if ckpt_every > 0:
            ckpt_start_time = time.time()
            logger.info(f"[{train_steps}] Wrap-up previous checkpoint")
            checkpoint_manager.wait_until_finished()
            logger.info("Start final checkpoint!")
            train_utils.save_checkpoint(
                train_steps, checkpoint_manager, ema_state, logger=logger)
            checkpoint_manager.wait_until_finished()
            checkpoint_time_meter.update(time.time() - ckpt_start_time)
            logger.info("Finished final checkpoint!")

        if FLAGS.multi_process:
            jax.distributed.shutdown()
        if args.get("profile", True):
            jax.profiler.stop_trace()
        logger.info("Done!")

if __name__ == '__main__':
    app.run(main)