import os
import jax
import flax
from flax.training import orbax_utils
from functools import partial
from pprint import pprint 
from flax import core, struct
import optax
from flax.training.train_state import TrainState
from typing import Any
import numpy as np
import jax.numpy as jnp
import logging
import importlib
import orbax
import tensorflow as tf 
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
from diffusion.models.layers.moe_mlp import compute_switch_loss
from diffusion.losses import create_diffusion
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from diffusion.utils import sharding 
import flax.linen as nn


def make_opt(args):
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) 
    # and a constant learning rate of 1e-4 in our paper):
    opt = optax.adamw(
        learning_rate=args.opt.lr, 
        weight_decay=args.opt.wd
    )
    if args.opt.grad_clip and args.opt.grad_clip > 0:
        opt = optax.chain(
            optax.clip_by_global_norm(args.opt.grad_clip),
            opt)
    if args.get("optax_multistep", False):
        opt = optax.MultiSteps(
            opt, 
            every_k_schedule=args.get("grad_acc", 1),
            use_grad_mean=True,
        )
    return opt


class EMATrainState(struct.PyTreeNode):
    train_state: TrainState
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    ema_decay: float = 0.9999

    def apply_gradients(self, *, grads, **kwargs):
        train_state = self.train_state.apply_gradients(grads=grads, **kwargs)
        ema_params = jax.tree.map(
            # x + (1-d) * (y-x) = x * d + y * (1 - d)
            lambda x, y: x + (1.0 - self.ema_decay) * (y - x),
            self.ema_params,
            train_state.params,
        )
        return self.replace(
            train_state=train_state,
            ema_params=ema_params
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema_decay=0.9999, **kwargs):
        train_state = TrainState.create(
            apply_fn=apply_fn, params=params, tx=tx)
        return cls(
           train_state=train_state, ema_params=params, ema_decay=ema_decay)


def create_model(args):
    # model_name, latent_size, num_classes, latent_dim):
    use_latent = args.get("use_latent", True)
    if use_latent:
        latent_size = args.image_size // 8
    else:
        latent_size = args.image_size
    # Create model:
    model_lib = importlib.import_module(
        f"diffusion.models.{args.model.package_name}")
    
    model = model_lib.Model(
        input_size=latent_size,
        num_classes=args.num_classes, 
        in_channels=args.latent_dim,
        **args.model)
    return model
   
    
def init_model(rng, data, model):
    x, y = data
    x = jnp.ones((1, *x.shape[-3:])).astype(x.dtype)
    y = jnp.ones((1,)).astype(y.dtype)
    t = jnp.ones(shape=y.shape, dtype=jnp.float32)
    rng, spl = jax.random.split(rng)
    spl1, spl2, spl3, spl4 = jax.random.split(spl, 4)
    params = model.init(
        {'params': spl1, 'dropout': spl2, "label_emb": spl3, "mt3": spl4},
        x, t, y, training=True)
    return params 

    
def create_train_state(args, mesh, rng, x, y, return_model=False, logging=None):
    # Create model:
    model = create_model(args)
    tx = make_opt(args)
    
    def make_ema_state(rng, x, y):
        params = init_model(rng, (x, y), model)
        ema_state = EMATrainState.create(
            apply_fn=None, params=params, tx=tx, 
            ema_decay=args.ema_decay)
        return ema_state
    
    # Obtaining the ema_state shapes
    rng, spl = jax.random.split(rng)
    abstract_ema_state = jax.eval_shape(make_ema_state, spl, x, y)
    logical_spec_ema_state = nn.get_partition_spec(abstract_ema_state)
    ps_rule = sharding.get_logical_partition_rules()
    ema_state_sharding = nn.logical_to_mesh_sharding(
        logical_spec_ema_state, mesh, ps_rule)

    x_partition, y_partition = sharding.get_data_partition_rules()
    x_sharding = NamedSharding(mesh, x_partition)
    y_sharding = NamedSharding(mesh, y_partition)
    make_ema_state_sharded = jax.jit(
        make_ema_state,  
        in_shardings=(
            NamedSharding(mesh, PS()),  # rng
            x_sharding, y_sharding),    # x, y sharding
        out_shardings=ema_state_sharding,
        # check_rep=False
    )
    rng, spl = jax.random.split(rng)
    ema_state = make_ema_state_sharded(spl, x, y)
    if return_model:
        return ema_state, model
    return ema_state, ema_state_sharding, model


# Train step
def make_train_step(args, model):
    pid = jax.process_index()
    # default: 1000 steps, linear noise schedule
    if not hasattr(args, "loss") :
        diffusion = create_diffusion(timestep_respacing="")  
    else:
        diffusion = create_diffusion(**args.loss)  

    def loss_fn(params, rng, x, model_kwargs):
        model_fn = partial(model.apply, params)
        loss_dict, t, aux_dict = diffusion.training_losses(
            model_fn, rng, x, model_kwargs=model_kwargs)
        # Compute MoE loss
        moe_info = aux_dict.get('moe', None)
        moe_args = args.loss.get('moe', None)
        moe_loss = 0.
        if moe_info is not None and moe_args is not None:
            moe_loss, z_loss = compute_switch_loss(
                moe_info, moe_info["experts"], 
                use_z_loss=True)
            moe_loss = moe_args.get("moe_loss_weight", 0) * moe_loss
            if z_loss is not None:
                moe_loss += moe_args.get("z_loss_weight", 0) * z_loss

        loss = loss_dict["loss"].mean() + moe_loss

        # Add moe_loss to the aux_dict
        loss_dict['moe_loss'] = moe_loss

        return loss, {"loss_dict": loss_dict, "t": t, "aux": aux_dict}
   
    def compute_grad(ema_state, x, y, rng):
        rng_d, rng_l, rng_mt3, rng_loss = jax.random.split(rng, 4)
        model_kwargs = dict(
            training=True, y=y, 
            rngs={'dropout': rng_d, "label_emb": rng_l, "mt3": rng_mt3},
            return_aux=args.return_aux)
        # [grads] are gradient averaged across current micro/mini batch
        (loss_val, all_aux), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(
            ema_state.train_state.params,
            rng_loss, x, model_kwargs) 
        # pjit is able to add the proper pmean here
        # # Gradient averaged across all data samples at current micro/mini batch
        # grads = jax.lax.pmean(grads, axis_name="dp")
        # loss_val = jax.lax.pmean(loss_val, axis_name="dp")
        return grads, loss_val, all_aux
   
    def compute_stats(grads, loss_val, all_aux):
        # Compute statistics
        t = all_aux["t"].astype(jnp.int32)
        loss_dict = jax.tree_util.tree_map(jnp.mean, all_aux["loss_dict"])
        loss_dict_per_t = jax.tree_util.tree_map(
            lambda loss: jnp.zeros((diffusion.num_timesteps,)).at[t].add(loss),
            loss_dict)
        t_count = jnp.zeros((diffusion.num_timesteps,)).at[t].add(1)
        aux = {
            "loss_dict": loss_dict,
            "t_count": t_count,
            "loss_per_t": loss_dict_per_t,
            "aux": all_aux["aux"],
        }
        # These variable are already pmeaned and/or do not need to be pmeaned
        gnorm = optax.global_norm(grads)
        aux.update({
            "loss_val": loss_val,
            "grad_norm": gnorm,
        })
        return aux
    
    if args.grad_acc > 1 and not args.get("optax_multistep", False):
        # Use scan to compute the gradient accumulation 
        # TODO: need to enforce ema_state is sharded!
        @jax.jit
        def _scan_(carry, data, ema_state):
            mini_x, mini_y, mini_rng = data
            if pid == 0:
                print(f"Local data shapes inside (gradacc={args.grad_acc}):")
                print(mini_x.shape, mini_y.shape)
            # micro_grad, loss_val, all_aux = compute_grad(...)
            # carry = jax.tree_util.tree_map(
            #     lambda a, g: a + g / float(args.grad_acc), carry, micro_grad)
            # return acc, (loss_val, all_aux)
            carry_out = compute_grad(
                ema_state, mini_x, mini_y, mini_rng)
            carry = jax.tree_util.tree_map(
                lambda a, g: a + g / float(args.grad_acc), carry, carry_out)
            loss_val = carry_out[1]
            return carry, loss_val
      
        def train_step_fn(ema_state, x, y, rng):
            # param_count = sum(
            #     x.size for x in
            #     jax.tree_util.tree_leaves(ema_state.train_state.params))
            # local_param_shapes = jax.tree_util.tree_map(
            #     lambda x: x.shape, ema_state.train_state.params 
            # )
            # if pid == 0:
            #     print(f"DiT Parameters Sharded: {param_count:,}")
            #     print(f"Local shapes :")
            #     pprint(local_param_shapes)
            #     print(f"Local data shapes (with gradacc={args.grad_acc}):")
            #     print(x.shape, y.shape)
            x_mini_lst = x.reshape(args.grad_acc, -1, *x.shape[1:])
            y_mini_lst = y.reshape(args.grad_acc, -1, *y.shape[1:])
            # rng_mini_lst = jnp.repeat(rng[None, ...], args.grad_acc, axis=0)
            rng_mini_lst = jax.random.split(rng, args.grad_acc)
            
            carry_shapes = jax.eval_shape(
                compute_grad, ema_state, 
                x_mini_lst[0], y_mini_lst[0], rng_mini_lst[0]) 
            if is_main():
                print("Carry shapes:")
                pprint(carry_shapes)
            carry_init = jax.tree_util.tree_map(
                lambda x: jnp.zeros(
                    shape=x.shape, dtype=x.dtype), carry_shapes)
            # TODO: this copying does not contain sharding information
            (grads, loss_val, all_aux), _ = jax.lax.scan(
                partial(_scan_, ema_state=ema_state), carry_init, 
                (x_mini_lst, y_mini_lst, rng_mini_lst))
            ema_state = ema_state.apply_gradients(grads=grads)
            # loss_val = jnp.mean(loss_val_lst)
            # all_aux = jax.tree_util.tree_map(
            #     lambda x: jnp.mean(x, axis=0), aux_lst)
            out_aux = compute_stats(grads, loss_val, all_aux)
            return ema_state, out_aux
    else:
        def train_step_fn(ema_state, x, y, rng):
            # param_count = sum(
            #     x.size for x in
            #     jax.tree_util.tree_leaves(ema_state.train_state.params))
            # local_param_shapes = jax.tree_util.tree_map(
            #     lambda x: x.shape, ema_state.train_state.params 
            # )
            # if pid == 0:
            #     print(f"DiT Parameters Sharded: {param_count:,}")
            #     print(f"Local shapes :")
            #     pprint(local_param_shapes)
            #     print(f"Local data shapes (outside):")
            #     print(x.shape, y.shape, rng.shape)
            grads, loss_val, all_aux = compute_grad(ema_state, x, y, rng)
            ema_state = ema_state.apply_gradients(grads=grads)
            out_aux = compute_stats(grads, loss_val, all_aux)
            return ema_state, out_aux
    return train_step_fn


def save_checkpoint(
    train_steps, checkpoint_manager, ema_state, logger=None):
    
    def _log_(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    
    _log_(f"Checkpoint saving at step: {train_steps}.")
    _log_(f"[{train_steps}] Wrap-up previous checkpoint")
    checkpoint_manager.wait_until_finished()
    _log_(f"[{train_steps}] Saving ...")
    ckpt = {"state": ema_state, "step": int(train_steps)}
    checkpoint_manager.save(train_steps, ckpt)
    _log_(f"Saved checkpoint to {checkpoint_manager.directory}")
   
    
def restore_checkpoint(
    checkpoint_manager, ema_state, logger=None, resume_step=None):
    
    def _log_(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
            
    ckpt_step = checkpoint_manager.latest_step()  # step = 4
    if resume_step is not None:
        _log_(f"Resuming from a specific step {resume_step}")
        all_steps = checkpoint_manager.all_steps()
        assert resume_step in all_steps, "%d %s" % (resume_step, all_steps)
        ckpt_step = resume_step
        
    _log_(os.listdir(checkpoint_manager.directory))
    _log_(f"Restoring from {checkpoint_manager.directory} at step {ckpt_step}")
    abstract_state = {"state": ema_state, 'step': -1}
    restore_args = orbax_utils.restore_args_from_target(abstract_state)
    ckpt = checkpoint_manager.restore(
        ckpt_step, items=abstract_state, 
        restore_kwargs={'restore_args': restore_args})
    ema_state = ckpt["state"]
    start_iter =  ckpt["step"] + 1
    _log_("Restored!")
    return ema_state, start_iter

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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    pid = jax.process_index()
    # TODO: this gets permission denied when flush into a gs:// bucket log file.
    # logging_dir = os.path.join(logging_dir, str(pid))
    # os.makedirs(logging_dir, mode=0o777, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[\033[34m%(asctime)s pid={pid}\033[0m] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        # handlers=[logging.StreamHandler(),
        #           logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger