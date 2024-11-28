"""
Testing: 
https://orbax.readthedocs.io/en/latest/orbax_checkpoint_101.html
https://jax.readthedocs.io/en/latest/multi_process.html
https://orbax.readthedocs.io/en/latest/async_checkpointing.html
"""
import flax
import os
import jax
import numpy as np
import os.path as osp
import jax.numpy as jnp
import orbax.checkpoint as ocp
from etils import epath
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


MULTI = True
if MULTI:
    jax.distributed.initialize()
    pid = jax.process_index()
    dc = jax.device_count()
else:
    pid = 0
    dc = 1
path = '/home/Grendel/outputs/my-checkpoints/'
# path = 'gs://dit-guandao/my-checkpoints'
# ckpt_mngr = ocp.CheckpointManager(path)

options = ocp.CheckpointManagerOptions(max_to_keep=1)
orbax_checkpointer = ocp.AsyncCheckpointer(
    ocp.PyTreeCheckpointHandler(), timeout_secs=50)
ckpt_mngr = ocp.CheckpointManager(path, orbax_checkpointer, options)

def compute(pid):
    ret = flax.core.frozen_dict.FrozenDict({
        'a': np.arange(100_000_000) + pid,
        'b': {
            'c': jax.process_index(),
            'd': jax.random.uniform(jax.random.PRNGKey(pid)) + np.arange(200_000_000),
        },
    })
    if MULTI:
        ret = jax.lax.pmean(ret, axis_name="devices")
    return ret 

def convert_array(x):
    if isinstance(x, jax.Array):
        return ocp.utils.fully_replicated_host_local_array_to_global_array(x)
    return x


for step in range(2):
    pids = pid * jnp.ones((jax.local_device_count(),), dtype=jnp.int32) * 2 + step
    if MULTI:
        my_tree = jax.pmap(compute, axis_name="devices")(pids)
    else:
        my_tree = compute(pids)
    # print(f"[{pid}/{dc}]", my_tree)
    # print(f"[{pid}/{dc}]", my_tree['a'])
    print(f"[{pid}/{dc}] Saving.")
    # my_tree_to_save = my_tree
    # my_tree_to_save = ocp.utils.fully_replicated_host_local_array_to_global_array(my_tree)
    my_tree_to_save = jax.tree_util.tree_map(convert_array, my_tree)
    ckpt_mngr.save(step, args=ocp.args.PyTreeSave(my_tree_to_save))
print("Done")