"""
Utility for sharding
"""
import re
import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.interpreters import pxla


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """An extended version of jax.tree_util.tree_map, where the mapped function
    f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r), 
        tree, *rest, is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    """Returns a pytree of PartitionSpec according to rules. Supports handling
    Flax TrainState and Optax optimizer state.
    """

    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """Don't partition scalar values."""
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                if jax.process_index() == 0:
                    print(name, rule, leaf.shape, ps)
                return ps
        raise ValueError(f"Partition rule not found for param: {name}")

    return named_tree_map(get_partition_spec, params, sep="/")


def get_default_partition_rules():
    """
    Borrow from https://github.com/test-time-training/ttt-lm-jax/blob/
                ac8cdc8a43b811afe23c27d7e82eed34a747b19c/ttt/models/
                model.py#L263C5-L298C10
    
        Partition rules. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
    """
    return (
        # Embeddings
        # ("x_embedder/embedding/kernel", PS(None, "tp")),
        # ("y_embedder/embedding_table/embedding", PS(None, "tp")),
        # ("t_embedder/dense1/kernel", PS("fsdp", "tp")),
        # ("t_embedder/dense2/kernel", PS("tp", "fsdp")),
        
        # Rule #1 if using rematscan, two padded dimension
        ("blocks/rollout_blocks/mlp/fc1/kernel", PS(None, None, "fsdp", "tp")),
        ("blocks/rollout_blocks/mlp/fc2/kernel", PS(None, None, "tp", "fsdp")),
        ("blocks/rollout_blocks/adaLN_mlp/kernel", PS(None, None, "fsdp", "tp")),
        ("blocks/rollout_blocks/attn/qkv/kernel", PS(None, None, "fsdp", "tp")),
        ("blocks/rollout_blocks/attn/proj/kernel", PS(None, None, "tp", "fsdp")),
        
        # Rule #2 if only using scan
        # # Output Head
        ("final_layer/adaLN_mlp/kernel", PS("fsdp", "tp")),
        # # AdaLN MLP
        ("adaLN_mlp/kernel", PS(None, "fsdp", "tp")),
        # # Attention/TTT KQVO
        ("attn/qkv/kernel", PS(None, "fsdp", "tp")),
        ("attn/proj/kernel", PS(None, "tp", "fsdp")),
        # # Feedforward MLP
        ("mlp/fc1/kernel", PS(None, "fsdp", "tp")),
        ("mlp/fc2/kernel", PS(None, "tp", "fsdp")),
        # Rule #3 if only using scan
        
        # The rest
        (".*", PS(None)), # duplicate
    )
    
    # # Original code from:
    # # https://github.com/test-time-training/ttt-lm-jax/blob/main/ttt/
    # # models/model.py#L263C5-L298C10 
    # return (
    #         # Embeddings
    #         ("model/wte/embedding", PS("mp", "fsdp")),
    #         # Attention/TTT
    #         ("seq_modeling_block/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
    #         ("seq_modeling_block/wo/kernel", PS("mp", "fsdp")),
    #         # TTT
    #         ("seq_modeling_block/ttt_norm/scale", PS(None)),
    #         ("seq_modeling_block/ttt_norm/bias", PS(None)),
    #         ("seq_modeling_block/post_norm/scale", PS(None)),
    #         ("seq_modeling_block/post_norm/bias", PS(None)),
    #         ("seq_modeling_block/learnable_ttt_lr/kernel", PS(None)),
    #         ("seq_modeling_block/learnable_ttt_lr/bias", PS(None)),
    #         ("seq_modeling_block/ttt_dense_0", PS(None)),
    #         ("seq_modeling_block/ttt_dense_1", PS(None)),
    #         ("seq_modeling_block/ttt_bias_0", PS(None)),
    #         ("seq_modeling_block/ttt_bias_1", PS(None)),
    #         # SwiGLU MLP
    #         ("feed_forward/w1/kernel", PS("fsdp", "mp")),
    #         ("feed_forward/w2/kernel", PS("mp", "fsdp")),
    #         ("feed_forward/w3/kernel", PS("fsdp", "mp")),
    #         # RMS Norm
    #         ("seq_norm/kernel", PS(None)),
    #         ("ffn_norm/kernel", PS(None)),
    #         # Output Head
    #         ("model/ln_f/kernel", PS(None)),
    #         ("lm_head/kernel", PS("fsdp", "mp")),
    #         (".*", PS(None)),
    #     )
    
def get_names_from_parition_spec(partition_specs):
    """Return axis names from partition specs."""
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))
    return list(names)


def names_in_current_mesh(*names):
    """Check if current mesh axes contain these names."""
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)

    
def with_sharding_constraint(x, partition_specs):
    """A smarter version of with_sharding_constraint that only applies the
    constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        print(partition_specs, axis_names)
        x = _with_sharding_constraint(x, partition_specs)
    return x


# https://flax.readthedocs.io/en/latest/guides/parallel_training/
# flax_on_pjit.html#logical-axis-annotation
def get_logical_partition_rules():
    return [
        ("B", ("dp", "fsdp")),        # data parallel 
        ("N", None),        # seq parallel
        ("D", "tp"),
        # AdaLN MLP
        ("adaln/inp", "fsdp"),
        ("adaln/out", "tp"),
        # MLP rules
        ("mlp/feat", "fsdp"),
        ("mlp/hidden", "tp"),
        # KQVO rules
        ("qkv/inp", "fsdp"),
        ("qkv/out", "tp"),
        ("proj/inp", "tp"),
        ("proj/out", "fsdp"),
    ]

    
def get_data_partition_rules():
    # return PS(("dp", "fsdp"), None, None, None), PS(("dp", "fsdp"))
    return PS(("dp", "fsdp"), None, None, None), PS(("dp", "fsdp"))


def get_data_recplicate_multiple(args):
    dp_dim = int(args.get("dp_dim", -1))
    fsdp_dim = int(args.get("fsdp_dim", 1))
    tp_dim = int(args.get("tp_dim", 1))
    if tp_dim < 0:
        tp_dim = jax.device_count() // dp_dim // fsdp_dim
    return tp_dim


def get_mesh(args):
    dp_dim = int(args.get("dp_dim", -1))
    fsdp_dim = int(args.get("fsdp_dim", 1))
    tp_dim = int(args.get("tp_dim", 1))
    # First two axies will be within the same device
    # Last axies will equals to #hosts
    device_array = np.array(jax.devices()).reshape((dp_dim, fsdp_dim, tp_dim))
    mesh = Mesh(device_array, ("dp", "fsdp", "tp")) 
    if jax.process_index() == 0:
        print("Device array.")
        print(device_array)
    # device_array = np.array(jax.devices()).reshape((tp_dim, fsdp_dim, dp_dim))
    # mesh = Mesh(device_array, ("tp", "fsdp", "dp")) 
    return mesh
