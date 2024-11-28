import jax
import numpy as np
import tensorflow as tf
from jax.sharding import Mesh

jax.distributed.initialize()
nproc = jax.process_count()
pid = jax.process_index()
p_pidx = 0

# Will only replicate across the model dimension
global_batch_size = 16
local_device_count = jax.local_device_count()
global_device_count = jax.device_count()
# dp, fsdp, mp = 4, 1, 4
dp, fsdp, mp = 2, 1, 8
# dp, fsdp, mp = 1, 1, 16
# assert mp % jax.local_device_count() == 0, "Duplicate = multiple of process."
# All devices
device_array = np.array(jax.devices()).reshape((dp, fsdp, mp))
mesh = Mesh(device_array, ("dp", "fsdp", "mp"))
print("=" * 80)
print("Mesh")
print(mesh)
print(device_array)
print("=" * 80)

# Will parallelize to (dp, fsdp) axies, so replicate the mp process
# So the data will be sharded into [dp] * [fsdp] shards
# NOTE: each process is going to have it's own dataloader
#       each dataloader is going to take a shard of the dataset
# NOTE: total number of shards for the dataset depends on how many dataloaders
#       are sharing data. For example:
# Mesh: 16,1,1  -> no data replication, each process bs = gbs//nprocess
# Mesh: 8, 1,2  -> no data replication, each process bs = gbs//nprocess
# Mesh: 4, 1,4  -> no data replication, each process bs = gbs//nprocess
# Mesh, 2, 1,8  -> 4 processes, 2 shards of the dataset, each 8 devices shared a shard
#                  2 process sharded a batch, bs = gbs//nprocess*2
num_data_shard = nproc // max(1, mp // local_device_count)
# shard_idx = pid % num_data_shard # assign each pidx to shardes
shard_idx = pid // num_data_shard # assign each pidx to shardes
local_batch_size = global_batch_size // num_data_shard
print("PID %d SID:%d bs=%d" % (pid, shard_idx, local_batch_size))

################################################################################
# Step 1: setup the Dataset for pure data parallelism (do once)
################################################################################
# Fake example data (replace with your Dataset)
ds = tf.data.Dataset.from_tensor_slices(
    [np.ones((1,)) * i for i in range(128)]
)

# if pid == p_pidx:
#     print("-" * 80)
#     print("Not sharded")
#     print("-" * 80)
#     print(ds)
#     for x in ds:
#         print(x)
#     print("-" * 80)
 
# ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())
ds = ds.shard(num_shards=num_data_shard, index=shard_idx)
ds = ds.batch(local_batch_size)

# # if pid == p_pidx:
# print("-" * 80)
# print("Sharded %d" % pid)
# print("-" * 80)
# print(ds)
# for x in ds:
#     print(x)
# print("-" * 80)

################################################################################
# Step 2: create a jax.Array of per-replica batches from the per-process batch
# produced from the Dataset (repeat every step). This can be used with batches
# produced by different data loaders as well!
################################################################################
# Grab just the first batch from the Dataset for this example
per_process_batch = ds.as_numpy_iterator().next()

sharding_init = jax.NamedSharding(
    mesh, jax.sharding.PartitionSpec(("dp", "fsdp")))
global_batch_array = jax.make_array_from_process_local_data(
    sharding_init, per_process_batch)

print("-" * 80)
print("PID=%d" % pid, "global array")
print("-" * 80)
print("Local device:", jax.local_devices())
print("Perprocess batch", per_process_batch.shape, per_process_batch[:, 0])
print(global_batch_array.sharding)
print(global_batch_array.shape)
print(global_batch_array.dtype)
print("Shard:", global_batch_array.addressable_shards)
print("-" * 80)
