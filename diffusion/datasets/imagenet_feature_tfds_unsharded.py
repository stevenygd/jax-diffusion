import jax
import tqdm
import numpy as np
import os.path as osp
import tensorflow as tf
import jax.numpy as jnp
from diffusion.utils import sharding
from jax.sharding import NamedSharding
tf.config.experimental.enable_op_determinism()
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")


def create_tf_dataset(
  record_file, global_batch_size, num_processes=1, seed=0, 
  multiplier=1.,
  repeat=True, shuffle=True, cache=False
):
  """
    [record_file] is a pattern of <dir_name>/%0.5d-of-%0.5d.tfrecords
  """
  files = tf.io.matching_files(record_file)
  files = tf.random.shuffle(files, seed=seed)
  shards = tf.data.Dataset.from_tensor_slices(files)
  if cache:
    raw_ds = tf.data.TFRecordDataset(files).cache() 
  else:
    raw_ds = shards.interleave(tf.data.TFRecordDataset)

  # Create a dictionary describing the features.
  def _parse_fn_(example_proto):
    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.string), 
    }
    parsed_ex = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.io.parse_tensor(parsed_ex["x"], out_type=tf.float32)
    x = tf.multiply(x, multiplier)
    y = parsed_ex["y"]
    return x, y

  ds = raw_ds.map(_parse_fn_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(buffer_size=10_000, seed=seed)
  if repeat:
    ds = ds.repeat()
  if num_processes > 0:
    local_batch_size = global_batch_size // num_processes 
    ds = ds.batch(local_batch_size)
    ds = ds.batch(num_processes)
  else:
    ds = ds.batch(global_batch_size)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


def make_data_loader(args, mesh, **kwargs):
    record_file = osp.join(args.data_dir, args.feature_path, "*.tfrecords")
    if "rng" in kwargs:
      seed = int(
        jax.random.randint(
          kwargs["rng"], shape=(1,), minval=0, maxval=100_000)[0]
      )
    else:
      seed = 0
      
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    global_batch_size = args.global_batch_size
    num_processes = (
        kwargs["num_devices"] if "num_devices" in kwargs 
        else jax.local_device_count())
    ds = create_tf_dataset(
      record_file, global_batch_size, 
      num_processes=num_processes,
      multiplier=args.scalar,
      seed=seed,
      cache=args.get("cache", False)
    )
    
    x_partition, y_partition = sharding.get_data_partition_rules()
    x_sharding = NamedSharding(mesh, x_partition)
    y_sharding = NamedSharding(mesh, y_partition)
    def make_jnp_array():
      for x, y in iter(ds):
        x, y = jnp.asarray(x), jnp.asarray(y)
        x = jax.device_put(x, x_sharding)
        y = jax.device_put(y, y_sharding)
        yield x, y
    loader = make_jnp_array()
    return ds, loader
  
  
if __name__ == "__main__":
    import time
    import tensorflow_datasets as tfds
    from omegaconf import OmegaConf
    from PIL import Image
    import jax.numpy as jnp
    import numpy as np 
    import lovely_jax
    lovely_jax.monkey_patch()
    
    # data_dir = "data/" 
    # feature_path = "imagenet256_flax_tfdata_sharded/"
    # scalar = 1. # var=0.698
    
    data_dir = "/mnt/gs/sci-guandao/ldm/epoch2/"
    feature_path = "imagenet256_tfdata_sharded-v2/"
    scalar = 0.08
    
    args = OmegaConf.create({
        "image_size": 256,
        "global_batch_size": 256
    })
    rng = jax.random.PRNGKey(int(time.time()))
    record_file = osp.join(data_dir, feature_path, "*.tfrecords")
    ds = create_tf_dataset(
      record_file, args.global_batch_size, num_processes=1, seed=0, 
      multiplier=scalar,
      shuffle=True, repeat=True, cache=True
    )
    
    # Computing the class distribution
    iters = 0
    cnt = jnp.zeros(1000)
    max_iters = 10_000 
    pbar = tqdm.tqdm(ds, total=max_iters)
    try:
      for _, y in pbar:
        if iters > max_iters:
          break
        iters += 1
        y = jnp.array(y).astype(jnp.int32).reshape(-1)
        cnt = cnt.at[y].add(1)
        pbar.set_description("zero=%s" % int(jnp.where(cnt > 0, 0, 1).sum()))
    except Exception as e:
      print("Exist.")
    cnt = np.array(cnt)
    print("Class distribution", 
          cnt.min(), cnt.mean(), cnt.max(), "(var=%s)" % cnt.var())
    print("zero=%s" % int(jnp.where(cnt > 0, 1, 0).sum()))
    # np.save("imagenet_class_distribution_tfrecord.npy", np.array(cnt))
    import json 
    json.dump(
      sorted(list(enumerate(cnt.tolist())), key=lambda x: x[1]),
      open("imagenet_class_distribution_tfrecord.json", "w"))
    
    # # Computing the mean and variance of the dataset
    # # ds, dsiter = make_data_loader(args, data_dir, rng=rng) 
    # print(ds)
    # max_iters = 1_000_000 
    # cnt, x_mean, x2_mean, y_acc = 0., 0., 0., []
    # for x, y in tqdm.tqdm(ds, total=max_iters // args.global_batch_size):
    #   if cnt > max_iters:
    #     break
    #   x, y = x.numpy(), y.numpy()
    #   # print(x.shape, x.max(), x.min(), x.mean(), x.var(), y.shape)
    #   x = x.reshape(-1, x.shape[-3], x.shape[-2] * x.shape[-1])  
    #   bs = x.shape[0]
    #   y = y.reshape(-1)
    #   cnt_ = cnt + x.shape[0]
    #   x_mean = x_mean * (cnt / float(cnt_)) + x.mean(axis=(0, -2, -1)) * (float(bs) / float(cnt_))
    #   x2_mean = x2_mean * (cnt / float(cnt_)) + jnp.square(x).mean(axis=(0, -2, -1)) * (float(bs) / float(cnt_))
    #   cnt = cnt_
    #   
    # print("Mean", x_mean)
    # x_var = (x2_mean - jnp.square(x_mean)) * float(cnt/(cnt - 1))
    # print("Var", x_var)