import jax
import tqdm
import numpy as np
import os.path as osp
import tensorflow as tf
import jax.numpy as jnp
from diffusion.utils import sharding
from jax.sharding import NamedSharding
tf.config.experimental.enable_op_determinism()


def create_tf_dataset(
    record_file, local_batch_size, num_data_shards, shard_idx,
    seed=0, cache=True, shuffle=True, repeat=True):
    """
        [record_file] is a pattern of <dir_name>/%0.5d-of-%0.5d.tfrecords
    """
    files = tf.io.matching_files(record_file)
    files = tf.random.shuffle(files, seed=seed)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shard(num_shards=num_data_shards, index=shard_idx)
    if cache:
        raw_ds = tf.data.TFRecordDataset(files).cache() 
    else:
        raw_ds = shards.interleave(
            tf.data.TFRecordDataset, 
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Create a dictionary describing the features.
    random_flip = tf.keras.layers.RandomFlip("horizontal", seed=seed)
    normalize = tf.keras.layers.Normalization(
            axis=-1, mean=[0.5, 0.5, 0.5],
            variance=[0.5 ** 2, 0.5 ** 2, 0.5 ** 2])
    def _parse_fn_(example_proto):
        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenFeature([], tf.string), 
        }
        parsed_ex = tf.io.parse_single_example(
            example_proto, feature_description)
        x = tf.io.decode_jpeg(parsed_ex["x"], channels=3)
        x = tf.cast(x, tf.float32) / 255.0
        x = random_flip(x)
        x = normalize(x)
        x = tf.transpose(x, [2, 0, 1])
        y = parsed_ex["y"]
        return x, y

    ds = raw_ds.map(_parse_fn_, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(local_batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_data_loader(args, mesh, **kwargs):
    record_file = osp.join(args.data_dir, args.feature_path, "*tfrecords*")
    if "rng" in kwargs:
      seed = int(
        jax.random.randint(
            kwargs["rng"], shape=(1,), minval=0, maxval=100_000)[0]
      )
    else:
      seed = 0
      
    print("Pixel dataloader seed", seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    # Only replicate the model parallel dimension
    mp = int(args.tp_dim)
    nproc = jax.process_count()
    pid = jax.process_index()
    global_batch_size = args.global_batch_size
    local_device_count = jax.local_device_count()
    num_proc_per_shard = max(1, mp // local_device_count)
    num_data_shards = nproc // num_proc_per_shard
    shard_idx = pid // num_proc_per_shard # assign each pidx to shardes
    local_batch_size = global_batch_size // num_data_shards
    ds = create_tf_dataset(
      record_file, local_batch_size, 
      num_data_shards=num_data_shards,
      shard_idx=shard_idx,
      seed=seed,
      cache=args.get("cache", False)
    )
    
    x_partition, y_partition = sharding.get_data_partition_rules()
    x_sharding = NamedSharding(mesh, x_partition)
    y_sharding = NamedSharding(mesh, y_partition)
    def make_jnp_array():
      for x, y in iter(ds):
        x, y = jnp.asarray(x), jnp.asarray(y)
        x = jax.make_array_from_process_local_data(x_sharding, x)
        y = jax.make_array_from_process_local_data(y_sharding, y)
        yield x, y
    loader = make_jnp_array()
    return ds, loader 


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from PIL import Image
    import jax.numpy as jnp
    import numpy as np 
    import lovely_jax
    import time
    lovely_jax.monkey_patch()
    
    args = OmegaConf.create({
        "data_dir": "data/",
        "feature_path": "imagenet256_pixels_tfdata_sharded/",
        "image_size": 256,
        "global_batch_size": 256
    })
    record_file = osp.join(args.data_dir, args.feature_path, "*.tfrecords")
    ds = create_tf_dataset(
      record_file, args.global_batch_size, num_devices=1, seed=0,
      cache=False, shuffle=False, repeat=False
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
    
    
    # # rng = jax.random.PRNGKey(int(time.time()))
    # rng = jax.random.PRNGKey(0)
    # ds, dsiter = make_data_loader(args, rng=rng) 
    # print(ds)
    # for i in range(2):
    #     x, y = next(dsiter)
    #     # x, y = jnp.array((x + 1) * 0.5), jnp.array(y)
    #     x, y = jnp.array(x), jnp.array(y)
    #     print(f"i={i}")
    #     print("X", x)
    #     print("Y", y)
    # for i in range(64):
    #     Image.fromarray(
    #         np.array(
    #             x.reshape(-1, *x.shape[2:])[i]
    #         ).transpose((1, 2, 0)).astype("uint8")
    #     ).save(f"tmp/test{i}.png")
    #     # print(f"Saved to tmp/test{i}.png")
    # # tfds.benchmark(ds, num_iter=100)