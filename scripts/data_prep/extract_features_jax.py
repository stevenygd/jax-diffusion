# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
```bash
python scripts/extract_features_jax.py \
    --data-path /mnt/Storage/data/imagenet/ILSVRC2012_img_train/ \
    --batch-size <make it large enough for your memory, 64 per 32GB/GPU>
```
A minimal training script for DiT using Flax.

https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_flax.py
https://huggingface.co/blog/sdxl_jax
"""
import jax
import flax
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import jax.numpy as jnp
from diffusers.models import FlaxAutoencoderKL
import tensorflow as tf
import tensorflow_datasets as tfds
import shutil
import os.path as osp


################################################################################
#                                 Image Preprocessing                         #
################################################################################

def center_crop_arr(image, image_size):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    crop_size = tf.minimum(height, width)
    offset_height = (height - crop_size) // 2
    offset_width = (width - crop_size) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)
    image = tf.image.resize(image, [image_size, image_size])
    return image

def preprocess(image, label, image_size):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = center_crop_arr(image, image_size)
    image = tf.image.random_flip_left_right(image)
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    return image, label

################################################################################
#                                 Extraction Loop                              #
################################################################################

def main(args):

    # Create autoencoder:
    print("\nCreating autoencoder...")
    assert args.image_size % 8 == 0, \
        "Image size must be divisible by 8 (for the VAE encoder)."
    
    vae_dir = args.vae_dir
    vae_config = np.load(os.path.join(vae_dir, 'config.npy'), allow_pickle=True).item()
    vae_params = np.load(os.path.join(vae_dir, 'params.npy'), allow_pickle=True).item()
    vae = FlaxAutoencoderKL.from_config(vae_config)

    # Setup data:
    print("Setting up data loader...")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False
    )
    dataset = dataset.map(lambda x, y: preprocess(x, y, args.image_size),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Create sharding
    print("Creating shards...")
    record_file = osp.join(osp.dirname(args.data_dir), "imagenet%d_flax_tfdata_sharded/%0.5d-of-%0.5d.tfrecords")
    record_file_dir = osp.dirname(record_file % (args.image_size, 0, 0))
    if osp.exists(record_file_dir):
        print(f"\tRemoving existing directory: {record_file_dir}")
        shutil.rmtree(record_file_dir, ignore_errors=True)
    os.makedirs(record_file_dir)
    n_shards = args.max_num_shards
    min_exs_per_shard = args.min_data_per_shard

    n_batches = tf.data.experimental.cardinality(dataset).numpy()
    exs_per_shard = max(int(np.ceil(min_exs_per_shard / args.batch_size)),
                        int(np.ceil(n_batches // n_shards)))
    num_shards = int(np.ceil(n_batches / exs_per_shard))
    assert n_batches - num_shards * exs_per_shard < exs_per_shard
    print("\n#examples=%d, #shards=%d, #ex/shard=%d"
          % (n_batches, num_shards, exs_per_shard))

    def extract(rng, x):
        rng = jax.random.fold_in(rng, jax.process_index())
        x = jnp.transpose(x, (0, 3, 1, 2)) # (device_bs, 256, 256, 3) -> (device_bs, 3, 256, 256)
        return vae.apply(
            {"params": vae_params},
            x, deterministic=False, rngs={"gaussian": rng},
            method=vae.encode
        ).latent_dist.sample(rng).transpose((0, 3, 1, 2)) * 0.18215 # (device_bs, 32, 32, 4) -> (device_bs, 4, 32, 32)
    extract = jax.pmap(extract, axis_name="devices")

    rng = jax.random.PRNGKey(args.global_seed)
    n_devices = jax.local_device_count()
    device_bs = args.batch_size // n_devices

    loader_iter = iter(enumerate(dataset))
    for shard_id in tqdm.tqdm(range(num_shards)):
        record_file_sharded = record_file % (args.image_size, shard_id, num_shards)
        print(record_file_sharded)
        with tf.io.TFRecordWriter(record_file_sharded) as writer:
            for _ in tqdm.tqdm(range(exs_per_shard), leave=False):
                try:
                    batch_id, (x, y) = next(loader_iter)
                    x_shape = x.shape[-3:]
                    x = jnp.array(x).reshape(n_devices, device_bs, *x_shape)
                    y = jnp.array(y).reshape(n_devices, device_bs)
                    rng_i = flax.jax_utils.replicate(
                        jax.random.fold_in(rng, batch_id))
                    z = extract(rng_i, x)
                    z_shape = z.shape[-3:]
                    z = z.reshape(args.batch_size, *z_shape)
                    y = y.reshape(-1)
                    for feature, label in zip(z, y):
                        feature = np.array(feature) # (4, 32, 32)
                        label = int(label)
                        tf_example = make_tf_example(feature, label)
                        writer.write(tf_example.SerializeToString())
                except StopIteration:
                    break


# Borrow from this: https://www.tensorflow.org/tutorials/load_data/tfrecord
# Create a dictionary with features that may be relevant.
def make_tf_example(features, label):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    feature = {
        'y': _int64_feature(label),
        "x": _bytes_feature(tf.io.serialize_tensor(features))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae-dir", type=str, required=True)

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--max-num-shards", type=int, default=1_000)
    parser.add_argument("--min-data-per-shard", type=int, default=1_000)
    args = parser.parse_args()

    main(args)