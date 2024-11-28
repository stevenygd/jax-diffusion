# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
```bash
python scripts/extract_perpixel_features_jax.py \
    --data-path /mnt/Storage/data/imagenet/ILSVRC2012_img_train/ \
    --batch-size <make it large enough for your memory, 64 per 32GB/GPU>
```
A minimal training script for DiT using Flax.

https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_flax.py
https://huggingface.co/blog/sdxl_jax
"""
import lovely_jax
import jax
import flax
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import shutil
import os.path as osp
lovely_jax.monkey_patch()


################################################################################
#                                 Helper Functions                             #
################################################################################


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


################################################################################
#                                 Training Loop                                #
################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    # Create model:
    print("Creating model...")
    assert args.image_size % 8 == 0, \
        "Image size must be divisible by 8 (for the VAE encoder)."

    # Setup data:
    print("Setting up datatraining...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create sharding
    record_file = args.tfrecord_pattern
    record_file_dir = osp.dirname(record_file % (args.image_size, 0, 0))
    shutil.rmtree(record_file_dir, ignore_errors=True)
    os.makedirs(record_file_dir)
    n_shards = args.max_num_shards
    min_exs_per_shard = args.min_data_per_shard

    # Sharding
    n_batches = len(loader)
    exs_per_shard = max(int(np.ceil(min_exs_per_shard / args.batch_size)),
                        int(np.ceil(n_batches // n_shards)))
    num_shards = int(np.ceil(n_batches / exs_per_shard))
    assert n_batches - num_shards * exs_per_shard < exs_per_shard
    print("#examples=%d, #shards=%d, #ex/shard=%d total=%d len(ds)=%d"
          % (n_batches, num_shards, exs_per_shard, 
             exs_per_shard * num_shards, len(loader)))
    breakpoint()

    print("Start storing...")
    loader_iter = iter(enumerate(loader))
    for shard_id in tqdm.tqdm(range(num_shards)):
        record_file_sharded = record_file % (args.image_size, shard_id, num_shards)
        print(record_file_sharded)
        with tf.io.TFRecordWriter(record_file_sharded) as writer:
            for _ in tqdm.tqdm(range(exs_per_shard), leave=False):
                try:
                    _, (x, y) = next(loader_iter)
                    x = (jnp.array(x) * 255).astype(jnp.uint8).transpose((0, 2, 3, 1))
                    y = jnp.array(y)
                    # Map input images to latent space + normalize latents:
                    x = x.reshape(-1, *x.shape[-3:])
                    y = y.reshape(-1)
                    for i in range(x.shape[0]):
                        features = np.array(x[i]).reshape(*x.shape[-3:])
                        label = int(y[i])
                        tf_example = make_tf_example(features, label)
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
        # "x": _bytes_feature(tf.io.serialize_tensor(features))
        "x": _bytes_feature(tf.io.encode_jpeg(features))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512],
                        default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)

    # Sharding
    parser.add_argument(
        "--tfrecord-pattern", type=str,
        default="data/imagenet%d_pixels_tfdata_sharded/%0.5d-of-%0.5d.tfrecords")
    parser.add_argument("--max-num-shards", type=int, default=1_000)
    parser.add_argument("--min-data-per-shard", type=int, default=1_000)
    args = parser.parse_args()
    main(args)