import os
import tqdm
import glob
import shutil
import numpy as np
import os.path as osp
import tensorflow as tf
import jax.numpy as jnp
import tensorflow_datasets as tfds


# Borrow from this: https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def make_tf_example(features, label, d, w, h):
  feature = {
      'y': _int64_feature(label),
      "x": _bytes_feature(tf.io.serialize_tensor(features))
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tf_record(
  data_root, record_file, max_num_data=-1, 
  n_shards=1_000, min_exs_per_shard=256):
  print("Data root:", data_root)
  print("Record_file format:", record_file)
  shutil.rmtree(osp.dirname(record_file), ignore_errors=True)
  os.makedirs(osp.dirname(record_file))
  
  # Get all the files, shard into chunks
  fnames = [osp.basename(f) for f in glob.glob(f"{data_root}/features/*.npy")]
  if max_num_data > 0:
    fnames = fnames[:max_num_data]
   
  # Sharding
  exs_per_shard = max(min_exs_per_shard, int(np.ceil(len(fnames) // n_shards)))
  num_shards = int(np.ceil(len(fnames) // exs_per_shard))
  assert len(fnames) - num_shards * exs_per_shard < exs_per_shard
  sharded_fnames = [
      fnames[i*num_shards:min(len(fnames), i*num_shards+exs_per_shard)]
      for i in range(num_shards)
    ]
  print("#examples=%d, #shards=%d, #ex/shard=%d" 
        % (len(fnames), num_shards, exs_per_shard))
 
  # TODO: this can be done in parallel 
  features_shape = None
  for shard_id, fnames_shard in tqdm.tqdm(
    enumerate(sharded_fnames), total=num_shards): 
    record_file_sharded = record_file % (shard_id, num_shards)
    print(record_file_sharded)
    with tf.io.TFRecordWriter(record_file_sharded) as writer:
        for fname in tqdm.tqdm(fnames_shard, leave=False):
            feature_file = osp.join(data_root, "features", fname)
            features = np.load(feature_file)[0]
            features_shape = features.shape
            d, w, h = features_shape
            label_file = osp.join(data_root, "labels", fname)
            label = np.load(label_file)[0]
            tf_example = make_tf_example(features, label, d, w, h)
            writer.write(tf_example.SerializeToString())
    
  # # Finally, save dataset statis
  # np.save(osp.join(osp.dirname(record_file), "stats.npy"), {
  #   "features_shape":  features_shape
  # })
    

def benchmark_reading_tf_dataset(record_file):
  """
    [record_file] is a pattern of <dir_name>/%0.5d-of-%0.5d.tfrecords
  """
  files = tf.io.matching_files(record_file)
  files = tf.random.shuffle(files)
  shards = tf.data.Dataset.from_tensor_slices(files)
  raw_ds = shards.interleave(tf.data.TFRecordDataset)
  raw_ds = raw_ds.shuffle(buffer_size=10000)

  # Create a dictionary describing the features.
  def _parse_fn_(example_proto):
    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.string), 
    }
    parsed_ex = tf.io.parse_single_example(example_proto, feature_description)
    return {
      "x": tf.io.parse_tensor(parsed_ex["x"], out_type=tf.float32),
      "y": parsed_ex["y"], 
    }

  ds = raw_ds.map(_parse_fn_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(256).prefetch(buffer_size=tf.data.AUTOTUNE)
  for raw_record in ds.take(10):
      print(raw_record["x"].shape, raw_record["x"].dtype)
      print(raw_record["y"].shape, raw_record["y"].dtype)
  tfds.benchmark(ds, batch_size=256)
  
  
if __name__ == "__main__":
  # Gather filenames
  root = "/home/guandaoyang/fast-DiT"
  data_root = f"{root}/data/imagenet_features/"
  record_file = f'{root}/data/imagenet256_tfdata_sharded/%0.5d-of-%0.5d.tfrecords'
  create_tf_record(data_root, record_file)
  # create_tf_record(data_root, record_file, max_num_data=10_002)
  record_file = f'{root}/data/imagenet256_tfdata_sharded/*.tfrecords'
  benchmark_reading_tf_dataset(record_file)