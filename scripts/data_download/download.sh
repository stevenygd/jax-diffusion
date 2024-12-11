#! /bin/bash

curr_dir=`pwd`

save_dir=`pwd`/data
mkdir -p ${save_dir}/imagenet512_flax_tfdata_sharded
cd ${save_dir}/imagenet512_flax_tfdata_sharded
for i in $(seq -f "%05g" 0 1001); do
    wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet512_flax_tfdata_sharded/${i}-of-01001.tfrecords
done
cd $curr_dir

# mkdir -p imagenet256_flax_tfdata_sharded
# cd imagenet256_flax_tfdata_sharded
# for i in $(seq -f "%05g" 0 1251); do
#     wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet256_flax_tfdata_sharded/${i}-of-01251.tfrecords
# done
# cd $curr_dir
~                  