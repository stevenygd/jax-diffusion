#! /bin/bash

curr_dir=`pwd`

save_dir=`pwd`/data
mkdir -p $save_dir/imagenet512_flax_tfdata_sharded
for i in $(seq -f "%05g" 0 1001); do
    wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet512_flax_tfdata_sharded/${i}-of-01001.tfrecords -P ${save_dir}/imagenet512_flax_tfdata_sharded/ &
done