#! /bin/bash

curr_dir=`pwd`

save_dir=`pwd`/data

mkdir -p $save_dir/imagenet256_flax_tfdata_sharded
for i in $(seq -f "%05g" 0 1251); do
        wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet256_flax_tfdata_sharded/${i}-of-01251.tfrecords -P ${save_dir}/imagenet256_flax_tfdata_sharded/ &
done

# mkdir -p $save_dir/imagenet256_pixels_tfdata_sharded
# for i in $(seq -f "%05g" 0 1001); do
#         wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet256_pixels_tfdata_sharded/${i}-of-01001.tfrecords -P ${save_dir}/imagenet256_pixels_tfdata_sharded/ &
# done

mkdir -p $save_dir/imagenet256_reference
wget https://guanda-stanford-data.s3.us-east-2.amazonaws.com/dit/data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz -P ${save_dir}/imagenet256_reference &
~                                                                                                                                                              