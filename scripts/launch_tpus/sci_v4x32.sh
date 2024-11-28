export USER_DIR=`pwd`
export USER_NAME=`whoami`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/${BUCKET_NAME}
export CODE_DIR="/mnt/disks/nfs/${USER_NAME}/jax-DiT/"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT/"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/mnt/disks/data"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

fusermount -u $BUCKET_MNT 
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
gcsfuse --implicit-dirs \
  -o rw \
  --stat-cache-ttl 10s \
  --type-cache-ttl 10s \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT

rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

# Create code directory
rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

# data_loader="imagenet_pixel_tfds" \ # 0.3 s/step bf cache
# data_loader="imagenet_pixel_tfds_unsharded" \ # 1.5 s/step bf cache 5xfaster

psize=2
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
python main_xm.py \
  --config diffusion/configs_mlc/latent/imgnet256_f8.py:dit_s \
  --config.data_dir=${DATA_DIR} \
  --config.global_batch_size=32 \
  --config.ckpt_every=10 \
  --config.log_every=5 \
  --config.total_iters=100 \
  --workdir=${BUCKET_MNT}/xm/ \
  --multi_process=True \
  --tpu=True \
  --gpu=False \
  --dp_dim=4 \
  --fsdp_dim=1 \
  --tp_dim=4

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main_xm.py \
#   --config diffusion/configs_mlc/latent/imgnet256_f8.py:uvit \
#   --config.data_dir=${DATA_DIR} \
#   --workdir=${BUCKET_MNT}/xm/ \
#   --multi_process=True \
#   --tpu=True \
#   --gpu=False \
#   --dp_dim=4 \
#   --fsdp_dim=1 \
#   --tp_dim=4

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-name config_pixels \
#   data_dir=${DATA_DIR} \
#   data_loader="imagenet_pixel_tfds" \
#   feature_path="imagenet256_pixels_tfdata_sharded/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_s \
#   model.patch_size=${psize} \
#   model.attn_kwargs.use_ar_mask=False \
#   +grad_clip=1 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   dp_dim=4 \
#   fsdp_dim=1 \
#   tp_dim=4 \
#   grad_acc=1


# # python test_loader.py
# python -m datasets.imagenet_pixel_tfds

# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# python main.py \
#   --config-name config_imgnet_only \
#   expr_name="dit/ldm-latents-epoch2/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_s \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=16 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1


# # python test_loader.py
# EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-S-2024-Oct-03-04-36-58"
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   dp_dim=16 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   +cache=True

#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   data_loader="imagenet_feature_tfds_sharded" \
#   +cache=True \
# [2024-10-04 05:59:08 pid=0] (step=0001500) loss=0.8284 diff=0.8284 moe=0.0000 gnorm=0.5630 
# steps/s=11.67 time(load,train,log)=(0.0304,0.0551,0.0015)I

#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   data_loader="imagenet_feature_tfds_sharded" \
# [2024-10-04 06:15:53 pid=0] (step=0002000) loss=0.7480 diff=0.7480 moe=0.0000 gnorm=1.6161 
# steps/s=8.82 time(load,train,log)=(0.0655,0.0477,0.0016)# 

# psize=8
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-name config_pixels \
#   expr_name="test/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_pixels_tfdata_sharded/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_s \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=8 \
#   +fsdp_dim=1 \
#   +tp_dim=2 \
#   grad_acc=1 \
#   global_batch_size=256


# psize=8
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# python main.py \
#   --config-name config_imgnet_only \
#   expr_name="test/dit/ldm-latents-epoch2/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_s \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=16 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=32
# 

# EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-S-2024-Oct-02-06-20-14"
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   dp_dim=16 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   +cache=True


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-name config_pixels \
#   expr_name="test/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_s \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=1 \
#   +fsdp_dim=8 \
#   +tp_dim=2 \
#   grad_acc=1 \
#   global_batch_size=16 \
#   ckpt_every=-1 \
#   total_iters=5
# 