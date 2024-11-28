USER_DIR=`pwd`
BUCKET_NAME=sci-guandao
export BUCKET_MNT="/mnt/disks/gs/${BUCKET_NAME}"
CODE_DIR="/checkpoint/guandao/jax-DiT"
LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
HYDRA_DIR=${USER_DIR}/hydra
WANDB_DIR=${USER_DIR}/wandb
DATA_DIR="/persistent_guandao"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

fusermount -u $BUCKET_MNT 
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chmod -R 777 $BUCKET_MNT
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


# Mon Oct 14
export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
psize=2
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config_pixels \
  expr_name="ssm-smpdiff/pixels-balanced/ps${psize}" \
  data_loader="imagenet_pixel_tfds" \
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  +cache=True \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model=ssm_s_uvit_smpdiff \
  model.patch_size=${psize} \
  dp_dim=256 \
  fsdp_dim=1 \
  tp_dim=1 \
  ckpt_every=5000 \
  max_ckpt_keep=70 \
  total_iters=400000


# # Thur Oct 10
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm-smpdiff/pixels-balanced-unsharded/ps${psize}" \
#   data_loader="imagenet_pixel_tfds_unsharded" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=ssm_b_4k_uvit_smpdiff \
#   dp_dim=256 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=70 \
#   total_iters=400000


# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v5/pixels-balanced/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_l \
#   dp_dim=256 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# EXPR_NAME="ttt/pixels/ps2-TTT-L-lmbd-mlp-2024-Sep-19-21-37-58"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   +profile_dir=null \
#   resume=True


# EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-S-2024-Oct-02-06-20-14"
# psize=1
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   dp_dim=256 \
#   fsdp_dim=1 \
#   tp_dim=1


# EXPR_NAME="ttt-mlp-bd/pixels/ps2-DiT-B-mt3lm-mlp-bd-2024-Aug-31-06-25-38"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   +dp_dim=256 \
#   +fsdp_dim=1 \
#   tp_dim=1 \
#   resume=True \
#   ckpt_every=10000 \
#   max_ckpt_keep=100 \
#   total_iters=800000


# # Sep 3
# EXPR_NAME="ttt-mlp-bd/pixels/ps2-DiT-B-mt3lm-mlp-bd-2024-Aug-31-06-25-38"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   resume=True \
#   ckpt_every=10000 \
#   total_iters=400000

# # L is too big :(
# psize=2
# JAX_TRACEBACK_FILTERING=off PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_l \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000
