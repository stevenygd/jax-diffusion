export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs-sci-guandao
export CODE_DIR="/checkpoint/guandao/jax-DiT"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/persistent"
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

rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR


# Oact 21
export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config_pixels \
  expr_name="uvit-ssm/pixels-balanced/uvit" \
  data_loader="imagenet_pixel_tfds" \
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  +cache=True \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model=uvit_ssm_s \
  grad_acc=2 \
  dp_dim=128 \
  fsdp_dim=1 \
  tp_dim=1 \
  ckpt_every=10000 \
  max_ckpt_keep=35 \
  total_iters=400000


# # Oact 16
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=8
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm-smpdiff/pixels-balanced/ps${psize}" \
#   data_loader="imagenet_pixel_tfds" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=ssm_b_uvit_smpdiff \
#   model.grad_checkpoint=False \
#   model.dit_block_kwargs.grad_checkpoint_attn=False \
#   model.dit_block_kwargs.grad_checkpoint_mlp=False \
#   model.dit_block_kwargs.grad_checkpoint_adamlp=False \
#   grad_acc=1 \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# # Oct 13
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# EXPR_NAME="ssm/pixels-balanced/ps4-SSM-S-2024-Oct-09-08-06-01"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   resume=True \
#   multi_process=True
# 


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
#   model=ssm_s_4k_uvit_smpdiff \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm/pixels-balanced/ps${psize}" \
#   data_loader="imagenet_pixel_tfds" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=ssm_s_16k \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm/pixels-balanced/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=ssm_s_16k \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000

# psize=2
# export DATA_DIR="${BUCKET_MNT}/data/"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v3/pixels-balanced-sharded-cached/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_pixels_tfdata_sharded/" \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_s \
#   +grad_clip=1 \
#   +dp_dim=128 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000
# 

# EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-B-2024-Oct-03-04-47-18"
# psize=1
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1
# 

# psize=1
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-name config_imgnet_only \
#   expr_name="dit/ldm-latents-epoch2/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_l \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=128 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main_fsdp.py \
#   --config-name config_pixels \
#   expr_name="ttt-v3/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_b \
#   +grad_clip=1 \
#   +dp_dim=128 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000

# Sep 8
# EXPR_NAME="ttt/pixels/ps2-TTT-S-lmbd-mlp-2024-Sep-04-21-16-01"
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
#   max_ckpt_keep=35 \
#   total_iters=400000

# # Sep 3
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_s \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# # Aug 30
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-mlp-bd/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_s \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# # Aug 25
# EXPR_NAME="rectflow-lognorm/pixels/ps2-DiT-Ti-mt3lm-mlp-bd-2024-Aug-23-02-54-58"
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
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=8 \
#   ckpt_every=10000 \
#   resume_step=190000 \
#   total_iters=1000000


# # Aug 22
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   loss=rectflow \
#   model="dit_tiny_mt3lm_mlp_bd" \
#   +model.learn_sigma=False \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=8 \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
 
# # Aug 20
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin-lm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mt3lm_lin_bd" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000
# 

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/pixels/ps${psize}_e2" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_ssm" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   model.attn_kwargs.d_expansion=2 \
#   total_iters=1000000
# 

# patch_size=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="SA/l${patch_size}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   image_size=512 \
#   model="dit_b" \
#   model.patch_size=${patch_size} \
#   total_iters=1000000

# psize=1
# niters=4
# for ilr in 1.0 0.1 0.01; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="mt3-mlp-niters${niters}-ilr${ilr}/l${psize}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_mttt_mlp_v0" \
#     model.patch_size=${psize} \
#     model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     total_iters=400000
# done

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm-vae/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_ssm" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vae" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000

# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="fixloader-linattn/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   model.patch_size=${psize} \
#   total_iters=1000000
# 
# # EXPR_NAME="linattn2-DiT-S-linattn-2024-Jul-14-05-23-47"
# EXPR_NAME="noscaleq-DiT-B-linattn-2024-Jul-12-08-43-22"
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
#   ckpt_every=25000 \
#   total_iters=4000000
