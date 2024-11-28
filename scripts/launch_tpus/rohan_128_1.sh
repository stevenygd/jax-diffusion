export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
<<<<<<< HEAD
export BUCKET_MNT=/mnt/gs/${BUCKET_NAME} # /mnt/disks/gs already broken :-)
=======
export BUCKET_MNT=/mnt/disks/gs/${BUCKET_NAME}
>>>>>>> uvit
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

<<<<<<< HEAD
# Sep 13
psize=2
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config_pixels \
  expr_name="ttt/pixels/ps${psize}" \
=======

# Oact 14
export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
psize=2
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config_pixels \
  expr_name="ttt-smpdiff/pixels-balanced/ps${psize}" \
  data_loader="imagenet_pixel_tfds" \
>>>>>>> uvit
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  +cache=True \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
<<<<<<< HEAD
  model=TTT_sb \
  model.patch_size=${psize} \
  +grad_clip=1.0 \
=======
  model=TTT_s_smpdiff_uvit \
  dp_dim=128 \
  fsdp_dim=1 \
  tp_dim=1 \
  model.patch_size=${psize} \
>>>>>>> uvit
  ckpt_every=10000 \
  max_ckpt_keep=35 \
  total_iters=400000


<<<<<<< HEAD
=======

# # Oct 13
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# EXPR_NAME="ssm-smpdiff/pixels-balanced-unsharded/ps4-SSM-S-2024-Oct-11-06-59-05"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model.dit_block_kwargs.grad_checkpoint_adamlp=True \
#   model.dit_block_kwargs.grad_checkpoint_mlp=True \
#   grad_acc=2 \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   resume=True \
#   multi_process=True
# 

# # Oact 13
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=4
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
#   model=ssm_s_4k_uvit_smpdiff \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# export EXPR_NAME="ttt-v3/pixels-balanced/ps2-TTT-S-lmbd-mlp-2024-Oct-03-09-02-21"
# export DATA_DIR="${BUCKET_MNT}/data/"
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
#   resume=True

# Oct 2
# psize=2
# export DATA_DIR="${BUCKET_MNT}/data/"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v3/pixels-balanced/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_pixels_tfdata_sharded/" \
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


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v3/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_sb \
#   +grad_clip=1 \
#   +dp_dim=32 \
#   +fsdp_dim=1 \
#   +tp_dim=4 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


# psize=2
# # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main_fsdp.py \
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_sb \
#   +grad_clip=1 \
#   +fsdp_dim=64 \
#   +mp_dim=-1 \
#   ckpt_every=10000 \
#   total_iters=400000


>>>>>>> uvit
# # Aug 28
# # XLA_PYTHON_CLIENT_MEM_FRACTION=.95 
# psize=2
# # XLA_PYTHON_CLIENT_ALLOCATOR=platform 
# HYDRA_FULL_ERROR=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name=config_pixels \
#   expr_name="rectflow-lognorm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_l_mt3lm_mlp_bd_memeff" \
#   model.patch_size=${psize} \
#   grad_acc=2 \
#   ckpt_every=10000 \
#   max_ckpt_keep=30 \
#   total_iters=400000


# # Aug 26
# EXPR_NAME="rectflow-lognorm/pixels/ps2-DiT-S-2024-Aug-25-17-33-48"
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
#   grad_clip=1 \
#   total_iters=1000000



# # TOO SLow, will find something else
# psize=1
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
#   model.grad_checkpoint=False \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=128 \
#   model.attn_kwargs.mttt_kwargs.max_sequence_length=65536 \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   grad_acc=2 \
#   ckpt_every=10000 \
#   max_ckpt_keep=30 \
#   total_iters=400000

# # Aug 25 - TOO SLOW! 11min 
# psize=1
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
#   model.grad_checkpoint=True \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=1 \
#   model.attn_kwargs.mttt_kwargs.max_sequence_length=65536 \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   max_ckpt_keep=30 \
#   total_iters=400000


# # Aug 22
# EXPR_NAME="mt3lm-mlp-bd-fixed/pixels/ps2-DiT-Ti-mt3lm-mlp-bd-2024-Aug-22-06-34-07"
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
#   total_iters=1000000

# # Aug 21 night
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lm-mlp-bd-fixed/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_tiny_mt3lm_mlp_bd" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000

# # Aug 21
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lm-mlp-bd/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_tiny_mt3lm_mlp_bd" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
 

# EXPR_NAME="mt3lin-bdar/latent/ps1-lr1.0-niters8-DiT-S-mt3lin-bdar-2024-Aug-20-21-05-56"
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
#   total_iters=1000000
# 

# psize=1
# niters=8
# for lr in 1.0 10 0.1; do
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin-bdar/latent/ps${psize}-lr${lr}-niters${niters}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mt3lin_bdar" \
#   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   model.patch_size=${psize} \
#   total_iters=400000
# done


# psize=2
# niters=64
# for lr in 1.0 10 0.1; do
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin-ar-vit/pixels/ps${psize}-lr${lr}-niters${niters}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_linear_ar" \
#   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=400000
# done

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm-vit/p${psize}" \
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
#   total_iters=1000000

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="fixloader-vae/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vae" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-s/p1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s" \
#   model.patch_size=4 \
#   total_iters=400000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="linattn-scaleqkv" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn_v1" \
#   total_iters=4000000
# 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="linattn-scaleqkv" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn_v1" \
#   total_iters=4000000
# 

# # Base 
# model_cfg=dit_s_mttt_mlp_v0
# n_iters=8
# ilr=1.0
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=4M-niters${n_iters}-ilr${ilr} \
#   model=${model_cfg} \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   opt.grad_clip=1.0 \
#   total_iters=4000000
