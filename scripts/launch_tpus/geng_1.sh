export USER_DIR=/home/Grendel
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs/${BUCKET_NAME}
export CODE_DIR="/checkpoint/guandao/jax-DiT"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
export OUTPUT_DIR=${BUCKET_MNT} # NOTE: this is /home/<username>/outputs
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

# Thur Oct 12
export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
psize=8
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config_pixels \
  expr_name="ssm-smpdiff/pixels-balanced-unsharded/ps${psize}" \
  data_loader="imagenet_pixel_tfds_unsharded" \
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  +cache=True \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model=ssm_s_uvit_smpdiff \
  dp_dim=256 \
  fsdp_dim=1 \
  tp_dim=1 \
  model.patch_size=${psize} \
  ckpt_every=10000 \
  max_ckpt_keep=35 \
  total_iters=400000

# # Test
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v5/pixels-balanced/ps${psize}" \
#   data_loader="imagenet_pixel_tfds" \
#   data_dir=${DATA_DIR} \
#   feature_path=${FEATURE_PATH} \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_s \
#   dp_dim=256 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# export EXPR_NAME="ttt-v3/pixels-balanced/ps2-TTT-B-lmbd-mlp-2024-Oct-03-08-27-26"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   dp_dim=256 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   resume=True \
#   multi_process=True

# # Sep 19
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main_fsdp.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_l \
#   +grad_clip=1 \
#   +dp_dim=64 \
#   +fsdp_dim=1 \
#   +tp_dim=4 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


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
#   model=TTT_b \
#   +grad_clip=1 \
#   +dp_dim=256 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000

# TTT-L v3 resumed!
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
#   dp_dim=64 \
#   fsdp_dim=1 \
#   tp_dim=4 \
#   +profile_dir=null \
#   ckpt_every=5000 \
#   max_ckpt_keep=140
# 
# # Sep 19
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main_fsdp.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_l \
#   +grad_clip=1 \
#   +dp_dim=64 \
#   +fsdp_dim=1 \
#   +tp_dim=4 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


# EXPR_NAME="ttt/pixels/ps2-TTT-B-lmbd-mlp-2024-Sep-05-07-27-42"
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
#   model=TTT_b \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-mlp-bd/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_b \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# EXPR_NAME="rectflow-lognorm/pixels/ps2-DiT-B-mt3lm-mlp-bd-2024-Aug-26-05-41-15"
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
#   +resume_step=100000 \
#   +grad_clip=1 \
#   ckpt_every=10000 \
#   total_iters=1000000


# 
# # Aug 25
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   loss=rectflow \
#   model="dit_b_mt3lm_mlp_bd" \
#   +model.learn_sigma=False \
#   model.grad_checkpoint=True \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=1 \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   max_ckpt_keep=30 \
#   total_iters=400000
# 

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
#   model="dit_s_memeff" \
#   +model.learn_sigma=False \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
 

# EXPR_NAME="udit/pixels/ps2-DiT-S-2024-Aug-18-09-11-35"
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
 
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_memeff" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000


# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="udit/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="u_dit_s" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000
 

# patch_size=1
# niters=32
# for lr in 1.0 0.1; do
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="TTT-AR/l${patch_size}-niters${niters}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   image_size=512 \
#   model="dit_s_mttt_ar" \
#   model.patch_size=${patch_size} \
#   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   total_iters=400000 \
#   ckpt_every=25000
# done


# patch_size=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="SA/l${patch_size}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   image_size=512 \
#   model="dit_s" \
#   model.patch_size=${patch_size} \
#   total_iters=1000000

# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3-lin/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_linear" \
#   model.patch_size=${psize} \
#   total_iters=1000000
# 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3-lin-xavier/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_linear_v1" \
#   model.patch_size=${psize} \
#   total_iters=1000000
# 

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_ssm" \
#   model.patch_size=${psize} \
#   total_iters=1000000
# 

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="fixloader-sa/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s" \
#   model.patch_size=${psize} \
#   total_iters=1000000
# 

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="fixloader-linattn-vit/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000


# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_s_mttt_linear \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=4000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b/2" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b" \
#   model.patch_size=2 \
#   total_iters=4000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b-la/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b_linattn" \
#   model.patch_size=1 \
#   total_iters=1000000
# 
# model_cfg=dit_s_mttt_linear_v1
# for ilr in 1.0 0.1 0.01; do
# 
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
# 
# 
#   # learn W init 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True
# 
#   # LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# 
#   # Winit + LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=learnW-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# done


# model_cfg=dit_s_mttt_linear
# for n_iters in 8 16; do
# for ilr in 1.0 0.1 0.01; do
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=niters${n_iters}-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
# done
# done

# model_cfg=dit_s_mttt_mlp_v0
# n_iters=4
# for ilr in 1.0 0.1 0.01; do
# 
#   # # Base - gives NaN, also breaks (directly terminate)
#   # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=False
# 
#   # learn W init 
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=gclip1.0-learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     opt.grad_clip=1.0 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=False \
#     model.attn_kwargs.mttt_kwargs.enc_residual=False
# 
#   # # LN + res 
#   # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=LNres-ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=True
# 
#   # # Winit + LN + res (NOTE this already ran)
#   # PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=learnW-LNres-ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=True
# done