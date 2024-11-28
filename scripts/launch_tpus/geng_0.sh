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


# Oct 16
export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
EXPR_NAME="ssm-smpdiff/pixels-balanced-unsharded/ps4-SSM-S-2024-Oct-11-06-59-05"
ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  --config-path . \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  dp_dim=256 \
  fsdp_dim=1 \
  tp_dim=1 \
  resume=True \
  multi_process=True \
  ckpt_every=5000 \
  max_ckpt_keep=70 \
  total_iters=400000


# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v5/pixels-balanced-unsharded/ps${psize}" \
#   data_loader="imagenet_pixel_tfds_unsharded" \
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

# # EXPR_NAME="ttt-mlp-bd/pixels/ps2-DiT-S-mt3lm-mlp-bd-2024-Aug-31-06-31-44"
# EXPR_NAME="ttt/pixels/ps2-TTT-S-lmbd-mlp-2024-Sep-04-21-16-01" # ttt-v2-S
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
#   +tp_dim=1 \
#   grad_acc=1 \
#   resume=True \
#   ckpt_every=10000 \
#   total_iters=1000000 \
#   max_ckpt_keep=100 \
#   +profile_dir=null


# EXPR_NAME="ttt/pixels/ps2-TTT-L-lmbd-mlp-2024-Sep-16-09-25-27"
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
#   dp_dim=64 \
#   fsdp_dim=1 \
#   tp_dim=4 \
#   grad_acc=1 \
#   resume=True \
#   ckpt_every=2500 \
#   max_ckpt_keep=140 \
#   +profile_dir=null


# # Sep 19
# EXPR_NAME="ttt-v3/pixels/ps2-TTT-XL-lmbd-mlp-2024-Sep-20-00-29-03"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   +profile_dir=null
# 

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
#   model=TTT_b \
#   +grad_clip=1 \
#   +dp_dim=256 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=70 \
#   ckpt_every=5000 \
#   total_iters=400000

# # Sep 17
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
#   model=TTT_bpp \
#   +grad_clip=1 \
#   +dp_dim=64 \
#   +fsdp_dim=2 \
#   +tp_dim=2 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


# EXPR_NAME="ttt/pixels/ps2-TTT-L-lmbd-mlp-2024-Sep-16-09-15-16"
# # EXPR_NAME="ttt/pixels/ps2-TTT-L-lmbd-mlp-2024-Sep-16-08-07-41"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main_fsdp.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   resume=True \
#   grad_clip=1 \
#   ckpt_every=10000 \
#   total_iters=1000000
# 

# # Test!
# psize=2
# # XLA_PYTHON_CLIENT_MEM_FRACTION=.75 \
# # XLA_PYTHON_CLIENT_ALLOCATOR=platform \
# # XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
#   +dp_dim=128 \
#   +fsdp_dim=1 \
#   +tp_dim=2 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=1 \
#   total_iters=2


# EXPR_NAME="rectflow-lognorm/t3lm-mlp-bd-fixed/pixels/ps2-DiT-S-mt3lm-mlp-bd-2024-Aug-24-00-08-40"
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
#   +resume_step=210000 \
#   grad_clip=1 \
#   ckpt_every=10000 \
#   total_iters=1000000



# # Aug 23
# # Will directly change to rect flow
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/t3lm-mlp-bd-fixed/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   loss=rectflow \
#   model="dit_s_mt3lm_mlp_bd" \
#   +model.learn_sigma=False \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000



# # Aug 23
# # NOTE: this cannot load anymore since the bug fixed for bd layer change
# #       its structure in a very complicated way... :-(
# EXPR_NAME="t3lm-mlp-bd/pixels/ps2-DiT-S-mt3lm-mlp-bd-2024-Aug-21-19-03-16"
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



# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="t3lm-mlp-bd/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mt3lm_mlp_bd" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
# 

# psize=4
# niters=64
# for lr in 1.0 0.1 0.01; do
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin-ar-vit-niter${niters}/pixels/ps${psize}-lr${lr}" \
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
# 


# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/pixels/ps${psize}" \
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

# # Broken, need to rerun :)
# EXPR_NAME="ssm-vit/p4-DiT-S-SSM-2024-Aug-06-09-16-26"
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


# patch_size=1
# niters=8
# for lr in 1.0 0.1; do
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="TTT-AR/l${patch_size}-niters${niters}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   image_size=256 \
#   model="dit_s_mttt_bdar" \
#   model.patch_size=${patch_size} \
#   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   total_iters=400000 \
#   ckpt_every=25000
# done


# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/l${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   image_size=512 \
#   multi_process=True \
#   model="dit_s_ssm" \
#   model.patch_size=${psize} \
#   total_iters=1000000

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

# psize=2
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

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b" \
#   model.patch_size=1 \
#   total_iters=4000000
 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-s-la/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   model.patch_size=1 \
#   total_iters=4000000
 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3-linear-scaleqkv" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_vit3_linear" \
#   total_iters=4000000
# 
# # Base model v2
# n_iters=1
# for ilr in 1.0 2.0 0.1; do
#   echo ${ilr}
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     multi_process=True \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     model=dit_s_mttt_mlp_v0 \
#     expr_name=mt3mlp-ilr${ilr}-niters${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.enc_layers=3 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters}
# done

# model_cfg=dit_s_mttt_linear_v1
# for ilr in 1.0 0.1 0.01; do
# 
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
#   
#   # learn W init 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True
#   
#   # LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
#   
#   # Winit + LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-learnW-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# done


# # Base 
# model_cfg=dit_s_mttt_mlp_v0
# n_iters=8
# # ilr=1.0 # doesn't work
# ilr=0.1
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
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


# EXPR_NAME="dit-s-p2-DiT-S-2024-Jul-13-19-03-47"
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
#   total_iters=8000000


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
#   # LN + res 
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=gclip1.0-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     opt.grad_clip=1.0 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=False \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
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