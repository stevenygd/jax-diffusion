export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs/sci-guandao
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
sudo chown `whoami`:`whoami` $BUCKET_MNT
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
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export EXPR_NAME="ttt-v3/pixels-balanced-sharded-cached/ps2-TTT-S-lmbd-mlp-2024-Oct-04-07-22-56"
ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
python main.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  --config-path . \
>>>>>>> uvit
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
<<<<<<< HEAD
  model=TTT_smm \
  model.patch_size=${psize} \
  +grad_clip=1.0 \
  ckpt_every=10000 \
  max_ckpt_keep=35 \
  total_iters=400000
=======
  dp_dim=128 \
  fsdp_dim=1 \
  tp_dim=1


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


# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# export EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-S-2024-Oct-02-06-20-14"
# psize=1
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/
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


# Sep 24, SSM-L, paused
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=ssm_l \
#   +grad_clip=1 \
#   +dp_dim=32 \
#   +fsdp_dim=1 \
#   +tp_dim=4 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=70 \
#   ckpt_every=5000 \
#   total_iters=400000


# # NO checkpoint, can't resume
# EXPR_NAME="ttt-v3/pixels/ps2-TTT-Bpp-lmbd-mlp-2024-Sep-21-09-36-07"
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


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt-v4/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   model=TTT_bpp \
#   +grad_clip=1 \
#   +dp_dim=32 \
#   +fsdp_dim=1 \
#   +tp_dim=4 \
#   grad_acc=1 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


# EXPR_NAME="ttt/pixels/ps2-TTT-Bpp-lmbd-mlp-2024-Sep-18-06-39-13"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# ls -d ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints/*/
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
#   dp_dim=32 \
#   fsdp_dim=2 \
#   tp_dim=2 \
#   grad_acc=1 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# psize=2
# # JAX_DISABLE_JIT=1 
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
#   +dp_dim=32 \
#   +fsdp_dim=2 \
#   +tp_dim=2 \
#   grad_acc=8 \
#   global_batch_size=256 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000


# EXPR_NAME="ttt/pixels/ps2-TTT-Ti-lmbd-mlp-2024-Sep-04-20-30-18"
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


# # Sep 4
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_tiny \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000
>>>>>>> uvit


# # Sep 4
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ttt/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=TTT_tiny \
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
#   model=TTT_tiny \
#   model.patch_size=${psize} \
#   +grad_clip=1.0 \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000


# # Aug 23 - resume
# EXPR_NAME="SA/pixels/ps2-DiT-Ti-2024-Aug-22-00-10-24"
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
#   total_iters=1000000


# # Aug 21
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="SA/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_tiny_memeff" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000



# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ttt-lm-mlp/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_lm_mlp" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
# 

# EXPR_NAME="SA/l2-DiT-S-2024-Aug-08-06-14-51"
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
#   total_iters=4000000
# 

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
#   image_size=512 \
#   model="dit_s_mttt_linear_ar" \
#   model.patch_size=${patch_size} \
#   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   total_iters=400000
# done


# EXPR_NAME="SA/l1-DiT-B-2024-Aug-08-19-49-24"
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
#   total_iters=4000000


# patch_size=1
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
#   +model.grad_checkpoint=True \
#   total_iters=1000000

# psize=1
# niters=8
# for ilr in 1.0 0.1; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="mt3-mlp-niters${niters}-ilr${ilr}/l${psize}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_mttt_mlp_v0" \
#     model.patch_size=${psize} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     total_iters=400000
# done

# psize=1
# niters=4
# for ilr in 1.0 0.1 0.01; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="mt3-lin-learnW-niters${niters}-ilr${ilr}/l${psize}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_mttt_linear_v1" \
#     model.patch_size=${psize} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     total_iters=1000000
# done


# psize=1
# niters=4
# for ilr in 1.0 0.1 0.01; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="mt3-lin-learnW-niters${niters}-ilr${ilr}/l${psize}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_mttt_linear_v1" \
#     model.patch_size=${psize} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     total_iters=1000000
# done

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="fixloader-vae/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vae" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="linattn/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin/1-base" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_linear" \
#   model.patch_size=1 \
#   total_iters=1000000
# 

# for ilr in 1.0 0.1 10.0 0.01; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="vit3lin-SGD-lr${ilr}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_vit3_linear" \
#     model.attn_kwargs.ttt_cfg.SGD=True \
#     model.attn_kwargs.ttt_cfg.inner_lr="[${ilr}]" \
#     total_iters=400000
# done

# ilr=1.0
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=noscaleq-4iters-ilr${ilr} \
#   model=dit_s_mttt_linear \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=4
# 

# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=dit-b-p4 \
#   model=dit_b \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=4
# 

# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=linattn2 \
#   model=dit_s_linattn \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True

# for ilr in 1.0 0.1 0.01; do 
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=mt3mlp \
#   model=dit_s_mttt_mlp_v0 \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=4 \
#   multi_process=True
# done

# EXPR_NAME="mt3mlp-DiT-S-mt3mlp-v0-2024-Jul-15-06-05-16"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   multi_process=True \
#   resume=True \
#   total_iters=4000000