#! /bin/bash

export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs # some existing directory already broken :)
export CODE_DIR="/root/research/jax-DiT/"
export LOCAL_CODE_DIR="/scratch/jax-DiT/"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=/scratch/hydra
export WANDB_DIR=/scratch/wandb
export DATA_DIR="/scratch/dit-data"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

# fusermount -u $BUCKET_MNT 
# sudo rm -rf $BUCKET_MNT
# sudo mkdir -p $BUCKET_MNT
# sudo chown `whoami`:`whoami` $BUCKET_MNT
# sudo chmod 777 $BUCKET_MNT
# gcsfuse --implicit-dirs \
#   -o rw \
#   --stat-cache-ttl 10s \
#   --type-cache-ttl 10s \
#   --dir-mode 777 \
#   --file-mode 777 \
#   --rename-dir-limit 5000000 \
#   $BUCKET_NAME $BUCKET_MNT

rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

rm -rf $LOCAL_CODE_DIR
mkdir -p $LOCAL_CODE_DIR
rsync -rv --delete \
  --exclude env/ \
  --exclude wandb/ \
  --exclude results/ \
  --exclude .git/ \
  --exclude */__pycache__/ \
  --exclude __pycache__/ \
  $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR
echo $LOCAL_CODE_DIR

# EXPR_NAME="rectflow-lognorm/latent/ps1-DiT-L-2024-Sep-03-09-42-33"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=False \
#   resume=True \
#   total_iters=400000

# DiT-L latent/1k
psize=1
# XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="rectflow-lognorm/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model="dit_l" \
  +model.dit_block_kwargs.mlp_dtype=bfloat16 \
  +model.dit_block_kwargs.adaln_mlp_dtype=bfloat16 \
  model.patch_size=${psize} \
  ckpt_every=10000 \
  log_every=100 \
  max_ckpt_keep=35 \
  return_aux=False \
  multi_process=False \
  model.grad_checkpoint=True \
  grad_acc=2 \
  total_iters=400000

# psize=1
# # XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_b" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   ckpt_every=25000 \
#   log_every=100 \
#   max_ckpt_keep=35 \
#   return_aux=False \
#   grad_acc=2 \
#   total_iters=1000000

# # Aug 26 evening - OOM, postpone
# psize=1
# # XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_l" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   log_every=100 \
#   max_ckpt_keep=35 \
#   return_aux=False \
#   grad_acc=8 \
#   total_iters=400000

# # Aug 24
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm-adm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_tiny_ssm" \
#   model.patch_size=${psize} \
#   total_iters=1000000


# # Aug 22 - wb5
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixel/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_tiny_memeff" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   grad_acc=4 \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000