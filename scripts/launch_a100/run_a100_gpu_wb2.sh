#! /bin/bash

export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
# export BUCKET_NAME=dit-guandao
# export BUCKET_MNT=/mnt/disks/gs/${BUCKET_NAME}
export BUCKET_MNT=/mnt/gs/${BUCKET_NAME}
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

mkdir $LOCAL_CODE_DIR
rsync -auv --delete \
  --exclude env/ \
  --exclude wandb/ \
  --exclude results/ \
  --exclude .git/ \
  $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

psize=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config \
  expr_name="ssm/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model="ssm_xl" \
  model.patch_size=${psize} \
  ckpt_every=10000 \
  max_ckpt_keep=35 \
  total_iters=400000 \
  grad_acc=32 \
  +test_checkpoint=True


# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="ssm_s" \
#   model.patch_size=${psize} \
#   ckpt_every=10000 \
#   max_ckpt_keep=35 \
#   total_iters=400000 \
#   grad_acc=2


# EXPR_NAME="ssm/pixels/ps4-SSM-S-2024-Sep-05-08-00-25"
# CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${EXPR_NAME}/checkpoints"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# XLA_PYTHON_CLIENT_MEM_FRACTION=.90 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   checkpoint_dir=${CHECKPOINT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=False \
#   resume=True \
#   grad_acc=2 \
#   total_iters=4000000

# Sep 5
# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.90 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="ssm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="ssm_s" \
#   model.patch_size=${psize} \
#   ckpt_every=25000 \
#   max_ckpt_keep=16 \
#   total_iters=400000 \
#   grad_acc=2

# # Aug 28
# psize=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm-zloss/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_ssm_moe" \
#   loss=rectflow_moe_zloss \
#   model.patch_size=${psize} \
#   grad_acc=2 \
#   max_ckpt_keep=16 \
#   ckpt_every=25000 \
#   total_iters=1000000

# # Aug 24
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_tiny_ssm" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000

# # Aug 22 - wb 2
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_ssm" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000


# # Aug 21 - wb 2
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_ssm" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000

# Aug 21 - wb1
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000
