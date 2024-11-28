#! /bin/bash

export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
# export BUCKET_NAME=dit-guandao
export BUCKET_MNT=/mnt/gs/${BUCKET_NAME}
export CODE_DIR="/root/research/jax-DiT/"
export LOCAL_CODE_DIR="/scratch/jax-DiT/"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=/scratch/hydra
export WANDB_DIR=/scratch/wandb
export DATA_DIR="/scratch/dit-data"
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

mkdir $LOCAL_CODE_DIR
rsync -auv --delete \
  --exclude env/ \
  --exclude wandb/ \
  --exclude results/ \
  --exclude .git/ \
  $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR


# Sep 20
psize=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-name config \
  expr_name="ssm/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model="ssm_b" \
  model.patch_size=${psize} \
  +test_checkpoint=True \
  ckpt_every=25000 \
  max_ckpt_keep=160 \
  grad_acc=2 \
  total_iters=4000000 \


# Some other time
# EXPR_NAME="dit/pixels/ps4-DiT-S-2024-Sep-05-07-34-31"
# CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/dit/pixels/ps4-DiT-S-2024-Sep-05-07-34-31/checkpoints"
# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.90 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="dit/pixels/ps${psize}" \
#   +wandb.expr_name=${EXPR_NAME} \
#   +wandb.project_name=mt3_res256 \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   resume=True \
#   checkpoint_dir=${CHECKPOINT_DIR} \
#   model="dit_s_mp" \
#   model.patch_size=${psize} \
#   ckpt_every=25000 \
#   max_ckpt_keep=16 \
#   total_iters=400000 \
#   grad_acc=16


# # Sep 5
# psize=4
# XLA_PYTHON_CLIENT_MEM_FRACTION=.90 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name config_pixels \
#   expr_name="dit/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_mp" \
#   model.patch_size=${psize} \
#   ckpt_every=25000 \
#   max_ckpt_keep=16 \
#   total_iters=400000 \
#   grad_acc=16

# # Aug 28
# psize=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm-moe/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_ssm_moe" \
#   loss=rectflow_moe \
#   model.patch_size=${psize} \
#   grad_acc=2 \
#   max_ckpt_keep=16 \
#   ckpt_every=25000 \
#   total_iters=1000000

# # Aug 26 
# EXPR_NAME="rectflow-lognorm/pixels/ps2-DiT-Ti-2024-Aug-25-08-19-48"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=False \
#   resume=True \
#   ckpt_every=10000 \
#   grad_acc=2 \
#   total_iters=4000000

# # Aug 24 - wb1
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_tiny" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000

# # Aug 21 - wb1
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s" \
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
