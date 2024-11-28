#! /bin/bash

export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs-sci-guandao # some existing directory already broken :)
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

mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

rm -rf $LOCAL_CODE_DIR
mkdir $LOCAL_CODE_DIR
rsync -auv --delete \
  --exclude env/ \
  --exclude wandb/ \
  --exclude results/ \
  --exclude .git/ \
  $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR
echo $LOCAL_CODE_DIR

# Aug 29
psize=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="rectflow-lognorm-moe/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model="dit_s_moe" \
  loss=rectflow_moe \
  model.patch_size=${psize} \
  grad_acc=2 \
  ckpt_every=25000 \
  total_iters=1000000

# # Aug 26 morning - too slow, never launch, also super low utility
# psize=1
# XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_l_ssm_memeff" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   grad_acc=1 \
#   total_iters=1000000


# # Aug 25 morning
# EXPR_NAME="ssm/pixels/ps2-dstate256-DiT-S-SSM-2024-Aug-22-06-23-18"
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