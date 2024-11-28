#! /bin/bash

export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs-sci-guandao
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
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 
# XLA_PYTHON_CLIENT_PREALLOCATE=false 
# XLA_PYTHON_CLIENT_ALLOCATOR=platform 
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="dit/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model=dit_xl \
  model.patch_size=${psize} \
  grad_acc=16 \
  max_ckpt_keep=35 \
  ckpt_every=10000 \
  total_iters=4000000 \
  +test_checkpoint=True

# # Aug 28
# psize=1
# # XLA_PYTHON_CLIENT_MEM_FRACTION=.95 
# # XLA_PYTHON_CLIENT_PREALLOCATE=false 
# # XLA_PYTHON_CLIENT_ALLOCATOR=platform 
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm-zloss/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model=dit_b_ssm_moe \
#   loss=rectflow_moe_zloss \
#   model.patch_size=${psize} \
#   grad_acc=8 \
#   max_ckpt_keep=35 \
#   ckpt_every=10000 \
#   total_iters=400000
