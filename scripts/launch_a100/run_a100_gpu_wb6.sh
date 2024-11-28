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

mkdir -p $LOCAL_CODE_DIR
rsync -rv \
  --exclude env/ \
  --exclude wandb/ \
  --exclude results/ \
  --exclude .git/ \
  --exclude */__pycache__/ \
  --exclude __pycache__/ \
  $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR
echo $LOCAL_CODE_DIR

# Sep 9, test whether gradacc=8 works for SSM-L
psize=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="dit-gradacc8/latent/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  model=dit_l_ssm \
  model.grad_checkpoint=True \
  +model.dit_block_kwargs.mlp_dtype=bfloat16 \
  +model.dit_block_kwargs.adaln_mlp_dtype=bfloat16 \
  model.patch_size=${psize} \
  return_aux=False \
  grad_acc=2 \
  ckpt_every=10000 \
  log_every=100 \
  max_ckpt_keep=40 \
  total_iters=400000

# # Sep 3 - 100% utility, but takes a year to finish :(
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name=config_pixels \
#   expr_name="mixprec/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model=ssm_s_16k \
#   model.patch_size=${psize} \
#   grad_acc=8 \
#   ckpt_every=10000 \
#   log_every=100 \
#   max_ckpt_keep=40 \
#   total_iters=400000

# # Sep 3 - 100% utility, but takes a year to finish :(
# psize=2
# XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-name=config_pixels \
#   expr_name="mixprec/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_memeff" \
#   model.patch_size=${psize} \
#   grad_acc=8 \
#   ckpt_every=10 \
#   log_every=1 \
#   max_ckpt_keep=40 \
#   total_iters=40


# # Sep 3 - failed to resume
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
#   grad_acc=8 \
#   max_ckpt_keep=40 \
#   total_iters=400000

# # Aug 22 
# psize=2
# dstate=256
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixel/ps${psize}-dstate${dstate}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_tiny_ssm" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   model.attn_kwargs.d_state=${dstate} \
#   grad_acc=4 \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000

# Aug 21 - wb 6
# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow/latent/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model="dit_s_mt3lm_mlp_bd" \
#   model.attn_kwargs.mttt_kwargs.max_sequence_length=1024 \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=32 \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   model.patch_size=${psize} \
#   total_iters=1000000
