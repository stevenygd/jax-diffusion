#! /bin/bash

export USER_DIR=`pwd`
export CODE_DIR="/root/research/jax-DiT/"
export LOCAL_CODE_DIR="/scratch/jax-DiT/"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=/scratch/hydra
export WANDB_DIR=/scratch/wandb
export DATA_DIR="/scratch/dit-data"
export REF_PATH="${BUCKET_MNT}/data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"
export EXPR_NAME="${1}"
export SEED=${2:-0}
export BS=${3:-64}
export STEPS=${4:-50}
export BUCKET_NAME=${5:-"sci-guandao"}
export BUCKET_MNT=/mnt/gs/${BUCKET_NAME}
export RESUME_STEP=${6:-0}
export CFG=${7:-1.5}

fusermount -u $BUCKET_MNT
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
sudo chmod 777 `dirname $BUCKET_MNT`
gcsfuse --implicit-dirs \
  -o rw \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT

# rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

# rm -rf $LOCAL_CODE_DIR
# rsync -rv \
#   --exclude env/ \
#   --exclude wandb/ \
#   --exclude results/ \
#   --exclude .git/ \
#   --exclude */__pycache__/ \
#   --exclude __pycache__/ \
#   $CODE_DIR $LOCAL_CODE_DIR
# cd $LOCAL_CODE_DIR
LOCAL_CODE_DIR=$CODE_DIR

ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python sample_ddp_jax.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  resume=${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  checkpoint_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints \
  experiment_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  hydra_dir=${HYDRA_DIR} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  wandb_dir=${WANDB_DIR} \
  resume=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  global_seed=${SEED} \
  multi_process=False \
  inference.remove_existing_sample_dir=False \
  inference.num_fid_samples=10000 \
  inference.num_sampling_steps=50 \
  inference.ref_batch=${REF_PATH} \
  inference.sleep_interval=6 \
  +inference.adm_eval_batch_size=256 \
  inference.per_proc_batch_size=${BS} \
  +inference.mode=ddp \
  +resume_step=${RESUME_STEP} \
  +inference.reeval_metrics=True \
  inference.cfg_scale=${CFG}
