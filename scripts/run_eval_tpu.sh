export USER_DIR=/home/`whoami`
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
rm -r $HYDRA_DIR $WANDB_DIR
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

# 1. wandb configuration
export WANDB_API_KEY="f18c16a3ee499fe6f8c2b73384c997467f4b1ffa"
export WANDB_PROJECT="mt3_highres"
export WANDB_TEAM="xnf"

# 2. directory of the codebase, output and stored data
CODE_DIR=/mnt/disks/nfs/ujinsong/jax-diffusion
OUT_DIR=/mnt/disks/gs/sci-guandao
VAE_DIR=/mnt/disks/gs/sci-guandao/vae

# 3. experiment name to resume
EXPR_NAME=my_train-DiT-XL-2024-Dec-11-04-02-15
RESUME_CHECKPOINT=-1 # set specific checkpoint to resume, -1 to resume all checkpoints

# 4. run evaluation, change other hyperparameters in the config file(if needed)
EXPR_DIR=${OUT_DIR}/checkpoints/${EXPR_NAME}
PYTHONPATH=${CODE_DIR} python ${CODE_DIR}/eval.py\
  --config-dir ${EXPR_DIR}/ \
  --config-path . \
  multi_process=False \
  +vae_dir=${VAE_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  resume=${EXPR_DIR} \
  +resume_step=${RESUME_CHECKPOINT} \
  inference.per_proc_batch_size=32 \
  inference.num_fid_samples=200