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
CODE_DIR=/home/`whoami`/jax-diffusion
OUT_DIR=/mnt/gs/sci-guandao
DATA_DIR=/mnt/gs/sci-guandao/data

# 3. model name and experiment name for current run
MODEL_NAME=dit_xl # should be one of the following: dit_l, dit_xl, ssm_l, ssm_xl
EXPR_NAME=my_train

if [ "$MODEL_NAME" != "dit_l" ] && [ "$MODEL_NAME" != "dit_xl" ] && [ "$MODEL_NAME" != "ssm_l" ] && [ "$MODEL_NAME" != "ssm_xl" ]; then
  echo "Invalid model name: ${MODEL_NAME}"
  exit 1
fi

# 4. run the training script, change other hyperparameters in the config file(if needed)
export CUDA_VISIBLE_DEVICES=0 
PYTHONPATH=${CODE_DIR} python ${CODE_DIR}/train.py \
  --config-name config \
  model=${MODEL_NAME} \
  expr_name=${EXPR_NAME} \
  data_dir=${DATA_DIR} \
  output_dir=${OUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  global_batch_size=16 \
  total_iters=100 \
  ckpt_every=30 \