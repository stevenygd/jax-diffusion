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
CODE_DIR=~/research/jax-diffusion
OUT_DIR=~/research/jax-diffusion/output
DATA_DIR=~/research/jax-diffusion/data

# 3. model name and experiment name for current run
MODEL_NAME=dit_xl # should be one of the following: dit_l, dit_xl, ssm_l, ssm_xl, dit_2b
EXPR_NAME=train_dit_xl_a100_40g

if [ "$MODEL_NAME" != "dit_l" ] && [ "$MODEL_NAME" != "dit_xl" ] && [ "$MODEL_NAME" != "ssm_l" ] && [ "$MODEL_NAME" != "ssm_xl" ] && [ "$MODEL_NAME" != "dit_2b" ]; then
  echo "Invalid model name: ${MODEL_NAME}"
  exit 1
fi

# 4. run the training script, change other hyperparameters in the config file(if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
PYTHONPATH=${CODE_DIR} python ${CODE_DIR}/train.py \
  --config-name config \
  model=${MODEL_NAME} \
  expr_name=${EXPR_NAME} \
  data_dir=${DATA_DIR} \
  output_dir=${OUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  global_batch_size=256 \
  total_iters=400000 \
  ckpt_every=5q000 \
  dp_dim=8 \
  grad_acc=32