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
DATA_DIR=/mnt/disks/data

# 3. model name and experiment name for current run
MODEL_NAME=dit_xl # should be one of the following: dit_l, dit_xl, ssm_l, ssm_xl
EXPR_NAME=my_train

if [ "$MODEL_NAME" != "dit_l" ] && [ "$MODEL_NAME" != "dit_xl" ] && [ "$MODEL_NAME" != "ssm_l" ] && [ "$MODEL_NAME" != "ssm_xl" ]; then
  echo "Invalid model name: ${MODEL_NAME}"
  exit 1
fi

# 4. run the training script, change other hyperparameters in the config file(if needed)
# export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
PYTHONPATH=${CODE_DIR} python ${CODE_DIR}/train.py \
  --config-name config \
  model=${MODEL_NAME} \
  expr_name=${EXPR_NAME} \
  data_dir=${DATA_DIR} \
  output_dir=${OUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  global_batch_size=128