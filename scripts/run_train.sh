export HYDRA_DIR=`pwd`/hydra
export WANDB_DIR=`pwd`/wandb
rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

# wandb configuration
export WANDB_API_KEY="f18c16a3ee499fe6f8c2b73384c997467f4b1ffa"
export WANDB_PROJECT="mt3_highres"
export WANDB_TEAM="xnf"

# directory of the codebase
CODE_DIR=/mnt/disks/nfs/ujinsong/jax-diffusion

# directory that stores the checkpoints upon training
OUT_DIR=/mnt/disks/gs/sci-guandao

# directory that reads the data from
DATA_DIR=/mnt/disks/data

# model name
MODEL_NAME=ssm_xl
EXPR_NAME=release

export LIBTPU_INIT_ARGS=" --xla_tpu_impure_oom_fast_exit_threshold=-1 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH=${CODE_DIR} python ${CODE_DIR}/train.py \
  --config-name config \
  model=${MODEL_NAME} \
  expr_name=${EXPR_NAME} \
  data_dir=${DATA_DIR} \
  output_dir=${OUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  global_batch_size=16 \
  ckpt_every=50 \
  total_iters=200