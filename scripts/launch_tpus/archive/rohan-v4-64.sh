export USER_DIR=`pwd`
export BUCKET_NAME=dit-guandao
export BUCKET_MNT=/mnt/disks/gs3 # /mnt/disks/gs already broken :-)
export CODE_DIR="/checkpoint/guandao/jax-DiT"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/persistent"
# export DATA_DIR="${BUCKET_MNT}/data"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

# fusermount -u $BUCKET_MNT 
# sudo rm -rf $BUCKET_MNT
# sudo mkdir -p $BUCKET_MNT
# sudo chmod -R 777 $BUCKET_MNT
# gcsfuse --implicit-dirs \
#   -o rw \
#   --stat-cache-ttl 10s \
#   --type-cache-ttl 10s \
#   --dir-mode 777 \
#   --file-mode 777 \
#   --rename-dir-limit 5000000 \
#   $BUCKET_NAME $BUCKET_MNT

echo "Hydra, wandb directory"
rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

# Create code directory
# echo "Coding directory"
# echo "...rm..."
# rm -rf $LOCAL_CODE_DIR
# echo "...cp..."
# cp -r $CODE_DIR $LOCAL_CODE_DIR
# cd $LOCAL_CODE_DIR
# echo `pwd`
cd $CODE_DIR

# export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

echo "Launching...."
psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python debug.py \
PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python debug_minimal.py \
  expr_name="test/debugv4" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  loss=rectflow \
  model="dit_tiny" \
  +model.learn_sigma=False \
  ckpt_every=10000 \
  log_every=1 \
  grad_acc=4 \
  return_aux=False \
  total_iters=400000
