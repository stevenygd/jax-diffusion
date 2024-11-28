USER_DIR=`pwd`
BUCKET_NAME=dit-guandao
# export BUCKET_MNT=${USER_DIR}/outputs
export BUCKET_MNT=/home/Grendel/outputs
# export BUCKET_MNT=/home/Grendel/outputs
CODE_DIR="/mnt/disks/nfs/guandao/jax-DiT"
LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
HYDRA_DIR=${USER_DIR}/hydra
WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/mnt/disks/data"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

# Make data directory
sudo rm -rf /persistent_1
sudo ln -s $DATA_DIR /persistent_1
sudo rm -rf /persistent_2
sudo ln -s $DATA_DIR /persistent_2

fusermount -u $BUCKET_MNT 
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chmod -R 777 $BUCKET_MNT
sudo chmod -R 777 `dirname $BUCKET_MNT`
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

rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

EXPR_NAME="linattn2-DiT-S-linattn-2024-Jul-14-05-23-47"
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  --config-path . \
  multi_process=True \
  resume=True \
  total_iters=4000000

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=test \
#   model=dit_s \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=2
# 
