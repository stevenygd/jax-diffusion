export BUCKET_MNT=/home/`whoami`/outputs
export USER_DIR=/home/`whoami`
export CODE_DIR="/mnt/disks/nfs/guandao/jax-DiT"
export RUN_DIR=/home/`whoami`/jax-DiT
export BUCKET_NAME=dit-guandao
export DATA_DIR="/mnt/disks/data"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

fusermount -u $BUCKET_MNT
rm -rf $BUCKET_MNT
mkdir -p $BUCKET_MNT
gcsfuse --implicit-dirs \
  -o rw \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT

OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
HYDRA_DIR=${USER_DIR}/hydra
WANDB_DIR=${USER_DIR}/wandb
rm -r $HYDRA_DIR $WANDB_DIR
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

rm -rf $RUN_DIR
cp -r $CODE_DIR $RUN_DIR
echo "Code in ${RUN_DIR}"
cd $RUN_DIR
PYTHONPATH=${RUN_DIR} WANDB_DIR=${WANDB_DIR} \
python main.py \
  model=dit_s_linattn \
  expr_name=DiT-Jax-v4x8 \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  image_size=256
