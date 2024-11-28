export BUCKET_MNT=/home/Grendel/outputs
export USER_DIR=/home/`whoami`
export CODE_DIR="/mnt/disks/nfs/guandao/jax-DiT"
export RUN_DIR=/home/`whoami`/jax-DiT
export BUCKET_NAME=dit-guandao
export DATA_DIR="/mnt/disks/data"
export REF_PATH="/mnt/disks/nfs/data/imagenet512_reference/VIRTUAL_imagenet512.npz"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"
export EXPR_NAME="${1}"
OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
HYDRA_DIR=${USER_DIR}/hydra
WANDB_DIR=${USER_DIR}/wandb

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

rm -r $HYDRA_DIR $WANDB_DIR
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

rm -rf $RUN_DIR
cp -r $CODE_DIR $RUN_DIR
echo "Code in ${RUN_DIR}"
cd $RUN_DIR
# inference.ref_batch=${DATA_DIR}/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz \
PYTHONPATH=${RUN_DIR} WANDB_DIR=${WANDB_DIR} python sample_ddp_jax.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  multi_process=True \
  global_seed=${2:-0} \
  inference.remove_existing_sample_dir=False \
  inference.num_fid_samples=10000 \
  inference.num_sampling_steps=512 \
  inference.ref_batch=${REF_PATH} \
  +inference.adm_eval_batch_size=256 \
  inference.per_proc_batch_size=32 # batch size for V4TPU 
