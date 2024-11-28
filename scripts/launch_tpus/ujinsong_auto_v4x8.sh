export USER_DIR=/home/`whoami`
export CODE_DIR="/mnt/disks/nfs/ujinsong/jax-DiT"
export RUN_DIR=/home/`whoami`/jax-DiT
export DATA_DIR="/mnt/disks/data"
export REF_PATH="/mnt/disks/nfs/data/imagenet512_reference/VIRTUAL_imagenet512.npz"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"
export EXPR_NAME="${1}"
export INFER_MODE="${2}"
export INFER_NSAMPLE="${3}"
export INFER_NSTEPS="${4}"
export SEED="${5}"
export IMAGE_SIZE="${6}"
export CFG_SCALE="${7}"
export PER_PROC_BATCH_SIZE="${8}"
export RESUME_STEP="${9:--1}"

if [ "$INFER_MODE" = "ddpm" ]; then
  if [ "$INFER_NSTEPS" -ne 128 ]; then
    echo "Assertion failed: INFER_NSTEPS should be 128 when INFER_MODE is 'ddpm'"
    exit 1
  fi
fi

force_log=false
if [ "$RESUME_STEP" -lt 0  ]; then
  pythonfile="sample_jax_log.py"
elif [ "$RESUME_STEP" -eq 0 ]; then
  pythonfile="sample_jax_log.py"
  force_log=true
else
  pythonfile="sample_jax_singleckpt.py"
fi

declare -A buckets
buckets["sci-guandao"]="/mnt/disks/gs2/outputs"
buckets["dit-guandao"]="/mnt/disks/gs1/outputs"

directory_exists=false
for BUCKET_NAME in "${!buckets[@]}"; do
  BUCKET_MNT=${buckets[$BUCKET_NAME]}

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

  if [ -d "${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints" ]; then
    echo "Directory ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints exists"
    directory_exists=true
    break
  fi
done

if [ "$directory_exists" = false ]; then
  echo "Directory not found in dit-guandao and sci-guandao."
  exit 1
fi

echo "${BUCKET_NAME} is mounted at ${BUCKET_MNT}"


if [ "$IMAGE_SIZE" -eq 256 ]; then
  export REF_BATCH_DIR=${BUCKET_MNT}/data/imagenet${IMAGE_SIZE}_reference/VIRTUAL_imagenet${IMAGE_SIZE}_labeled.npz
  # export REF_BATCH_DIR=${DATA_DIR}/imagenet${IMAGE_SIZE}_reference/VIRTUAL_imagenet${IMAGE_SIZE}_labeled.npz
elif [ "$IMAGE_SIZE" -eq 512 ]; then
  export REF_BATCH_DIR=${DATA_DIR}/imagenet${IMAGE_SIZE}_reference/VIRTUAL_imagenet${IMAGE_SIZE}.npz
else
  echo "Assertion failed: IMAGE_SIZE should be 256 or 512"
  exit 1
fi


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
PYTHONPATH=${RUN_DIR} WANDB_DIR=${WANDB_DIR} python ${pythonfile} \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  --config-path . \
  multi_process=False \
  global_seed=${SEED} \
  checkpoint_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints \
  experiment_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  hydra_dir=${HYDRA_DIR} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  wandb_dir=${WANDB_DIR} \
  resume=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  +resume_step=${RESUME_STEP} \
  inference.remove_existing_sample_dir=False \
  inference.num_fid_samples=${INFER_NSAMPLE} \
  +inference.mode=${INFER_MODE} \
  inference.num_sampling_steps=${INFER_NSTEPS} \
  inference.ref_batch=${REF_BATCH_DIR} \
  +inference.adm_eval_batch_size=256 \
  inference.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \
  inference.sleep_interval=1\
  inference.cfg_scale=${CFG_SCALE} \
  +force_log=${force_log}