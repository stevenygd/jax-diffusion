export CODE_DIR=/home/guandaoyang/jax-DiT
export BUCKET_NAME=${5:-"dit-guandao"}
export BUCKET_MNT=/mnt/gs/${BUCKET_NAME}
# export RUN_DIR=/home/`whoami`/jax-DiT
export DATA_DIR="/home/guandaoyang/jax-DiT/data"
export REF_PATH="${BUCKET_MNT}/data/imagenet256_reference/VIRTUAL_imagenet256_labeled.npz"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"
export EXPR_NAME="${1}"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=./hydra
export WANDB_DIR=./wandb
export SEED=${2:-0}
export BS=${3:-32}
export TFBS=${4:-256}

# fusermount -u $BUCKET_MNT
# sudo rm -rf $BUCKET_MNT
# sudo mkdir -p $BUCKET_MNT
# sudo chown -R `whoami`:`whoami` $BUCKET_MNT
# sudo chmod 777 $BUCKET_MNT
# sudo chmod 777 `dirname $BUCKET_MNT`
# gcsfuse --implicit-dirs \
#   -o rw \
#   --dir-mode 777 \
#   --file-mode 777 \
#   --rename-dir-limit 5000000 \
#   $BUCKET_NAME $BUCKET_MNT


# rm -rf $RUN_DIR
# cp -r $CODE_DIR $RUN_DIR
# echo "Code in ${RUN_DIR}"
# cd $RUN_DIR
# PYTHONPATH=${RUN_DIR} WANDB_DIR=${WANDB_DIR} python sample_ddp_jax.py \
python sample_ddp_jax.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  resume=${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  checkpoint_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints \
  experiment_dir=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  hydra_dir=${HYDRA_DIR} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  wandb_dir=${WANDB_DIR} \
  resume=${BUCKET_MNT}/checkpoints/${EXPR_NAME} \
  global_seed=${SEED} \
  multi_process=False \
  inference.remove_existing_sample_dir=False \
  inference.num_fid_samples=10000 \
  inference.num_sampling_steps=50 \
  inference.ref_batch=${REF_PATH} \
  inference.sleep_interval=60 \
  +inference.adm_eval_batch_size=${TFBS} \
  inference.per_proc_batch_size=${BS} \
  +inference.mode=ddp \
  +resume_step=0 \
  +inference.reeval_metrics=True
