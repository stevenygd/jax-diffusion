export USER_DIR=`pwd`
export BUCKET_NAME=sci-guandao
export BUCKET_MNT=/mnt/disks/gs-sci-guandao 
export CODE_DIR="/checkpoint/guandao/jax-DiT"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
export OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/persistent"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

fusermount -u $BUCKET_MNT 
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
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


# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   loss=rectflow \
#   model="dit_l_mt3lm_mlp_bd_memeff" \
#   +model.learn_sigma=False \
#   model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=256 \
#   model.attn_kwargs.mttt_kwargs.max_sequence_length=16384 \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   max_ckpt_keep=30 \
#   total_iters=400000



# Aug 25, too slow, 8s/step, postpone
psize=1
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="rectflow-lognorm/pixels/ps${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  loss=rectflow \
  model="dit_tiny_mt3lm_mlp_bd_memeff" \
  +model.learn_sigma=False \
  model.attn_kwargs.mttt_kwargs.remat_mini_batch_group_size=1 \
  model.attn_kwargs.mttt_kwargs.max_sequence_length=65536 \
  data_loader=imagenet_pixel_tfds \
  feature_path=imagenet256_pixels_tfdata_sharded/ \
  model.patch_size=${psize} \
  +model.patch_type="vit" \
  use_latent=False \
  latent_dim=3 \
  ckpt_every=5000 \
  max_ckpt_keep=30 \
  total_iters=400000
