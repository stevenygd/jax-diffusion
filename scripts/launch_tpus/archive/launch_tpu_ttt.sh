USER_DIR=/home/Grendel
BUCKET_NAME=dit-guandao
export BUCKET_MNT=${USER_DIR}/outputs
CODE_DIR="/checkpoint/guandao/jax-DiT"
LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
OUTPUT_DIR=$BUCKET_MNT # NOTE: this is /home/<username>/outputs
HYDRA_DIR=${USER_DIR}/hydra
WANDB_DIR=${USER_DIR}/wandb

fusermount -u $BUCKET_MNT 
rm -rf $BUCKET_MNT
mkdir -p $BUCKET_MNT
chmod -R 777 $BUCKET_MNT
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
DATA_DIR="/persistent"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

psize=2
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="DiT-L/L${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model="dit_l" \
  image_size=256 \
  model.patch_size=${psize} \
  total_iters=1000000

# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="linattn/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin/1-base" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_mttt_linear" \
#   model.patch_size=1 \
#   total_iters=1000000
# 

# for ilr in 1.0 0.1 10.0 0.01; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="vit3lin-SGD-lr${ilr}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model="dit_s_vit3_linear" \
#     model.attn_kwargs.ttt_cfg.SGD=True \
#     model.attn_kwargs.ttt_cfg.inner_lr="[${ilr}]" \
#     total_iters=400000
# done

# ilr=1.0
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=noscaleq-4iters-ilr${ilr} \
#   model=dit_s_mttt_linear \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=4
# 

# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=dit-b-p4 \
#   model=dit_b \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=4
# 

# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=linattn2 \
#   model=dit_s_linattn \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True

# for ilr in 1.0 0.1 0.01; do 
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=mt3mlp \
#   model=dit_s_mttt_mlp_v0 \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=4 \
#   multi_process=True
# done

# EXPR_NAME="mt3mlp-DiT-S-mt3mlp-v0-2024-Jul-15-06-05-16"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   multi_process=True \
#   resume=True \
#   total_iters=4000000