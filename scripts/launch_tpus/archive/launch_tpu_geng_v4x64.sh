export USER_DIR=`pwd`
export BUCKET_NAME=dit-guandao
export BUCKET_MNT=/mnt/disks/outputs
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
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
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

# Create code directory
rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

patch_size=2
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="DiT-L/L${patch_size}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model=dit_l \
  model.patch_size=${patch_size} \
  global_batch_size=32 \
  model.depth=16 \
  log_every=1 \
  total_iters=4000000


# psize=4
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3lin/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_s_mttt_linear \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=4000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b/2" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b" \
#   model.patch_size=2 \
#   total_iters=4000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b-la/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b_linattn" \
#   model.patch_size=1 \
#   total_iters=1000000
# 
# model_cfg=dit_s_mttt_linear_v1
# for ilr in 1.0 0.1 0.01; do
# 
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
# 
# 
#   # learn W init 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True
# 
#   # LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# 
#   # Winit + LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=learnW-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# done


# model_cfg=dit_s_mttt_linear
# for n_iters in 8 16; do
# for ilr in 1.0 0.1 0.01; do
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=niters${n_iters}-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
# done
# done

# model_cfg=dit_s_mttt_mlp_v0
# n_iters=4
# for ilr in 1.0 0.1 0.01; do
# 
#   # # Base - gives NaN, also breaks (directly terminate)
#   # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=False
# 
#   # learn W init 
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=gclip1.0-learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     opt.grad_clip=1.0 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=False \
#     model.attn_kwargs.mttt_kwargs.enc_residual=False
# 
#   # # LN + res 
#   # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=LNres-ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=False \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=True
# 
#   # # Winit + LN + res (NOTE this already ran)
#   # PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name=learnW-LNres-ilr${ilr} \
#   #   model=${model_cfg} \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   #   model.attn_kwargs.mttt_kwargs.learnable_init=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_ln=True \
#   #   model.attn_kwargs.mttt_kwargs.enc_residual=True
# done