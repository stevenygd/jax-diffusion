export USER_DIR=`pwd`
export BUCKET_NAME=dit-guandao
export BUCKET_MNT=/mnt/disks/gs2
export CODE_DIR="/checkpoint/jingnathanyan/jax-DiT"
export LOCAL_CODE_DIR="${USER_DIR}/jax-DiT"
export OUTPUT_DIR=${BUCKET_MNT} # NOTE: this is /home/<username>/outputs
export HYDRA_DIR=${USER_DIR}/hydra
export WANDB_DIR=${USER_DIR}/wandb
export DATA_DIR="/persistent"
export WANDB_API_KEY="6299728a4fd7921829280c9703fd72c48459dda7"

fusermount -u $BUCKET_MNT 
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chmod -R 777 $BUCKET_MNT
gcsfuse --implicit-dirs \
  -o rw \
  --stat-cache-ttl 10s \
  --type-cache-ttl 10s \
  --dir-mode 777 \
  --file-mode 777 \

rm -r $HYDRA_DIR $WANDB_DIR 
mkdir -p $HYDRA_DIR
mkdir -p $WANDB_DIR

rm -rf $LOCAL_CODE_DIR
cp -r $CODE_DIR $LOCAL_CODE_DIR
cd $LOCAL_CODE_DIR

psize=1
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
  expr_name="fixloader-sa/l${psize}" \
  data_dir=${DATA_DIR} \
  output_dir=${OUTPUT_DIR} \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  model="dit_s" \
  +model.attn_type="ssm" \
  model.patch_size=${psize} \
  total_iters=1000000


# psize=2
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

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b" \
#   model.patch_size=1 \
#   total_iters=4000000
 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-s-la/1" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_linattn" \
#   model.patch_size=1 \
#   total_iters=4000000
 
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="mt3-linear-scaleqkv" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_vit3_linear" \
#   total_iters=4000000
# 
# # Base model v2
# n_iters=1
# for ilr in 1.0 2.0 0.1; do
#   echo ${ilr}
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     multi_process=True \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     model=dit_s_mttt_mlp_v0 \
#     expr_name=mt3mlp-ilr${ilr}-niters${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.enc_layers=3 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters}
# done

# model_cfg=dit_s_mttt_linear_v1
# for ilr in 1.0 0.1 0.01; do
# 
#   # Base 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr}
#   
#   # learn W init 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-learnW-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True
#   
#   # LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
#   
#   # Winit + LN + res 
#   PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=4iters-learnW-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.n_iters=4 \
#     model.attn_kwargs.mttt_kwargs.learnable_init=True \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
# done


# # Base 
# model_cfg=dit_s_mttt_mlp_v0
# n_iters=8
# # ilr=1.0 # doesn't work
# ilr=0.1
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=4M-niters${n_iters}-ilr${ilr} \
#   model=${model_cfg} \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   opt.grad_clip=1.0 \
#   total_iters=4000000


# EXPR_NAME="dit-s-p2-DiT-S-2024-Jul-13-19-03-47"
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   resume=True \
#   total_iters=8000000


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
#   # LN + res 
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name=gclip1.0-LNres-ilr${ilr} \
#     model=${model_cfg} \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     opt.grad_clip=1.0 \
#     model.attn_kwargs.mttt_kwargs.n_iters=${n_iters} \
#     model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#     model.attn_kwargs.mttt_kwargs.learnable_init=False \
#     model.attn_kwargs.mttt_kwargs.enc_ln=True \
#     model.attn_kwargs.mttt_kwargs.enc_residual=True
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