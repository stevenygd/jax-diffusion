export USER_DIR=/home/Grendel
export BUCKET_NAME=sci-guandao
# export BUCKET_NAME=dit-guandao
export BUCKET_MNT=/mnt/disks/${BUCKET_NAME}
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
sudo chmod -R 777 $BUCKET_MNT
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

export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
export EXPR_NAME="ttt-v3/pixels-balanced-sharded-cached/ps2-TTT-B-lmbd-mlp-2024-Oct-04-07-27-06"
ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/
FEATURE_PATH="imagenet256_pixels_tfdata_sharded_balanced/imagenet256_pixels_tfdata_sharded"
PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
python main.py \
  --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
  --config-path . \
  data_dir=${DATA_DIR} \
  feature_path=${FEATURE_PATH} \
  output_dir=${OUTPUT_DIR} \
  results_dir=${OUTPUT_DIR}/checkpoints \
  hydra_dir=${HYDRA_DIR} \
  wandb_dir=${WANDB_DIR} \
  multi_process=True \
  dp_dim=128 \
  fsdp_dim=1 \
  tp_dim=1


# export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# export DATA_DIR="${BUCKET_MNT}/ldm/epoch2"
# export EXPR_NAME="dit/ldm-latents-epoch2/ps1-DiT-L-2024-Oct-03-04-47-37"
# psize=1
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   +cache=True \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model.patch_size=${psize} \
#   dp_dim=128 \
#   fsdp_dim=1 \
#   tp_dim=1 \
#   max_ckpt_keep=100 \
#   ckpt_every=50000 \
#   total_iters=4000000


# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} \
# python main.py \
#   --config-name config_imgnet_only \
#   expr_name="dit/ldm-latents-epoch2/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   feature_path="imagenet256_tfdata_sharded-v2/" \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model=dit_b \
#   model.patch_size=${psize} \
#   +grad_clip=1 \
#   +dp_dim=128 \
#   +fsdp_dim=1 \
#   +tp_dim=1 \
#   grad_acc=1 \
#   global_batch_size=256



# EXPR_NAME="ttt/pixels/ps2-TTT-Spp-lmbd-mlp-2024-Sep-13-20-25-49"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   results_dir=${OUTPUT_DIR}/checkpoints \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   +profile_dir=null


# # Aug 25: Run DiT-Tiny RectFlow-Lognorm Pixel / 2 16k ctx length
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="rectflow-lognorm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   use_latent=False \
#   latent_dim=3 \
#   model="dit_tiny_memeff" \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   +model.learn_sigma=False \
#   loss=rectflow \
#   ckpt_every=10000 \
#   total_iters=400000

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/pixels/ps${psize}-dstate256" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_ssm" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   model.attn_kwargs.d_state=256 \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000
# 

# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_ssm" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   use_latent=False \
#   latent_dim=3 \
#   ckpt_every=10000 \
#   total_iters=400000


# # Doesn't fit into memroy sadly :(
# psize=2
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit/pixels/ps${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   +model.patch_type="vit" \
#   +model.grad_checkpoint=True \
#   +model.attn_type="jax" \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000
 


# patch_size=1
# niters=8
# for lr in 1.0 5.0 0.1; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="TTT-AR/l${patch_size}-niters${niters}" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     image_size=512 \
#     model="dit_s_mttt_mlp_v0" \
#     model.patch_size=${patch_size} \
#     model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#     model.attn_kwargs.mttt_kwargs.lr=${lr} \
#     total_iters=400000
# 
#   # PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   #   expr_name="TTT-AR/l${patch_size}-niters${niters}" \
#   #   data_dir=${DATA_DIR} \
#   #   output_dir=${OUTPUT_DIR} \
#   #   hydra_dir=${HYDRA_DIR} \
#   #   wandb_dir=${WANDB_DIR} \
#   #   multi_process=True \
#   #   image_size=512 \
#   #   model="dit_s_mttt_bdar" \
#   #   model.patch_size=${patch_size} \
#   #   model.attn_kwargs.mttt_kwargs.n_iters=${niters} \
#   #   model.attn_kwargs.mttt_kwargs.lr=${lr} \
#   #   total_iters=400000
# done


# # NOTE this doesn't have checkpoints yet...
# EXPR_NAME="ssm/p1-DiT-B-SSM-2024-Aug-09-06-43-51"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
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
#   total_iters=4000000


# psize=1
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="ssm/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   image_size=512 \
#   model=dit_b_ssm \
#   ckpt_every=25000 \
#   total_iters=400000


# EXPR_NAME="dit-s/2-DiT-S-2024-Jul-23-18-39-14"
# ls ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/config.yaml
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   --config-path . \
#   data_dir=${DATA_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   resume=True \
#   total_iters=4000000
# 
# psize=8
# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="dit-b/p${psize}" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_b" \
#   data_loader=imagenet_pixel_tfds \
#   feature_path=imagenet256_pixels_tfdata_sharded/ \
#   model.patch_size=${psize} \
#   use_latent=False \
#   latent_dim=3 \
#   total_iters=1000000
# 

# PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name="vit3lin/1-base" \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True \
#   model="dit_s_vit3_linear" \
#   model.patch_size=1 \
#   total_iters=1000000
# 

# # for ilr in 1 0.1 10 0.01; do
# for ilr in 1; do
#   PYTHONPATH=${LOCAL_CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#     expr_name="vit3lin" \
#     data_dir=${DATA_DIR} \
#     output_dir=${OUTPUT_DIR} \
#     hydra_dir=${HYDRA_DIR} \
#     wandb_dir=${WANDB_DIR} \
#     multi_process=True \
#     model=dit_s_vit3_linear \
#     model.attn_kwargs.ttt_cfg.inner_itr=4 \
#     model.attn_kwargs.ttt_cfg.inner_lr="[${ilr}, ${ilr}, ${ilr}, ${ilr}]" \
#     total_iters=400000
# done

# cd $CODE_DIR

# # Linear attention tiny
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=noscaleq \
#   model=dit_tiny_linattn \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True
 
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=noscaleq \
#   model=dit_tiny_mttt_linear \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   multi_process=True

# for ilr in 0.1 0.01; do
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
# done

# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   expr_name=mt3match2 \
#   model=dit_s_mttt_linear \
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
#   multi_process=True
# done

# EXPR_NAME="mt3mlp-DiT-S-mt3mlp-v0-2024-Jul-15-06-05-16"
# ilr=1.0
# PYTHONPATH=${CODE_DIR} WANDB_DIR=${WANDB_DIR} python main.py \
#   --config-dir ${BUCKET_MNT}/checkpoints/${EXPR_NAME}/ \
#   expr_name=mt3mlp \
#   model=dit_s_mttt_mlp_v0 \
#   data_dir=${DATA_DIR} \
#   output_dir=${OUTPUT_DIR} \
#   hydra_dir=${HYDRA_DIR} \
#   wandb_dir=${WANDB_DIR} \
#   model.attn_kwargs.mttt_kwargs.lr=${ilr} \
#   model.attn_kwargs.mttt_kwargs.n_iters=4 \
#   multi_process=True \
#   resume=True \
#   total_iters=4000000
# done