#! /bin/bash

# export TPU_NAME=ttt-512-0
# export ZONE=europe-west4-b
# export TPU_DISK_0=disk-guandao
# export TPU_TYPE=v5litepod-256
export TPU_NAME=ttt-64-1
export ZONE=us-central2-b
export TPU_DISK_0=disk-guandao-v4
export PROJECT_ID=molten-unison-414123
export NFS_IP=10.164.0.2
export TPU_TYPE=v4-64

# # DELETE TPU
# gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}  --project=${PROJECT_ID}
# gcloud alpha compute tpus queued-resources delete  ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE} --async
# 
# # Create TPU
# gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
#    --node-id=${TPU_NAME} \
#    --project=${PROJECT_ID} \
#    --zone=${ZONE} \
#    --accelerator-type=${TPU_TYPE} \
#    --runtime-version=tpu-ubuntu2204-base

# Mount data disk
gcloud alpha compute tpus tpu-vm attach-disk ${TPU_NAME} \
    --disk=${TPU_DISK_0} --zone=${ZONE}  \
    --project=${PROJECT_ID}  --mode=read-only
gcloud compute tpus tpu-vm scp /checkpoint/guandao/jax-DiT/scripts/install_data_disk_karan.sh \
   --worker=all --zone=${ZONE} \
   ${TPU_NAME}:/home/Grendel 
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
   --worker=all --zone=${ZONE} \
   --command  "bash ~/install_data_disk_karan.sh"

# Mount NFS
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
   --worker=all --zone=${ZONE}\
   --command="sudo apt-get update --allow-releaseinfo-change && sudo apt-get -y update && sudo apt-get -y install nfs-common"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mkdir -p /checkpoint"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo chmod ugo+rw /checkpoint"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command="showmount $NFS_IP -e"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
    --command="sudo mount $NFS_IP:/checkpoint /checkpoint"
 
# Install project env
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
   --worker=all --zone=${ZONE} \
   --command "bash /checkpoint/guandao/jax-DiT/scripts/install_env.sh"

# # THESE ARE ALL GOOD GCSFUSE FUCK IT UP
# # Install gcsfush
# gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
#    --worker=all --zone=${ZONE} \
#    --command "bash /checkpoint/guandao/jax-DiT/scripts/install_gcsfuse.sh"

