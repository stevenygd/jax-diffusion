#! /bin/bash

# Configuration
export TPU_NAME=v4-128-p
export ZONE=us-central2-b
export DISK_NAME=us-central2-b-imagenet
export NFS_IP=10.47.236.66

# # Step 0: create TPU
# gcloud alpha compute tpus tpu-vm create $TPU_NAME \
# 	--zone=$ZONE \
# 	--accelerator-type='v4-128' \
# 	--preemptible \
# 	--version='tpu-ubuntu2204-base'

# Step 1: mount data disk (read-only disk)
gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME --zone=$ZONE \
  --disk=$DISK_NAME \
  --mode=read-only # if data disk

# NOTE: when remount, the disk will increment, so it might not be /dev/sdb
#       this requires some additional attention if there are many nodes
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mkdir -p /mnt/disks/data"
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mount -o ro,noload /dev/sdb /mnt/disks/data"


# Step 2: mount code disk (network file system)
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
    --command="sudo apt-get update --allow-releaseinfo-change && sudo apt-get -y update && sudo apt-get -y install nfs-common"

gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mkdir -p /mnt/disks/nfs"
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo chmod ugo+rw /mnt/disks/nfs/"
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
    --command="showmount $NFS_IP -e"
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
    --command="sudo mount $NFS_IP:/nfs_us_c2b /mnt/disks/nfs/"

# Step 3: install TPU environment
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="bash /mnt/disks/nfs/guandao/jax-DiT/scripts/install_env.sh"

# Step 4: install fscfushion 
gcloud compute tpus tpu-vm ssh guandao@$TPU_NAME --zone=$ZONE --worker=all \
	--command="bash /mnt/disks/nfs/guandao/jax-DiT/scripts/install_gcsfuse.sh"