#! /bin/bash

# Configuration
export TPU_NAME=${1}
export ZONE=us-central2-b
export DISK_NAME=us-central2-b-imagenet
export NFS_IP=35.186.52.1
export USER_NAME=`whoami`

# Step 0: create TPU
gcloud compute tpus tpu-vm create $TPU_NAME \
	--zone=$ZONE \
	--accelerator-type=v4-8 \
    --preemptible \
	--version=tpu-ubuntu2204-base

# Step 1: mount data disk (read-only disk)
gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME --zone=$ZONE \
  --disk=$DISK_NAME \
  --mode=read-only # if data disk

# NOTE: when remount, the disk will increment, so it might not be /dev/sdb
#       this requires some additional attention if there are many nodes
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mkdir -p /mnt/disks/data"
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mount -o ro,noload /dev/sdb /mnt/disks/data"


# Step 2: mount code disk (network file system)
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
    --command="sudo apt-get update --allow-releaseinfo-change && sudo apt-get -y update && sudo apt-get -y install nfs-common"

gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo mkdir -p /mnt/disks/nfs"
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="sudo chmod ugo+rw /mnt/disks/nfs/"
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
    --command="showmount $NFS_IP -e"
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
    --command="sudo mount -o rw,vers=3 $NFS_IP:/nfs_us_c2b /mnt/disks/nfs/"

# Step 3: install TPU environment
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="bash /mnt/disks/nfs/${USER_NAME}/jax-DiT/scripts/install_env.sh"

# Step 4: install fscfushion
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
	--command="bash /mnt/disks/nfs/${USER_NAME}/jax-DiT/scripts/install_gcsfuse.sh"

# Step 5: set shortcut
gcloud compute tpus tpu-vm ssh ${USER_NAME}@$TPU_NAME --zone=$ZONE --worker=all \
    --command="function_definitions='
    export CODE_ROOT=/mnt/disks/nfs/ujinsong
    function exprcheck() {
        echo \"[ experiment name: \$EXPR_NAME ]\"
        local CKPT_STEP=\$1
        bash \$CODE_ROOT/jax-DiT/scripts/eval_check.sh \$EXPR_NAME \$CKPT_STEP
    }

    function tpucheck() {
        local input=\$1
        if [ \"\$input\" != \"0\" ]; then
            bash \$CODE_ROOT/jax-DiT/scripts/load_data_ujinsong.sh
        fi
        gcsfuse
        lsblk
        ls \$CODE_ROOT
        ls /mnt/disks/data
		ls /mnt/disks/gs1/outputs
    }

    function tpubusy() {
        sudo lsof -w /dev/accel* | tail -n +2 | awk '\''{print \$2}'\'' | sort -u | while read pid; do
            echo \"PID: \$pid\"
            sudo cat /proc/\$pid/cmdline
            echo
        done
        tmux ls
    }
    '
    echo \"\$function_definitions\" >> ~/.bashrc && source ~/.bashrc"