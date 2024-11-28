#! /bin/bash

# Install gcsfuse for checkpoints
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse
echo `gcsfuse --help`

# BUCKET_NAME="sci-guandao"
# BUCKET_MNT="/mnt/disks/gs/${BUCKET_NAME}"
# # # Mount bucket
# fusermount -u $BUCKET_MNT
# sudo rm -rf $BUCKET_MNT
# sudo mkdir -p $BUCKET_MNT
# sudo chmod 777 $BUCKET_MNT
# gcsfuse --implicit-dirs \
#   -o rw \
#   --dir-mode 777 \
#   --file-mode 777 \
#   --rename-dir-limit 5000000 \
#   $BUCKET_NAME $BUCKET_MNT
# 