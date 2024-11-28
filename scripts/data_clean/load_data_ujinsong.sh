export BUCKET_MNT="/mnt/disks/gs1/outputs"
export BUCKET_NAME="dit-guandao"

fusermount -u $BUCKET_MNT
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
sudo chmod 777 `dirname $BUCKET_MNT`
gcsfuse --implicit-dirs \
  -o rw \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT

export BUCKET_MNT="/mnt/disks/gs2/outputs"
export BUCKET_NAME="sci-guandao"

fusermount -u $BUCKET_MNT
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
sudo chmod 777 `dirname $BUCKET_MNT`
gcsfuse --implicit-dirs \
  -o rw \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT

export BUCKET_MNT="/mnt/disks/pexel-bucket"
export BUCKET_NAME="pexel-bucket"

fusermount -u $BUCKET_MNT
sudo rm -rf $BUCKET_MNT
sudo mkdir -p $BUCKET_MNT
sudo chown -R `whoami`:`whoami` $BUCKET_MNT
sudo chmod 777 $BUCKET_MNT
sudo chmod 777 `dirname $BUCKET_MNT`
gcsfuse --implicit-dirs \
  -o rw \
  --dir-mode 777 \
  --file-mode 777 \
  --rename-dir-limit 5000000 \
  $BUCKET_NAME $BUCKET_MNT