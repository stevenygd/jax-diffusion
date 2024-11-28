POD_TYPE='v3-512' # v3-8, v3-512, v4-64, v5-128, v5-256
PROJECT='trc' # wang, rohan, trc, yann, geng
NUMBER=0
POD_NAME=${PROJECT}-${POD_TYPE}-${NUMBER}
DISK_NAME=disk-${POD_NAME}

DISK_NAME=disk-guandao

if [[ $POD_TYPE == *"v4"* ]]; then
    ZONE="us-central2-b"
elif [[ $POD_TYPE == *"v3"* ]]; then
    ZONE="europe-west4-a"
else
    ZONE="europe-west4-b"
fi

if [[ $POD_TYPE == *"v3"* ]] || [[ $POD_TYPE == *"v4"* ]]; then
    DEV_NUM=$(($(echo $POD_TYPE | sed 's/.*-//' | awk '{print $1}') / 8))
else
    DEV_NUM=$(($(echo $POD_TYPE | sed 's/.*-//' | awk '{print $1}') / 4))
fi

##############
# Create Pod #
##############

if [[ ${POD_TYPE} =~ "v3" ]]; then
    gcloud compute tpus tpu-vm create ${POD_NAME} \
        --zone=${ZONE} \
        --project=${!PROJECT} \
        --accelerator-type=${POD_TYPE} \
        --version=tpu-vm-base \
        --preemptible
elif [[ ${POD_TYPE} =~ "v4" ]]; then
    gcloud compute tpus tpu-vm create ${POD_NAME} \
        --zone=${ZONE} \
        --project=${!PROJECT} \
        --accelerator-type=${POD_TYPE} \
        --version=tpu-ubuntu2204-base
elif [[ ${POD_TYPE} =~ "v5-128" ]]; then
    gcloud compute tpus queued-resources delete ${POD_NAME} --project ${!PROJECT} --zone ${ZONE} --async
    gcloud alpha compute tpus queued-resources create ${POD_NAME} \
        --node-id=${POD_NAME} \
        --project=${!PROJECT} \
        --zone=${ZONE} \
        --accelerator-type=v5litepod-128 \
        --runtime-version=tpu-ubuntu2204-base \
        --reserved
    sleep 5m
elif [[ ${POD_TYPE} =~ "v5-256" ]]; then
    gcloud compute tpus queued-resources delete ${POD_NAME} --project ${!PROJECT} --zone ${ZONE} --async
    gcloud alpha compute tpus queued-resources create ${POD_NAME} \
        --node-id=${POD_NAME} \
        --project=${!PROJECT} \
        --zone=${ZONE} \
        --accelerator-type=v5litepod-256 \
        --runtime-version=tpu-ubuntu2204-base  \
        --reserved
    sleep 5m
fi


gcloud alpha compute tpus tpu-vm attach-disk ${POD_NAME} --disk=${DISK_NAME} --zone=${ZONE} --project=${!PROJECT} --mode=read-only &&
ATTACHED=$(($(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME} --project=${!PROJECT} --zone=${ZONE} --worker=all --command "sudo lsblk | grep sdb" | wc -l) == ${DEV_NUM}))
if [[ $ATTACHED -ne 1 ]]
then
  echo "Persistent disk has not been attached to all hosts in TPU pod! Exiting..."
  exit 1
fi

##############
# Mount Disk #
##############

gcloud alpha compute tpus tpu-vm scp ${PWD}/client/mount_disk.sh ${POD_NAME}:/home/karandalal --project=${!PROJECT} --zone=${ZONE} --worker=all

gcloud alpha compute tpus tpu-vm ssh ${POD_NAME} --project=${!PROJECT} --zone=${ZONE} --worker=all --command "chmod +x /home/karandalal/mount_disk.sh && bash /home/karandalal/mount_disk.sh"

MOUNTED=$(($(gcloud alpha compute tpus tpu-vm ssh ${POD_NAME} --project=${!PROJECT} --zone=${ZONE} --worker=all --command "sudo lsblk | grep /persistent" | wc -l) == ${DEV_NUM}))
if [[ $MOUNTED -ne 1 ]]
then
  echo "Persistent disk has not been mounted to All hosts in TPU pod! Exiting..."
  exit 1
fi

#############
# Setup NFS #
#############

sudo apt install -y nfs-kernel-server
sudo exportfs -ar
sudo systemctl restart nfs-kernel-server.service

gcloud alpha compute tpus tpu-vm scp ${PWD}/client/nfs_client.sh ${POD_NAME}:/home/karandalal --project=${!PROJECT} --zone=${ZONE}  --worker=all
gcloud alpha compute tpus tpu-vm ssh ${POD_NAME} --project=${!PROJECT} --zone=${ZONE}  --worker=all --command "sudo mkdir -p /checkpoint"

function setup_nfs {
    gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=all --command "chmod +x /home/karandalal/nfs_client.sh && bash /home/karandalal/nfs_client.sh ${IP}"

    SHARED=$(gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=all --command "df | grep ${IP}:/checkpoint" | wc -l)
    if [[ $SHARED -ne ${DEV_NUM} ]]; then
        echo "NFS client has not been set up on all hosts in TPU pod! Checking individual workers..."

        for ((i=0; i<$DEV_NUM; i++)); do
            WORKER_SETUP=$(gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=${i} --command "df | grep ${IP}:/checkpoint" | wc -l)
            if [[ $WORKER_SETUP -eq 0 ]]; then
                echo "Worker ${i} failed. Attempting to fix..."

                PID=$(gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=${i} --command "sudo fuser /var/lib/dpkg/lock-frontend")
                gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=${i} --command "sudo kill -9 ${PID}"
                gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=${i} --command "chmod +x /home/karandalal/nfs_client.sh && bash /home/karandalal/nfs_client.sh ${IP}"

            fi
        done

        SHARED=$(gcloud alpha compute tpus tpu-vm ssh "${POD_NAME}" --project="${!PROJECT}" --zone="${ZONE}" --worker=all --command "df | grep ${IP}:/checkpoint" | wc -l)
        if [[ $SHARED -ne ${DEV_NUM} ]]; then
            echo "NFS client has not been set up on all hosts in TPU pod after retry! Exiting..."
            exit 1
        fi
    fi
}

setup_nfs
Collapse