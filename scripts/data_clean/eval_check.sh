EXPR_NAME="${1}"
CKPT_STEP="${2}"
CFG=1.0

SEED=0
IMAGE_SIZE=256
NSMP=10000
STEP=50
MODE="rectflow"

declare -A buckets
buckets["dit-guandao"]="/mnt/disks/gs1/outputs"
buckets["sci-guandao"]="/mnt/disks/gs2/outputs"

directory_exists=false
for BUCKET_NAME in "${!buckets[@]}"; do
  BUCKET_MNT=${buckets[$BUCKET_NAME]}
  if [ -d "${BUCKET_MNT}/checkpoints/${EXPR_NAME}/checkpoints" ]; then
    directory_exists=true
    break
  fi
done

if [ "$directory_exists" = false ]; then
  echo "Directory not found in dit-guandao and sci-guandao."
  exit 1
fi

SAMPLE_DIR="$BUCKET_MNT/checkpoints/$EXPR_NAME/samples"
CHECKPOINT_DIR="$BUCKET_MNT/checkpoints/$EXPR_NAME/checkpoints"

echo "checkpoints: " && ls $CHECKPOINT_DIR | sort -n | tr '\n' ' '
echo " "
echo "samples: " && ls $SAMPLE_DIR | sort -n | tr '\n' ' '
echo " "

SAMPLE_FOLDER_SUFFIX="size-${IMAGE_SIZE}-cfg-${CFG}-seed-${SEED}-step-${STEP}-nsmp-${NSMP}-${MODE}"


if [ -z "$CKPT_STEP" ]; then
  echo "CKPT_STEP argument is empty. Brief status of all checkpoints:"

  NPZ_SUFFIX="size-${IMAGE_SIZE}-cfg-${CFG}-seed-${SEED}-step-${STEP}-nsmp-${NSMP}-${MODE}.npz"
  METRIC_SUFFIX="size-${IMAGE_SIZE}-cfg-${CFG}-seed-${SEED}-step-${STEP}-nsmp-${NSMP}-${MODE}-metrics.npy"

  for SUBDIR in "$SAMPLE_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
      SUBDIR_NAME=$(basename "$SUBDIR")
      if [ -d "$SAMPLE_DIR/$SUBDIR_NAME" ]; then
        FOLDER_FOUND=false
        for SUBSUBDIR in "$SUBDIR"/*; do
          if [ -d "$SUBSUBDIR" ] && [[ "$SUBSUBDIR" == *"$SAMPLE_FOLDER_SUFFIX" ]]; then
            FOLDER_FOUND=true
            SAMPLE_PNG_DIR="$SUBSUBDIR"
            break
          fi
        done
        if [ "$FOLDER_FOUND" = false ]; then
          # echo "$SUBDIR_NAME: Sampling not started"
          continue
        fi
        echo "$SUBDIR_NAME:"
        
        NPZ_FOUND=false
        for FILE in "$SAMPLE_DIR/$SUBDIR_NAME"/*; do
          if [[ "$FILE" == *"$NPZ_SUFFIX" ]]; then
            NPZ_FOUND=true
          fi
        done
        if [ "$NPZ_FOUND" = false ]; then
          echo "   Missing $NPZ_SUFFIX"
        fi

        METRIC_FOUND=false
        for FILE in "$SAMPLE_DIR/$SUBDIR_NAME"/*; do
          if [[ "$FILE" == *"$METRIC_SUFFIX" ]]; then
            METRIC_FOUND=true
          fi
        done
        if [ "$METRIC_FOUND" = false ]; then
          echo "   Missing $METRIC_SUFFIX"
        fi
        # if [ "$NPZ_FOUND" = true ] && [ "$METRIC_FOUND" = true ]; then
        #   echo "   Sampling complete!"
        # fi
      fi
    fi
  done
  exit 1
fi

for SUBDIR in "$SAMPLE_DIR"/*; do
  if [ -d "$SUBDIR" ]; then
    SUBDIR_NAME=$(basename "$SUBDIR")
    if [ -d "$SAMPLE_DIR/$SUBDIR_NAME" ]; then
      FOLDER_FOUND=false
      for SUBSUBDIR in "$SUBDIR"/*; do
        if [ -d "$SUBSUBDIR" ] && [[ "$SUBSUBDIR" == *"$SAMPLE_FOLDER_SUFFIX" ]]; then
          FOLDER_FOUND=true
          SAMPLE_PNG_DIR="$SUBSUBDIR"
          break
        fi
      done
    fi
  fi
done

# Find the most recent sample directory
if [ -z "$SAMPLE_PNG_DIR" ]; then
    echo "Error: SAMPLE_DIR is not found for $CKPT_STEP."
    exit 1
fi

# Find the most recently modified files and print their details
find "$SAMPLE_PNG_DIR" -type f -printf "%T@ %p\n" | sort -n | tail -n 3 | while read -r time path; do
    formatted_time=$(TZ="America/Los_Angeles" date -d @"$time" "+%Y-%m-%d %H:%M:%S %Z")
    file_name=$(basename "$path")
    echo "$formatted_time      $file_name"
done
echo " "

python /mnt/disks/nfs/ujinsong/jax-DiT/scripts/data_clean/proc_check.py $EXPR_NAME $CKPT_STEP\
  --seed $SEED --cfg $CFG --step $STEP