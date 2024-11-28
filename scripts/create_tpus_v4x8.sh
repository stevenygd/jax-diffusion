#!/bin/bash

# Array of TPU names
TPU_NAMES=("v4-8-p-1" "v4-8-p-2" "v4-8-p-3" "v4-8-p-4" "v4-8-p-5" "v4-8-p-6" "v4-8-p-7" "v4-8-p-8" "v4-8-p-9" "v4-8-p-10" "v4-8-p-11" "v4-8-p-12" "v4-8-p-13" "v4-8-p-14" "v4-8-p-15" "v4-8-p-16")
SCRIPT_PATH="/mnt/disks/nfs/ujinsong/jax-DiT/scripts/create_tpu_ujinsong_v4x8.sh"

# Function to check if a TPU exists
function tpu_exists {
  local TPU_NAME=$1
  AVAILABLE_TPUS=$(gcloud compute tpus list --format="value(name)")
  if echo "$AVAILABLE_TPUS" | grep -q "$TPU_NAME"; then
    return 0
  else
    return 1
  fi
}

for TPU_NAME in "${TPU_NAMES[@]}"; do
  while true; do
    if tpu_exists "$TPU_NAME"; then
      echo "TPU $TPU_NAME is already available. Moving to the next one..."
      break
    else
      echo "Creating TPU $TPU_NAME..."
      bash "$SCRIPT_PATH" "$TPU_NAME"
      sleep 120  # Wait for a while before checking again
    fi
  done
done

echo "All TPUs have been created"