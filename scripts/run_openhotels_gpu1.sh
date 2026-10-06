#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
run_dir=${RUN_DIR:-"$repo_dir/output/openhotels-classification"}
data_dir=${DATA_DIR:-"$repo_dir/data/OpenHotels-Updated"}

# Relative paths are interpreted from the checkout, regardless of the caller.
cd "$repo_dir"

mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
export PYTHONUNBUFFERED=1

exec "$python_bin" main.py \
  --dataset openhotels \
  --data_dir "$data_dir" \
  --save_dir "$run_dir" \
  --architecture "${ARCHITECTURE:-vit_small_r26_s32_224}" \
  --num_images "${NUM_IMAGES:-4}" \
  --image_size "${IMAGE_SIZE:-224}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-16}" \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --lr "${LR:-0.01}" \
  --wd "${WEIGHT_DECAY:-5e-4}" \
  --num_epochs "${NUM_EPOCHS:-50}" \
  --eval_every "${EVAL_EVERY:-5}" \
  --skip_initial_eval true \
  --log_interval "${LOG_INTERVAL:-100}" \
  --use_mutual_distillation_loss true \
  "$@"
