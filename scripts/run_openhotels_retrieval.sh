#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
run_dir=${RUN_DIR:-"$repo_dir/output/openhotels-retrieval-epshn"}
cache_dir=${EMBEDDING_CACHE_DIR:-"$repo_dir/cache/openhotels-retrieval"}
data_dir=${DATA_DIR:-"$repo_dir/data/OpenHotels-Updated"}
checkpoint=${CLASSIFICATION_CHECKPOINT:-"$repo_dir/output/openhotels-classification/best.pth"}
physical_gpu=${GPU_ID:-0}  # The selected physical GPU becomes logical cuda:0.
hotels_per_batch=${HOTELS_PER_BATCH:-24}
log_interval=${LOG_INTERVAL:-20}
num_epochs=${NUM_EPOCHS:-20}
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ -n "${ADDITIONAL_EPOCHS:-}" ]]; then
  resume_args+=(--additional_epochs "$ADDITIONAL_EPOCHS")
fi

# Relative paths are interpreted from the checkout, regardless of the caller.
cd "$repo_dir"
mkdir -p "$run_dir" "$cache_dir"

# CUDA renumbers the selected physical GPU to logical cuda:0 in this process.
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONUNBUFFERED=1

exec "$python_bin" retrieval_main.py "${resume_args[@]}" \
  --device cuda:0 \
  --data_dir "$data_dir" \
  --save_dir "$run_dir" \
  --checkpoint "$checkpoint" \
  --architecture "${ARCHITECTURE:-vit_small_r26_s32_224}" \
  --image_size "${IMAGE_SIZE:-224}" \
  --model_max_views "${MODEL_MAX_VIEWS:-4}" \
  --min_query_views "${MIN_QUERY_VIEWS:-2}" \
  --max_query_views "${MAX_QUERY_VIEWS:-4}" \
  --positive_gallery_size "${POSITIVE_GALLERY_SIZE:-4}" \
  --hotels_per_batch "$hotels_per_batch" \
  --num_epochs "$num_epochs" \
  --loss "${RETRIEVAL_LOSS:-epshn}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --view_loss_weight "${VIEW_LOSS_WEIGHT:-0.25}" \
  --classification_loss_weight "${CLASSIFICATION_LOSS_WEIGHT:-0.1}" \
  --classification_decay_epochs "${CLASSIFICATION_DECAY_EPOCHS:-5}" \
  --lr "${LR:-0.001}" \
  --weight_decay "${WEIGHT_DECAY:-0.0005}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --num_workers "${NUM_WORKERS:-16}" \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --gallery_batch_size "${GALLERY_BATCH_SIZE:-128}" \
  --query_batch_size "${QUERY_BATCH_SIZE:-8}" \
  --embedding_cache_dir "$cache_dir" \
  --eval_every "${EVAL_EVERY:-5}" \
  --log_interval "$log_interval" \
  "$@"
