#!/usr/bin/env bash
# Start another retrieval training phase with a larger hotel batch.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
source_run=${SOURCE_RUN_DIR:-"$repo_dir/output/openhotels-retrieval-epshn"}
export RUN_DIR=${RUN_DIR:-"$repo_dir/output/openhotels-retrieval-epshn-batch48-continuation"}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-"$source_run/last.pth"}
export HOTELS_PER_BATCH=${HOTELS_PER_BATCH:-48}
export ADDITIONAL_EPOCHS=${ADDITIONAL_EPOCHS:-20}
export EMBEDDING_CACHE_DIR=${EMBEDDING_CACHE_DIR:-"$repo_dir/cache/openhotels-retrieval-batch48-continuation"}

# CLI overrides must also govern which output/checkpoint receives a baseline.
launch_args=("$@")
for ((arg_index = 0; arg_index < ${#launch_args[@]}; arg_index++)); do
  case "${launch_args[arg_index]}" in
    --save_dir|--resume)
      option=${launch_args[arg_index]}
      if ((arg_index + 1 >= ${#launch_args[@]})); then
        printf '%s requires a path.\n' "$option" >&2
        exit 2
      fi
      arg_index=$((arg_index + 1))
      if [[ "$option" == --save_dir ]]; then
        RUN_DIR=${launch_args[arg_index]}
      else
        RESUME_CHECKPOINT=${launch_args[arg_index]}
      fi
      ;;
    --save_dir=*) RUN_DIR=${launch_args[arg_index]#--save_dir=} ;;
    --resume=*) RESUME_CHECKPOINT=${launch_args[arg_index]#--resume=} ;;
  esac
done

mkdir -p "$RUN_DIR"
# Preserve the source run's validated best only when resuming that same run.
# A custom checkpoint from another run must never inherit an unrelated model.
if [[ ! -f "$RUN_DIR/best.pth" && -f "$source_run/best.pth" ]] && \
   [[ "$RESUME_CHECKPOINT" -ef "$source_run/last.pth" || \
      "$RESUME_CHECKPOINT" -ef "$source_run/best.pth" ]]; then
  cp "$source_run/best.pth" "$RUN_DIR/best.pth"
fi
exec bash "$repo_dir/scripts/run_openhotels_retrieval_monitored.sh" "$@"
