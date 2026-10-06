#!/usr/bin/env bash
# Run inside screen; preserve console output and periodically sample progress.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
run_dir=${RUN_DIR:-"$repo_dir/output/openhotels-retrieval-epshn"}
physical_gpu=${GPU_ID:-0}
monitor_interval=${MONITOR_INTERVAL:-30}
if [[ ! "$monitor_interval" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'MONITOR_INTERVAL must be a positive integer number of seconds.' >&2
  exit 2
fi

# Keep monitor files beside the actual training output when the CLI overrides it.
launch_args=("$@")
for ((arg_index = 0; arg_index < ${#launch_args[@]}; arg_index++)); do
  case "${launch_args[arg_index]}" in
    --save_dir)
      if ((arg_index + 1 >= ${#launch_args[@]})); then
        printf '%s\n' '--save_dir requires a path.' >&2
        exit 2
      fi
      arg_index=$((arg_index + 1))
      run_dir=${launch_args[arg_index]}
      ;;
    --save_dir=*) run_dir=${launch_args[arg_index]#--save_dir=} ;;
  esac
done
run_tag=$(date -u +%Y%m%dT%H%M%SZ)-$$
run_prefix="$run_dir/run-$run_tag"
console_log="$run_prefix.console.log"
monitor_log="$run_prefix.monitor.log"
status_file="$run_prefix.status"
pid_file="$run_prefix.pid"
exit_file="$run_prefix.exit"
monitor_pid=
training_pid=
tee_pid=
mkdir -p "$run_dir"
ln -sfn -- "$(basename -- "$status_file")" "$run_dir/monitor.status"
ln -sfn -- "$(basename -- "$monitor_log")" "$run_dir/monitor.log"
ln -sfn -- "$(basename -- "$console_log")" "$run_dir/monitor.console.log"

snapshot() {
  local state=$1
  {
    printf 'time=%s state=%s training_pid=%s\n' "$(date -u +%FT%TZ)" "$state" "$training_pid"
    printf '%s\n' '--- Latest console progress ---'
    tail -n 8 "$console_log" 2>/dev/null || true
    printf '%s\n' "--- Physical GPU $physical_gpu (index, name, utilization %, memory used/total MiB) ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -i "$physical_gpu" --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>&1 || true
    else
      printf '%s\n' 'nvidia-smi is unavailable; GPU statistics were skipped.'
    fi
  } | tee -a "$monitor_log" > "$status_file.tmp"
  mv -f -- "$status_file.tmp" "$status_file"
}

stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=
  fi
}

cleanup() {
  stop_monitor
  if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
    kill -TERM "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  if [[ -n "$tee_pid" ]]; then
    wait "$tee_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Console: %s\nMonitor: %s\nStatus: %s\n' "$console_log" "$monitor_log" "$status_file"
exec 3> >(tee -a "$console_log")
tee_pid=$!
bash "$repo_dir/scripts/run_openhotels_retrieval.sh" "$@" >&3 2>&1 &
training_pid=$!
exec 3>&-
printf '%s\n' "$training_pid" > "$pid_file"
(
  timer_pid=
  trap 'if [[ -n "$timer_pid" ]]; then kill "$timer_pid" 2>/dev/null || true; wait "$timer_pid" 2>/dev/null || true; fi; exit' TERM INT
  while kill -0 "$training_pid" 2>/dev/null; do
    snapshot running
    sleep "$monitor_interval" &
    timer_pid=$!
    wait "$timer_pid"
    timer_pid=
  done
) &
monitor_pid=$!

if wait "$training_pid"; then exit_code=0; else exit_code=$?; fi
stop_monitor
wait "$tee_pid" || true
tee_pid=
printf '%s\n' "$exit_code" > "$exit_file"
snapshot "exited:$exit_code"
printf 'Training exited with code %s; logs: %s\n' "$exit_code" "$run_prefix"
exit "$exit_code"
