#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <config.yaml> <timing-output-directory>" >&2
  exit 2
fi

config_path=$1
timing_dir=$2
mkdir -p "$timing_dir"

log_path="$timing_dir/run_delivery.log"
timings_path="$timing_dir/stage_timings.csv"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

write_timings() {
  {
    printf 'stage_index,stage,elapsed_seconds\n'
    sed -nE 's/^\[([0-9]+)\] ([^:]+): completed in ([0-9.]+)s$/\1,\2,\3/p' "$log_path"
  } > "$timings_path"
}
trap write_timings EXIT

printf 'started_at_utc=%s\nconfig=%s\n' "$started_at" "$config_path" | tee "$log_path"
python -m dongfeng_registration.cli run-delivery --config "$config_path" 2>&1 | tee -a "$log_path"
