#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
timing_root=${1:-"$repo_dir/../outputs/all_remaining_data_20260907"}

cd "$repo_dir"
source .venv/bin/activate
export TORCH_HOME=/mnt/workspace/Perception_Models/torch
export HF_HOME=/mnt/workspace/Perception_Models/pose/huggingface
export MPLCONFIGDIR=/mnt/workspace/Pose_Dynamic_Compensation/.cache/matplotlib
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

configs=(
  configs/t3_20260425_061115.yaml
  configs/t3_20260425_062616.yaml
  configs/t3_20260425_143257.yaml
  configs/m18_20251215104909.yaml
  configs/m18_20251220130904.yaml
)

mkdir -p "$timing_root"
for config_path in "${configs[@]}"; do
  run_name=$(basename "$config_path" .yaml)
  ./scripts/run_delivery_timed.sh "$config_path" "$timing_root/$run_name"
done
