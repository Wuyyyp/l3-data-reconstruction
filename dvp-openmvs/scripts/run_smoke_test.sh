#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scene="${1:-diagnostics/smoke_retest}"
python scripts/make_smoke_scene.py "$scene" --width 832
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ./build/APD "$scene" 0
python scripts/check_smoke_output.py "$scene"
