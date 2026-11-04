#!/usr/bin/env bash
set -euo pipefail
# Optional isolated installation. Does not change the existing Python packages.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${STREETGS_TOOLS:-$(dirname "$ROOT")/.tools}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available physical GPU}"
export PYTHONPATH="$TOOLS/gsplat153${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$TOOLS/gsplat153/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$TOOLS/gsplat153_build"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$ROOT"
exec "${PYTHON:-/usr/bin/python}" "$@"
