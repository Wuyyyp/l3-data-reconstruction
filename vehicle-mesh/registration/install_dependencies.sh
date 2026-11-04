#!/usr/bin/env bash
# Install the standalone multiview registration task dependencies.
# Usage: bash install_dependencies.sh [--python PATH] [--venv DIR] [--torch-index-url URL]
# Example: bash tests/multiview_registration/install_dependencies.sh --venv /opt/multiview-env --torch-index-url https://download.pytorch.org/whl/cu128
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR=""
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    --venv) [[ $# -ge 2 ]] || { echo "--venv requires a directory" >&2; exit 2; }; VENV_DIR="$2"; shift 2 ;;
    --torch-index-url) [[ $# -ge 2 ]] || { echo "--torch-index-url requires a URL" >&2; exit 2; }; TORCH_INDEX_URL="$2"; shift 2 ;;
    -h|--help) sed -n '1,6p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('SAM3 requires Python 3.12 or newer; select a compatible interpreter with --python')
PY
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install 'setuptools<82'
"$PYTHON_BIN" -m pip install 'torch>=2.7' 'torchvision>=0.22' --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install -r "$TASK_DIR/requirements.txt"
"$PYTHON_BIN" - <<'PY'
import sys
import torch, trimesh, pyrender, sam3
if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 7): raise SystemExit('SAM3 requires PyTorch 2.7 or newer')
print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
print(f"trimesh={trimesh.__version__}, pyrender={pyrender.__version__}, sam3={sam3.__file__}")
try:
    import os
    os.environ['PYOPENGL_PLATFORM']='egl'
    from OpenGL import EGL
except Exception as error:
    print(f"WARNING: EGL shared libraries are unavailable; install the OS EGL/GLU runtime before rendering: {error}")
PY
echo "Set SAM3_CHECKPOINT to the local SAM3 checkpoint before running registration."
