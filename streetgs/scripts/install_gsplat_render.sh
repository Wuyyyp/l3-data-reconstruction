#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${STREETGS_TOOLS:-$(dirname "$ROOT")/.tools}"
SOURCE="$TOOLS/gsplat-1.5.3-source"
PYTHON="${PYTHON:-/usr/bin/python}"
mkdir -p "$TOOLS"
if [[ ! -d "$SOURCE" ]]; then
  git clone --branch v1.5.3 --depth 1 --recursive https://github.com/nerfstudio-project/gsplat.git "$SOURCE"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == 937e29912570c372bed6747a5c9bf85fed877bae ]]
BUILD_NO_CUDA=1 "$PYTHON" -m pip install --no-deps --target "$TOOLS/gsplat153" \
  "$SOURCE" ninja==1.13.2 jaxtyping==0.2.19 rich==13.7.1 \
  markdown-it-py==3.0.0 mdurl==0.1.2 pygments==2.17.2 typing_extensions==4.13.2
# Torch 1.13 JIT defaults to C++14; gsplat 1.5.3's setup.py specifies C++17,
# but its lazy JIT path omits the standard. Match the official build standard.
"$PYTHON" - "$TOOLS/gsplat153/gsplat/cuda/_backend.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
for a,b in [('extra_cflags = [opt_level, "-Wno-attributes"]',
             'extra_cflags = [opt_level, "-Wno-attributes", "-std=c++17"]'),
            ('extra_cuda_cflags = [opt_level]',
             'extra_cuda_cflags = [opt_level, "-std=c++17"]')]:
    assert a in s or b in s, 'Unexpected gsplat JIT source'
    s=s.replace(a,b)
p.write_text(s)
PY
echo 'Installed isolated dependencies. First render compiles the CUDA extension.'
