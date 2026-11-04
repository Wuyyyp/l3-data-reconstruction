#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; keep source tooling Linux-only.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
install_dir=${VCG_INSTALL_DIR:-"$repo_root/tools/vcg"}
vcg_commit=658ba36d0a5666650da6e066b4794efc5a463407

if [[ ! -d "$install_dir/.git" ]]; then
  mkdir -p "$install_dir"
  git -C "$install_dir" init
  git -C "$install_dir" remote add origin https://github.com/cdcseacave/VCG.git
  git -C "$install_dir" fetch --depth 1 origin "$vcg_commit"
  git -C "$install_dir" checkout --detach FETCH_HEAD
fi

actual_commit=$(git -C "$install_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$vcg_commit" ]]; then
  echo "VCG source must be commit $vcg_commit, got $actual_commit" >&2
  exit 1
fi

header_path="$install_dir/vcg/complex/complex.h"
if [[ ! -f "$header_path" ]]; then
  echo "VCG header was not installed: $header_path" >&2
  exit 1
fi
