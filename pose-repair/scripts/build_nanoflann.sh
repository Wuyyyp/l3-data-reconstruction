#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; keep CMake package discovery Linux-only.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${NANOFLANN_SOURCE_DIR:-"$repo_root/.tools/nanoflann-1.6.3-source"}
build_dir=${NANOFLANN_BUILD_DIR:-"$repo_root/.tools/nanoflann-1.6.3-build"}
install_dir=${NANOFLANN_INSTALL_DIR:-"$repo_root/tools/nanoflann"}
nanoflann_tag=v1.6.3

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch "$nanoflann_tag" --depth 1 \
    https://github.com/jlblancoc/nanoflann.git "$source_dir"
fi

actual_tag=$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)
if [[ "$actual_tag" != "$nanoflann_tag" ]]; then
  echo "nanoflann source must be tag $nanoflann_tag, got ${actual_tag:-untagged}" >&2
  exit 1
fi

cmake -S "$source_dir" -B "$build_dir" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DNANOFLANN_BUILD_EXAMPLES=OFF \
  -DNANOFLANN_BUILD_TESTS=OFF
cmake --build "$build_dir" --target install -j "${NANOFLANN_BUILD_JOBS:-4}"

config_path=$(find "$install_dir" -type f -name nanoflannConfig.cmake -print -quit)
if [[ -z "$config_path" ]]; then
  echo "nanoflannConfig.cmake was not installed below: $install_dir" >&2
  exit 1
fi
