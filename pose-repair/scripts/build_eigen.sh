#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; CMake can mistake Windows vcpkg packages
# for Linux dependencies when those paths are inherited.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${EIGEN_SOURCE_DIR:-"$repo_root/.tools/eigen-3.4.0-source"}
build_dir=${EIGEN_BUILD_DIR:-"$repo_root/.tools/eigen-3.4.0-build"}
install_dir=${EIGEN_INSTALL_DIR:-"$repo_root/tools/eigen"}
eigen_tag=3.4.0

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch "$eigen_tag" --depth 1 \
    https://gitlab.com/libeigen/eigen.git "$source_dir"
fi

actual_tag=$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)
if [[ "$actual_tag" != "$eigen_tag" ]]; then
  echo "Eigen source must be tag $eigen_tag, got ${actual_tag:-untagged}" >&2
  exit 1
fi

cmake -S "$source_dir" -B "$build_dir" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_TESTING=OFF
cmake --build "$build_dir" --target install -j "${EIGEN_BUILD_JOBS:-4}"

config_path="$install_dir/share/eigen3/cmake/Eigen3Config.cmake"
if [[ ! -f "$config_path" ]]; then
  echo "Eigen3Config.cmake was not installed: $config_path" >&2
  exit 1
fi
