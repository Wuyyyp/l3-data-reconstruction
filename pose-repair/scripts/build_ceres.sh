#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; CMake can mistake Windows packages for
# Linux dependencies when those paths are inherited.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${CERES_SOURCE_DIR:-"$repo_root/.tools/ceres-2.2.0-source"}
build_dir=${CERES_BUILD_DIR:-"$repo_root/.tools/ceres-2.2.0-build"}
install_dir=${CERES_INSTALL_DIR:-"$repo_root/tools/ceres"}
eigen_install_dir=${EIGEN_INSTALL_DIR:-"$repo_root/tools/eigen"}
eigen3_dir="$eigen_install_dir/share/eigen3/cmake"
ceres_tag=2.2.0

bash "$repo_root/scripts/build_eigen.sh"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch "$ceres_tag" --depth 1 \
    https://github.com/ceres-solver/ceres-solver.git "$source_dir"
fi

actual_tag=$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)
if [[ "$actual_tag" != "$ceres_tag" ]]; then
  echo "Ceres source must be tag $ceres_tag, got ${actual_tag:-untagged}" >&2
  exit 1
fi

cmake -S "$source_dir" -B "$build_dir" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
  -DEigen3_DIR="$eigen3_dir" \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DOCUMENTATION=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_CUDA=OFF \
  -DMINIGLOG=OFF \
  -DSUITESPARSE=ON \
  -DLAPACK=ON
cmake --build "$build_dir" --target install -j "${CERES_BUILD_JOBS:-4}"

config_path="$install_dir/lib/cmake/Ceres/CeresConfig.cmake"
if [[ ! -f "$config_path" ]]; then
  echo "CeresConfig.cmake was not installed: $config_path" >&2
  exit 1
fi
