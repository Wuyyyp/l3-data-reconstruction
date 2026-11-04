#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; CMake can mistake Windows vcpkg packages
# for Linux dependencies when those paths are inherited.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${COLMAP_SOURCE_DIR:-"$repo_root/.tools/colmap-4.2.0-source"}
build_dir=${COLMAP_BUILD_DIR:-"$repo_root/.tools/colmap-4.2.0-build"}
install_dir=${COLMAP_ABSOLUTE_POSE_PRIOR_DIR:-"$repo_root/tools/colmap_pose_prior"}
patch_file="$repo_root/third_party/colmap_absolute_pose_prior_4.2.0.patch"
oiio_compat_patch_file="$repo_root/third_party/colmap_openimageio_2_1_compat_4.2.0.patch"
eigen_install_dir=${EIGEN_INSTALL_DIR:-"$repo_root/tools/eigen"}
eigen3_dir="$eigen_install_dir/share/eigen3/cmake"
ceres_install_dir=${CERES_INSTALL_DIR:-"$repo_root/tools/ceres"}
ceres_dir="$ceres_install_dir/lib/cmake/Ceres"

bash "$repo_root/scripts/build_eigen.sh"
bash "$repo_root/scripts/build_ceres.sh"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch 4.2.0 --depth 1 https://github.com/colmap/colmap.git "$source_dir"
fi

expected_commit=be5e29168d4aff238409d60424812df66aac919f
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "COLMAP source must be 4.2.0 commit $expected_commit, got $actual_commit" >&2
  exit 1
fi

if ! rg -q 'absolute_pose_prior_bundle_adjuster' "$source_dir/src/colmap/exe/colmap.cc"; then
  git -C "$source_dir" apply "$patch_file"
fi
if ! rg -q '#define OIIO_MAKE_VERSION' "$source_dir/src/colmap/util/oiio_utils.h"; then
  git -C "$source_dir" apply "$oiio_compat_patch_file"
fi

# Ubuntu 20.04 ships OpenImageIO without the CMake config required by COLMAP 4.2.
install -m 0644 "$repo_root/cmake/FindOpenImageIO.cmake" "$source_dir/cmake/FindOpenImageIO.cmake"

cmake -S "$source_dir" -B "$build_dir" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
  -DEigen3_DIR="$eigen3_dir" \
  -DCeres_DIR="$ceres_dir" \
  -DCUDA_ENABLED=OFF \
  -DGUI_ENABLED=OFF \
  -DCGAL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF
cmake --build "$build_dir" --target colmap -j "${COLMAP_BUILD_JOBS:-4}"

mkdir -p "$install_dir/bin"
install -m 0755 "$build_dir/src/colmap/exe/colmap" "$install_dir/bin/colmap"
"$install_dir/bin/colmap" help | rg 'absolute_pose_prior_bundle_adjuster'
