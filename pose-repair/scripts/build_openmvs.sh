#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; CMake can mistake Windows vcpkg packages
# for Linux dependencies when those paths are inherited.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${OPENMVS_SOURCE_DIR:-"$repo_root/.tools/openmvs-2.4.0-source"}
build_dir=${OPENMVS_BUILD_DIR:-"$repo_root/.tools/openmvs-2.4.0-build"}
install_dir=${OPENMVS_INSTALL_DIR:-"$repo_root/tools/openmvs"}
enable_cuda=${OPENMVS_ENABLE_CUDA:-OFF}
cuda_architectures=${OPENMVS_CUDA_ARCHITECTURES:-native}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
cuda_compiler=${CUDACXX:-}
case "$enable_cuda" in
  ON|OFF) ;;
  *)
    echo "OPENMVS_ENABLE_CUDA must be ON or OFF, got: $enable_cuda" >&2
    exit 1
    ;;
esac
if [[ "$enable_cuda" == "ON" ]]; then
  if [[ -d "$cuda_home/bin" ]]; then
    export PATH="$cuda_home/bin:$PATH"
  fi
  if [[ -z "$cuda_compiler" ]]; then
    cuda_compiler=$(command -v nvcc || true)
  fi
  if [[ -z "$cuda_compiler" ]]; then
    echo "CUDA build requested but nvcc was not found; set CUDA_HOME or CUDACXX" >&2
    exit 1
  fi
fi
openmvs_tag=v2.4.0
eigen_install_dir=${EIGEN_INSTALL_DIR:-"$repo_root/tools/eigen"}
eigen3_dir="$eigen_install_dir/share/eigen3/cmake"
cgal_install_dir=${CGAL_INSTALL_DIR:-"$repo_root/tools/cgal"}
boost_install_dir=${BOOST_INSTALL_DIR:-"$repo_root/tools/boost"}
nanoflann_install_dir=${NANOFLANN_INSTALL_DIR:-"$repo_root/tools/nanoflann"}
vcg_install_dir=${VCG_INSTALL_DIR:-"$repo_root/tools/vcg"}
jpegxl_patch_file="$repo_root/third_party/openmvs_optional_jpegxl_2.4.0.patch"
ubuntu_compat_patch_file="$repo_root/third_party/openmvs_ubuntu_20_04_compat_2.4.0.patch"

bash "$repo_root/scripts/build_eigen.sh"
bash "$repo_root/scripts/build_boost.sh"
bash "$repo_root/scripts/build_cgal.sh"
bash "$repo_root/scripts/build_nanoflann.sh"
bash "$repo_root/scripts/build_vcg.sh"

nanoflann_config=$(find "$nanoflann_install_dir" -type f -name nanoflannConfig.cmake -print -quit)
if [[ -z "$nanoflann_config" ]]; then
  echo "nanoflannConfig.cmake was not installed below: $nanoflann_install_dir" >&2
  exit 1
fi
nanoflann_dir=$(dirname "$nanoflann_config")
cgal_config=$(find "$cgal_install_dir" -type f -name CGALConfig.cmake -print -quit)
if [[ -z "$cgal_config" ]]; then
  echo "CGALConfig.cmake was not installed below: $cgal_install_dir" >&2
  exit 1
fi
cgal_dir=$(dirname "$cgal_config")

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch "$openmvs_tag" --depth 1 --recurse-submodules \
    https://github.com/cdcseacave/openMVS.git "$source_dir"
fi

actual_tag=$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)
if [[ "$actual_tag" != "$openmvs_tag" ]]; then
  echo "OpenMVS source must be tag $openmvs_tag, got ${actual_tag:-untagged}" >&2
  exit 1
fi
git -C "$source_dir" submodule update --init --recursive

if git -C "$source_dir" apply --reverse --check "$jpegxl_patch_file" >/dev/null 2>&1; then
  : # Patch is already applied.
elif git -C "$source_dir" apply --check "$jpegxl_patch_file"; then
  git -C "$source_dir" apply "$jpegxl_patch_file"
else
  echo "OpenMVS JPEG XL compatibility patch does not apply cleanly" >&2
  exit 1
fi

if git -C "$source_dir" apply --reverse --check "$ubuntu_compat_patch_file" >/dev/null 2>&1; then
  : # Patch is already applied.
elif git -C "$source_dir" apply --check "$ubuntu_compat_patch_file"; then
  git -C "$source_dir" apply "$ubuntu_compat_patch_file"
else
  echo "OpenMVS Ubuntu 20.04 compatibility patch does not apply cleanly" >&2
  exit 1
fi

cmake_args=(
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$install_dir"
  -DCMAKE_BUILD_RPATH='$ORIGIN/../lib'
  -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
  -DEigen3_DIR="$eigen3_dir"
  -DCMAKE_PREFIX_PATH="$boost_install_dir"
  -DBOOST_ROOT="$boost_install_dir"
  -DBoost_ROOT="$boost_install_dir"
  -DBoost_NO_SYSTEM_PATHS=ON
  -DCGAL_DIR="$cgal_dir"
  -Dnanoflann_DIR="$nanoflann_dir"
  -DVCG_ROOT="$vcg_install_dir"
  -DOpenMVS_USE_CUDA="$enable_cuda"
  -DOpenMVS_BUILD_VIEWER=OFF
  -DOpenMVS_ENABLE_TESTS=OFF
  -DOpenMVS_USE_PYTHON=OFF
)
if [[ "$enable_cuda" == "ON" ]]; then
  cmake_args+=(
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_architectures"
    -DCMAKE_CUDA_COMPILER="$cuda_compiler"
  )
fi
if [[ -n "${VCPKG_ROOT:-}" ]]; then
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
fi

cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" --target InterfaceCOLMAP DensifyPointCloud \
  -j "${OPENMVS_BUILD_JOBS:-4}"

mkdir -p "$install_dir/bin" "$install_dir/lib"
for executable in InterfaceCOLMAP DensifyPointCloud; do
  built_path=$(find "$build_dir/bin" -maxdepth 2 -type f -name "$executable" -print -quit)
  if [[ -z "$built_path" ]]; then
    echo "OpenMVS executable was not built: $executable" >&2
    exit 1
  fi
  install -m 0755 "$built_path" "$install_dir/bin/$executable"
done

while IFS= read -r -d '' library; do
  cp -a "$library" "$install_dir/lib/"
done < <(find "$build_dir/lib" -maxdepth 2 \( -type f -o -type l \) -name '*.so*' -print0)

interface_help=$("$install_dir/bin/InterfaceCOLMAP" --help 2>&1 || true)
densify_help=$("$install_dir/bin/DensifyPointCloud" --help 2>&1 || true)
grep -E 'input-file|input path|COLMAP' <<<"$interface_help" >/dev/null
grep -E 'resolution-level|number-views' <<<"$densify_help" >/dev/null
