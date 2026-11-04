#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; keep CMake package discovery Linux-only.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${CGAL_SOURCE_DIR:-"$repo_root/.tools/cgal-6.0.1-source"}
build_dir=${CGAL_BUILD_DIR:-"$repo_root/.tools/cgal-6.0.1-build"}
install_dir=${CGAL_INSTALL_DIR:-"$repo_root/tools/cgal"}
boost_install_dir=${BOOST_INSTALL_DIR:-"$repo_root/tools/boost"}
cgal_tag=v6.0.1

bash "$repo_root/scripts/build_boost.sh"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --branch "$cgal_tag" --depth 1 \
    https://github.com/CGAL/cgal.git "$source_dir"
fi

actual_tag=$(git -C "$source_dir" describe --tags --exact-match 2>/dev/null || true)
if [[ "$actual_tag" != "$cgal_tag" ]]; then
  echo "CGAL source must be tag $cgal_tag, got ${actual_tag:-untagged}" >&2
  exit 1
fi

cmake -S "$source_dir" -B "$build_dir" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_PREFIX_PATH="$boost_install_dir" \
  -DBOOST_ROOT="$boost_install_dir" \
  -DBoost_ROOT="$boost_install_dir" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DCGAL_HEADER_ONLY=ON \
  -DBUILD_TESTING=OFF
cmake --build "$build_dir" --target install -j "${CGAL_BUILD_JOBS:-4}"

config_path=$(find "$install_dir" -type f -name CGALConfig.cmake -print -quit)
if [[ -z "$config_path" ]]; then
  echo "CGALConfig.cmake was not installed below: $install_dir" >&2
  exit 1
fi
