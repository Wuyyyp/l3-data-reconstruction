#!/usr/bin/env bash
set -euo pipefail

# WSL appends Windows paths by default; keep build tools and package discovery Linux-only.
export PATH=${LINUX_BUILD_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=1.84.0
archive_name=boost_1_84_0.tar.bz2
archive=${BOOST_ARCHIVE:-"$repo_root/.tools/$archive_name"}
partial_archive="${archive}.partial"
source_dir=${BOOST_SOURCE_DIR:-"$repo_root/.tools/boost-1.84.0-source"}
install_dir=${BOOST_INSTALL_DIR:-"$repo_root/tools/boost"}
sha256=cc4b893acf645c9d4b698e9a0f08ca8846aa5d6c68275c14c3e7949c24109454
zstd_include_dir=${ZSTD_INCLUDE_DIR:-/usr/include}
zstd_lib_dir=${ZSTD_LIB_DIR:-"/usr/lib/$(gcc -print-multiarch)"}
boost_iostreams_lib="$install_dir/lib/libboost_iostreams.so"

if [[ -f "$install_dir/include/boost/version.hpp" ]] && \
   [[ -f "$install_dir/lib/libboost_program_options.so" ]] && \
   [[ -f "$boost_iostreams_lib" ]] && \
   nm -D --demangle "$boost_iostreams_lib" | grep 'zstd_base' >/dev/null; then
  exit 0
fi

if [[ ! -f "$zstd_include_dir/zstd.h" ]] || \
   [[ ! -f "$zstd_lib_dir/libzstd.so" ]]; then
  echo "Zstandard development files are required; install libzstd-dev" >&2
  exit 1
fi

mkdir -p "$(dirname "$archive")"
if [[ ! -f "$archive" ]] || \
   ! printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status; then
  if [[ -f "$archive" && ! -f "$partial_archive" ]]; then
    mv "$archive" "$partial_archive"
  fi
  downloaded=false
  for attempt in $(seq 1 10); do
    if curl --fail --location --continue-at - \
      "https://archives.boost.org/release/$version/source/$archive_name" \
      --output "$partial_archive"; then
      downloaded=true
      break
    fi
    echo "Boost download attempt $attempt failed; resuming in 2 seconds" >&2
    sleep 2
  done
  if [[ "$downloaded" != true ]]; then
    echo "Boost archive download failed after 10 attempts" >&2
    exit 1
  fi
  printf '%s  %s\n' "$sha256" "$partial_archive" | sha256sum --check -
  mv "$partial_archive" "$archive"
fi
printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check -

if [[ ! -f "$source_dir/boost/version.hpp" ]]; then
  mkdir -p "$source_dir"
  tar -xjf "$archive" --strip-components=1 -C "$source_dir"
fi

cd "$source_dir"
./bootstrap.sh \
  --prefix="$install_dir" \
  --with-libraries=filesystem,iostreams,program_options,serialization,system,thread
./b2 \
  --prefix="$install_dir" \
  variant=release \
  link=shared \
  runtime-link=shared \
  threading=multi \
  -sZSTD_INCLUDE="$zstd_include_dir" \
  -sZSTD_LIBPATH="$zstd_lib_dir" \
  -a \
  install \
  -j "${BOOST_BUILD_JOBS:-4}"

test -f "$install_dir/include/boost/version.hpp"
test -f "$install_dir/lib/libboost_program_options.so"
nm -D --demangle "$boost_iostreams_lib" | grep 'zstd_base' >/dev/null
