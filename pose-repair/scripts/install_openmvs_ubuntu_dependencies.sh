#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root, for example: sudo bash scripts/install_openmvs_ubuntu_dependencies.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config \
  libboost-iostreams-dev \
  libboost-program-options-dev \
  libboost-serialization-dev \
  libboost-system-dev \
  libcgal-dev \
  libgmp-dev \
  libjpeg-dev \
  libmpfr-dev \
  libopencv-dev \
  libpng-dev \
  libtiff-dev \
  libzstd-dev \
  zlib1g-dev
