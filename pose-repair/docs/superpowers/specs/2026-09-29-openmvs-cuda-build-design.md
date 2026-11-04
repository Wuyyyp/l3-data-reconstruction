# OpenMVS CUDA Build Design

## Goal

Provide a project-local CUDA build of OpenMVS 2.4.0 for RTX 4090 dense
reconstruction while preserving the existing CPU build as a fallback.

## Build Layout

- `tools/openmvs/` remains the CPU installation.
- `tools/openmvs-cuda/` is the CUDA installation.
- `OPENMVS_ENABLE_CUDA=ON` selects CUDA; the default remains `OFF` so existing
  deployments without a CUDA toolkit continue to build.
- `OPENMVS_CUDA_ARCHITECTURES=89` targets the RTX 4090. The value is forwarded
  to `CMAKE_CUDA_ARCHITECTURES` only for CUDA builds.
- `OPENMVS_INSTALL_DIR` and `OPENMVS_BUILD_DIR` keep CPU and CUDA artifacts
  isolated.

## Runtime Configuration

CUDA profiles point `openmvs_interface_colmap_executable` and
`openmvs_densify_point_cloud_executable` at `tools/openmvs-cuda/bin` and set
`openmvs_gpu_device: "0"`. CPU profiles omit the device option.

## Verification

1. Script tests verify the configurable CMake flags and separate install path.
2. Configure and build under WSL with a CUDA toolkit that supports `sm_89`.
3. Confirm CUDA `DensifyPointCloud --help` exposes `--gpu-device`.
4. Run a bounded five-frame reconstruction and observe the process on GPU 0.
5. Remove depth-map artifacts from the interrupted CPU run and restart the
   4,186-image dense stage using the existing scene, masks, and sparse points.

## Failure Handling

The CUDA build fails early when `nvcc` is missing. The CPU binaries are never
overwritten, so the existing CPU pipeline remains available.
