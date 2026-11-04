# OpenMVS CUDA Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate an isolated CUDA-enabled OpenMVS 2.4.0 installation for RTX 4090 dense reconstruction.

**Architecture:** Extend the existing build script with environment-controlled CUDA flags while retaining CPU defaults. Install CUDA binaries separately, select them through the existing OpenMVS executable configuration, validate on a bounded dataset, then restart the full dense stage from the reusable COLMAP/OpenMVS workspace.

**Tech Stack:** Bash, CMake/Ninja, CUDA Toolkit, OpenMVS 2.4.0, pytest, WSL2.

---

### Task 1: Make the OpenMVS build CUDA-configurable

**Files:**
- Modify: `tests/test_environment_scripts.py`
- Modify: `scripts/build_openmvs.sh`

- [ ] **Step 1: Write the failing script test**

Assert that `build_openmvs.sh` reads `OPENMVS_ENABLE_CUDA`, conditionally sets
`OpenMVS_USE_CUDA`, forwards `OPENMVS_CUDA_ARCHITECTURES`, and rejects CUDA
builds without `nvcc`.

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
.\.conda\envs\dongfeng-registration\python.exe -m pytest tests/test_environment_scripts.py -q
```

Expected: failure because the script still hard-codes `OpenMVS_USE_CUDA=OFF`.

- [ ] **Step 3: Implement the build flags**

Add `OPENMVS_ENABLE_CUDA` and `OPENMVS_CUDA_ARCHITECTURES`, validate their
values, append the corresponding CMake arguments, and keep `OFF` as the default.

- [ ] **Step 4: Verify the tests pass**

Run the same pytest command and expect all tests to pass.

### Task 2: Add a CUDA validation profile

**Files:**
- Create: `configs/t3_openmvs_cuda_smoke.yaml`
- Modify: `tests/test_colmap_config_profiles.py`
- Modify: `docs/colmap_reconstruction.md`

- [ ] **Step 1: Write a failing config test**

Require a bounded five-frame profile using `tools/openmvs-cuda`, backend
`openmvs`, and GPU device `0`.

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
.\.conda\envs\dongfeng-registration\python.exe -m pytest tests/test_colmap_config_profiles.py -q
```

Expected: failure because the CUDA profile does not exist.

- [ ] **Step 3: Add the profile and documentation**

Copy the tested T3 smoke settings, change only the OpenMVS executable paths,
output directory, and GPU device. Document the WSL CUDA toolkit prerequisite and
the separate CPU/CUDA build commands.

- [ ] **Step 4: Verify profile tests pass**

Run the same pytest command and expect all tests to pass.

### Task 3: Build and validate the CUDA runtime

**Files:**
- Generated: `tools/openmvs-cuda/`
- Generated: `.tools/openmvs-2.4.0-build-cuda/`

- [ ] **Step 1: Install a WSL CUDA toolkit supporting compute capability 8.9**

Install toolkit-only packages; do not install or replace the Windows/WSL GPU
driver.

- [ ] **Step 2: Build into the isolated CUDA paths**

Run:

```bash
OPENMVS_ENABLE_CUDA=ON \
OPENMVS_CUDA_ARCHITECTURES=89 \
OPENMVS_BUILD_DIR=/mnt/d/DongFeng/Code/.tools/openmvs-2.4.0-build-cuda \
OPENMVS_INSTALL_DIR=/mnt/d/DongFeng/Code/tools/openmvs-cuda \
bash scripts/build_openmvs.sh
```

- [ ] **Step 3: Verify the CUDA executable**

Confirm `DensifyPointCloud --help` contains `gpu-device` and run the bounded
CUDA profile while observing GPU utilization with `nvidia-smi`.

### Task 4: Restart the full dense reconstruction

**Files:**
- Reuse: `D:/DongFeng/outputs/T3_openmvs_full/scene.mvs`
- Reuse: `D:/DongFeng/outputs/T3_openmvs_full/colmap_workspace/`
- Generate: `D:/DongFeng/outputs/T3_openmvs_full/scene_dense.mvs`
- Generate: `D:/DongFeng/outputs/T3_openmvs_full/scene_dense.ply`
- Generate: `D:/DongFeng/outputs/T3_openmvs_full/fused.ply`

- [ ] **Step 1: Remove only aborted CPU dense artifacts**

Delete generated `.dmap`, `scene_dense.mvs`, and `scene_dense.ply` files while
preserving `scene.mvs`, the COLMAP workspace, masks, and sparse triangulation.

- [ ] **Step 2: Run CUDA DensifyPointCloud**

Use GPU 0 and the established resolution/view/fusion parameters.

- [ ] **Step 3: Publish the CloudCompare-compatible PLY**

Run the project publisher to remove OpenMVS-only per-vertex list fields while
preserving XYZ, RGB, and normals.

- [ ] **Step 4: Verify outputs**

Check the process exit code, point count, PLY header, file size, and direct
CloudCompare command-line loading. Run the focused OpenMVS and configuration
test suites before reporting completion.
