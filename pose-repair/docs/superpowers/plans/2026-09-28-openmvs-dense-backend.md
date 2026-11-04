# OpenMVS Dense Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenMVS v2.4.0 as a configuration-selected dense point-cloud backend after the existing absolute-pose-prior COLMAP bundle adjustment, then validate it on a bounded T3 run.

**Architecture:** Keep sparse reconstruction and BA in `colmap_reconstruction.py`, then dispatch the common `sparse_ba` model to either the existing COLMAP dense path or a focused OpenMVS adapter. The adapter uses COLMAP only for image undistortion, converts the workspace through `InterfaceCOLMAP`, applies native OpenMVS masks, runs `DensifyPointCloud`, and publishes the native PLY at the existing `dense/fused.ply` contract.

**Tech Stack:** Python 3.11, pytest, YAML, COLMAP 4.2.0, OpenMVS 2.4.0, CMake/Ninja, Ubuntu WSL.

---

## File Structure

- Create `src/dongfeng_registration/openmvs_backend.py`: OpenMVS command builders, mask naming, subprocess orchestration, timing, PLY validation, and report payload.
- Modify `src/dongfeng_registration/colmap_reconstruction.py`: configuration fields, backend validation, and dense-backend dispatch after BA.
- Create `tests/test_openmvs_backend.py`: adapter unit and orchestration tests.
- Modify `tests/test_colmap_reconstruction.py`: configuration default, parsing, invalid value, and dispatch contract tests.
- Create `scripts/build_openmvs.sh`: reproducible OpenMVS v2.4.0 headless build and install.
- Modify `scripts/build_colmap_absolute_pose_prior.sh`: explicit CPU/headless options and portable dependency diagnostics if required by the WSL build.
- Modify `.gitignore`: ignore generated OpenMVS binaries.
- Modify `configs/t3.yaml`: document the default COLMAP backend and OpenMVS executable/parameter defaults.
- Create `configs/t3_openmvs_smoke.yaml`: bounded Linux/WSL T3 validation profile.
- Modify `docs/colmap_reconstruction.md`: build, configuration, masks, outputs, and validation instructions.

### Task 1: Configuration Contract

**Files:**
- Modify: `tests/test_colmap_reconstruction.py`
- Modify: `src/dongfeng_registration/colmap_reconstruction.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests that assert the default is COLMAP, all OpenMVS values parse from YAML,
and an unknown backend raises `ValueError`:

```python
def test_colmap_config_defaults_to_colmap_mvs_backend(tmp_path: Path) -> None:
    config = load_colmap_config(_write_minimal_config(tmp_path))
    assert config.mvs_backend == "colmap"


def test_colmap_config_reads_openmvs_settings(tmp_path: Path) -> None:
    config = load_colmap_config(_write_openmvs_config(tmp_path))
    assert config.mvs_backend == "openmvs"
    assert config.openmvs_resolution_level == 1
    assert config.openmvs_max_resolution == 1280
    assert config.openmvs_min_resolution == 640
    assert config.openmvs_number_views == 8
    assert config.openmvs_number_views_fuse == 3
    assert config.openmvs_gpu_device == "0"


def test_colmap_config_rejects_unknown_mvs_backend(tmp_path: Path) -> None:
    path = _write_minimal_config(tmp_path, extra="mvs_backend: unknown")
    with pytest.raises(ValueError, match="mvs_backend"):
        load_colmap_config(path)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_colmap_reconstruction.py -q
```

Expected: failures because `ColmapReconstructionConfig` has no OpenMVS fields
and invalid values are not rejected.

- [ ] **Step 3: Implement configuration parsing**

Add typed fields for the backend, executable paths, and six stable dense
controls. Validate `mvs_backend` against `{"colmap", "openmvs"}` before
constructing the dataclass. Resolve executables through `_executable_path` so a
bare executable name can come from `PATH` and a relative path remains relative
to the selected YAML.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/test_colmap_reconstruction.py -q` and expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_colmap_reconstruction.py src/dongfeng_registration/colmap_reconstruction.py
git commit -m "feat: configure OpenMVS dense backend"
```

### Task 2: OpenMVS Commands And Mask Contract

**Files:**
- Create: `tests/test_openmvs_backend.py`
- Create: `src/dongfeng_registration/openmvs_backend.py`

- [ ] **Step 1: Write failing command tests**

Define tests for these public helpers:

```python
def test_interface_colmap_command_uses_undistorted_workspace() -> None:
    command = build_interface_colmap_command(
        executable=Path("InterfaceCOLMAP"),
        colmap_workspace=Path("dense/openmvs/colmap_workspace"),
        scene_path=Path("dense/openmvs/scene.mvs"),
    )
    assert command == [
        "InterfaceCOLMAP", "-i", "dense/openmvs/colmap_workspace",
        "-o", "dense/openmvs/scene.mvs", "--image-folder",
        "dense/openmvs/colmap_workspace/images",
    ]


def test_densify_command_applies_zero_label_mask() -> None:
    command = build_densify_point_cloud_command(settings, paths)
    assert command[command.index("--mask-path") + 1] == str(paths.mask_root)
    assert command[command.index("--ignore-mask-label") + 1] == "0"
    assert command[command.index("--resolution-level") + 1] == "1"
    assert command[command.index("--max-resolution") + 1] == "1280"
    assert command[command.index("--number-views-fuse") + 1] == "3"
```

Add a mask test that creates `camera/100.png`, prepares an undistorted
`camera/100.jpg`, and verifies output `camera/100.jpg.mask.png` preserves
binary values and exact image dimensions.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_openmvs_backend.py -q` and expect import failure because
the module does not exist.

- [ ] **Step 3: Implement focused helpers**

Create immutable settings/path dataclasses and these functions:

```python
def build_interface_colmap_command(...) -> list[str]: ...
def build_densify_point_cloud_command(...) -> list[str]: ...
def prepare_openmvs_masks(...) -> dict[str, int]: ...
def count_binary_ply_vertices(path: Path) -> int: ...
```

Use nearest-neighbor mask resizing. Require one source mask for every
undistorted image. Parse the PLY header through `element vertex N` without
loading point data into memory.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/test_openmvs_backend.py -q` and expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_openmvs_backend.py src/dongfeng_registration/openmvs_backend.py
git commit -m "feat: add OpenMVS command and mask adapter"
```

### Task 3: Backend Orchestration And Common Output

**Files:**
- Modify: `tests/test_openmvs_backend.py`
- Modify: `tests/test_colmap_reconstruction.py`
- Modify: `src/dongfeng_registration/openmvs_backend.py`
- Modify: `src/dongfeng_registration/colmap_reconstruction.py`

- [ ] **Step 1: Write failing orchestration tests**

Patch `subprocess.run` with a recorder that materializes expected scene and PLY
outputs. Assert OpenMVS mode calls, in order:

```text
colmap image_undistorter
InterfaceCOLMAP
DensifyPointCloud
```

Assert no command contains `patch_match_stereo` or `stereo_fusion`, the native
PLY is copied to `dense/fused.ply`, and the report includes backend, timings,
file size, and positive vertex count.

Also test missing scene and zero-vertex PLY failures with focused error text.

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_openmvs_backend.py tests/test_colmap_reconstruction.py -q
```

Expected: failures because orchestration and dispatch do not exist.

- [ ] **Step 3: Implement `run_openmvs_dense`**

The function resets `dense/openmvs`, times each command with
`time.perf_counter`, validates each expected artifact immediately, prepares
masks after undistortion, copies `scene_dense.ply` atomically to
`dense/fused.ply`, and returns:

```python
{
    "mvs_backend": "openmvs",
    "fused_ply": str(fused_ply),
    "dense_point_count": vertex_count,
    "dense_file_size_bytes": fused_ply.stat().st_size,
    "dense_stage_seconds": {"undistort": ..., "convert": ..., "densify": ...},
    "openmvs": {"scene": ..., "dense_scene": ..., "native_ply": ..., "masks": ...},
}
```

- [ ] **Step 4: Dispatch after BA**

Extract the existing COLMAP dense block into `_run_colmap_dense(...)`, then in
`_run_ba_and_dense(...)` dispatch only after the common BA report is complete.
Merge the selected backend report into the common report. Preserve the existing
COLMAP command sequence and output path exactly.

- [ ] **Step 5: Verify GREEN and regression**

Run:

```powershell
pytest tests/test_openmvs_backend.py tests/test_colmap_reconstruction.py -q
```

Expected: all tests pass and existing COLMAP tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add tests/test_openmvs_backend.py tests/test_colmap_reconstruction.py src/dongfeng_registration/openmvs_backend.py src/dongfeng_registration/colmap_reconstruction.py
git commit -m "feat: dispatch COLMAP and OpenMVS dense backends"
```

### Task 4: Runtime Validation

**Files:**
- Modify: `tests/test_colmap_reconstruction.py`
- Modify: `src/dongfeng_registration/colmap_reconstruction.py`

- [ ] **Step 1: Write failing executable-validation tests**

Test that OpenMVS executables are not required for `run_dense: false` or the
COLMAP backend, but are required for `run_dense: true` plus
`mvs_backend: openmvs`. Error messages must name the missing executable field.

- [ ] **Step 2: Verify RED**

Run the individual tests and confirm failure because only COLMAP and custom BA
are currently validated.

- [ ] **Step 3: Add conditional validation**

Validate OpenMVS tools immediately after COLMAP/custom-BA checks, before any
output directory is reset. Do not probe executables by running them during
normal configuration loading.

- [ ] **Step 4: Verify GREEN and full unit suite**

Run:

```powershell
pytest tests/test_openmvs_backend.py tests/test_colmap_reconstruction.py tests/test_cli.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_colmap_reconstruction.py src/dongfeng_registration/colmap_reconstruction.py
git commit -m "test: validate OpenMVS runtime requirements"
```

### Task 5: Reproducible Build Scripts

**Files:**
- Create: `scripts/build_openmvs.sh`
- Modify: `scripts/build_colmap_absolute_pose_prior.sh`
- Modify: `.gitignore`
- Test: `tests/test_environment_scripts.py`

- [ ] **Step 1: Write failing script contract tests**

Read both scripts as text and assert fixed tags/commits, headless options,
project-local install paths, and post-build help checks. Assert `.gitignore`
contains `/tools/openmvs/`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_environment_scripts.py -q`; expect missing OpenMVS script
and ignore rule failures.

- [ ] **Step 3: Add the OpenMVS build script**

The script must:

1. clone OpenMVS tag `v2.4.0` with submodules into `.tools/openmvs-2.4.0-source`;
2. configure Release/headless with Ninja and a project-local install prefix;
3. build only `InterfaceCOLMAP` and `DensifyPointCloud` plus required libraries;
4. install/copy required binaries and shared libraries to `tools/openmvs`;
5. set an origin-relative RPATH or wrapper-free library layout;
6. run both binaries with `--help` and require successful option output.

Keep source/build paths overrideable through environment variables. Keep the
COLMAP script pinned to commit `be5e29168d4aff238409d60424812df66aac919f` and
make CPU/headless flags explicit.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run:

```powershell
pytest tests/test_environment_scripts.py -q
wsl.exe -d Ubuntu-20.04 -- bash -n /mnt/d/DongFeng/Code/scripts/build_openmvs.sh
wsl.exe -d Ubuntu-20.04 -- bash -n /mnt/d/DongFeng/Code/scripts/build_colmap_absolute_pose_prior.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_openmvs.sh scripts/build_colmap_absolute_pose_prior.sh tests/test_environment_scripts.py .gitignore
git commit -m "build: add reproducible OpenMVS toolchain"
```

### Task 6: Config Profiles And Documentation

**Files:**
- Modify: `configs/t3.yaml`
- Create: `configs/t3_openmvs_smoke.yaml`
- Modify: `docs/colmap_reconstruction.md`
- Modify: `tests/test_colmap_config_profiles.py`

- [ ] **Step 1: Write failing profile tests**

Assert `configs/t3.yaml` explicitly uses `mvs_backend: colmap`. Assert the smoke
profile uses `openmvs`, a positive bounded `frame_count`, project-local tool
paths, and OpenMVS mask-compatible settings.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_colmap_config_profiles.py -q` and expect missing-key and
missing-profile failures.

- [ ] **Step 3: Add profiles and documentation**

Set explicit backend/default values in T3. Add a smoke profile that uses Linux
paths but keeps data and output roots overridable by copying the file for local
execution. Document both pipelines, explain why `image_undistorter` remains,
list mask polarity and native/common outputs, and provide build/run commands.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/test_colmap_config_profiles.py -q`.

- [ ] **Step 5: Commit**

```bash
git add configs/t3.yaml configs/t3_openmvs_smoke.yaml docs/colmap_reconstruction.md tests/test_colmap_config_profiles.py
git commit -m "docs: add OpenMVS reconstruction profile"
```

### Task 7: Unit And Regression Verification

**Files:**
- No production file changes unless a test exposes a defect.

- [ ] **Step 1: Run focused tests**

```powershell
pytest tests/test_openmvs_backend.py tests/test_colmap_reconstruction.py tests/test_colmap_config_profiles.py tests/test_environment_scripts.py tests/test_cli.py -q
```

- [ ] **Step 2: Run the full suite**

```powershell
pytest -q
```

Record pre-existing unrelated failures separately. Any new failure caused by
this feature must be fixed before proceeding.

- [ ] **Step 3: Verify repository hygiene**

```powershell
git diff --check
git status --short
```

Expect no generated binary or WSL build directory to be tracked.

### Task 8: WSL Tool Builds

**Files:**
- Generated and ignored: `.tools/`, `tools/colmap_pose_prior/bin/`, `tools/openmvs/`

- [ ] **Step 1: Complete Ubuntu dependencies**

Install the documented compiler and library packages inside Ubuntu WSL. Correct
the WSL DNS only if the current resolver still fails, retaining a backup of the
original resolver file.

- [ ] **Step 2: Build and validate custom COLMAP**

```bash
cd /mnt/d/DongFeng/Code
bash scripts/build_colmap_absolute_pose_prior.sh
tools/colmap_pose_prior/bin/colmap help | rg absolute_pose_prior_bundle_adjuster
```

Expected: the custom command name is printed.

- [ ] **Step 3: Build and validate OpenMVS**

```bash
cd /mnt/d/DongFeng/Code
bash scripts/build_openmvs.sh
tools/openmvs/bin/InterfaceCOLMAP --help
tools/openmvs/bin/DensifyPointCloud --help
```

Expected: both commands load all shared libraries and print usage.

### Task 9: T3 Small-Batch Validation

**Files:**
- Local-only config derived from: `configs/t3_openmvs_smoke.yaml`
- Generated output: `D:/DongFeng/outputs/T3_openmvs_smoke/`

- [ ] **Step 1: Resolve WSL-accessible inputs**

Point the local smoke copy at existing T3 canonical data, semantic cache, and
delivery output through `/mnt/d/DongFeng/...` paths. Do not modify the committed
portable profile with machine-only paths.

- [ ] **Step 2: Run the complete bounded reconstruction**

```bash
cd /mnt/d/DongFeng/Code
python -m dongfeng_registration.cli reconstruct-colmap \
  --config /mnt/d/DongFeng/outputs/T3_openmvs_smoke/t3_openmvs_smoke.local.yaml
```

The run must include masked matching, triangulation, absolute-pose-prior BA,
COLMAP-to-OpenMVS conversion, and OpenMVS densification.

- [ ] **Step 3: Validate artifacts and command log**

Check the report and native files against all seven acceptance requirements in
the design. Parse `dense/fused.ply` vertex count and verify no COLMAP PatchMatch
or fusion command was invoked.

- [ ] **Step 4: Summarize results**

Report selected images, registered BA images, sparse points, dense points,
native/common PLY sizes, stage timings, GPU/CPU mode, and any quality limitation.
Do not claim geometric accuracy without a reference surface.

### Task 10: Completion Review

**Files:**
- Review all files changed since commit `5ba1938`.

- [ ] **Step 1: Audit every design requirement**

Map configuration, backend replacement, masks, common output, build tools,
tests, documentation, and real-data evidence to concrete files or command
outputs.

- [ ] **Step 2: Run final verification**

Re-run focused tests, `git diff --check`, custom COLMAP help, OpenMVS help, and
the PLY/report validator against the smoke output.

- [ ] **Step 3: Commit any final fixes**

Use a focused commit message and leave generated binaries/data ignored.
