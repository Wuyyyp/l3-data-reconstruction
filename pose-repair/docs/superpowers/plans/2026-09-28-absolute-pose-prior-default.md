# Absolute Pose Prior Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every shipped configuration and the configuration loader default to enabled absolute-pose-prior COLMAP bundle adjustment.

**Architecture:** Keep the existing standard and absolute-prior BA command implementations. Change only the loader fallback and explicit shipped configuration values, with a repository-wide YAML contract test preventing future profiles from silently reverting to standard or disabled BA.

**Tech Stack:** Python, pytest, PyYAML, COLMAP configuration YAML

---

### Task 1: Lock the Default Loader Behavior

**Files:**
- Modify: `tests/test_colmap_reconstruction.py`
- Modify: `src/dongfeng_registration/colmap_reconstruction.py`

- [x] **Step 1: Write the failing test**

Add a test that loads a minimal COLMAP configuration without `ba_mode` and asserts:

```python
assert config.ba_mode == "absolute_pose_prior"
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```powershell
.conda\envs\dongfeng-registration\python.exe -m pytest -q tests/test_colmap_reconstruction.py -k defaults_to_absolute_pose_prior
```

Expected: failure because the current fallback is `standard`.

- [x] **Step 3: Change the loader fallback**

Change:

```python
payload.get("ba_mode", "standard")
```

to:

```python
payload.get("ba_mode", "absolute_pose_prior")
```

- [x] **Step 4: Run the test to verify it passes**

Run the same targeted pytest command and expect one passing test.

### Task 2: Enforce the Shipped Configuration Contract

**Files:**
- Create: `tests/test_colmap_config_profiles.py`
- Modify: `configs/*.yaml`

- [x] **Step 1: Write the failing repository-wide profile test**

Load every `configs/*.yaml` file with PyYAML and assert:

```python
assert colmap["run_bundle_adjustment"] is True
assert colmap["ba_mode"] == "absolute_pose_prior"
assert colmap["ba_position_prior_sigma_m"] == 0.1
assert colmap["ba_rotation_prior_sigma_deg"] == 0.5
```

For filenames containing `windows`, assert the executable is:

```text
D:/DongFeng/Code/tools/colmap_pose_prior/bin/colmap.exe
```

For all other profiles, assert:

```text
../tools/colmap_pose_prior/bin/colmap
```

- [x] **Step 2: Run the profile test to verify it fails**

Run:

```powershell
.conda\envs\dongfeng-registration\python.exe -m pytest -q tests/test_colmap_config_profiles.py
```

Expected: failures listing profiles that omit or disable absolute-prior BA.

- [x] **Step 3: Update every shipped YAML profile**

Under each top-level `colmap:` block, set:

```yaml
run_bundle_adjustment: true
ba_mode: absolute_pose_prior
absolute_pose_prior_colmap_executable: <platform-specific path>
ba_position_prior_sigma_m: 0.1
ba_rotation_prior_sigma_deg: 0.5
```

Preserve all unrelated model, dataset, matching, optimization, and dense reconstruction settings.

- [x] **Step 4: Run the profile test to verify it passes**

Run the same profile pytest command and expect all profiles to pass.

### Task 3: Regression Verification

**Files:**
- Verify: `src/dongfeng_registration/colmap_reconstruction.py`
- Verify: `configs/*.yaml`
- Verify: `tests/test_colmap_reconstruction.py`
- Verify: `tests/test_colmap_config_profiles.py`

- [x] **Step 1: Run COLMAP tests**

```powershell
.conda\envs\dongfeng-registration\python.exe -m pytest -q tests/test_colmap_reconstruction.py tests/test_colmap_config_profiles.py
```

- [x] **Step 2: Validate YAML and whitespace**

```powershell
git diff --check
```

- [x] **Step 3: Inspect the final diff**

Confirm that only the loader fallback, COLMAP profile settings, tests, and implementation documentation changed.
