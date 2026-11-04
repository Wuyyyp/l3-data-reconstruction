# Dynamic Compensation Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate dynamic-object compensation while preserving all numerical and file-format outputs.

**Architecture:** Extend the existing raw-lidar compensation script with deterministic frame-local track extraction, a bounded decompressed-cloud cache, reusable timestamp-cluster selections, and optimized output handling. Keep the current motion model and public deliverables unchanged.

**Tech Stack:** Python 3.11, NumPy, SciPy, optional python-lzf, pytest.

---

### Task 1: Lock PCD and interpolation behavior

**Files:**
- Modify: `tests/test_dynamic_object_compensation.py`
- Modify: `src/dongfeng_registration/dynamic_object_compensation.py`

- [x] Add failing tests proving native LZF output validation, Python fallback, streaming PCD writes, and exact interpolation endpoints.
- [x] Run the focused tests and verify the expected failures.
- [x] Add optional native LZF dispatch, streaming PCD writing, and endpoint fast paths.
- [x] Run the focused tests and verify they pass.

### Task 2: Reuse track-prepass work

**Files:**
- Modify: `tests/test_dynamic_object_compensation.py`
- Modify: `scripts/compensate_dynamic_raw_lidars.py`

- [x] Add failing tests for bounded all-or-nothing cloud caching and reusable object-cluster selections.
- [x] Run the focused tests and verify the expected failures.
- [x] Implement frame-local extraction, deterministic merging, cache reuse, and cluster reuse.
- [x] Run the focused tests and verify they pass.

### Task 3: Parallelize track extraction and deduplicate single-lidar output

**Files:**
- Modify: `tests/test_dynamic_object_compensation.py`
- Modify: `scripts/compensate_dynamic_raw_lidars.py`

- [x] Add failing tests for deterministic parallel extraction and hard-link/copy output behavior.
- [x] Run the focused tests and verify the expected failures.
- [x] Apply `track_workers` to frame extraction and add single-lidar hard-link output mode.
- [x] Run the focused tests and verify they pass.

### Task 4: Propagate configuration and timing

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Modify: `configs/t3.yaml`
- Modify: `configs/t3_windows_full37.yaml`
- Modify: `configs/t3_windows_subset21.yaml`
- Modify: `configs/l29.yaml`
- Modify: `configs/l29_windows.yaml`
- Modify: `configs/m18.yaml`
- Modify: `configs/m18_windows.yaml`
- Modify: `pyproject.toml`

- [x] Add failing delivery tests for config and timing propagation.
- [x] Run the focused tests and verify the expected failures.
- [x] Pass the vehicle config and timing path to dynamic compensation and add conservative per-vehicle defaults.
- [x] Add the optional native LZF performance dependency.
- [x] Run delivery and config tests.

### Task 5: Regression and benchmark verification

**Files:**
- Create: `benchmark_results/dynamic_compensation_acceleration_20261008/report.md`

- [x] Run the complete dynamic-compensation test module.
- [x] Run the complete pytest suite (553 passed, 15 skipped, 8 unrelated existing failures after excluding one missing historical tool module).
- [x] Benchmark T3 100 frames and M18 5 frames with the optimized path.
- [x] Compare PCD hashes/arrays and summary rows against the baseline outputs.
- [x] Record timings, speedups, memory settings, and any residual risks in the report.
