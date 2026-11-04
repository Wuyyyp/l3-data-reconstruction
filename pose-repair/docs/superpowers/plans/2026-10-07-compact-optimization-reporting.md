# Compact Optimization Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve exact full-retained optimization metrics while replacing the long CPU reporting tail and duplicate full Parquet outputs with GPU evaluation, vectorized summaries, outlier-only details, and deterministic 1% QC samples.

**Architecture:** The optimizer continues evaluating every retained observation in memory, but `auto` selects Torch/CUDA when supported. Reporting writes exact aggregate metrics, a compact outlier table, and a deterministic image-pair QC table; the separate QC command consumes that compact handoff directly. Delivery configuration supplies one QC ratio to both stages.

**Tech Stack:** Python, pandas, NumPy, PyTorch, Typer, PyArrow/Parquet, pytest, YAML.

---

### Task 1: Backend selection and exact vectorized summaries

**Files:**
- Modify: `src/dongfeng_registration/extrinsic_optimization.py`
- Test: `tests/test_extrinsic_optimization.py`

- [ ] Add failing tests proving `auto` selects Torch when CUDA and camera models are supported, falls back to legacy otherwise, and reports the resolved backend.
- [ ] Run the focused tests and verify they fail because `auto` currently always selects legacy.
- [ ] Implement one backend resolver used by both reporting metadata and evaluation dispatch.
- [ ] Add a failing parity test comparing the existing edge summary with a vectorized implementation.
- [ ] Replace the per-group Python loop with grouped aggregate operations and verify exact counts plus numerically equivalent metrics.
- [ ] Run focused backend and summary tests.

### Task 2: Compact optimization artifacts

**Files:**
- Modify: `src/dongfeng_registration/extrinsic_optimization.py`
- Modify: `src/dongfeng_registration/cli/__init__.py`
- Test: `tests/test_extrinsic_optimization.py`
- Test: `tests/test_cli.py`

- [ ] Add failing tests for deterministic 1% image-pair selection and the combined QC schema.
- [ ] Add failing tests requiring only `qc_matches.parquet` and `reprojection_outliers.parquet`, with no full before/after or inlier Parquet files.
- [ ] Implement deterministic group sampling using stable ordering and evenly spaced group indices.
- [ ] Build the compact QC table from selected rows while retaining exact full-data summaries in memory.
- [ ] Change outlier diagnostics to serialize only outliers and exact aggregate/group metrics.
- [ ] Add `--projection-qc-sample-ratio` to the optimization CLI and pass it into the optimizer.
- [ ] Add progress messages and sub-phase timings for summary, observation export, image-extrinsic export, and pkl export.
- [ ] Run focused artifact and CLI tests.

### Task 3: Compact projection QC reader and pipeline wiring

**Files:**
- Modify: `scripts/generate_reprojection_qc.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Modify: `configs/l29.yaml`
- Modify: `configs/l29_windows.yaml`
- Modify: `configs/m18.yaml`
- Modify: `configs/m18_windows.yaml`
- Modify: `configs/t3.yaml`
- Modify: `configs/t3_windows_full37.yaml`
- Modify: `configs/t3_windows_subset21.yaml`
- Test: `tests/test_delivery_profiles.py`
- Test: `tests/test_visual_diagnostics.py`

- [ ] Add a failing test that projection QC reads `qc_matches.parquet` and does not resample preselected groups.
- [ ] Add a failing pipeline test requiring the same configured ratio in optimize and projection-QC stage arguments.
- [ ] Implement compact-table loading with legacy full-table compatibility.
- [ ] Wire `projection_qc_sample_ratio` into optimization and mark compact QC as pre-sampled.
- [ ] Set all vehicle delivery profiles to `0.01`.
- [ ] Run focused QC and pipeline tests.

### Task 4: Regression and performance verification

**Files:**
- Modify: `docs/pose_correction.md`
- Create: `benchmark_results/compact_optimization_reporting_20261007.json`

- [ ] Run the focused test set covering optimization, CLI, delivery profiles, and visualization.
- [ ] Run the full relevant pytest modules and record failures, distinguishing pre-existing failures.
- [ ] Replay existing T3 evaluated observations to compare old and new global/edge/outlier metrics.
- [ ] Benchmark vectorized summaries and compact artifact output against the recorded 271.4-second evaluation and 412.2-second reporting baseline.
- [ ] Verify the optimized pkl and image-extrinsic export contracts are unchanged.
- [ ] Update the workflow documentation with retained artifacts, compact QC behavior, and phase progress.
- [ ] Review the diff for unrelated changes and perform a final code review.
