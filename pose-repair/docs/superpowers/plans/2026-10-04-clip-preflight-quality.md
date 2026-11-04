# Clip Preflight Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sampled LoMA preflight that runs before dynamic compensation, diagnoses initial intra/inter-camera pose quality, and reuses its LoMA results during the later full matching stage.

**Architecture:** The preflight stage builds the same intra baseline and inter relation pair identities as the normal matcher, selects up to 20 temporally stratified pairs per relation, and runs LoMA with the same mandatory semantic-mask path as the normal matcher. It stores one raw parquet per cache group plus a manifest keyed by camera, relation, frame endpoints, backend settings, and pose field. The full matcher loads those records, applies the current semantic masks, and runs LoMA only for missing pair identities. A separate report module computes RANSAC, reprojection, valid-depth, and connectivity metrics and returns PASS/WARN/FAIL before expensive stages.

**Tech Stack:** Python, pandas/parquet, OpenCV, Pydantic/YAML configuration, Typer CLI, pytest.

---

### Task 1: Add preflight configuration and pure sampling/cache contracts

**Files:**
- Modify: `src/dongfeng_registration/config.py`
- Create: `src/dongfeng_registration/clip_preflight.py`
- Create: `tests/test_clip_preflight.py`

- [x] Write failing tests for temporal stratification, total-pair caps, pair-key stability, and cache round trips.
- [x] Add a typed `PreflightDiagnosticConfig` with 20 pairs per relation, fixed seed, 800 total-pair cap, mandatory semantic-mask matching, and configurable thresholds.
- [x] Implement deterministic pair selection and a parquet/manifest cache keyed by pair identity.
- [x] Run the focused tests and then the existing config tests.

### Task 2: Expose the normal matcher’s pair identities and cache reuse

**Files:**
- Modify: `src/dongfeng_registration/staged_pipeline.py`
- Modify: `src/dongfeng_registration/match_cache.py`
- Test: `tests/test_clip_preflight.py`, `tests/test_staged_pipeline.py`

- [x] Add optional pair selection and preflight-cache parameters to intra and inter staged matching without changing default behavior.
- [x] Require semantic masks for every preflight match and reuse valid semantic cache entries.
- [x] When a full match run sees a cached preflight pair, load its backend points, apply the current semantic masks, recompute normalized coordinates, and skip LoMA inference.
- [x] Record `matching_source` as `preflight_reused` or `full_loma` in pair metrics and manifests.
- [x] Test that a cached pair does not call the backend and that semantic filtering still happens after reuse.

### Task 3: Implement preflight execution and quality report

**Files:**
- Modify: `src/dongfeng_registration/clip_preflight.py`
- Modify: `src/dongfeng_registration/staged_pipeline.py`
- Create: `tests/test_clip_preflight.py`

- [x] Build sampled intra baseline pairs and inter relation pairs from the same pose/calibration helpers used by the full matcher.
- [x] Before dynamic compensation and full-dataset segmentation, generate or reuse semantic masks for the sampled images and run sampled LoMA pairs with those masks.
- [x] Re-evaluate each sample against the initial pose using image RANSAC, reprojection error, Sampson error, valid depth, and ray gap.
- [x] Aggregate overall, camera, pair/relation, and temporal metrics without merging intra and inter into one opaque score.
- [x] Emit `summary.json`, `report.md`, `sampled_pairs.csv`, `pair_metrics.csv`, `raw_match_diagnostics.parquet`, and optional bad-pair visualizations.
- [x] Implement PASS/WARN/FAIL using retained reprojection median, pair-level and relation-level failure fractions; retain P95 as a diagnostic-only metric.
- [x] Stop the delivery pipeline before dynamic compensation on FAIL unless `fail_action: report_only` is configured.

### Task 4: Add CLI and delivery-stage integration

**Files:**
- Modify: `src/dongfeng_registration/cli/__init__.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Modify: `README.md`
- Modify: `docs/pose_correction.md`
- Test: `tests/test_cli.py`, `tests/test_delivery_pipeline.py`

- [x] Add `diagnose-clip-quality --config ...` using only the minimal canonical pose/image inputs and the preflight cache root.
- [x] Add `preflight-quality` before `dynamic-compensation` in `run-delivery`.
- [x] Make the stage use config values instead of requiring runtime threshold arguments; use `fail_action: report_only` as the explicit debugging override.
- [x] Document the two-stage cache and the fact that preflight generates or reuses semantic masks only for sampled images before the later full semantic stage.

### Task 5: Verification

**Files:**
- Modify: `tests/test_delivery_pipeline.py`

- [x] Run focused preflight and staged-matcher tests.
- [x] Run the complete pytest suite; the current workspace still has 8 pre-existing/environment failures unrelated to preflight.
- [x] Run `run-delivery --dry-run` on a Windows config and verify stage order is `normalize -> preflight-quality -> dynamic-compensation`.
- [x] Run a small real-data preflight and verify no duplicate backend call for reused pairs.
- [x] Inspect the git diff and preserve unrelated working-tree changes.
