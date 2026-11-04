# Match Filtering Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the combined LoMA matching and retained-match filtering wall time without changing any retained row, rejection reason, adaptive threshold, or pair metric.

**Architecture:** Keep OpenCV RANSAC and all existing quality semantics, but replace repeated per-column pandas mutations with an array-first result builder. Add bounded relation-level process parallelism for cache refresh and a producer/consumer path that lets CPU filtering of a completed relation overlap the next GPU LoMA relation. Configuration controls worker count and overlap, with deterministic single-worker fallback.

**Tech Stack:** Python, NumPy, pandas, OpenCV, concurrent.futures, pytest, Parquet.

---

### Task 1: Lock Down Result Equivalence

**Files:**
- Modify: `tests/test_staged_pipeline.py`

- [ ] Add a regression test that evaluates deterministic intra/inter fixture matches through the reference implementation and the new array-first implementation.
- [ ] Assert exact equality for retained masks, boolean stage columns, rejection reasons, candidate counts, and pair ordering.
- [ ] Assert tight numeric equality for navigation geometry, reprojection, ray distance, adaptive thresholds, medians, and MAD values.
- [ ] Run the focused tests and confirm they fail because the new implementation entry point does not exist.

### Task 2: Build Array-First Pair Evaluation

**Files:**
- Modify: `src/dongfeng_registration/match_quality.py`
- Modify: `src/dongfeng_registration/staged_pipeline.py`
- Test: `tests/test_staged_pipeline.py`

- [ ] Add helpers that collect derived NumPy arrays and scalar columns before constructing one DataFrame result.
- [ ] Refactor navigation geometry, threshold annotation, and quality filtering to use those helpers without changing formulas or dtypes visible in stored Parquet files.
- [ ] Preserve the existing public evaluator signatures and pair-metric schema.
- [ ] Run equivalence tests, existing staged-pipeline tests, and a 500-pair T3 benchmark.

### Task 3: Add Bounded Relation-Level Parallel Filtering

**Files:**
- Modify: `src/dongfeng_registration/config.py`
- Modify: `src/dongfeng_registration/staged_pipeline.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Modify: `configs/t3.yaml`
- Modify: `configs/t3_windows_full37.yaml`
- Modify: `configs/t3_windows_subset21.yaml`
- Modify: `configs/l29.yaml`
- Modify: `configs/l29_windows.yaml`
- Modify: `configs/m18.yaml`
- Modify: `configs/m18_windows.yaml`
- Test: `tests/test_staged_pipeline.py`
- Test: `tests/test_delivery_profiles.py`

- [ ] Add `matching.filter_workers` with default `4`, minimum `1`, and deterministic worker-local initialization.
- [ ] Dispatch independent completed relation/cache groups to a process pool; keep all pairs within each group ordered.
- [ ] Merge results in original relation order and use atomic output replacement so interrupted workers cannot leave valid-looking partial caches.
- [ ] Verify one-worker and four-worker outputs are exactly equivalent.

### Task 4: Overlap GPU Matching and CPU Filtering

**Files:**
- Modify: `src/dongfeng_registration/staged_pipeline.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Test: `tests/test_staged_pipeline.py`
- Test: `tests/test_delivery_profiles.py`

- [ ] Add `matching.overlap_filtering` with default `true`.
- [ ] Submit filtering after each relation raw cache is finalized while the persistent LoMA process advances to the next relation.
- [ ] Apply backpressure at `filter_workers` pending jobs and propagate worker failures before optimization begins.
- [ ] Confirm overlap-off and overlap-on output trees contain identical retained Parquet and quality metrics.

### Task 5: Benchmark and Verify

**Files:**
- Modify: `scripts/benchmark_match_filtering.py`
- Create: `benchmark_results/T3_match_filtering_acceleration_20261008/report.md`

- [ ] Record array-only, four-worker, and overlap-enabled filtering times on the same T3 raw cache.
- [ ] Run the full T3 LoMA+filter stage with the same config and cold output directories.
- [ ] Compare all retained files by row count, stable key order, retained/rejection columns, and numeric tolerances.
- [ ] Run the full pytest suite and report separate LoMA, filtering, overlap, and combined wall-time changes.
