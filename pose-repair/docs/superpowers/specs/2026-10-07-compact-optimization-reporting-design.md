# Compact Optimization Reporting Design

## Goal

Reduce the long CPU-only tail after multicamera optimization while preserving exact
before/after metrics over every retained match, the existing optimized calibration
artifacts, actionable outlier details, and deterministic 1% reprojection QC.

## Current Problem

The T3 full run evaluates 1,207,104 retained observations after optimization and then
writes three approximately 225 MB Parquet files. The optimization itself takes about
518 seconds, while post-optimization evaluation and reporting take about 684 seconds.
The default `auto` evaluation backend currently resolves to the legacy CPU evaluator,
and edge summaries iterate over thousands of groups in Python.

The full before/after observation tables are not delivered. They are only consumed by
the following reprojection QC stage, which samples a small subset of image pairs. The
full inlier table is also almost a complete duplicate of the after table.

## Design

### Exact full-data evaluation

All retained observations remain part of before/after evaluation and all reported
global, edge, and outlier metrics. `auto` evaluation selects the Torch backend when a
CUDA device is available and all camera models are supported; otherwise it falls back
to the legacy evaluator. The selected backend is recorded in `summary.json`.

Torch and legacy evaluation are required to agree within explicit numerical tolerances
on representative undistorted-camera fixtures. Backend fallback remains functional on
CPU-only systems and unsupported camera models.

### Compact QC handoff

The optimizer deterministically selects image-pair groups for QC using the configured
sample ratio. It evaluates all observations for metrics but writes only the selected
groups to `observations/qc_matches.parquet`. This table contains match identity and
coordinates plus before/after reprojection coordinates and errors.

The projection QC stage reads this compact table directly and does not sample again.
The sampling manifest records total and selected image-pair counts and the effective
ratio.

### Retained outputs

Keep:

- `metrics/summary.json`
- `metrics/edge_summary.csv`
- `metrics/reprojection_outlier_diagnostics.json`
- `metrics/reprojection_outlier_groups.csv`
- loss, camera delta, disturbance, and absorption metrics
- `observations/reprojection_outliers.parquet`
- `observations/qc_matches.parquet`
- optimized YAML, image extrinsics, and optimized pkl

Remove:

- `observations/before_matches.parquet`
- `observations/after_matches.parquet`
- `observations/after_matches_reprojection_inliers.parquet`
- `observations/after_matches_reprojection_outliers.parquet`

### Reporting acceleration

Edge summaries use vectorized grouped aggregation rather than one Python call per
group. Outlier diagnostics compute exact metrics from the in-memory evaluated table
but write only outlier rows. Full tables are never serialized.

Progress messages identify before evaluation, optimization, after evaluation,
reporting, QC sample export, and deliverable export boundaries.

## Configuration

`projection_qc_sample_ratio` is the single source of truth for post-optimization QC
sampling and defaults to `0.01` in vehicle delivery configurations. The delivery
pipeline passes it to optimization and invokes projection QC in pre-sampled mode.

## Compatibility

The projection QC reader accepts the new compact table. A clear error explains that
legacy full tables are required only when compact QC data is absent; legacy output
directories can still be visualized without rerunning optimization.

No optimization objective, retained-match selection, pair sampling, pkl field, or
camera pose calculation changes.

## Verification

- Unit-test exact summary and edge-summary parity with the previous implementation.
- Unit-test Torch auto-selection and legacy fallback.
- Unit-test deterministic image-pair sampling and compact QC schema.
- Unit-test that full and inlier Parquet files are absent and outliers remain.
- Unit-test projection QC reading compact data without secondary sampling.
- Run targeted optimization and delivery-pipeline tests.
- Replay the existing T3 retained cache or evaluated observations to compare metrics,
  artifacts, storage, and post-optimization phase timings.
