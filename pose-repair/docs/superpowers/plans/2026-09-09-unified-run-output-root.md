# Unified Run Output Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store all non-cache artifacts from one dataset run below one configured output root while leaving semantic and match caches in `data_cache`.

**Architecture:** Add `delivery.run_output_root` as the single non-cache output root. `build_delivery_stages` derives dynamic compensation, dynamic QC, optimization, projection QC, delivery, and optional COLMAP locations from named subdirectories. Legacy explicit output paths remain accepted only as an override for older configurations.

**Tech Stack:** Python, PyYAML, pytest.

---

### Task 1: Derive runtime artifact paths

**Files:**
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write the failing test**

```python
def test_build_delivery_stages_derives_all_non_cache_outputs_from_run_root(tmp_path):
    config = write_delivery_config(tmp_path, run_output_root="run-output")
    stages = build_delivery_stages(config)
    commands = {stage.name: stage.command for stage in stages}

    assert str(tmp_path / "run-output" / "dynamic_compensation") in commands["dynamic-compensation"]
    assert str(tmp_path / "run-output" / "projection_qc" / "dynamic_pointcloud") in commands["dynamic-pointcloud-qc"]
    assert str(tmp_path / "run-output" / "joint_optimization") in commands["optimize"]
    assert str(tmp_path / "run-output" / "deliverables") in commands["export-deliverables"]
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_delivery_run_output_root_groups_non_cache_artifacts -q`

Expected: FAIL because `run_output_root` is not yet mapped to the stage output directories.

- [x] **Step 3: Implement path derivation**

```python
run_output_root = _as_path(root, cfg.get("run_output_root"))
if run_output_root is not None:
    dynamic_output_root = run_output_root / "dynamic_compensation"
    optimization_output_root = run_output_root / "joint_optimization"
    projection_qc_root = run_output_root / "projection_qc"
    deliverable_root = run_output_root / "deliverables"
```

Derive the optional COLMAP output root as `run_output_root / "colmap"`, without changing semantic or match-cache roots.

- [x] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_cli.py::test_delivery_run_output_root_groups_non_cache_artifacts -q`

Expected: PASS.

### Task 2: Make example configurations use the unified root

**Files:**
- Modify: `configs/t3.yaml` or the applicable current T3 delivery profile
- Modify: `configs/l29_windows.yaml`
- Modify: `configs/t3.yaml`
- Modify: `configs/l29.yaml`
- Modify: `configs/m18.yaml`

- [x] **Step 1: Replace scattered delivery output settings with a single root**

```yaml
delivery:
  run_output_root: ../outputs/T3_run
  semantic_cache_root: ../data_cache/T3/semantic_masks
  match_cache_root: ../data_cache/T3/matches
```

Remove only explicit dynamic/optimization/QC/deliverable output root values. Keep cache paths unchanged.

- [x] **Step 2: Run configuration parsing tests**

Run: `python -c "from dongfeng_registration.delivery_pipeline import build_delivery_stages; ..."`

Expected: PASS.

### Task 3: Document the output contract

**Files:**
- Modify: `README.md`
- Modify: `docs/project_workflow.md`

- [x] **Step 1: Document the run directory tree**

```text
<run_output_root>/
  dynamic_compensation/
  joint_optimization/
  projection_qc/
  colmap/
  deliverables/
```

State that semantic masks and LoMA/raw/retained matches remain under `data_cache` and are reusable caches, not run deliverables.

- [x] **Step 2: Run relevant regression tests**

Run: `python -m pytest tests/test_colmap_reconstruction.py tests/test_cli.py::test_staged_commands_are_registered tests/test_cli.py::test_run_delivery_dry_run_prints_full_mainline tests/test_cli.py::test_run_delivery_can_slice_stages_in_dry_run tests/test_cli.py::test_delivery_run_output_root_groups_non_cache_artifacts -q`

Expected: PASS for tests that do not depend on removed historical paths.
