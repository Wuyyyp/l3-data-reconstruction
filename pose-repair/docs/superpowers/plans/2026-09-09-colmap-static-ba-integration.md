# COLMAP Static BA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-driven COLMAP reconstruction that masks dynamic semantic pixels before feature extraction and matching, then runs external-pose BA and optional dense MVS.

**Architecture:** A new `colmap_reconstruction` module owns COLMAP data preparation and execution. The Typer CLI exposes it directly, while the delivery pipeline conditionally runs the same command after delivery export. Semantic cache class masks are converted into COLMAP static-valid masks before the matching database is created.

**Tech Stack:** Python 3.11, Typer, PyYAML, NumPy, Pillow, SQLite/COLMAP CLI, pytest.

---

### Task 1: Test static mask preparation

**Files:**
- Create: `tests/test_colmap_reconstruction.py`
- Create: `src/dongfeng_registration/colmap_reconstruction.py`

- [x] Write a test that creates semantic metadata and class masks and expects an image-relative `camera/frame.png` valid mask with dynamic pixels set to zero.
- [x] Run `pytest tests/test_colmap_reconstruction.py -q` and verify it fails because the module is missing.
- [x] Implement `build_static_valid_masks` and its metadata validation.
- [x] Re-run the test and verify it passes.

### Task 2: Test forced masking during feature extraction

**Files:**
- Modify: `tests/test_colmap_reconstruction.py`
- Modify: `src/dongfeng_registration/colmap_reconstruction.py`

- [x] Write a test asserting each generated `feature_extractor` command contains `--ImageReader.mask_path` and the static-mask root.
- [x] Run that test and verify it fails.
- [x] Implement command construction and matching database rebuild using those commands.
- [x] Re-run the test and verify it passes.

### Task 3: Add CLI and delivery integration

**Files:**
- Modify: `src/dongfeng_registration/cli/__init__.py`
- Modify: `src/dongfeng_registration/delivery_pipeline.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_code_delivery_contract.py`

- [x] Write tests for CLI command registration and an optional `colmap-reconstruction` delivery stage following export.
- [x] Run the targeted tests and verify they fail.
- [x] Implement the command and stage construction.
- [x] Re-run the tests and verify they pass.

### Task 4: Ship config and documentation

**Files:**
- Create: `configs/colmap.yaml`
- Modify: `README.md`
- Modify: `docs/project_workflow.md`

- [x] Add a self-contained parameterized YAML template and document CSV input variants, static masking, BA constraints, and dense options.
- [x] Add config-load tests where appropriate and run focused tests.

### Task 5: Verify and smoke test

**Files:**
- Test: `tests/test_colmap_reconstruction.py`

- [x] Run all COLMAP and CLI/delivery tests.
- [x] Run `python -m dongfeng_registration.cli reconstruct-colmap --help`.
- [x] Produce a 20-frame configuration from available T3 canonical input and run through static feature extraction/matching and BA; verify output artifacts and mask coverage.
