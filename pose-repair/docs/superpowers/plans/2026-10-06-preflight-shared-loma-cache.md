# Preflight Shared LoMa Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Share the configured LoMa GPU feature cache between preflight and formal matching in the delivery process.

**Architecture:** The `match-delivery` command runs preflight and persistent matching in one Python process. Preflight passes stable image keys to the existing `run_match_backend` API, stores sampled pair results in `PreflightMatchCache`, and persistent matching reuses those results while retaining the same process-level backend/cache. The earlier standalone preflight subprocess is removed from the default delivery stage list.

**Tech Stack:** Python, Typer, pandas, pytest, existing LoMa backend/cache.

---

### Task 1: Lock down shared-cache behavior with tests

**Files:**
- Modify: `tests/test_clip_preflight.py`
- Modify: `tests/test_delivery_profiles.py`

- [x] **Step 1: Add a preflight call assertion**

Assert that the preflight backend call receives `first_image_key` and `second_image_key` equal to the two source image paths.

- [x] **Step 2: Add delivery ordering assertion**

Stub preflight and persistent matching, invoke the matching delivery command helper, and assert preflight is called before persistent matching.

- [x] **Step 3: Run the focused tests and verify they fail**

Run `pytest tests/test_clip_preflight.py tests/test_delivery_profiles.py -q`.

Expected: the new assertions fail because the current preflight call lacks cache keys and `match-delivery` does not invoke preflight.

### Task 2: Pass cache context from preflight

**Files:**
- Modify: `src/dongfeng_registration/clip_preflight.py:1041`

- [x] **Step 1: Pass stable image keys**

Add `first_image_key=str(first.path)` and `second_image_key=str(second.path)` to the existing `run_match_backend` call. Do not create a separate cache or alter matching configuration.

- [x] **Step 2: Run the focused preflight tests**

Run `pytest tests/test_clip_preflight.py -q` and verify they pass.

### Task 3: Integrate preflight into the persistent matching process

**Files:**
- Modify: `src/dongfeng_registration/cli/__init__.py:472-532`
- Modify: `src/dongfeng_registration/delivery_pipeline.py:217-228`
- Modify: `tests/test_delivery_profiles.py`

- [x] **Step 1: Invoke preflight before persistent matching**

In `match_delivery`, call `run_preflight_from_delivery_config(config.resolve())` before `run_persistent_matching_stage`. Raise the same exit condition as `diagnose_clip_quality` when the report is `FAIL` and `fail_action` is not `report_only`.

- [x] **Step 2: Remove the duplicate standalone preflight stage**

Do not add the separate `preflight-quality` subprocess in `build_delivery_stages`; `match-delivery` is now the single process boundary for preflight plus formal matching.

- [x] **Step 3: Preserve stage naming compatibility**

Keep `STAGE_ORDER` and the public `diagnose-clip-quality` command unchanged. Existing explicit diagnostic runs remain available; the full delivery path uses the integrated command.

- [x] **Step 4: Run focused delivery tests**

Run `pytest tests/test_delivery_profiles.py tests/test_cli.py -q`.

### Task 4: Full verification and cache evidence

**Files:**
- No new production files.

- [x] **Step 1: Run the regression suite**

Run `pytest -q` from `D:\DongFeng\Code`.

- [x] **Step 2: Run a small integrated delivery test**

Use a test delivery configuration with a small frame window and inspect the match metrics for `feature_cache_gpu_hit_count > 0` after preflight and formal matching execute in one process.

- [x] **Step 3: Confirm no unrelated files changed**

Run `git status --short` and review only the files listed above plus the design/plan documents.
