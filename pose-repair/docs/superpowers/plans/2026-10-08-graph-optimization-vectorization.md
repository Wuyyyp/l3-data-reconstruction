# Graph Optimization Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse unique observation endpoints and vectorize mean/common losses in the PyTorch graph optimizer.

**Architecture:** Build immutable endpoint and padded-group layouts before optimization. The optimizer loop composes each unique camera-to-world transform once and evaluates grouped means through batched tensor reductions.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, pytest.

---

### Task 1: Lock endpoint and grouped-mean equivalence

**Files:**
- Modify: `tests/test_extrinsic_optimization.py`
- Modify: `src/dongfeng_registration/extrinsic_optimization.py`

- [x] Add tests for unique endpoint indexing, gathered transform equivalence, ragged group means, and gradients.
- [x] Run the focused tests and verify they fail because the new helpers do not exist.
- [x] Implement the endpoint and padded-group helpers.
- [x] Run the focused tests and verify they pass.

### Task 2: Integrate both fast paths

**Files:**
- Modify: `src/dongfeng_registration/extrinsic_optimization.py`
- Modify: `tests/test_extrinsic_optimization.py`

- [x] Replace per-observation navigation pose construction with unique endpoint poses.
- [x] Compose unique endpoint camera-to-world transforms once per iteration and gather them for matches.
- [x] Replace Python grouped-loss loops with padded tensor reductions.
- [x] Add endpoint/group counts to solver diagnostics.
- [x] Run the complete optimization test module.

### Task 3: Benchmark and validate T3

**Files:**
- Create: `benchmark_results/graph_optimization_vectorization_20261008/report.md`

- [x] Run matching 20-step and 100-step baseline/optimized T3 profiles.
- [x] Compare loss histories and final image camera-to-world poses.
- [x] Record speedup, numerical differences, and residual bottlenecks.
