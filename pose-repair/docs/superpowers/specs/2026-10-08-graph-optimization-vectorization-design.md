# Graph Optimization Vectorization Design

## Goal

Reduce T3 multi-camera graph-optimization runtime without changing the sampled
matches, optimization variables, loss terms, weights, bounds, schedule, early
stopping, or mode-projection behavior.

## Design

Each observation endpoint is identified by `(camera, frame, timestamp_ns)`.
Navigation pose lookup and camera-to-world composition are performed once for
each unique endpoint, then gathered by observation index. Frame and body
disturbance variables retain their existing `(camera, frame)` and `frame` keys.

Mean and frame-common penalties retain equal weighting per logical group.
Ragged groups are represented by padded index and validity-mask tensors. Their
means are calculated in one tensor operation, avoiding hundreds of Python-loop
CUDA launches while keeping deterministic reduction order.

## Compatibility

- Duplicate endpoints map to the same computed transform.
- Different timestamps for the same camera/frame remain distinct endpoints.
- Missing cameras and ragged frame groups remain supported.
- The NumPy mode-projection implementation and all exported formats are unchanged.

## Verification

- Unit tests compare endpoint expansion against the former per-observation formula.
- Unit tests compare vectorized group means and gradients against the loop formula.
- Existing optimizer tests must pass.
- T3 20-step and 100-step runs compare runtime, loss history, and exported poses.
