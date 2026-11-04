# Dynamic Compensation Acceleration Design

## Goal

Reduce dynamic-object compensation wall time without changing compensated point coordinates, point order, point count, PCD fields, box payloads, or fallback behavior.

## Design

The track prepass remains the source of object observations. It will additionally retain the immutable PCD clouds and object timestamp-cluster indices needed by the compensation pass. A bounded all-or-nothing cache avoids partial-cache behavior that depends on frame order: if the configured memory budget is exceeded, cached clouds are discarded and the existing reread path is used. Compact cluster selections remain reusable because they are small.

Track extraction is parallelized by frame. Each worker produces frame-local samples and selections; the caller merges and sorts them deterministically. The configured frame worker count remains independent from the track worker count.

PCD `binary_compressed` decoding uses an optional native `lzf` module when available and falls back to the existing validated Python decoder. PCD writing streams the header and ndarray rather than constructing another full payload in memory.

Exact interpolation endpoints return the original transform directly. Reference-time object transforms are calculated once per frame, lidar, and object rather than once per timestamp cluster.

For a single lidar, fused and per-lidar PCD bytes are identical. The fused file is written once and the per-lidar output is created as a hard link, with a normal copy fallback when links are unsupported.

## Configuration

`dynamic_compensation` supports:

- `workers`: compensation-frame workers.
- `track_workers`: track-prepass workers.
- `cloud_cache_max_gb`: maximum retained decompressed PCD bytes; `0` disables cloud caching.
- `single_lidar_output_mode`: `hardlink` or `write`.

The delivery pipeline passes the same vehicle config to the compensation script and writes a stage timing JSON file.

## Verification

Unit tests cover cache overflow, deterministic parallel track merging, cluster reuse, native-LZF fallback, streaming PCD writes, endpoint interpolation, hard-link output, and delivery config propagation. T3 and M18 subset benchmarks compare output hashes/arrays and stage timings before and after the change.
