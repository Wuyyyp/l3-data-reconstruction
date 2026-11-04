Ground splitting integration (2026-10-27)
Canonical trainer: /mnt/zhoukaixuan_workspace/code/street_gaussians/lib/trainers/improvedgs_ground_static
Prepared-data launcher: run_prepared.py --direct
Set GROUND_RUN_ROOT to a NEW prepared experiment root (converted, input_manifest.json, actors.json, prepared, ground_masks, deps, input_validation.json).
Ground splitting is required. GROUND_SPLIT=0 is rejected in the canonical model. The launcher explicitly supplies GROUND_SPLIT=1.
Default: iterations 500 through 15000, every 100 iterations; absolute screen-space gradient threshold 0.0002; maximum 8192 parents per event; cap 1200000 ground points; longest tangent axis split; normal thickness stays 0.001m.
Warmup ending does not freeze ground geometry. Splitting requires eligible gradients and available point capacity; not every scheduled check necessarily creates children.
Completed historical experiments and their code snapshots remain unchanged. sample00024 run.py now dispatches to this canonical trainer; run_historical_no_split.py preserves the old invocation.
The legacy improvedgs_dynamic ground experiment and original generic trainers are not this supported static ground entry.
