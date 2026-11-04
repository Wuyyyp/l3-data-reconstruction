---
name: multiview-vehicle-registration
description: Run the standalone seven-view registration task for world-coordinate dynamic motor vehicles.
---

# Seven-view vehicle registration

This is a self-contained task package under `tests/multiview_registration/`. Do not import or depend on modules from the application repository. Read [README.md](README.md) for the input layout, environment setup, and the two-step command sequence.

The workflow has two public Python interfaces:

1. `select_dynamic_vehicles.py` estimates world motion from ego-coordinate tracks transformed by each frame's ego-to-world pose. It selects distinct dynamic motor vehicles only, writes seven box videos and trajectory evidence, and creates a YAML model mapping that a person must fill in. Its category whitelist defaults to `Car` and `Suv` for l3 assets and `Vehicle` for Waymo assets.
2. `run_registration.py` validates that mapping and registers each selected vehicle across all its visible source frames. It uses one shared per-frame pose across seven views and temporal ground/upright/smoothness constraints.

Install dependencies with `install_dependencies.sh`; set `SAM3_CHECKPOINT` to the external checkpoint file before registration. Keep `trimesh` and `pyrender` use inside this task. Do not change the main application pipeline or modify the source sequence. Use a new output directory for a fresh run and `--resume` only for continuing that run.
