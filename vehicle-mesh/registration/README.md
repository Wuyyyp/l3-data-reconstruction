# Seven-view dynamic vehicle registration

This directory is a self-contained task package. It imports no application modules from the surrounding repository. Inputs are supplied as a sequence directory; vehicle models are selected by editing the generated YAML. The only external model runtime is SAM3, installed from its upstream package, with a separately supplied local checkpoint.

## Install

```bash
cd /path/to/multiview_registration
bash install_dependencies.sh --venv /path/to/multiview-venv --torch-index-url https://download.pytorch.org/whl/cu128
source /path/to/multiview-venv/bin/activate
export SAM3_CHECKPOINT=/path/to/sam3_checkpoint.pt
```

Use Python 3.12 or newer, a CUDA-compatible PyTorch 2.7+ runtime, and system EGL/GLU libraries for offscreen rendering. For a preconfigured environment, pass its interpreter with `--python`; `--torch-index-url` can select a different CUDA wheel index. The installation check reports whether CUDA is available and warns if EGL cannot load. SAM3 weights require separate access approval and are not bundled.

## Step 1: select dynamic vehicles

Input needs `manifest.json`, `result_interpolation_optimized_first101.pkl`, and each video named by the manifest. Dynamic status is computed from ego-coordinate track centers transformed to world coordinates by the per-frame ego-to-world pose. Only `VEHICLE_*` tracks passing the world-trajectory speed, displacement, continuity, and visibility checks are selected; bicycles and stationary road vehicles are excluded. The count is the number of distinct dynamic motor-vehicle IDs across the seven views.

```bash
python select_dynamic_vehicles.py \
  --input /path/to/sequence --output /path/to/selected --jobs 4
```

The output contains one box video per camera, selected IDs, trajectory statistics, `class_whitelist.txt`, and `registration.yaml`. The whitelist initially maps `Car` and `Suv` to the l3 dataset, and `Vehicle` to Waymo. For each ID, fill `objects.<ID>.category` with one allowed category and `model_path` with its model file. Paths may be absolute or relative to the YAML file. Edit the copied whitelist to add or remove categories.

`--input` selects the source sequence, `--output` selects a new output directory, and `--jobs` controls parallel CPU video generation. Thresholds can be adjusted with `--speed-threshold`, `--min-observations`, and `--min-displacement`. This step uses CPU only.

## Step 2: register all visible frames

```bash
python run_registration.py \
  --input /path/to/sequence \
  --config /path/to/selected/registration.yaml \
  --output /path/to/registration --jobs 2 --gpus 0 1
```

`--input` names the original sequence, `--config` names the edited model mapping, `--output` names a fresh run directory, `--jobs` caps concurrent GPU workers and validation workers, and `--gpus` lists physical GPU IDs. `--resume` continues a partially completed output. `--evaluations` and `--static-evaluations` tune pose-search budgets.

Every selected track is registered at every source frame where it is visible. Each pose is shared across all calibrated views for that frame; temporal optimization constrains ground contact, upright orientation, and smooth motion. Outputs include exact per-object masks, the optimized per-frame pose PKL, seven registered videos, and validation reports. Videos use H.264 High, `yuv420p`, and faststart.

The geometry path uses `trimesh` and `pyrender` only within this task package. Camera and track interpretation follow the input PKL fields documented in the package code; no Build, Reconstruction, or Registration pipeline modules are imported.

## Include stationary or occluded vehicles

The default selection remains dynamic-only. To register all annotated motor vehicles, supply a reviewed model mapping and a JSON evidence file containing `selected_ids`, `dynamic_ids`, and a `policy` starting with `all_`, then pass `--include-static --selection-evidence /path/to/selection_statistics.json`. Each selected ID must have a model in the YAML; one-frame and fully occluded tracks retain an annotation/temporal estimate when image evidence is unavailable.

Optionally pass `--scene-depth-root /path/to/converted` for compensated LiDAR foreground occlusion. This directory must contain `lidar_depth/{frame:06d}_{camera_id:02d}.npy` with `mask` and `value` arrays matching the input image size. The seven camera IDs are left front=0, right front=1, rear=2, left rear=3, right rear=4, front fov30=9, front fov120=10. Sparse samples are expanded by 10 pixels with a 2 metre depth tolerance; this is approximate visibility, not a reconstructed scene surface.

Always use a fresh `--output` directory. Use `--resume` only to continue that same run. The separate `scripts/refine_mesh_*` interfaces accept `--asset` (or `MESH_ASSET_PATH`); NV batch interfaces take explicit data roots and checkpoint arguments. They do not automatically interpret this package's prepared PKL sequence format.


## Two separate mask products (world displacement > 1 m)

`run_registration.py` now always produces both mask products before mesh fitting:

* `mask-videos/`: existing SAM3 projected-box matching and recovery for `obj_selected.txt`; source track IDs and the old selection policy are unchanged.
* `moving-object-masks/`: final whole-object instances. A rider and their motorcycle/e-bike/cycle, including attached covers, are one `rider_vehicle` instance with one ID and colour. Independent pedestrians remain `person` instances. Source track IDs unite detections across prompts; joint text + projected-box SAM3 refinement recovers the full rider/vehicle outline. Source `Huge_vehicle`, `Vehicle_else`, and `Firetruck` annotations are included in vehicle matching; `Tricycle` is included in cycle/rider matching.
* `moving-object-candidates/`: internal SAM3 candidates for `vehicle`, `person`, `bicycle`, `motorcycle`, `electric scooter`, `electric bicycle`, and `tricycle`. Component masks here are intermediate evidence, not final object instances. No selected mesh track or asset is required.

Run only the two mask products, using the same prepared input and model mapping as registration:

```bash
source /data/envs/3dbbox-mesh/activate.sh
python mesh/run_registration.py --input /path/to/prepared_input \
  --config /path/to/selection/registration.yaml --output /path/to/fresh_output \
  --gpus 4 --jobs 1 --masks-only --motion-displacement-m 1 \
  --include-static --selection-evidence /path/to/selection/selection_statistics.json \
  --scene-depth-root /path/to/converted
```

Omit `--include-static` and `--selection-evidence` when using the original dynamic-only selection. Mask-only mode does not fit/export a mesh pose PKL. The normal command without `--masks-only` continues the existing mesh fitting/rendering pipeline; it still consumes only `mask-videos/`.

The new motion rule is **maximum Euclidean distance between any two world XYZ positions of a target within the input clip > 1 metre**. Exactly 1 metre is static. This is not image displacement, accumulated path length, or net first-to-last displacement. A moving track's masks are retained over its detected frames, including temporary stops within the clip.

Metric motion evidence, in priority order:

1. Stable association to a compatible source 3D annotation track (one-to-one projected-box matching per prompt/frame). Apply the optimized ego-to-world pose to annotation object centres and evaluate their whole-clip displacement. This also covers pedestrian/rider/two-wheeler annotations; mesh selection is irrelevant.
2. For unassociated detections, use unexpanded sparse LiDAR inside the eroded instance mask and calibrated camera-to-world poses to estimate visible-surface world centroids. Truncated masks and insufficient depth are excluded from this estimate. A median visible-surface radius above 10 m indicates mixed foreground/background depth for these road-user classes and is `uncertain`. Surface sampling can change with viewpoint: near-threshold displacements that do not exceed the surface-extent uncertainty margin also remain `uncertain`.
3. Without adequate metric evidence, mark `uncertain`; never replace missing evidence with image motion or a filled box. The strict 1 m rule applies to trajectories with sufficient evidence. LiDAR fallback is an estimate, not a guaranteed object-centre measurement.

New outputs:

```text
moving-object-masks/
  config.json                       # parameters and input provenance
  manifest.json                     # coverage, timings, rule, limitations
  masks/<camera>/<frame>.npz         # only moving instances, bbox + packbits
  masks/<camera>/assignments.json    # whole entity IDs, source members and motion evidence
  union-masks/<camera>/<frame>.png   # lossless full-resolution union, 0/255
  videos/<camera>.mp4               # binary-mask preview (lossy)
  overlays/<camera>.mp4             # source + moving-mask preview
  review/                           # first/middle/last frame overlays
```

Final matched IDs are `obj_<source_track_id>` and combine all associated components. A person associated with a rider track is never exported as a separate pedestrian. Unassociated IDs remain camera-local; contained unassociated body/cycle components are consumed by the whole rider instance. Calibrated box depth intervals reject a rider mask mostly covered by a clearly nearer static cycle, avoiding transfer of its foreground silhouette to a hidden moving track. Ambiguous association, occlusion and failed joint refinement can still cause missed or incomplete masks; per-frame provenance distinguishes joint refinement from component-union fallback. A class label cannot reliably distinguish electric and conventional two-wheelers. The original per-prompt evidence is retained only under `moving-object-candidates/`.

The new segmenter can also run alone without any model mapping:

```bash
python mesh/utils/segment_moving_objects.py --input /path/to/prepared_input \
  --output /path/to/moving_candidates --gpus 4 --displacement-m 1 \
  --scene-depth-root /path/to/converted
python mesh/utils/assemble_moving_objects.py --candidates /path/to/moving_candidates \
  --output /path/to/new_moving_masks --gpus 4
```

Use a fresh output for changed inputs/settings. An interrupted identical run resumes at completed prompt/camera markers. To validate the geometry, threshold, instance encoding and independent output filtering:

```bash
python mesh/utils/test_motion_masks.py
python mesh/utils/test_whole_entities.py
```
