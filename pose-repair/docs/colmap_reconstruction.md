# COLMAP Static-Feature Reconstruction

`reconstruct-colmap` consumes the `deliverables/` directory of one dataset run. Its
settings live in the selected vehicle configuration under the top-level `colmap:` key.
It does not use the LoMA retained matches. The input pose is selected from one of
these files:

```text
<delivery_root>/image_extrinsics/image_camera_to_world_before.csv
<delivery_root>/image_extrinsics/image_camera_to_world_after.csv
```

Set `pose_variant: before` or `pose_variant: after` in `configs/<vehicle>.yaml`.
The default is `after`. Normal vehicle runs include COLMAP pose-prior BA
(`enabled: true`, `run_bundle_adjustment: true`) with `run_dense: false`.
Dense reconstruction is opt-in via `run_dense: true`; explicit OpenMVS smoke
profiles remain dense tests.

The runtime requires a COLMAP 4.2 executable configured through
`colmap_executable`; it does not require the optional `pycolmap` Python package.
Dense reconstruction is selected with `mvs_backend: colmap | openmvs`. The
default and all normal vehicle profiles use `colmap`; OpenMVS is opt-in.
Older delivery layouts containing one CSV per camera under
`image_extrinsics/before/` or `image_extrinsics/after/` are read automatically.

## Full pose-prior BA

`ba_mode: absolute_pose_prior` performs the final BA using the delivery CSV as
a full metric SE(3) prior for every image. It uses COLMAP's official
`AbsolutePosePriorCostFunctor`: the CSV `T_camera_to_world` is inverted to
COLMAP's `sensor_from_world`, then its rotation and translation residuals are
weighted separately by `ba_rotation_prior_sigma_deg` and
`ba_position_prior_sigma_m`. Camera intrinsics and sensor-to-rig extrinsics are
fixed; image poses and, when `ba_refine_points3d: true`, 3D points are jointly
optimized.

The stock COLMAP command does not expose this BA mode. Build the project-local
extension once; it does not replace the system `colmap` executable:

```bash
cd /mnt/workspace/Pose_Dynamic_Compensation/code
bash scripts/build_colmap_absolute_pose_prior.sh
```

The script checks out the fixed COLMAP 4.2.0 source commit, applies
`third_party/colmap_absolute_pose_prior_4.2.0.patch`, and writes the resulting
executable to `tools/colmap_pose_prior/bin/colmap`. Configure the final BA as:

```yaml
colmap:
  colmap_executable: colmap
  ba_mode: absolute_pose_prior
  absolute_pose_prior_colmap_executable: ../tools/colmap_pose_prior/bin/colmap
  ba_position_prior_sigma_m: 0.1
  ba_rotation_prior_sigma_deg: 0.5
```

Use `ba_mode: standard` only when the extension is unavailable. That fallback
has no pose prior and can move disconnected or weakly constrained cameras far
from the delivery poses. The reconstruction report records the selected BA mode
and both prior standard deviations.

## Dense reconstruction backends

Both backends consume the same `sparse_after` model produced after the custom
absolute-pose-prior BA and publish the final point cloud at
`<output_root>/dense/fused.ply`.

The default COLMAP route runs:

```text
sparse_after -> image_undistorter -> patch_match_stereo -> stereo_fusion
```

The optional OpenMVS route runs:

```text
sparse_after -> COLMAP image_undistorter -> InterfaceCOLMAP -> DensifyPointCloud
```

`image_undistorter` remains in the OpenMVS route only to prepare the standard
COLMAP workspace consumed by `InterfaceCOLMAP`. OpenMVS replaces COLMAP
`patch_match_stereo` and `stereo_fusion`; those commands are not run when
`mvs_backend: openmvs`.

Build the pinned, headless OpenMVS 2.4.0 tools under Linux or WSL:

```bash
cd /mnt/workspace/Pose_Dynamic_Compensation/code
sudo bash scripts/install_openmvs_ubuntu_dependencies.sh
bash scripts/build_openmvs.sh
```

The default build is CPU-only and installs under the project-local
`tools/openmvs` directory. To keep a CUDA build isolated from that fallback,
install a WSL CUDA toolkit containing `nvcc`, then run:

```bash
OPENMVS_ENABLE_CUDA=ON \
OPENMVS_CUDA_ARCHITECTURES=89 \
OPENMVS_BUILD_DIR="$PWD/.tools/openmvs-2.4.0-build-cuda" \
OPENMVS_INSTALL_DIR="$PWD/tools/openmvs-cuda" \
bash scripts/build_openmvs.sh
```

Compute capability `89` targets RTX 4090. Override it for other GPUs. The CUDA
toolkit is required only to build the CUDA variant; the Windows NVIDIA driver
continues to provide WSL GPU access.

Select OpenMVS with:

```yaml
mvs_backend: openmvs
openmvs_interface_colmap_executable: ../tools/openmvs/bin/InterfaceCOLMAP
openmvs_densify_point_cloud_executable: ../tools/openmvs/bin/DensifyPointCloud
openmvs_resolution_level: 1
openmvs_max_resolution: 1280
openmvs_min_resolution: 640
openmvs_number_views: 8
openmvs_number_views_fuse: 3
openmvs_gpu_device: cpu  # CPU-only build; omit this option or use cpu
```

`configs/t3_openmvs_smoke.yaml` is a bounded five-frame validation profile.
Copy it before changing machine-local data paths; the committed profile keeps
project-relative tool and data paths.
`configs/t3_openmvs_cuda_smoke.yaml` is the equivalent profile for the isolated
CUDA build and selects GPU device `0`.

## Dynamic masking

Before database creation, the command converts the semantic cache's class-index
masks into two COLMAP masks. A valid pixel is 255 and a masked pixel is 0.

`output_root/matching/pose_valid_masks` is used by `feature_extractor` through
`--ImageReader.mask_path`. Its `colmap_pose_mask_classes` normally includes
`vegetation` and `sky`, so these regions do not create features, matches, tracks,
or BA observations.

`output_root/matching/dense_valid_masks` is used by `stereo_fusion` through
`--StereoFusion.mask_path`. Its `colmap_dense_mask_classes` includes `sky` and
moving-object classes, but intentionally excludes `vegetation`. Consequently,
vegetation can appear in the dense cloud while sky and moving objects are omitted.

For OpenMVS, the same dense masks are resized with nearest-neighbor sampling to
the undistorted image dimensions and written beside each undistorted image as
`dense/openmvs/colmap_workspace/images/<camera>/<image>.jpg.mask.png`. Keeping
the camera subdirectory avoids collisions when different cameras use the same
image basename. Mask value `0` means excluded and `255` means retained;
`DensifyPointCloud` discovers these sidecars and is called with
`--ignore-mask-label 0`.

OpenMVS's native `scene_dense.ply` includes per-vertex `view_indices` and
`view_weights` list properties. The published `dense/fused.ply` removes these
OpenMVS-only tracking fields while preserving XYZ, RGB, and normals, so the
point cloud can be opened directly in CloudCompare and other standard PLY
readers. The native PLY remains under `dense/openmvs/` for diagnosis.

Both class lists are configurable under `colmap:`. A matching database is rebuilt
on every run; reusing an unmasked database would violate this workflow.

## Run

Set `delivery.run_output_root` in `configs/<vehicle>.yaml`, together with canonical data,
semantic cache, and COLMAP executable paths, then run:

```powershell
python -m dongfeng_registration.cli reconstruct-colmap --config configs\m18.yaml
```

Run the bounded OpenMVS profile under Linux or WSL with:

```bash
python -m dongfeng_registration.cli reconstruct-colmap \
  --config configs/t3_openmvs_smoke.yaml
```

For a bounded validation run, set `start_frame: 0`, `frame_count: 20`, and
`run_dense: false`. This still performs static-only feature extraction, matching,
triangulation, and external-pose BA, but skips the long MVS stages.

To include this after normal delivery export, add this to the `delivery` mapping
of the vehicle configuration:

```yaml
delivery:
  # A top-level colmap: mapping in this same file enables this final stage.
```

## Output

```text
<run_output_root>/deliverables/
  result_interpolation_optimized.pkl
  compensated_perception/
  image_extrinsics/
  metrics/
  optimization_diagnostics/
  projection_qc/
  manifest.json
  README_delivery.md
  colmap-ba/
    sparse_before/input/
    sparse_before/triangulated_binary/
    sparse_before/filtered_binary/
    sparse_after/
    image_camera_to_world_after_ba.csv
    pose_delta.csv
    matching/
    logs/
    STATUS.json
    timing_summary.json
    report.json
    README.md
    colmap_reconstruction_report.json  # legacy report alias
    dense/fused.ply                    # only when run_dense=true
```

This matches the `samples-restored/l3-data/<clip>/colmap-ba/` delivery layout.
Standalone configurations also default to `<delivery_root>/colmap-ba/` when
`output_root` is omitted. An explicit CLI output override is still supported.
The base PKL and `image_extrinsics/*` retain their pre-BA delivery values;
consume `colmap-ba/image_camera_to_world_after_ba.csv` for BA camera-to-world
poses. Original annotation fields and per-image optimized pose fields are preserved.

Read `STATUS.json` before using the model. `report.json` records `selected_frames`,
`available_images`, and `coverage` (`partial` or `complete`, relative to the input
pose CSV). Partial validation never claims to optimize unselected frames.
`timing_summary.json` and `logs/` contain sparse-stage wall times, commands,
return codes and output. Failed reruns clear old success reports.
The root manifest and README link to these BA outputs; repeated delivery exports
retain that index when the BA directory exists.


OpenMVS additionally retains its native artifacts for diagnosis:

```text
<output_root>/dense/openmvs/
  scene.mvs
  scene_dense.mvs
  scene_dense.ply
  colmap_workspace/
    images/<camera>/<image>.jpg.mask.png
```

The report records `mvs_backend`, dense point count, final PLY size, and
per-stage timings. The first OpenMVS implementation produces a dense point
cloud only; mesh reconstruction and texturing are not part of this pipeline.

`pose_delta.csv` records each external-pose correction relative to the
selected input CSV. BA keeps intrinsic parameters and sensor-to-rig calibration
fixed. It releases image rig-to-world poses and can jointly refine triangulated
points when `ba_refine_points3d` is enabled.
