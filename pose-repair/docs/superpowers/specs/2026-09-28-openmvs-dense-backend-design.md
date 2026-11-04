# OpenMVS Dense Backend Design

## Goal

Add OpenMVS as a configuration-selected replacement for the complete COLMAP
dense reconstruction algorithm after bundle adjustment. Preserve the existing
COLMAP backend and the project-local absolute-pose-prior COLMAP executable.
Validate the new backend on a bounded T3 dataset run.

The first version produces a dense point cloud only. Mesh reconstruction,
refinement, and texturing are outside this change.

## Pipeline Boundary

The sparse pipeline remains unchanged:

1. Semantic masks exclude configured classes during feature extraction.
2. COLMAP extracts and matches features.
3. COLMAP triangulates and filters sparse points.
4. `tools/colmap_pose_prior/bin/colmap` runs
   `absolute_pose_prior_bundle_adjuster`.
5. The selected MVS backend consumes the same `sparse_ba` model.

`mvs_backend: colmap` keeps the current pipeline:

```text
image_undistorter -> patch_match_stereo -> stereo_fusion
```

`mvs_backend: openmvs` uses:

```text
image_undistorter -> InterfaceCOLMAP -> DensifyPointCloud
```

`image_undistorter` is retained in the OpenMVS route only as the official
COLMAP-to-OpenMVS preparation step. COLMAP does not estimate depth maps or fuse
points in this route. OpenMVS performs the complete dense algorithm.

OpenMVS is pinned to v2.4.0. This version provides native image-mask support
for dense point-cloud segmentation and includes the corrected COLMAP pixel
center conventions required by the bridge.

## Configuration

The `colmap` mapping accepts these fields:

```yaml
mvs_backend: colmap  # colmap | openmvs

openmvs_interface_colmap_executable: ../tools/openmvs/bin/InterfaceCOLMAP
openmvs_densify_point_cloud_executable: ../tools/openmvs/bin/DensifyPointCloud
openmvs_resolution_level: 1
openmvs_max_resolution: 1280
openmvs_min_resolution: 640
openmvs_number_views: 8
openmvs_number_views_fuse: 3
openmvs_gpu_device: cpu  # CPU-only build; omit this option or use cpu
```

The default backend is `colmap`, preserving all existing configuration
behavior. OpenMVS executable validation is conditional: those files are only
required when `run_dense: true` and `mvs_backend: openmvs`.

The first version exposes only stable, high-impact OpenMVS controls. Additional
OpenMVS command-line parameters are not accepted as an arbitrary string because
that would weaken configuration validation and reproducibility.

## Dynamic And Sky Masks

The existing dense valid masks use `0` for an excluded pixel and `255` for a
retained pixel. The OpenMVS workspace receives one geometrically corresponding
mask per undistorted image, named with OpenMVS's `.mask.png` convention and
stored beside that image. Camera subdirectories are preserved so equal image
basenames from different cameras cannot collide.

`DensifyPointCloud` discovers the sidecar masks without `--mask-path` and runs
with `--ignore-mask-label 0`. Missing masks are fatal. There is no silent
unmasked fallback and no silent fallback to the COLMAP backend.

Mask generation reuses the established dense mask class list and the existing
undistorted-workspace sizing behavior so both MVS backends apply the same class
policy.

## Files And Reporting

Both backends publish a common final path:

```text
<output_root>/dense/fused.ply
```

OpenMVS native artifacts are retained for diagnosis:

```text
<output_root>/dense/openmvs/
  colmap_workspace/
    images/<camera>/<image>.jpg.mask.png
  scene.mvs
  scene_dense.mvs
  scene_dense.ply
```

The reconstruction report records:

- selected `mvs_backend`;
- executable paths and effective dense parameters;
- preparation, conversion, and densification elapsed time;
- final PLY path, file size, and vertex count when readable.

A missing or empty scene, missing or empty PLY, failed subprocess, or incomplete
mask set aborts the run with a focused error.

## Build And Installation

The Linux/WSL build remains reproducible and project-local:

```text
tools/colmap_pose_prior/bin/colmap
tools/openmvs/bin/InterfaceCOLMAP
tools/openmvs/bin/DensifyPointCloud
```

The existing `scripts/build_colmap_absolute_pose_prior.sh` continues to build
the pinned custom COLMAP 4.2.0 source and verify that
`absolute_pose_prior_bundle_adjuster` is registered.

A new OpenMVS build script checks out v2.4.0, performs a headless build, installs
the two required applications and their runtime libraries under
`tools/openmvs`, and verifies both executable help commands. Generated binaries
remain ignored by Git.

Install the Ubuntu/WSL build dependencies and build the CPU-only tools with:

```bash
sudo bash scripts/install_openmvs_ubuntu_dependencies.sh
bash scripts/build_openmvs.sh
```

## Tests

Development follows test-first coverage for:

- configuration defaults and OpenMVS field parsing;
- invalid backend rejection;
- conditional executable validation;
- `InterfaceCOLMAP` command construction;
- `DensifyPointCloud` command construction, including mask polarity;
- backend dispatch without running COLMAP PatchMatch in OpenMVS mode;
- common `dense/fused.ply` publication;
- report fields and output validation.

The real-data check uses a separate T3 smoke configuration with a bounded frame
range. It runs the normal masked sparse pipeline, absolute-pose-prior BA, and
OpenMVS densification. Acceptance requires:

1. the custom COLMAP help output contains
   `absolute_pose_prior_bundle_adjuster`;
2. every selected BA image is registered;
3. `InterfaceCOLMAP` creates a non-empty `scene.mvs`;
4. `DensifyPointCloud` creates non-empty native MVS and PLY outputs;
5. `dense/fused.ply` exists, has a positive vertex count, and is reported;
6. no COLMAP `patch_match_stereo` or `stereo_fusion` command runs in OpenMVS
   mode;
7. elapsed times, selected image count, and point count are summarized.

If resources permit, the exact same `sparse_ba` model is also passed to the
COLMAP dense backend for a point-count and runtime comparison. This comparison
is diagnostic, not a pass/fail accuracy claim; geometric accuracy requires an
independent reference surface that the current dataset does not provide.

## Documentation

`docs/colmap_reconstruction.md` will describe both backends, Linux/WSL build
commands, configuration examples, output files, mask semantics, and the bounded
validation workflow. Vehicle configurations keep `mvs_backend: colmap` unless
the user explicitly selects OpenMVS.
