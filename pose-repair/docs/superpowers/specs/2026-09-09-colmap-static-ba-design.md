# COLMAP Static-Feature BA Design

## Goal

Integrate COLMAP external-pose BA and dense reconstruction into the project. Dynamic semantic classes must be excluded before COLMAP feature extraction, so they cannot create matches, triangulated tracks, or BA observations.

## Data Flow

1. Read the selected delivery pose CSV (`image_camera_to_world_before.csv` or `image_camera_to_world_after.csv`) and the canonical dataset pkl.
2. Read semantic cache class-index masks and metadata, derive one binary static-valid mask per input image. Static pixels are 255; configured dynamic classes are 0.
3. Rebuild the COLMAP database. Every `feature_extractor` call receives `--ImageReader.mask_path`; matching consequently operates only on static features.
4. Write initial cameras and image poses, triangulate, and run external-pose-only BA. Intrinsics, rig sensor calibration, and points remain fixed.
5. Undistort images and run a photometric seed followed by geometric PatchMatch. Provide static masks to fusion as well, so dense output excludes dynamic pixels.

## Interfaces

`dongfeng-registration reconstruct-colmap --config configs/colmap.yaml` is the direct entry point. The COLMAP config names the canonical dataset, semantic cache, delivery directory, selected pose variant, output root, executable, matching settings, BA scope, and MVS settings.

The vehicle delivery configuration can optionally include `colmap_config`. When present, `run-delivery` invokes `reconstruct-colmap` after `export-deliverables`.

## Validation

Unit tests cover static-mask output naming and polarity, the compulsory feature-mask command option, and delivery-stage placement. A 20-frame run rebuilds a database, performs static-only feature extraction/matching and BA, and verifies that the sparse model, mask directory, BA corrections, and reconstruction report exist. Dense MVS is disabled for this smoke run to keep it bounded; the production configuration enables it.
