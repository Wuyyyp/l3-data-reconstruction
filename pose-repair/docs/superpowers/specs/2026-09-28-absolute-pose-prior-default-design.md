# Absolute Pose Prior Default Design

## Goal

Make COLMAP absolute-pose-prior bundle adjustment the default for every shipped
configuration while retaining standard COLMAP bundle adjustment as an explicit
fallback.

## Configuration

Every YAML file under `configs/` will enable bundle adjustment and declare:

```yaml
colmap:
  run_bundle_adjustment: true
  ba_mode: absolute_pose_prior
  ba_position_prior_sigma_m: 0.1
  ba_rotation_prior_sigma_deg: 0.5
```

Linux configurations will use:

```yaml
absolute_pose_prior_colmap_executable: ../tools/colmap_pose_prior/bin/colmap
```

Windows configurations will use:

```yaml
absolute_pose_prior_colmap_executable: D:/DongFeng/Code/tools/colmap_pose_prior/bin/colmap.exe
```

The existing stock `colmap_executable` remains unchanged because feature
extraction, matching, triangulation, filtering, and dense reconstruction still
use the standard COLMAP executable.

## Runtime Behavior

`load_colmap_config()` will default an omitted `ba_mode` to
`absolute_pose_prior`. Users can explicitly set `ba_mode: standard` when the
extended executable is unavailable. The existing preflight check will continue
to fail clearly when absolute-pose-prior BA is enabled but its executable is
missing.

## Validation

Tests will verify that:

1. An omitted `ba_mode` resolves to `absolute_pose_prior`.
2. Every shipped configuration enables bundle adjustment and selects
   `absolute_pose_prior`.
3. Windows configurations use the `.exe` path and other configurations use the
   Linux path.
4. Existing COLMAP reconstruction tests continue to pass.

