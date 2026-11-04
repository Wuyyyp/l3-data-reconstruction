from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dongfeng_registration.dynamic_object_compensation import read_pcd_cloud  # noqa: E402
from dongfeng_registration.lidar_scan_registration import (  # noqa: E402
    direction_adaptive_lidar_point_filter_mask,
    interpolate_ego_world_pose,
    lidar_point_filter_mask,
    transform_point_records,
    voxel_downsample_indices,
    write_csv,
    write_json,
    write_point_records_ply_ascii,
)
from dongfeng_registration.semantic_lidar_filter import (  # noqa: E402
    DEFAULT_LIDAR_DYNAMIC_CLASSES,
    SemanticClassMaskIndex,
    dynamic_point_mask_from_semantic_views,
)


DEFAULT_T3_PKL = Path(
    r"D:\DongFeng\outputs\T3_joint_mainline_test\deliverables\result_interpolation_optimized.pkl"
)
DEFAULT_T3_LIDAR_ROOT = Path(r"D:\DongFeng\data\T3_canonical\lidar\front_lidar_compensation")
DEFAULT_T3_OUTPUT = Path(r"D:\DongFeng\outputs\T3_front_lidar_optimized_ego_world")
# Compensated front-LiDAR PCDs are aligned to their filename timestamp in ego
# coordinates, so fusion must use the scan-time ego pose rather than camera offset poses.
DEFAULT_T3_EGO_FIELD = "ego2global_transformation_matrix"
DEFAULT_T3_SEMANTIC_CACHE = Path(r"D:\DongFeng\data_cache\T3\semantic_masks")


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def timestamp_to_seconds(value: int | float) -> float:
    raw = float(value)
    if abs(raw) >= 1.0e18:
        return raw / 1.0e9
    if abs(raw) >= 1.0e14:
        return raw / 1.0e6
    return raw


def pcd_filename_reference_time_s(path: Path) -> float:
    """Return the compensation reference time encoded by a timestamp-named PCD."""

    stem = Path(path).stem
    if not stem.isdigit():
        raise ValueError(f"PCD filename does not encode a numeric timestamp: {path.name}")
    return timestamp_to_seconds(int(stem))


def output_ply_path(output_dir: Path, output_ply_name: str) -> Path:
    name = Path(str(output_ply_name))
    if name.name != str(output_ply_name) or name.suffix.lower() != ".ply":
        raise ValueError("output_ply_name must be a .ply filename without directory components")
    return Path(output_dir) / name


def output_metadata_path(output_dir: Path, output_name: str, *, suffix: str) -> Path:
    name = Path(str(output_name))
    if name.name != str(output_name) or name.suffix.lower() != suffix:
        raise ValueError(
            f"metadata output name must be a {suffix} filename without directory components"
        )
    return Path(output_dir) / name


def resolve_lidar_path(info: dict[str, Any], *, lidar_root: Path, lidar_name: str) -> Path:
    sensor = info["sensors"]["lidar"][lidar_name]
    return Path(lidar_root) / Path(str(sensor["data_path"])).name


def world_from_lidar(
    ego_to_world: np.ndarray,
    *,
    info: dict[str, Any],
    lidar_name: str,
    point_coordinate_frame: str,
) -> np.ndarray:
    if point_coordinate_frame == "ego":
        return np.asarray(ego_to_world, dtype=np.float64).reshape(4, 4)
    lidar_to_ego = np.asarray(
        info["sensors"]["lidar"][lidar_name]["extrinsic"],
        dtype=np.float64,
    ).reshape(4, 4)
    return np.asarray(ego_to_world, dtype=np.float64).reshape(4, 4) @ lidar_to_ego


def parse_csv_values(value: str) -> frozenset[str]:
    return frozenset(item.strip() for item in str(value).split(",") if item.strip())


def semantic_camera_views(
    info: dict[str, Any],
    *,
    cameras: tuple[str, ...],
    semantic_index: SemanticClassMaskIndex,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, Any]], list[str]]:
    views: dict[str, tuple[np.ndarray, np.ndarray, Any]] = {}
    missing: list[str] = []
    for camera in cameras:
        camera_data = info.get("sensors", {}).get("cams", {}).get(camera)
        if not isinstance(camera_data, dict) or camera_data.get("timestamp") is None:
            missing.append(camera)
            continue
        semantic = semantic_index.load(camera, camera_data["timestamp"])
        if semantic is None:
            missing.append(camera)
            continue
        views[camera] = (
            np.asarray(camera_data["extrinsic"], dtype=np.float64),
            np.asarray(camera_data["cam_intrinsic"], dtype=np.float64),
            semantic,
        )
    return views, missing


def motion_direction_ego(
    infos: list[dict[str, Any]],
    *,
    frame_index: int,
    ego_field: str,
    ego_to_world: np.ndarray,
) -> np.ndarray:
    """Estimate the local horizontal driving direction from neighboring ego poses."""

    current = int(frame_index)
    left = max(0, current - 1)
    right = min(len(infos) - 1, current + 1)
    if left == right:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    first = np.asarray(infos[left][ego_field], dtype=np.float64).reshape(4, 4)
    second = np.asarray(infos[right][ego_field], dtype=np.float64).reshape(4, 4)
    delta_world = second[:3, 3] - first[:3, 3]
    delta_ego = np.asarray(ego_to_world, dtype=np.float64).reshape(4, 4)[:3, :3].T @ delta_world
    delta_ego[2] = 0.0
    if not np.all(np.isfinite(delta_ego)) or float(np.linalg.norm(delta_ego)) < 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return delta_ego / np.linalg.norm(delta_ego)


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    infos = load_infos(args.pkl)
    selected = infos[int(args.start) : int(args.start) + int(args.count)]
    if not selected:
        raise ValueError("selected frame range is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_cameras = tuple(
        item.strip() for item in str(args.semantic_cameras).split(",") if item.strip()
    )
    semantic_index = (
        SemanticClassMaskIndex(
            args.semantic_cache_root,
            backend=args.semantic_backend,
            cache_key=args.semantic_cache_key or None,
            dynamic_classes=parse_csv_values(args.semantic_dynamic_classes),
        )
        if args.filter_dynamic_semantic
        else None
    )
    record_clouds: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    interpolation_sources: Counter[str] = Counter()

    for local_index, info in enumerate(selected):
        frame_index = int(args.start) + local_index
        pcd_path = resolve_lidar_path(info, lidar_root=args.lidar_root, lidar_name=args.lidar_name)
        if not pcd_path.exists():
            raise FileNotFoundError(pcd_path)
        cloud = read_pcd_cloud(pcd_path)
        reference_time_s = pcd_filename_reference_time_s(pcd_path)
        reference_source = "pcd_filename_timestamp"
        ego_to_world, interpolation_source = interpolate_ego_world_pose(
            infos,
            time_s=reference_time_s,
            ego_field=args.ego_field,
        )
        world_transform = world_from_lidar(
            ego_to_world,
            info=info,
            lidar_name=args.lidar_name,
            point_coordinate_frame=args.point_coordinate_frame,
        )
        range_ceiling_mean_m: float | None = None
        motion_direction = None
        if args.range_mode == "fixed":
            keep = lidar_point_filter_mask(
                cloud.xyz,
                min_range_m=args.min_range_m,
                max_range_m=args.max_range_m,
                min_z_m=args.min_z_m,
                max_z_m=args.max_z_m,
            )
        else:
            motion_direction = motion_direction_ego(
                infos,
                frame_index=frame_index,
                ego_field=args.ego_field,
                ego_to_world=ego_to_world,
            )
            keep, per_point_max_range = direction_adaptive_lidar_point_filter_mask(
                cloud.xyz,
                motion_direction_ego=motion_direction,
                min_range_m=args.min_range_m,
                longitudinal_max_range_m=args.longitudinal_max_range_m,
                lateral_max_range_m=args.lateral_max_range_m,
                gamma=args.range_gamma,
            )
            xyz = cloud.xyz
            keep &= (xyz[:, 2] >= float(args.min_z_m)) & (xyz[:, 2] <= float(args.max_z_m))
            range_ceiling_mean_m = float(np.mean(per_point_max_range))
        filtered_records = cloud.data[keep]
        semantic_details: dict[str, dict[str, int]] = {}
        missing_semantic_cameras: list[str] = []
        semantic_removed_count = 0
        if semantic_index is not None:
            views, missing_semantic_cameras = semantic_camera_views(
                info,
                cameras=semantic_cameras,
                semantic_index=semantic_index,
            )
            if views:
                dynamic, semantic_details = dynamic_point_mask_from_semantic_views(
                    np.column_stack(
                        [filtered_records["x"], filtered_records["y"], filtered_records["z"]]
                    ),
                    camera_views=views,
                    dilation_pixels=args.semantic_dilation_pixels,
                )
                semantic_removed_count = int(np.count_nonzero(dynamic))
                filtered_records = filtered_records[~dynamic]
        transformed = transform_point_records(filtered_records, world_transform)
        record_clouds.append(transformed)
        interpolation_sources[interpolation_source] += 1
        pkl_time_s = timestamp_to_seconds(info["timestamp"])
        frame_rows.append(
            {
                "frame_index": frame_index,
                "pcd_path": str(pcd_path),
                "input_points": int(len(cloud.data)),
                "range_filtered_points": int(np.count_nonzero(keep)),
                "range_mode": str(args.range_mode),
                "mean_per_point_max_range_m": range_ceiling_mean_m,
                "motion_direction_ego_x": None if motion_direction is None else float(motion_direction[0]),
                "motion_direction_ego_y": None if motion_direction is None else float(motion_direction[1]),
                "semantic_removed_points": semantic_removed_count,
                "retained_points": int(len(transformed)),
                "semantic_views_loaded": int(len(semantic_details)),
                "semantic_views_missing": ";".join(missing_semantic_cameras),
                "scan_reference_time_s": reference_time_s,
                "scan_reference_source": reference_source,
                "pkl_time_s": pkl_time_s,
                "scan_minus_pkl_ms": 1000.0 * (reference_time_s - pkl_time_s),
                "ego_interpolation_source": interpolation_source,
            }
        )
        print(
            f"[frame] {frame_index} points={len(transformed)} semantic_removed={semantic_removed_count} "
            f"scan_minus_pkl_ms={1000.0 * (reference_time_s - pkl_time_s):.3f} "
            f"ego={interpolation_source}"
        )

    fuse_start = time.perf_counter()
    fused_records = np.concatenate(record_clouds)
    fused_xyz = np.column_stack(
        [fused_records["x"], fused_records["y"], fused_records["z"]]
    ).astype(np.float32)
    indices = voxel_downsample_indices(fused_xyz, args.output_voxel_size_m)
    fused_down = fused_records[indices]
    ply_path = output_ply_path(output_dir, args.output_ply_name)
    frame_pose_csv_path = output_metadata_path(
        output_dir,
        args.frame_pose_csv_name,
        suffix=".csv",
    )
    summary_path = output_metadata_path(
        output_dir,
        args.summary_name,
        suffix=".json",
    )
    write_point_records_ply_ascii(ply_path, fused_down)
    fuse_time_s = time.perf_counter() - fuse_start

    write_csv(frame_pose_csv_path, frame_rows)
    summary = {
        "pkl": str(args.pkl),
        "ego_field": str(args.ego_field),
        "lidar_root": str(args.lidar_root),
        "lidar_name": str(args.lidar_name),
        "point_coordinate_frame": str(args.point_coordinate_frame),
        "lidar_to_ego_extrinsic_applied": str(args.point_coordinate_frame) == "lidar",
        "pcd_align_ratio": None,
        "output_coordinate_frame": "world",
        "start": int(args.start),
        "count": int(len(selected)),
        "input_points": int(len(fused_records)),
        "output_points": int(len(fused_down)),
        "semantic_filter": {
            "enabled": bool(args.filter_dynamic_semantic),
            "cache_root": str(args.semantic_cache_root),
            "backend": str(args.semantic_backend),
            "cache_key": semantic_index.cache_key if semantic_index is not None else None,
            "cameras": list(semantic_cameras),
            "dynamic_classes": sorted(parse_csv_values(args.semantic_dynamic_classes)),
            "dilation_pixels": int(args.semantic_dilation_pixels),
            "range_filtered_points": int(sum(row["range_filtered_points"] for row in frame_rows)),
            "removed_points": int(sum(row["semantic_removed_points"] for row in frame_rows)),
            "retained_points": int(sum(row["retained_points"] for row in frame_rows)),
        },
        "range_filter": {
            "mode": str(args.range_mode),
            "distance_definition": "Euclidean point distance in the ego-coordinate input cloud",
            "min_range_m": float(args.min_range_m),
            "max_range_m": float(args.max_range_m) if args.range_mode == "fixed" else None,
            "longitudinal_max_range_m": float(args.longitudinal_max_range_m)
            if args.range_mode == "direction_adaptive"
            else None,
            "lateral_max_range_m": float(args.lateral_max_range_m)
            if args.range_mode == "direction_adaptive"
            else None,
            "gamma": float(args.range_gamma) if args.range_mode == "direction_adaptive" else None,
            "motion_direction": "ego-frame translation direction from adjacent ego poses; stationary fallback is +x"
            if args.range_mode == "direction_adaptive"
            else None,
            "mean_per_point_max_range_m": float(
                np.nanmean([row["mean_per_point_max_range_m"] for row in frame_rows])
            )
            if args.range_mode == "direction_adaptive"
            else None,
        },
        "interpolation_sources": dict(interpolation_sources),
        "scan_minus_pkl_ms": {
            "median": float(np.median([row["scan_minus_pkl_ms"] for row in frame_rows])),
            "min": float(np.min([row["scan_minus_pkl_ms"] for row in frame_rows])),
            "max": float(np.max([row["scan_minus_pkl_ms"] for row in frame_rows])),
        },
        "timing": {
            "fuse_and_write_time_s": fuse_time_s,
            "total_time_s": time.perf_counter() - started,
        },
        "outputs": {
            "world_fused_ply": str(ply_path),
            "frame_pose_interpolation_csv": str(frame_pose_csv_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    print(f"[done] outputs -> {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse LiDAR scans in world coordinates from an optimized result_interpolation pkl without ICP/GICP."
    )
    parser.add_argument("--pkl", type=Path, default=DEFAULT_T3_PKL)
    parser.add_argument("--lidar-root", type=Path, default=DEFAULT_T3_LIDAR_ROOT)
    parser.add_argument("--lidar-name", default="front_lidar")
    parser.add_argument("--ego-field", default=DEFAULT_T3_EGO_FIELD)
    parser.add_argument("--point-coordinate-frame", choices=("lidar", "ego"), default="ego")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_T3_OUTPUT)
    parser.add_argument(
        "--output-ply-name",
        default="optimized_ego_interpolated_world_fused.ply",
    )
    parser.add_argument("--frame-pose-csv-name", default="frame_pose_interpolation.csv")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=598)
    parser.add_argument("--min-range-m", type=float, default=1.5)
    parser.add_argument("--max-range-m", type=float, default=80.0)
    parser.add_argument("--range-mode", choices=("fixed", "direction_adaptive"), default="fixed")
    parser.add_argument("--longitudinal-max-range-m", type=float, default=60.0)
    parser.add_argument("--lateral-max-range-m", type=float, default=100.0)
    parser.add_argument("--range-gamma", type=float, default=1.5)
    parser.add_argument("--min-z-m", type=float, default=-float("inf"))
    parser.add_argument("--max-z-m", type=float, default=float("inf"))
    parser.add_argument("--output-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--filter-dynamic-semantic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic-cache-root", type=Path, default=DEFAULT_T3_SEMANTIC_CACHE)
    parser.add_argument("--semantic-backend", default="mask2former")
    parser.add_argument("--semantic-cache-key", default="")
    parser.add_argument(
        "--semantic-cameras",
        default="center_camera_fov30,center_camera_fov120,left_front_camera,left_rear_camera,rear_camera,right_front_camera,right_rear_camera",
    )
    parser.add_argument(
        "--semantic-dynamic-classes",
        default=",".join(sorted(DEFAULT_LIDAR_DYNAMIC_CLASSES)),
    )
    parser.add_argument("--semantic-dilation-pixels", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
