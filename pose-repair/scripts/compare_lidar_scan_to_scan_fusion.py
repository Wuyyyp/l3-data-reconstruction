from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dongfeng_registration.lidar_scan_registration import (  # noqa: E402
    bev_image,
    chain_pair_transforms,
    filter_lidar_points,
    fused_cloud_stats,
    lidar_point_filter_mask,
    nearest_neighbor_stats,
    normalize_registration_method,
    optimize_world_pose_graph,
    resolve_point_coordinate_frame,
    relative_to_anchor_transforms,
    summarize_pair_metrics,
    transform_points,
    transform_point_records,
    voxel_downsample_indices,
    voxel_downsample_numpy,
    write_csv,
    write_json,
    write_point_records_ply_ascii,
    write_ply_ascii,
)
from dongfeng_registration.dynamic_object_compensation import read_pcd_cloud  # noqa: E402

DEFAULT_MIN_Z_M = -float("inf")
DEFAULT_MAX_Z_M = float("inf")
DEFAULT_T3_CANONICAL_ROOT = Path(r"D:\DongFeng\data\T3_canonical")
DEFAULT_T3_STANDARD_PKL = DEFAULT_T3_CANONICAL_ROOT / "result_interpolation_standard.pkl"
DEFAULT_T3_FRONT_LIDAR_ROOT = (
    DEFAULT_T3_CANONICAL_ROOT / "lidar" / "front_lidar_compensation"
)
DEFAULT_T3_OUTPUT_DIR = Path(r"D:\DongFeng\outputs\T3_front_lidar_multiframe_fusion")


def build_t3_defaults() -> dict[str, Path]:
    return {
        "pkl": DEFAULT_T3_STANDARD_PKL,
        "lidar_root": DEFAULT_T3_FRONT_LIDAR_ROOT,
        "output_dir": DEFAULT_T3_OUTPUT_DIR,
    }


def registration_methods_to_run(method: str) -> list[str]:
    value = str(method).strip().lower()
    if value == "both":
        return ["icp", "gicp"]
    if value == "all":
        return ["translation_only", "icp", "gicp"]
    return [normalize_registration_method(value)]


def build_pose_graph_edge_pairs(frame_count: int, frame_gaps: list[int]) -> list[tuple[int, int]]:
    """Return unique (target, source) pairs for local overlap constraints."""

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for gap in sorted({int(item) for item in frame_gaps if int(item) > 0}):
        for source_index in range(gap, int(frame_count)):
            pair = (source_index - gap, source_index)
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_lidar_path(info: dict[str, Any], *, lidar_root: Path, lidar_name: str) -> Path:
    lidar_data = info["sensors"]["lidar"][lidar_name]
    data_path = str(lidar_data["data_path"])
    return Path(lidar_root) / Path(data_path).name


def world_from_lidar(
    info: dict[str, Any],
    *,
    lidar_name: str,
    point_coordinate_frame: str,
) -> np.ndarray:
    ego_to_world = np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64)
    if str(point_coordinate_frame) == "ego":
        return ego_to_world
    lidar_to_ego = np.asarray(
        info["sensors"]["lidar"][lidar_name]["extrinsic"],
        dtype=np.float64,
    )
    return ego_to_world @ lidar_to_ego


def make_o3d_cloud(points: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def run_multiscale_translation_only_icp(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    max_iterations: list[int],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    transform = np.asarray(init_transform, dtype=np.float64).reshape(4, 4).copy()
    details: list[dict[str, float | int]] = []
    for level, (voxel, max_corr, max_iter) in enumerate(
        zip(voxel_sizes, max_correspondence_distances, max_iterations, strict=True)
    ):
        source_down = voxel_downsample_numpy(source_points, voxel)
        target_down = voxel_downsample_numpy(target_points, voxel)
        if len(source_down) == 0 or len(target_down) == 0:
            details.append(
                {
                    "level": int(level),
                    "registration_method": "translation_only",
                    "voxel_size_m": float(voxel),
                    "max_correspondence_m": float(max_corr),
                    "fitness": float("nan"),
                    "inlier_rmse": float("nan"),
                    "correspondence_count": 0,
                }
            )
            continue
        from scipy.spatial import cKDTree

        current = transform.copy()
        tree = cKDTree(target_down)
        last_distances = np.array([], dtype=np.float64)
        last_inlier_count = 0
        for _ in range(int(max_iter)):
            transformed = transform_points(source_down, current).astype(np.float64)
            distances, indices = tree.query(transformed, k=1, workers=-1)
            valid = np.isfinite(distances)
            if not np.any(valid):
                break
            distances = np.asarray(distances[valid], dtype=np.float64)
            indices = np.asarray(indices[valid], dtype=np.int64)
            inliers = distances <= float(max_corr)
            last_inlier_count = int(np.count_nonzero(inliers))
            if last_inlier_count == 0:
                break
            residual = target_down[indices[inliers]] - transformed[valid][inliers]
            delta_t = np.median(residual, axis=0)
            if not np.all(np.isfinite(delta_t)):
                break
            current[:3, 3] += delta_t
            last_distances = distances[inliers]
            if float(np.linalg.norm(delta_t)) < 1.0e-4:
                break
        fitness = float(last_inlier_count / max(len(source_down), 1))
        rmse = (
            float(np.sqrt(np.mean(last_distances**2))) if last_distances.size else float("nan")
        )
        details.append(
            {
                "level": int(level),
                "registration_method": "translation_only",
                "voxel_size_m": float(voxel),
                "max_correspondence_m": float(max_corr),
                "fitness": fitness,
                "inlier_rmse": rmse,
                "correspondence_count": int(last_inlier_count),
            }
        )
        transform = current
    return transform, details


def run_multiscale_icp(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    registration_method: str,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    max_iterations: list[int],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    transform = np.asarray(init_transform, dtype=np.float64).reshape(4, 4).copy()
    method = normalize_registration_method(registration_method)
    try:
        import open3d as o3d
    except Exception:
        return run_multiscale_translation_only_icp(
            source_points=source_points,
            target_points=target_points,
            init_transform=transform,
            voxel_sizes=voxel_sizes,
            max_correspondence_distances=max_correspondence_distances,
            max_iterations=max_iterations,
        )

    if method == "translation_only":
        return run_multiscale_translation_only_icp(
            source_points=source_points,
            target_points=target_points,
            init_transform=transform,
            voxel_sizes=voxel_sizes,
            max_correspondence_distances=max_correspondence_distances,
            max_iterations=max_iterations,
        )

    details: list[dict[str, float | int]] = []
    for level, (voxel, max_corr, max_iter) in enumerate(
        zip(voxel_sizes, max_correspondence_distances, max_iterations, strict=True)
    ):
        source_down = voxel_downsample_numpy(source_points, voxel)
        target_down = voxel_downsample_numpy(target_points, voxel)
        source_cloud = make_o3d_cloud(source_down)
        target_cloud = make_o3d_cloud(target_down)
        radius = max(float(max_corr) * 2.0, float(voxel) * 2.5)
        source_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        target_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        if method == "gicp":
            result = o3d.pipelines.registration.registration_generalized_icp(
                source_cloud,
                target_cloud,
                float(max_corr),
                transform,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(max_iter)
                ),
            )
        else:
            result = o3d.pipelines.registration.registration_icp(
                source_cloud,
                target_cloud,
                float(max_corr),
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(max_iter)
                ),
            )
        transform = np.asarray(result.transformation, dtype=np.float64)
        details.append(
            {
                "level": int(level),
                "registration_method": method,
                "voxel_size_m": float(voxel),
                "max_correspondence_m": float(max_corr),
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "correspondence_count": int(len(result.correspondence_set)),
            }
        )
    return transform, details


def read_lidar_frames(
    infos: list[dict[str, Any]],
    *,
    lidar_root: Path,
    lidar_name: str,
    start: int,
    count: int,
    min_range_m: float,
    max_range_m: float,
    min_z_m: float,
    max_z_m: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[Path], list[int]]:
    raw_frames: list[np.ndarray] = []
    filtered_frames: list[np.ndarray] = []
    filtered_records: list[np.ndarray] = []
    paths: list[Path] = []
    timestamps: list[int] = []
    for frame_index in range(int(start), min(len(infos), int(start) + int(count))):
        info = infos[frame_index]
        path = resolve_lidar_path(info, lidar_root=lidar_root, lidar_name=lidar_name)
        if not path.exists():
            raise FileNotFoundError(path)
        cloud = read_pcd_cloud(path)
        points = cloud.xyz
        keep = lidar_point_filter_mask(
            points,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            min_z_m=min_z_m,
            max_z_m=max_z_m,
        )
        filtered = points[keep]
        raw_frames.append(points)
        filtered_frames.append(filtered)
        filtered_records.append(cloud.data[keep].copy())
        paths.append(path)
        timestamps.append(int(info["sensors"]["lidar"][lidar_name]["timestamp"]))
    return raw_frames, filtered_frames, filtered_records, paths, timestamps


def compare_scan_to_scan(args: argparse.Namespace) -> Path:
    total_start = time.perf_counter()
    infos = load_infos(args.pkl)
    selected_infos = infos[int(args.start) : int(args.start) + int(args.count)]
    if len(selected_infos) < 2:
        raise ValueError("--count must select at least 2 frames")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    _raw_frames, filtered_frames, filtered_records, pcd_paths, lidar_timestamps = read_lidar_frames(
        infos,
        lidar_root=args.lidar_root,
        lidar_name=args.lidar_name,
        start=args.start,
        count=args.count,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        min_z_m=args.min_z_m,
        max_z_m=args.max_z_m,
    )
    load_time_s = time.perf_counter() - load_start
    point_coordinate_frame = resolve_point_coordinate_frame(
        args.point_coordinate_frame,
        lidar_name=args.lidar_name,
        lidar_root=args.lidar_root,
    )
    world_poses = [
        world_from_lidar(
            info,
            lidar_name=args.lidar_name,
            point_coordinate_frame=point_coordinate_frame,
        )
        for info in selected_infos
    ]
    edge_pairs = build_pose_graph_edge_pairs(
        len(filtered_frames),
        args.pose_graph_frame_gaps,
    )
    registration_start = time.perf_counter()
    pose_graph_edges: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    all_level_rows: list[dict[str, Any]] = []
    for target_index, source_index in edge_pairs:
        source = filtered_frames[source_index]
        target = filtered_frames[target_index]
        init_prev_from_curr = np.linalg.inv(world_poses[target_index]) @ world_poses[source_index]
        before_source_in_target = transform_points(source, init_prev_from_curr)
        before_stats = nearest_neighbor_stats(
            before_source_in_target,
            target,
            max_distance_m=args.metric_max_distance_m,
        )
        icp_start = time.perf_counter()
        measured_prev_from_curr, level_rows = run_multiscale_icp(
            source_points=source,
            target_points=target,
            init_transform=init_prev_from_curr,
            registration_method=args.registration_method,
            voxel_sizes=args.icp_voxel_sizes,
            max_correspondence_distances=args.icp_max_correspondence_distances,
            max_iterations=args.icp_max_iterations,
        )
        icp_total_time_s = time.perf_counter() - icp_start
        after_source_in_target = transform_points(source, measured_prev_from_curr)
        after_stats = nearest_neighbor_stats(
            after_source_in_target,
            target,
            max_distance_m=args.metric_max_distance_m,
        )
        delta = np.linalg.inv(init_prev_from_curr) @ measured_prev_from_curr
        translation_delta_m = float(np.linalg.norm(delta[:3, 3]))
        rotation_delta_deg = float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
        )
        source_frame = int(args.start) + source_index
        target_frame = int(args.start) + target_index
        final_fitness = float(level_rows[-1]["fitness"])
        final_inlier_rmse = float(level_rows[-1]["inlier_rmse"])
        accepted = (
            final_fitness >= float(args.pose_graph_min_fitness)
            and final_inlier_rmse <= float(args.pose_graph_max_inlier_rmse_m)
        )
        if accepted:
            edge_weight = max(final_fitness, 0.05) / max(final_inlier_rmse, 0.05) ** 2
            pose_graph_edges.append(
                {
                    "target_index": target_index,
                    "source_index": source_index,
                    "target_from_source": measured_prev_from_curr,
                    "weight": edge_weight,
                }
            )
        pair_rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "frame_gap": source_index - target_index,
                "source_timestamp_ns": lidar_timestamps[source_index],
                "target_timestamp_ns": lidar_timestamps[target_index],
                "source_points": int(len(source)),
                "target_points": int(len(target)),
                "registration_method": normalize_registration_method(args.registration_method),
                "before_nn_median_m": before_stats["median_m"],
                "before_nn_rmse_m": before_stats["rmse_m"],
                "before_inlier_ratio": before_stats["inlier_ratio"],
                "after_nn_median_m": after_stats["median_m"],
                "after_nn_rmse_m": after_stats["rmse_m"],
                "after_inlier_ratio": after_stats["inlier_ratio"],
                "final_fitness": final_fitness,
                "final_inlier_rmse": final_inlier_rmse,
                "final_correspondence_count": level_rows[-1]["correspondence_count"],
                "pose_graph_edge_accepted": accepted,
                "pose_graph_edge_weight": edge_weight if accepted else 0.0,
                "delta_from_init_translation_m": translation_delta_m,
                "delta_from_init_rotation_deg": rotation_delta_deg,
                "icp_total_time_s": icp_total_time_s,
            }
        )
        for row in level_rows:
            item = dict(row)
            item["source_frame"] = source_frame
            item["target_frame"] = target_frame
            all_level_rows.append(item)
        print(
            "[pair] "
            f"{target_frame}->{source_frame} "
            f"before_rmse={before_stats['rmse_m']:.3f} "
            f"after_rmse={after_stats['rmse_m']:.3f} "
            f"fitness={level_rows[-1]['fitness']:.3f} "
            f"delta_t={translation_delta_m:.3f}m "
            f"delta_r={rotation_delta_deg:.3f}deg "
            f"accepted={accepted} "
            f"time={icp_total_time_s:.2f}s"
        )
    registration_time_s = time.perf_counter() - registration_start
    graph_start = time.perf_counter()
    optimized_world_poses, pose_graph_summary = optimize_world_pose_graph(
        world_poses,
        pose_graph_edges,
        navigation_translation_weight=args.pose_graph_navigation_translation_weight,
        navigation_rotation_weight=args.pose_graph_navigation_rotation_weight,
        rotation_residual_scale_m_per_rad=args.pose_graph_rotation_residual_scale_m_per_rad,
        max_nfev=args.pose_graph_max_nfev,
    )
    graph_optimization_time_s = time.perf_counter() - graph_start
    global_edge_rows: list[dict[str, Any]] = []
    for edge in pose_graph_edges:
        target_index = int(edge["target_index"])
        source_index = int(edge["source_index"])
        measured = np.asarray(edge["target_from_source"], dtype=np.float64)
        navigation_relative = np.linalg.inv(world_poses[target_index]) @ world_poses[source_index]
        optimized_relative = (
            np.linalg.inv(optimized_world_poses[target_index]) @ optimized_world_poses[source_index]
        )
        navigation_error = np.linalg.inv(measured) @ navigation_relative
        optimized_error = np.linalg.inv(measured) @ optimized_relative
        global_edge_rows.append(
            {
                "target_frame": int(args.start) + target_index,
                "source_frame": int(args.start) + source_index,
                "frame_gap": source_index - target_index,
                "edge_weight": float(edge["weight"]),
                "before_relative_translation_error_m": float(
                    np.linalg.norm(navigation_error[:3, 3])
                ),
                "after_relative_translation_error_m": float(
                    np.linalg.norm(optimized_error[:3, 3])
                ),
                "before_relative_rotation_error_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (np.trace(navigation_error[:3, :3]) - 1.0) / 2.0,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "after_relative_rotation_error_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (np.trace(optimized_error[:3, :3]) - 1.0) / 2.0,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
            }
        )

    before_clouds = [
        transform_points(points, transform)
        for points, transform in zip(filtered_frames, world_poses, strict=True)
    ]
    after_clouds = [
        transform_points(points, transform)
        for points, transform in zip(filtered_frames, optimized_world_poses, strict=True)
    ]
    before_record_clouds = [
        transform_point_records(records, transform)
        for records, transform in zip(filtered_records, world_poses, strict=True)
    ]
    after_record_clouds = [
        transform_point_records(records, transform)
        for records, transform in zip(filtered_records, optimized_world_poses, strict=True)
    ]
    before_fused = np.vstack(before_clouds).astype(np.float32)
    after_fused = np.vstack(after_clouds).astype(np.float32)
    before_fused_records = np.concatenate(before_record_clouds)
    after_fused_records = np.concatenate(after_record_clouds)
    before_down_indices = voxel_downsample_indices(before_fused, args.output_voxel_size_m)
    after_down_indices = voxel_downsample_indices(after_fused, args.output_voxel_size_m)
    before_fused_down = before_fused[before_down_indices]
    after_fused_down = after_fused[after_down_indices]
    before_fused_records_down = before_fused_records[before_down_indices]
    after_fused_records_down = after_fused_records[after_down_indices]

    write_start = time.perf_counter()
    write_point_records_ply_ascii(
        output_dir / "original_navigation_world_fused.ply",
        before_fused_records_down,
    )
    method = normalize_registration_method(args.registration_method)
    write_point_records_ply_ascii(
        output_dir / f"global_pose_graph_{method}_world_fused.ply",
        after_fused_records_down,
    )
    write_csv(output_dir / "pair_metrics.csv", pair_rows)
    write_csv(output_dir / "icp_level_metrics.csv", all_level_rows)
    write_csv(output_dir / "pose_graph_edge_metrics.csv", global_edge_rows)

    before_bev = bev_image(
        before_fused_down,
        resolution_m=args.bev_resolution_m,
        xlim=(args.bev_x_min_m, args.bev_x_max_m),
        ylim=(args.bev_y_min_m, args.bev_y_max_m),
    )
    after_bev = bev_image(
        after_fused_down,
        resolution_m=args.bev_resolution_m,
        xlim=(args.bev_x_min_m, args.bev_x_max_m),
        ylim=(args.bev_y_min_m, args.bev_y_max_m),
    )
    label_height = 48
    canvas = np.zeros((before_bev.shape[0] + label_height, before_bev.shape[1] * 2, 3), dtype=np.uint8)
    canvas[label_height:, : before_bev.shape[1]] = before_bev
    canvas[label_height:, before_bev.shape[1] :] = after_bev
    cv2.putText(canvas, "before: original navigation world poses", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"after: global pose graph {method.upper()} world poses", (before_bev.shape[1] + 16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(output_dir / "compare_bev.png"), canvas)
    write_time_s = time.perf_counter() - write_start

    summary = {
        "pkl": str(args.pkl),
        "lidar_root": str(args.lidar_root),
        "lidar_name": str(args.lidar_name),
        "registration_method": method,
        "point_coordinate_frame": point_coordinate_frame,
        "output_coordinate_frame": "world",
        "start": int(args.start),
        "count": int(len(filtered_frames)),
        "pcd_paths": [str(path) for path in pcd_paths],
        "original_navigation_world_fused_points": int(len(before_fused_down)),
        "global_pose_graph_world_fused_points": int(len(after_fused_down)),
        "original_navigation_world_fused_stats": fused_cloud_stats(before_fused_down, voxel_xy_m=args.metric_xy_voxel_m),
        "global_pose_graph_world_fused_stats": fused_cloud_stats(after_fused_down, voxel_xy_m=args.metric_xy_voxel_m),
        "pair_metrics_summary": summarize_pair_metrics(pair_rows),
        "pose_graph": {
            **pose_graph_summary,
            "candidate_edge_count": int(len(edge_pairs)),
            "accepted_edge_count": int(len(pose_graph_edges)),
            "frame_gaps": [int(item) for item in args.pose_graph_frame_gaps],
            "min_fitness": float(args.pose_graph_min_fitness),
            "max_inlier_rmse_m": float(args.pose_graph_max_inlier_rmse_m),
            "navigation_translation_weight": float(args.pose_graph_navigation_translation_weight),
            "navigation_rotation_weight": float(args.pose_graph_navigation_rotation_weight),
            "rotation_residual_scale_m_per_rad": float(args.pose_graph_rotation_residual_scale_m_per_rad),
            "before_relative_translation_error_median": float(
                np.median([row["before_relative_translation_error_m"] for row in global_edge_rows])
            ) if global_edge_rows else float("nan"),
            "after_relative_translation_error_median": float(
                np.median([row["after_relative_translation_error_m"] for row in global_edge_rows])
            ) if global_edge_rows else float("nan"),
            "before_relative_rotation_error_deg_median": float(
                np.median([row["before_relative_rotation_error_deg"] for row in global_edge_rows])
            ) if global_edge_rows else float("nan"),
            "after_relative_rotation_error_deg_median": float(
                np.median([row["after_relative_rotation_error_deg"] for row in global_edge_rows])
            ) if global_edge_rows else float("nan"),
        },
        "timing": {
            "load_time_s": load_time_s,
            "registration_time_s": registration_time_s,
            "pose_graph_optimization_time_s": graph_optimization_time_s,
            "write_and_qc_time_s": write_time_s,
            "total_time_s": time.perf_counter() - total_start,
        },
        "icp_voxel_sizes": [float(item) for item in args.icp_voxel_sizes],
        "icp_max_correspondence_distances": [
            float(item) for item in args.icp_max_correspondence_distances
        ],
        "outputs": {
            "original_navigation_world_fused_ply": str(output_dir / "original_navigation_world_fused.ply"),
            "global_pose_graph_world_fused_ply": str(output_dir / f"global_pose_graph_{method}_world_fused.ply"),
            "compare_bev_png": str(output_dir / "compare_bev.png"),
            "pair_metrics_csv": str(output_dir / "pair_metrics.csv"),
            "icp_level_metrics_csv": str(output_dir / "icp_level_metrics.csv"),
            "pose_graph_edge_metrics_csv": str(output_dir / "pose_graph_edge_metrics.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[done] outputs -> {output_dir}")
    return output_dir


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare before/after LiDAR multi-frame fusion using scan-to-scan ICP."
    )
    defaults = build_t3_defaults()
    parser.add_argument(
        "--pkl",
        type=Path,
        default=defaults["pkl"],
    )
    parser.add_argument(
        "--lidar-root",
        type=Path,
        default=defaults["lidar_root"],
    )
    parser.add_argument("--lidar-name", default="front_lidar")
    parser.add_argument(
        "--point-coordinate-frame",
        default="auto",
        choices=("auto", "lidar", "ego"),
        help=(
            "Coordinate frame of loaded points. auto treats perception as ego/body "
            "points and regular LiDARs as sensor-frame points."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument(
        "--registration-method",
        default="both",
        choices=("translation_only", "translation-only", "translation", "icp", "gicp", "both", "all"),
        help=(
            "scan-to-scan registration backend: translation-only, point-to-plane ICP, "
            "generalized ICP, both ICP/GICP, or all three methods saved separately."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--min-range-m", type=float, default=1.5)
    parser.add_argument("--max-range-m", type=float, default=80.0)
    parser.add_argument("--min-z-m", type=float, default=DEFAULT_MIN_Z_M)
    parser.add_argument("--max-z-m", type=float, default=DEFAULT_MAX_Z_M)
    parser.add_argument(
        "--icp-voxel-sizes",
        type=parse_float_list,
        default=parse_float_list("1.0,0.5,0.25"),
    )
    parser.add_argument(
        "--icp-max-correspondence-distances",
        type=parse_float_list,
        default=parse_float_list("2.0,1.0,0.5"),
    )
    parser.add_argument(
        "--icp-max-iterations",
        type=parse_int_list,
        default=parse_int_list("40,30,20"),
    )
    parser.add_argument(
        "--pose-graph-frame-gaps",
        type=parse_int_list,
        default=parse_int_list("1,5,10"),
        help="Frame gaps used to add overlapping scan-registration edges to the global pose graph.",
    )
    parser.add_argument(
        "--pose-graph-min-fitness",
        type=float,
        default=0.20,
        help="Discard an ICP/GICP edge when final fitness is below this value.",
    )
    parser.add_argument(
        "--pose-graph-max-inlier-rmse-m",
        type=float,
        default=0.50,
        help="Discard an ICP/GICP edge when final inlier RMSE exceeds this value.",
    )
    parser.add_argument(
        "--pose-graph-navigation-translation-weight",
        type=float,
        default=1.0,
        help="World-translation prior weight relative to scan-registration residuals.",
    )
    parser.add_argument(
        "--pose-graph-navigation-rotation-weight",
        type=float,
        default=1.0,
        help="World-rotation prior weight relative to scan-registration residuals.",
    )
    parser.add_argument(
        "--pose-graph-rotation-residual-scale-m-per-rad",
        type=float,
        default=5.0,
        help="Meters used to balance one radian of rotation residual against translation.",
    )
    parser.add_argument(
        "--pose-graph-max-nfev",
        type=int,
        default=300,
        help="Maximum robust least-squares evaluations for global world-pose refinement.",
    )
    parser.add_argument("--output-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--metric-max-distance-m", type=float, default=0.5)
    parser.add_argument("--metric-xy-voxel-m", type=float, default=0.25)
    parser.add_argument("--bev-resolution-m", type=float, default=0.05)
    parser.add_argument("--bev-x-min-m", type=float, default=-10.0)
    parser.add_argument("--bev-x-max-m", type=float, default=80.0)
    parser.add_argument("--bev-y-min-m", type=float, default=-25.0)
    parser.add_argument("--bev-y-max-m", type=float, default=25.0)
    args = parser.parse_args()
    if not (
        len(args.icp_voxel_sizes)
        == len(args.icp_max_correspondence_distances)
        == len(args.icp_max_iterations)
    ):
        raise ValueError(
            "--icp-voxel-sizes, --icp-max-correspondence-distances and "
            "--icp-max-iterations must have the same length"
        )
    return args


def main() -> None:
    args = parse_args()
    methods = registration_methods_to_run(args.registration_method)
    if len(methods) == 1:
        args.registration_method = methods[0]
        compare_scan_to_scan(args)
        return
    output_root = Path(args.output_dir)
    for method in methods:
        method_args = argparse.Namespace(**vars(args))
        method_args.registration_method = method
        method_args.output_dir = output_root / method
        compare_scan_to_scan(method_args)


if __name__ == "__main__":
    main()
