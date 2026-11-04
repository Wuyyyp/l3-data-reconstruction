from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PairRegistrationResult:
    source_index: int
    target_index: int
    init_transform: np.ndarray
    measured_transform: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_count: int
    before_nn_median_m: float
    before_nn_rmse_m: float
    after_nn_median_m: float
    after_nn_rmse_m: float


def normalize_registration_method(method: str) -> str:
    value = str(method).strip().lower()
    if value in {"translation-only", "translation_only", "translation"}:
        return "translation_only"
    if value not in {"icp", "gicp", "translation_only"}:
        raise ValueError("registration method must be 'translation_only', 'icp', or 'gicp'")
    return value


def normalize_point_coordinate_frame(value: str) -> str:
    frame = str(value).strip().lower()
    if frame not in {"auto", "lidar", "ego"}:
        raise ValueError("point coordinate frame must be 'auto', 'lidar', or 'ego'")
    return frame


def resolve_point_coordinate_frame(
    value: str,
    *,
    lidar_name: str,
    lidar_root: str | Path | None = None,
) -> str:
    frame = normalize_point_coordinate_frame(value)
    if frame != "auto":
        return frame
    if str(lidar_name) == "perception":
        return "ego"
    if lidar_root is not None and Path(lidar_root).name.endswith("_compensation"):
        return "ego"
    return "lidar"


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.astype(np.float32)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_h = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    return (matrix @ points_h.T).T[:, :3].astype(np.float32)


def transform_point_records(records: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform x/y/z in a structured point array while preserving all attributes."""

    names = records.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("point records must contain x/y/z fields")
    updated = records.copy()
    xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float32)
    transformed = transform_points(xyz, transform)
    updated["x"] = transformed[:, 0]
    updated["y"] = transformed[:, 1]
    updated["z"] = transformed[:, 2]
    return updated


def chain_pair_transforms(source_to_previous: list[np.ndarray]) -> list[np.ndarray]:
    transforms = [np.eye(4, dtype=np.float64)]
    current = np.eye(4, dtype=np.float64)
    for edge in source_to_previous:
        current = current @ np.asarray(edge, dtype=np.float64).reshape(4, 4)
        transforms.append(current.copy())
    return transforms


def relative_to_anchor_transforms(world_from_lidar: list[np.ndarray]) -> list[np.ndarray]:
    matrices = [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in world_from_lidar]
    if not matrices:
        return []
    anchor_inv = np.linalg.inv(matrices[0])
    return [anchor_inv @ matrix for matrix in matrices]


def _timestamp_to_seconds(value: int | float) -> float:
    raw = float(value)
    if abs(raw) >= 1.0e18:
        return raw / 1.0e9
    if abs(raw) >= 1.0e14:
        return raw / 1.0e6
    return raw


def interpolate_ego_world_pose(
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    time_s: float,
    ego_field: str,
) -> tuple[np.ndarray, str]:
    """Interpolate an ego-to-world pose field at a LiDAR scan reference time.

    Translation is linear and rotation is interpolated on SO(3). Times outside the
    pkl interval use the nearest available record so boundary scans remain usable.
    """
    if not infos:
        raise ValueError("at least one pkl info is required for ego interpolation")
    times = np.asarray([_timestamp_to_seconds(info["timestamp"]) for info in infos], dtype=np.float64)
    if np.any(np.diff(times) < 0.0):
        raise ValueError("pkl infos must be ordered by timestamp for ego interpolation")
    if any(ego_field not in info for info in infos):
        raise KeyError(f"requested ego pose field is missing: {ego_field}")
    target_time = float(time_s)
    if target_time <= times[0]:
        return np.asarray(infos[0][ego_field], dtype=np.float64).reshape(4, 4), "nearest_first"
    if target_time >= times[-1]:
        return np.asarray(infos[-1][ego_field], dtype=np.float64).reshape(4, 4), "nearest_last"
    right = int(np.searchsorted(times, target_time, side="right"))
    left = right - 1
    if target_time == times[left]:
        return np.asarray(infos[left][ego_field], dtype=np.float64).reshape(4, 4), "exact"
    alpha = (target_time - times[left]) / (times[right] - times[left])
    first = np.asarray(infos[left][ego_field], dtype=np.float64).reshape(4, 4)
    second = np.asarray(infos[right][ego_field], dtype=np.float64).reshape(4, 4)
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    rotations = Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]], axis=0))
    output[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    return output, "interpolated"


def _transform_from_pose_delta(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    transform[:3, 3] = vector[3:]
    return transform


def _pose_delta_from_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]
    )


def optimize_world_pose_graph(
    navigation_world_poses: list[np.ndarray],
    edges: list[dict[str, Any]],
    *,
    navigation_translation_weight: float,
    navigation_rotation_weight: float,
    rotation_residual_scale_m_per_rad: float,
    max_nfev: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Globally refine world poses from navigation priors and scan-registration edges.

    The first frame remains fixed at its navigation world pose. Later frames receive
    left-multiplied SE(3) corrections in world coordinates, avoiding the cumulative
    drift caused by chaining every scan-to-scan registration edge.
    """
    navigation = [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in navigation_world_poses]
    if not navigation:
        return [], {"success": True, "edge_count": 0, "variable_count": 0, "nfev": 0}
    if len(navigation) == 1:
        return [navigation[0].copy()], {"success": True, "edge_count": 0, "variable_count": 0, "nfev": 0}

    frame_count = len(navigation)
    for edge in edges:
        target_index = int(edge["target_index"])
        source_index = int(edge["source_index"])
        if not (0 <= target_index < frame_count and 0 <= source_index < frame_count):
            raise ValueError("pose-graph edge index is outside navigation_world_poses")
        if target_index == source_index:
            raise ValueError("pose-graph edge must connect distinct frames")

    rotation_scale = float(rotation_residual_scale_m_per_rad)
    translation_weight = float(navigation_translation_weight)
    rotation_weight = float(navigation_rotation_weight)
    if rotation_scale <= 0.0:
        raise ValueError("rotation_residual_scale_m_per_rad must be positive")
    if translation_weight < 0.0 or rotation_weight < 0.0:
        raise ValueError("navigation prior weights must be non-negative")

    def poses_from_parameters(parameters: np.ndarray) -> list[np.ndarray]:
        optimized = [navigation[0].copy()]
        values = np.asarray(parameters, dtype=np.float64).reshape(frame_count - 1, 6)
        for frame_index, delta in enumerate(values, start=1):
            optimized.append(_transform_from_pose_delta(delta) @ navigation[frame_index])
        return optimized

    def scaled_residual(transform: np.ndarray) -> np.ndarray:
        delta = _pose_delta_from_transform(transform)
        delta[:3] *= rotation_scale
        return delta

    prior_enabled = translation_weight > 0.0 or rotation_weight > 0.0

    def residuals(parameters: np.ndarray) -> np.ndarray:
        optimized = poses_from_parameters(parameters)
        pieces: list[np.ndarray] = []
        if prior_enabled:
            values = np.asarray(parameters, dtype=np.float64).reshape(frame_count - 1, 6)
            prior_scale = np.array(
                [
                    np.sqrt(rotation_weight) * rotation_scale,
                    np.sqrt(rotation_weight) * rotation_scale,
                    np.sqrt(rotation_weight) * rotation_scale,
                    np.sqrt(translation_weight),
                    np.sqrt(translation_weight),
                    np.sqrt(translation_weight),
                ],
                dtype=np.float64,
            )
            pieces.extend(delta * prior_scale for delta in values)
        for edge in edges:
            target_index = int(edge["target_index"])
            source_index = int(edge["source_index"])
            target_from_source = np.asarray(edge["target_from_source"], dtype=np.float64).reshape(4, 4)
            predicted = np.linalg.inv(optimized[target_index]) @ optimized[source_index]
            error = np.linalg.inv(target_from_source) @ predicted
            pieces.append(np.sqrt(float(edge.get("weight", 1.0))) * scaled_residual(error))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)

    variable_count = 6 * (frame_count - 1)
    residual_count = 6 * (((frame_count - 1) if prior_enabled else 0) + len(edges))
    sparsity = lil_matrix((residual_count, variable_count), dtype=np.int8)
    row_offset = 0
    if prior_enabled:
        for frame_index in range(1, frame_count):
            column_start = 6 * (frame_index - 1)
            sparsity[row_offset : row_offset + 6, column_start : column_start + 6] = 1
            row_offset += 6
    for edge in edges:
        for frame_index in {int(edge["target_index"]), int(edge["source_index"])}:
            if frame_index > 0:
                column_start = 6 * (frame_index - 1)
                sparsity[row_offset : row_offset + 6, column_start : column_start + 6] = 1
        row_offset += 6

    start = np.zeros(variable_count, dtype=np.float64)
    initial_cost = float(0.5 * np.sum(np.square(residuals(start))))
    solver = least_squares(
        residuals,
        start,
        jac_sparsity=sparsity.tocsr(),
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        verbose=0,
    )
    optimized = poses_from_parameters(solver.x)
    corrections = [np.eye(4, dtype=np.float64)]
    corrections.extend(
        _transform_from_pose_delta(values)
        for values in np.asarray(solver.x, dtype=np.float64).reshape(frame_count - 1, 6)
    )
    correction_values = np.asarray([_pose_delta_from_transform(item) for item in corrections])
    return optimized, {
        "success": bool(solver.success),
        "message": str(solver.message),
        "edge_count": int(len(edges)),
        "variable_count": int(variable_count),
        "nfev": int(solver.nfev),
        "initial_cost": initial_cost,
        "final_cost": float(solver.cost),
        "max_world_correction_translation_m": float(np.linalg.norm(correction_values[:, 3:], axis=1).max()),
        "max_world_correction_rotation_deg": float(np.degrees(np.linalg.norm(correction_values[:, :3], axis=1)).max()),
    }


def filter_lidar_points(
    points: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    min_z_m: float,
    max_z_m: float,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if values.size == 0:
        return values.copy()
    ranges = np.linalg.norm(values.astype(np.float64), axis=1)
    keep = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(min_range_m))
        & (ranges <= float(max_range_m))
        & (values[:, 2] >= float(min_z_m))
        & (values[:, 2] <= float(max_z_m))
    )
    return values[keep]


def lidar_point_filter_mask(
    points: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    min_z_m: float,
    max_z_m: float,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if values.size == 0:
        return np.zeros((len(values),), dtype=bool)
    ranges = np.linalg.norm(values.astype(np.float64), axis=1)
    return (
        np.isfinite(values).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(min_range_m))
        & (ranges <= float(max_range_m))
        & (values[:, 2] >= float(min_z_m))
        & (values[:, 2] <= float(max_z_m))
    )


def direction_adaptive_lidar_point_filter_mask(
    points: np.ndarray,
    *,
    motion_direction_ego: np.ndarray,
    min_range_m: float,
    longitudinal_max_range_m: float,
    lateral_max_range_m: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter ego-frame points with a longitudinal-to-lateral range envelope.

    The per-point ceiling is ``longitudinal + (lateral - longitudinal) *
    sin(theta)**gamma``, where theta is the horizontal angle from the vehicle
    motion direction. Both forward and rearward longitudinal rays use the
    longitudinal ceiling; lateral rays use the lateral ceiling.
    """

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if float(lateral_max_range_m) < float(longitudinal_max_range_m):
        raise ValueError("lateral_max_range_m must be at least longitudinal_max_range_m")
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive")
    if values.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

    direction = np.asarray(motion_direction_ego, dtype=np.float64).reshape(3).copy()
    direction[2] = 0.0
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm < 1.0e-9:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction /= direction_norm

    ranges = np.linalg.norm(values.astype(np.float64), axis=1)
    horizontal = values[:, :2].astype(np.float64)
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    direction_xy = direction[:2]
    dot_abs = np.ones(len(values), dtype=np.float64)
    nonzero_horizontal = horizontal_norm > 1.0e-9
    if np.any(nonzero_horizontal):
        unit_horizontal = horizontal[nonzero_horizontal] / horizontal_norm[nonzero_horizontal, None]
        dot_abs[nonzero_horizontal] = np.abs(unit_horizontal @ direction_xy)
    lateral_fraction = np.sqrt(np.clip(1.0 - np.square(dot_abs), 0.0, 1.0))
    per_point_max_range = (
        float(longitudinal_max_range_m)
        + (float(lateral_max_range_m) - float(longitudinal_max_range_m))
        * np.power(lateral_fraction, float(gamma))
    )
    keep = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(min_range_m))
        & (ranges <= per_point_max_range)
    )
    return keep, per_point_max_range.astype(np.float32)


def direction_adaptive_lidar_point_filter_mask(
    points: np.ndarray,
    *,
    motion_direction_ego: np.ndarray,
    min_range_m: float,
    longitudinal_max_range_m: float,
    lateral_max_range_m: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter ego-frame points with a longitudinal-to-lateral range envelope.

    The per-point ceiling is ``longitudinal + (lateral - longitudinal) *
    sin(theta)**gamma``, where theta is the horizontal angle from the vehicle
    motion direction. Both forward and rearward longitudinal rays use the
    longitudinal ceiling; lateral rays use the lateral ceiling.
    """

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if float(lateral_max_range_m) < float(longitudinal_max_range_m):
        raise ValueError("lateral_max_range_m must be at least longitudinal_max_range_m")
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive")
    if values.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)

    direction = np.asarray(motion_direction_ego, dtype=np.float64).reshape(3).copy()
    direction[2] = 0.0
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm < 1.0e-9:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction /= direction_norm

    ranges = np.linalg.norm(values.astype(np.float64), axis=1)
    horizontal = values[:, :2].astype(np.float64)
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    direction_xy = direction[:2]
    dot_abs = np.ones(len(values), dtype=np.float64)
    nonzero_horizontal = horizontal_norm > 1.0e-9
    if np.any(nonzero_horizontal):
        unit_horizontal = horizontal[nonzero_horizontal] / horizontal_norm[nonzero_horizontal, None]
        dot_abs[nonzero_horizontal] = np.abs(unit_horizontal @ direction_xy)
    lateral_fraction = np.sqrt(np.clip(1.0 - np.square(dot_abs), 0.0, 1.0))
    per_point_max_range = (
        float(longitudinal_max_range_m)
        + (float(lateral_max_range_m) - float(longitudinal_max_range_m))
        * np.power(lateral_fraction, float(gamma))
    )
    keep = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(min_range_m))
        & (ranges <= per_point_max_range)
    )
    return keep, per_point_max_range.astype(np.float32)


def voxel_downsample_numpy(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    voxel = float(voxel_size_m)
    if values.size == 0 or voxel <= 0:
        return values.copy()
    keys = np.floor(values / voxel).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return values[unique_indices]


def voxel_downsample_indices(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    voxel = float(voxel_size_m)
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if voxel <= 0:
        return np.arange(len(values), dtype=np.int64)
    keys = np.floor(values / voxel).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return unique_indices.astype(np.int64, copy=False)


def nearest_neighbor_stats(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_distance_m: float,
) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float32).reshape(-1, 3)
    if source.size == 0 or target.size == 0:
        return {
            "count": 0,
            "median_m": float("nan"),
            "rmse_m": float("nan"),
            "inlier_ratio": float("nan"),
        }
    tree = cKDTree(target)
    distances, _indices = tree.query(source, k=1, workers=-1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return {
            "count": 0,
            "median_m": float("nan"),
            "rmse_m": float("nan"),
            "inlier_ratio": float("nan"),
        }
    inliers = distances <= float(max_distance_m)
    selected = distances[inliers]
    if selected.size == 0:
        selected = distances
    return {
        "count": int(distances.size),
        "median_m": float(np.median(selected)),
        "rmse_m": float(np.sqrt(np.mean(selected * selected))),
        "inlier_ratio": float(np.mean(inliers)),
    }


def fused_cloud_stats(points: np.ndarray, *, voxel_xy_m: float) -> dict[str, float | int]:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if values.size == 0:
        return {
            "points": 0,
            "occupied_xy_voxels": 0,
            "mean_points_per_xy_voxel": float("nan"),
            "median_z_std_m": float("nan"),
            "mean_z_std_m": float("nan"),
        }
    keys = np.floor(values[:, :2] / float(voxel_xy_m)).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    z_sorted = values[order, 2]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys_sorted, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(keys_sorted)]
    counts = ends - starts
    z_stds = np.array(
        [float(np.std(z_sorted[start:end])) for start, end in zip(starts, ends) if end - start >= 2],
        dtype=np.float64,
    )
    return {
        "points": int(len(values)),
        "occupied_xy_voxels": int(len(starts)),
        "mean_points_per_xy_voxel": float(np.mean(counts)),
        "median_z_std_m": float(np.median(z_stds)) if z_stds.size else float("nan"),
        "mean_z_std_m": float(np.mean(z_stds)) if z_stds.size else float("nan"),
    }


def summarize_pair_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "pair_count": 0,
            "mean_icp_time_s": float("nan"),
            "median_icp_time_s": float("nan"),
            "max_icp_time_s": float("nan"),
            "max_translation_m": float("nan"),
            "max_translation_source_frame": -1,
            "max_translation_target_frame": -1,
            "max_rotation_deg": float("nan"),
            "max_rotation_source_frame": -1,
            "max_rotation_target_frame": -1,
        }
    times = np.asarray([float(row["icp_total_time_s"]) for row in rows], dtype=np.float64)
    translations = np.asarray(
        [float(row["delta_from_init_translation_m"]) for row in rows],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [float(row["delta_from_init_rotation_deg"]) for row in rows],
        dtype=np.float64,
    )
    max_t_index = int(np.nanargmax(translations))
    max_r_index = int(np.nanargmax(rotations))
    max_t_row = rows[max_t_index]
    max_r_row = rows[max_r_index]
    return {
        "pair_count": int(len(rows)),
        "mean_icp_time_s": float(np.nanmean(times)),
        "median_icp_time_s": float(np.nanmedian(times)),
        "max_icp_time_s": float(np.nanmax(times)),
        "max_translation_m": float(translations[max_t_index]),
        "max_translation_source_frame": int(max_t_row["source_frame"]),
        "max_translation_target_frame": int(max_t_row["target_frame"]),
        "max_rotation_deg": float(rotations[max_r_index]),
        "max_rotation_source_frame": int(max_r_row["source_frame"]),
        "max_rotation_target_frame": int(max_r_row["target_frame"]),
    }


def write_ply_ascii(path: Path, points: np.ndarray) -> None:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(values)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        np.savetxt(handle, values, fmt="%.6f %.6f %.6f")


def _ply_property_type(dtype: np.dtype) -> str:
    kind = dtype.kind
    size = dtype.itemsize
    if kind == "f":
        return "double" if size == 8 else "float"
    if kind == "i":
        if size <= 1:
            return "char"
        if size <= 2:
            return "short"
        return "int"
    if kind == "u":
        if size <= 1:
            return "uchar"
        if size <= 2:
            return "ushort"
        return "uint"
    raise ValueError(f"unsupported PLY property dtype: {dtype}")


def write_point_records_ply_ascii(path: Path, records: np.ndarray) -> None:
    """Write structured point records to ASCII PLY, preserving scalar attributes."""

    names = records.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("point records must contain x/y/z fields")
    field_order = ["x", "y", "z"] + [name for name in names if name not in {"x", "y", "z"}]
    scalar_fields: list[str] = []
    for name in field_order:
        dtype = records.dtype.fields[name][0]
        if dtype.shape == ():
            scalar_fields.append(name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(records)}\n")
        for name in scalar_fields:
            handle.write(f"property {_ply_property_type(records.dtype.fields[name][0])} {name}\n")
        handle.write("end_header\n")
        if len(records) == 0:
            return
        columns = [np.asarray(records[name]).reshape(-1) for name in scalar_fields]
        matrix = np.column_stack(columns)
        formats = []
        for name in scalar_fields:
            dtype = records.dtype.fields[name][0]
            formats.append("%.9g" if dtype.kind == "f" else "%d")
        np.savetxt(handle, matrix, fmt=formats)


def bev_image(
    points: np.ndarray,
    *,
    resolution_m: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    width = max(1, int(np.ceil((xlim[1] - xlim[0]) / float(resolution_m))))
    height = max(1, int(np.ceil((ylim[1] - ylim[0]) / float(resolution_m))))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if values.size == 0:
        return image
    keep = (
        (values[:, 0] >= xlim[0])
        & (values[:, 0] <= xlim[1])
        & (values[:, 1] >= ylim[0])
        & (values[:, 1] <= ylim[1])
    )
    values = values[keep]
    if values.size == 0:
        return image
    xs = np.floor((values[:, 0] - xlim[0]) / float(resolution_m)).astype(np.int32)
    ys = np.floor((values[:, 1] - ylim[0]) / float(resolution_m)).astype(np.int32)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(height - 1 - ys, 0, height - 1)
    z = values[:, 2]
    z_norm = np.clip((z - np.percentile(z, 5)) / max(np.percentile(z, 95) - np.percentile(z, 5), 1e-6), 0, 1)
    colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[ys, xs] = colors[:, 0, :]
    return image


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
