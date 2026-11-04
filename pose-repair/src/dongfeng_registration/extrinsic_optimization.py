from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera_models import CameraCalibration
from .camera_models import project_points
from .config import load_config
from .dataset import Dataset
from .evaluation import summarize_errors
from .geometry import (
    epipolar_errors,
    essential_from_relative_pose,
    fundamental_from_essential,
    ray_geometry_metrics,
    triangulate_and_reproject,
)
from .navigation import NavigationTrajectory, navigation_time_offset_ns
from .pkl_pose import (
    calibrations_from_result_interpolation,
    frame_metadata_from_image_root,
    frame_metadata_from_result_interpolation,
    navigation_from_result_interpolation,
    resolve_result_interpolation_image_root,
    resolve_result_interpolation_path,
)
from .result_interpolation_export import generate_optimized_result_interpolation_pkl
from .time_varying_rotation import (
    _torch_rotvec_to_matrix,
    _torch_skew,
    _torch_transform_inverse,
)


@dataclass
class MultiCameraExtrinsicOptimizationResult:
    output: Path
    optimized_sensor_to_car: dict[str, np.ndarray]
    before_matches: pd.DataFrame
    after_matches: pd.DataFrame
    before_summary: dict[str, float | int]
    after_summary: dict[str, float | int]
    edge_summary: pd.DataFrame
    solver_status: dict[str, object]


@dataclass
class MultiCameraAlternatingOptimizationResult:
    output: Path
    frame_rotvecs: dict[tuple[str, int], np.ndarray]
    before_matches: pd.DataFrame
    after_block_matches: pd.DataFrame
    before_summary: dict[str, float | int]
    after_block_summary: dict[str, float | int]
    block_summary: pd.DataFrame
    solver_status: dict[str, object]


@dataclass(frozen=True)
class _UniqueEndpointLayout:
    keys: tuple[tuple[str, int, int], ...]
    first_indices: np.ndarray
    second_indices: np.ndarray


@dataclass(frozen=True)
class _PaddedGroupLayout:
    indices: np.ndarray
    valid_mask: np.ndarray
    counts: np.ndarray


def _build_unique_endpoint_layout(sampled: pd.DataFrame) -> _UniqueEndpointLayout:
    required = {
        "camera1",
        "frame1",
        "timestamp1_ns",
        "camera2",
        "frame2",
        "timestamp2_ns",
    }
    missing = sorted(required.difference(sampled.columns))
    if missing:
        raise ValueError(f"sampled observations missing endpoint columns: {missing}")

    first = pd.MultiIndex.from_arrays(
        [
            sampled["camera1"].astype(str),
            sampled["frame1"].to_numpy(dtype=np.int64),
            sampled["timestamp1_ns"].to_numpy(dtype=np.int64),
        ],
        names=("camera", "frame", "timestamp_ns"),
    )
    second = pd.MultiIndex.from_arrays(
        [
            sampled["camera2"].astype(str),
            sampled["frame2"].to_numpy(dtype=np.int64),
            sampled["timestamp2_ns"].to_numpy(dtype=np.int64),
        ],
        names=first.names,
    )
    unique = first.append(second).unique().sort_values()
    keys = tuple(
        (str(camera), int(frame), int(timestamp_ns))
        for camera, frame, timestamp_ns in unique.tolist()
    )
    return _UniqueEndpointLayout(
        keys=keys,
        first_indices=np.asarray(unique.get_indexer(first), dtype=np.int64),
        second_indices=np.asarray(unique.get_indexer(second), dtype=np.int64),
    )


def _build_padded_group_layout(
    groups: Sequence[Sequence[int]],
) -> _PaddedGroupLayout:
    normalized = [tuple(int(index) for index in group) for group in groups if group]
    if not normalized:
        return _PaddedGroupLayout(
            indices=np.empty((0, 0), dtype=np.int64),
            valid_mask=np.empty((0, 0), dtype=bool),
            counts=np.empty((0,), dtype=np.int64),
        )

    counts = np.asarray([len(group) for group in normalized], dtype=np.int64)
    indices = np.zeros((len(normalized), int(counts.max())), dtype=np.int64)
    valid_mask = np.zeros_like(indices, dtype=bool)
    for row, group in enumerate(normalized):
        width = len(group)
        indices[row, :width] = group
        valid_mask[row, :width] = True
    return _PaddedGroupLayout(
        indices=indices,
        valid_mask=valid_mask,
        counts=counts,
    )


def _torch_group_means(values, indices, valid_mask, counts):
    gathered = values[indices]
    mask = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    return (gathered * mask).sum(dim=1) / counts.to(dtype=values.dtype).unsqueeze(-1)


def _torch_compose_unique_endpoint_poses(
    endpoint_body,
    *,
    body_delta,
    endpoint_body_indices,
    sensor_to_car,
    endpoint_camera_indices,
    frame_delta,
    endpoint_frame_indices,
):
    poses = endpoint_body
    if body_delta is not None:
        poses = poses @ body_delta[endpoint_body_indices]
    poses = poses @ sensor_to_car[endpoint_camera_indices]
    if frame_delta is not None:
        poses = poses @ frame_delta[endpoint_frame_indices]
    return poses


def _resolve_torch_dtype(torch, dtype_name: str):
    normalized = str(dtype_name).lower()
    mapping = {
        "float64": torch.float64,
        "double": torch.float64,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    resolved = mapping[normalized]
    canonical = "float64" if resolved == torch.float64 else "float32"
    return resolved, canonical


def _delta_transform(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = np.asarray(parameters[3:6], dtype=np.float64)
    return transform


def _apply_sensor_to_car_delta(
    sensor_to_car: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    return sensor_to_car @ _delta_transform(parameters)


def _camera_to_world_from_sensor_to_car(
    navigation: NavigationTrajectory,
    sensor_to_car: np.ndarray,
    timestamp_ns: int,
    navigation_offset_ns: int = 0,
) -> np.ndarray:
    return navigation.pose_at(int(timestamp_ns) + int(navigation_offset_ns)) @ sensor_to_car


def _rotation_delta_transform(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        raise ValueError("frame disturbance requires at least 3 rotation values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    if len(values) >= 6:
        transform[:3, 3] = values[3:6]
    return transform


def _disturbance_from_transform(transform: np.ndarray, dof: int) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    values = np.zeros(int(dof), dtype=np.float64)
    values[:3] = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
    if int(dof) >= 6:
        values[3:6] = matrix[:3, 3]
    return values


def _camera_frame_common_mode_loss_numpy(
    frame_rotvecs: dict[tuple[str, int], np.ndarray],
    *,
    dof: int,
    rotation_weight: float,
    translation_weight: float,
    min_cameras: int = 2,
) -> float:
    groups: dict[int, list[np.ndarray]] = {}
    for (_camera, frame), value in frame_rotvecs.items():
        groups.setdefault(int(frame), []).append(
            np.asarray(value, dtype=np.float64).reshape(-1)[: int(dof)]
        )
    losses: list[float] = []
    for values in groups.values():
        if len(values) < int(min_cameras):
            continue
        mean_value = np.mean(np.vstack(values), axis=0)
        loss = max(float(rotation_weight), 0.0) * float(np.mean(mean_value[:3] ** 2))
        if int(dof) == 6:
            loss += max(float(translation_weight), 0.0) * float(
                np.mean(mean_value[3:6] ** 2)
            )
        losses.append(loss)
    return float(np.mean(losses)) if losses else 0.0


def _project_time_varying_modes_numpy(
    *,
    fixed_parameters: np.ndarray,
    frame_parameters: np.ndarray,
    body_parameters: np.ndarray | None,
    frame_keys: list[tuple[str, int]] | tuple[tuple[str, int], ...],
    camera_to_index: dict[str, int],
    body_frame_to_index: dict[int, int],
    frame_mean_groups: list[list[int]] | tuple[list[int], ...],
    frame_common_groups: list[list[int]] | tuple[list[int], ...],
    frame_dof: int = 6,
    body_dof: int = 6,
    absorb_frame_common: bool = True,
    absorb_camera_long_term: bool = True,
    sensor_to_car_by_camera: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Project explainable modes out of camera frame disturbances in-place.

    This helper implements the small-angle reparameterization used during torch
    optimization.  The same-frame multi-camera mean is moved into the body
    disturbance, and each camera's long-term mean is moved into its fixed
    extrinsic delta.  In first-order vector space this preserves
    fixed + body + frame for every camera-frame variable.
    """

    frame_dof = int(frame_dof)
    body_dof = int(body_dof)
    fixed_dim = min(frame_dof, int(fixed_parameters.shape[1]))
    body_dim = (
        min(frame_dof, body_dof, int(body_parameters.shape[1]))
        if body_parameters is not None and body_parameters.size
        else 0
    )
    report: dict[str, object] = {
        "enabled": True,
        "frame_common_projection_count": 0,
        "camera_long_term_projection_count": 0,
        "frame_common_absorbed_norm_sum": 0.0,
        "camera_long_term_absorbed_norm_sum": 0.0,
    }

    if sensor_to_car_by_camera is not None:
        base_sensor_to_car = {
            str(camera): np.asarray(value, dtype=np.float64)
            for camera, value in sensor_to_car_by_camera.items()
        }
        current_sensor_to_car = {}
        for camera, camera_index in camera_to_index.items():
            if str(camera) not in base_sensor_to_car:
                continue
            current_sensor_to_car[str(camera)] = (
                base_sensor_to_car[str(camera)]
                @ _rotation_delta_transform(fixed_parameters[int(camera_index)])
            )
        frame_rotvecs = {
            (str(camera), int(frame)): np.asarray(frame_parameters[index], dtype=np.float64).copy()
            for index, (camera, frame) in enumerate(frame_keys)
        }
        body_frame_disturbances = (
            {
                int(frame): np.asarray(body_parameters[index], dtype=np.float64).copy()
                for frame, index in body_frame_to_index.items()
            }
            if body_parameters is not None and body_parameters.size
            else {}
        )

        if absorb_frame_common and body_parameters is not None and body_parameters.size:
            frame_rotvecs, body_frame_disturbances, common_report = (
                _absorb_camera_common_frame_disturbance(
                    frame_rotvecs=frame_rotvecs,
                    body_frame_disturbances=body_frame_disturbances,
                    sensor_to_car_by_camera=current_sensor_to_car,
                    min_cameras=2,
                    min_median_cos=-1.0,
                    min_positive_projection_fraction=0.0,
                    body_disturbance_dof=body_dof,
                )
            )
            common_count = int(common_report.get("frames_absorbed_rotation", 0)) + int(
                common_report.get("frames_absorbed_translation", 0)
            )
            report["frame_common_projection_count"] = int(
                common_report.get("frames_absorbed_rotation", 0)
                or common_report.get("frames_absorbed_translation", 0)
            )
            report["frame_common_absorbed_norm_sum"] = float(
                sum(
                    float(item.get("rotation_common_norm_rad", 0.0) or 0.0)
                    + float(item.get("translation_common_norm_m", 0.0) or 0.0)
                    for item in common_report.get("per_frame", [])
                    if not item.get("skip_reason")
                )
            )
            report["frame_common_projection_event_count"] = int(common_count)

        if absorb_camera_long_term:
            current_sensor_to_car, frame_rotvecs, long_report = (
                _absorb_camera_long_term_frame_disturbance(
                    frame_rotvecs=frame_rotvecs,
                    sensor_to_car_by_camera=current_sensor_to_car,
                    min_frames=1,
                    min_median_cos=-1.0,
                    min_positive_projection_fraction=0.0,
                    min_mean_to_rms=0.0,
                )
            )
            report["camera_long_term_projection_count"] = int(
                long_report.get("cameras_absorbed_rotation", 0)
                or long_report.get("cameras_absorbed_translation", 0)
            )
            report["camera_long_term_absorbed_norm_sum"] = float(
                sum(
                    float(item.get("rotation_common_norm_rad", 0.0) or 0.0)
                    + float(item.get("translation_common_norm_m", 0.0) or 0.0)
                    for item in long_report.get("per_camera", [])
                    if not item.get("skip_reason")
                )
            )

        for index, key in enumerate(frame_keys):
            frame_parameters[index, :frame_dof] = frame_rotvecs[key][:frame_dof]
        if body_parameters is not None and body_parameters.size:
            for frame, index in body_frame_to_index.items():
                if int(frame) in body_frame_disturbances:
                    body_parameters[index, :body_dof] = body_frame_disturbances[int(frame)][
                        :body_dof
                    ]
        for camera, camera_index in camera_to_index.items():
            if str(camera) not in current_sensor_to_car or str(camera) not in base_sensor_to_car:
                continue
            fixed_delta = (
                np.linalg.inv(base_sensor_to_car[str(camera)])
                @ current_sensor_to_car[str(camera)]
            )
            fixed_parameters[int(camera_index), :6] = _disturbance_from_transform(
                fixed_delta,
                6,
            )
        return report

    if absorb_frame_common and body_dim > 0:
        for indices in frame_common_groups:
            if len(indices) < 2:
                continue
            frames = {int(frame_keys[index][1]) for index in indices}
            if len(frames) != 1:
                continue
            body_index = body_frame_to_index.get(frames.pop())
            if body_index is None:
                continue
            index_array = np.asarray(indices, dtype=np.int64)
            mean_value = frame_parameters[index_array, :body_dim].mean(axis=0)
            if not np.all(np.isfinite(mean_value)):
                continue
            frame_parameters[index_array, :body_dim] -= mean_value
            body_parameters[body_index, :body_dim] += mean_value
            report["frame_common_projection_count"] = int(
                report["frame_common_projection_count"]
            ) + 1
            report["frame_common_absorbed_norm_sum"] = float(
                report["frame_common_absorbed_norm_sum"]
            ) + float(np.linalg.norm(mean_value))

    if absorb_camera_long_term and fixed_dim > 0:
        for indices in frame_mean_groups:
            if not indices:
                continue
            cameras = {str(frame_keys[index][0]) for index in indices}
            if len(cameras) != 1:
                continue
            camera_index = camera_to_index.get(cameras.pop())
            if camera_index is None:
                continue
            index_array = np.asarray(indices, dtype=np.int64)
            mean_value = frame_parameters[index_array, :fixed_dim].mean(axis=0)
            if not np.all(np.isfinite(mean_value)):
                continue
            frame_parameters[index_array, :fixed_dim] -= mean_value
            fixed_parameters[camera_index, :fixed_dim] += mean_value
            report["camera_long_term_projection_count"] = int(
                report["camera_long_term_projection_count"]
            ) + 1
            report["camera_long_term_absorbed_norm_sum"] = float(
                report["camera_long_term_absorbed_norm_sum"]
            ) + float(np.linalg.norm(mean_value))

    return report


def _vector_coherence(
    vectors: np.ndarray,
    common: np.ndarray,
    *,
    min_median_cos: float,
    min_positive_projection_fraction: float,
) -> tuple[bool, float, float]:
    values = np.asarray(vectors, dtype=np.float64)
    common = np.asarray(common, dtype=np.float64).reshape(-1)
    common_norm = float(np.linalg.norm(common))
    if values.size == 0 or common_norm <= 1e-12:
        return False, float("nan"), float("nan")
    norms = np.linalg.norm(values, axis=1)
    valid = norms > max(1e-12, float(np.percentile(norms, 20)) * 0.25)
    if not np.any(valid):
        return False, float("nan"), float("nan")
    cosines = values[valid] @ common / (np.maximum(norms[valid], 1e-12) * common_norm)
    median_cos = float(np.median(cosines))
    positive_fraction = float(np.mean(cosines > 0.0))
    passes = (
        median_cos >= float(min_median_cos)
        and positive_fraction >= float(min_positive_projection_fraction)
    )
    return bool(passes), median_cos, positive_fraction


def _absorb_camera_common_frame_disturbance(
    *,
    frame_rotvecs: dict[tuple[str, int], np.ndarray],
    body_frame_disturbances: dict[int, np.ndarray],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    min_cameras: int = 3,
    min_median_cos: float = 0.5,
    min_positive_projection_fraction: float = 0.8,
    body_disturbance_dof: int = 6,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[int, np.ndarray], dict[str, object]]:
    """Move coherent per-frame, cross-camera camera disturbances into body disturbances.

    The camera-frame disturbance for camera k at frame i is converted to an
    equivalent body-frame transform by conjugation:

        T_BC(k) * ΔT_C(k, i) * inv(T_BC(k))

    A coherent per-frame average of those body-equivalent disturbances is then
    multiplied into ΔT_B(i), while the inverse common transform is removed from
    every camera residual. This re-parameterizes the same camera-to-world poses.
    """

    new_frame_rotvecs = {
        (str(camera), int(frame)): np.asarray(value, dtype=np.float64).copy()
        for (camera, frame), value in frame_rotvecs.items()
    }
    new_body_disturbances = {
        int(frame): np.asarray(value, dtype=np.float64).copy()
        for frame, value in body_frame_disturbances.items()
    }
    frames = sorted({int(frame) for _camera, frame in new_frame_rotvecs})
    per_frame: list[dict[str, object]] = []
    frames_absorbed_rotation = 0
    frames_absorbed_translation = 0
    min_cameras = max(1, int(min_cameras))
    body_disturbance_dof = int(body_disturbance_dof)

    for frame in frames:
        keys = [
            (camera, frame)
            for camera in sorted(sensor_to_car_by_camera)
            if (camera, frame) in new_frame_rotvecs
        ]
        if len(keys) < min_cameras:
            per_frame.append(
                {
                    "frame": frame,
                    "camera_count": len(keys),
                    "absorbed_rotation": False,
                    "absorbed_translation": False,
                    "skip_reason": "not_enough_cameras",
                }
            )
            continue

        body_equivalent_vectors: list[np.ndarray] = []
        body_equivalent_transforms: dict[str, np.ndarray] = {}
        frame_dof = max(len(new_frame_rotvecs[key]) for key in keys)
        for camera, _frame in keys:
            sensor_to_car = np.asarray(sensor_to_car_by_camera[camera], dtype=np.float64)
            camera_delta = _rotation_delta_transform(new_frame_rotvecs[(camera, frame)])
            body_equivalent = sensor_to_car @ camera_delta @ np.linalg.inv(sensor_to_car)
            body_equivalent_transforms[camera] = body_equivalent
            body_equivalent_vectors.append(_disturbance_from_transform(body_equivalent, 6))

        vectors = np.vstack(body_equivalent_vectors)
        rotation_common = vectors[:, :3].mean(axis=0)
        translation_common = vectors[:, 3:6].mean(axis=0)
        absorb_rotation, rotation_median_cos, rotation_positive_fraction = _vector_coherence(
            vectors[:, :3],
            rotation_common,
            min_median_cos=min_median_cos,
            min_positive_projection_fraction=min_positive_projection_fraction,
        )
        absorb_translation = False
        translation_median_cos = float("nan")
        translation_positive_fraction = float("nan")
        if frame_dof >= 6 and body_disturbance_dof >= 6:
            (
                absorb_translation,
                translation_median_cos,
                translation_positive_fraction,
            ) = _vector_coherence(
                vectors[:, 3:6],
                translation_common,
                min_median_cos=min_median_cos,
                min_positive_projection_fraction=min_positive_projection_fraction,
            )
        common_vector = np.zeros(6, dtype=np.float64)
        if absorb_rotation:
            common_vector[:3] = rotation_common
        if absorb_translation:
            common_vector[3:6] = translation_common
        if not absorb_rotation and not absorb_translation:
            per_frame.append(
                {
                    "frame": frame,
                    "camera_count": len(keys),
                    "absorbed_rotation": False,
                    "absorbed_translation": False,
                    "rotation_common_norm_rad": float(np.linalg.norm(rotation_common)),
                    "translation_common_norm_m": float(np.linalg.norm(translation_common)),
                    "rotation_median_cos": rotation_median_cos,
                    "translation_median_cos": translation_median_cos,
                    "rotation_positive_projection_fraction": rotation_positive_fraction,
                    "translation_positive_projection_fraction": translation_positive_fraction,
                    "skip_reason": "incoherent_common_mode",
                }
            )
            continue

        common_transform = _rotation_delta_transform(common_vector)
        common_inverse = np.linalg.inv(common_transform)
        body_old = _rotation_delta_transform(
            new_body_disturbances.get(
                frame,
                np.zeros(body_disturbance_dof, dtype=np.float64),
            )
        )
        new_body_disturbances[frame] = _disturbance_from_transform(
            body_old @ common_transform,
            body_disturbance_dof,
        )
        for camera, _frame in keys:
            sensor_to_car = np.asarray(sensor_to_car_by_camera[camera], dtype=np.float64)
            residual_body = common_inverse @ body_equivalent_transforms[camera]
            residual_camera = np.linalg.inv(sensor_to_car) @ residual_body @ sensor_to_car
            new_frame_rotvecs[(camera, frame)] = _disturbance_from_transform(
                residual_camera,
                frame_dof,
            )
        if absorb_rotation:
            frames_absorbed_rotation += 1
        if absorb_translation:
            frames_absorbed_translation += 1
        per_frame.append(
            {
                "frame": frame,
                "camera_count": len(keys),
                "absorbed_rotation": bool(absorb_rotation),
                "absorbed_translation": bool(absorb_translation),
                "rotation_common_norm_rad": float(np.linalg.norm(rotation_common)),
                "rotation_common_norm_deg": float(np.rad2deg(np.linalg.norm(rotation_common))),
                "translation_common_norm_m": float(np.linalg.norm(translation_common)),
                "rotation_median_cos": rotation_median_cos,
                "translation_median_cos": translation_median_cos,
                "rotation_positive_projection_fraction": rotation_positive_fraction,
                "translation_positive_projection_fraction": translation_positive_fraction,
                "skip_reason": "",
            }
        )

    report = {
        "enabled": True,
        "min_cameras": int(min_cameras),
        "min_median_cos": float(min_median_cos),
        "min_positive_projection_fraction": float(min_positive_projection_fraction),
        "frames_considered": int(len(frames)),
        "frames_absorbed_rotation": int(frames_absorbed_rotation),
        "frames_absorbed_translation": int(frames_absorbed_translation),
        "per_frame": per_frame,
    }
    return new_frame_rotvecs, new_body_disturbances, report


def _absorb_camera_long_term_frame_disturbance(
    *,
    frame_rotvecs: dict[tuple[str, int], np.ndarray],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    min_frames: int = 30,
    min_median_cos: float = 0.5,
    min_positive_projection_fraction: float = 0.8,
    min_mean_to_rms: float = 0.35,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], np.ndarray], dict[str, object]]:
    """Move coherent per-camera, cross-time camera disturbances into fixed extrinsics.

    For each camera k, a stable cross-time component of ΔT_C(k, i) is absorbed
    into T_BC(k), while its inverse is removed from every per-frame residual:

        T_BC'(k) = T_BC(k) * ΔT_common(k)
        ΔT_C'(k, i) = inv(ΔT_common(k)) * ΔT_C(k, i)

    This preserves T_BC(k) * ΔT_C(k, i) for every frame.
    """

    new_sensor_to_car = {
        str(camera): np.asarray(value, dtype=np.float64).copy()
        for camera, value in sensor_to_car_by_camera.items()
    }
    new_frame_rotvecs = {
        (str(camera), int(frame)): np.asarray(value, dtype=np.float64).copy()
        for (camera, frame), value in frame_rotvecs.items()
    }
    per_camera: list[dict[str, object]] = []
    cameras_absorbed_rotation = 0
    cameras_absorbed_translation = 0
    min_frames = max(1, int(min_frames))

    for camera in sorted(new_sensor_to_car):
        keys = sorted(
            [key for key in new_frame_rotvecs if key[0] == camera],
            key=lambda item: item[1],
        )
        if len(keys) < min_frames:
            per_camera.append(
                {
                    "camera": camera,
                    "frame_count": len(keys),
                    "absorbed_rotation": False,
                    "absorbed_translation": False,
                    "skip_reason": "not_enough_frames",
                }
            )
            continue

        frame_dof = max(len(new_frame_rotvecs[key]) for key in keys)
        vectors = np.vstack(
            [
                _disturbance_from_transform(
                    _rotation_delta_transform(new_frame_rotvecs[key]),
                    6,
                )
                for key in keys
            ]
        )
        rotation_common = vectors[:, :3].mean(axis=0)
        rotation_norms = np.linalg.norm(vectors[:, :3], axis=1)
        rotation_rms = float(np.sqrt(np.mean(rotation_norms**2)))
        rotation_mean_to_rms = float(np.linalg.norm(rotation_common) / (rotation_rms + 1e-12))
        absorb_rotation, rotation_median_cos, rotation_positive_fraction = _vector_coherence(
            vectors[:, :3],
            rotation_common,
            min_median_cos=min_median_cos,
            min_positive_projection_fraction=min_positive_projection_fraction,
        )
        absorb_rotation = bool(
            absorb_rotation and rotation_mean_to_rms >= float(min_mean_to_rms)
        )

        translation_common = vectors[:, 3:6].mean(axis=0)
        translation_norms = np.linalg.norm(vectors[:, 3:6], axis=1)
        translation_rms = float(np.sqrt(np.mean(translation_norms**2)))
        translation_mean_to_rms = float(
            np.linalg.norm(translation_common) / (translation_rms + 1e-12)
        )
        absorb_translation = False
        translation_median_cos = float("nan")
        translation_positive_fraction = float("nan")
        if frame_dof >= 6:
            (
                absorb_translation,
                translation_median_cos,
                translation_positive_fraction,
            ) = _vector_coherence(
                vectors[:, 3:6],
                translation_common,
                min_median_cos=min_median_cos,
                min_positive_projection_fraction=min_positive_projection_fraction,
            )
            absorb_translation = bool(
                absorb_translation
                and translation_mean_to_rms >= float(min_mean_to_rms)
            )

        common_vector = np.zeros(6, dtype=np.float64)
        if absorb_rotation:
            common_vector[:3] = rotation_common
        if absorb_translation:
            common_vector[3:6] = translation_common
        if not absorb_rotation and not absorb_translation:
            per_camera.append(
                {
                    "camera": camera,
                    "frame_count": len(keys),
                    "absorbed_rotation": False,
                    "absorbed_translation": False,
                    "rotation_common_norm_rad": float(np.linalg.norm(rotation_common)),
                    "translation_common_norm_m": float(np.linalg.norm(translation_common)),
                    "rotation_mean_to_rms": rotation_mean_to_rms,
                    "translation_mean_to_rms": translation_mean_to_rms,
                    "rotation_median_cos": rotation_median_cos,
                    "translation_median_cos": translation_median_cos,
                    "rotation_positive_projection_fraction": rotation_positive_fraction,
                    "translation_positive_projection_fraction": translation_positive_fraction,
                    "skip_reason": "incoherent_or_small_long_term_mode",
                }
            )
            continue

        common_transform = _rotation_delta_transform(common_vector)
        common_inverse = np.linalg.inv(common_transform)
        new_sensor_to_car[camera] = new_sensor_to_car[camera] @ common_transform
        for key in keys:
            residual = common_inverse @ _rotation_delta_transform(new_frame_rotvecs[key])
            new_frame_rotvecs[key] = _disturbance_from_transform(residual, frame_dof)
        if absorb_rotation:
            cameras_absorbed_rotation += 1
        if absorb_translation:
            cameras_absorbed_translation += 1
        per_camera.append(
            {
                "camera": camera,
                "frame_count": len(keys),
                "absorbed_rotation": bool(absorb_rotation),
                "absorbed_translation": bool(absorb_translation),
                "rotation_common_norm_rad": float(np.linalg.norm(rotation_common)),
                "rotation_common_norm_deg": float(np.rad2deg(np.linalg.norm(rotation_common))),
                "translation_common_norm_m": float(np.linalg.norm(translation_common)),
                "rotation_mean_to_rms": rotation_mean_to_rms,
                "translation_mean_to_rms": translation_mean_to_rms,
                "rotation_median_cos": rotation_median_cos,
                "translation_median_cos": translation_median_cos,
                "rotation_positive_projection_fraction": rotation_positive_fraction,
                "translation_positive_projection_fraction": translation_positive_fraction,
                "skip_reason": "",
            }
        )

    report = {
        "enabled": True,
        "min_frames": int(min_frames),
        "min_median_cos": float(min_median_cos),
        "min_positive_projection_fraction": float(min_positive_projection_fraction),
        "min_mean_to_rms": float(min_mean_to_rms),
        "cameras_considered": int(len(new_sensor_to_car)),
        "cameras_absorbed_rotation": int(cameras_absorbed_rotation),
        "cameras_absorbed_translation": int(cameras_absorbed_translation),
        "per_camera": per_camera,
    }
    return new_sensor_to_car, new_frame_rotvecs, report


def _compose_camera_to_world(
    *,
    navigation: NavigationTrajectory,
    timestamp_ns: int,
    sensor_to_car: np.ndarray,
    navigation_offset_ns: int = 0,
    body_disturbance: np.ndarray | None = None,
    camera_disturbance: np.ndarray | None = None,
) -> np.ndarray:
    body_to_world = navigation.pose_at(int(timestamp_ns) + int(navigation_offset_ns))
    if body_disturbance is not None:
        body_to_world = body_to_world @ _rotation_delta_transform(body_disturbance)
    camera_to_body = np.asarray(sensor_to_car, dtype=np.float64)
    if camera_disturbance is not None:
        camera_to_body = camera_to_body @ _rotation_delta_transform(camera_disturbance)
    return body_to_world @ camera_to_body


def _matrix_fields(transform: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(transform, dtype=np.float64)
    return {
        f"T_cw_{row}{column}": float(matrix[row, column])
        for row in range(4)
        for column in range(4)
    }


def _frame_metadata_from_matches(matches: pd.DataFrame) -> dict[str, dict[int, dict[str, object]]]:
    normalized = _normalize_multicamera_columns(matches)
    metadata: dict[str, dict[int, dict[str, object]]] = {}
    for camera_col, frame_col, timestamp_col in (
        ("camera1", "frame1", "timestamp1_ns"),
        ("camera2", "frame2", "timestamp2_ns"),
    ):
        if not {camera_col, frame_col, timestamp_col}.issubset(normalized.columns):
            continue
        for row in normalized[[camera_col, frame_col, timestamp_col]].drop_duplicates().itertuples(
            index=False
        ):
            camera = str(getattr(row, camera_col))
            frame = int(getattr(row, frame_col))
            timestamp_ns = int(getattr(row, timestamp_col))
            metadata.setdefault(camera, {})[frame] = {
                "timestamp_ns": timestamp_ns,
                "image_filename": "",
                "image_path": "",
            }
    return metadata


def _frame_metadata_from_dataset(
    dataset_root: Path,
    cameras: list[str] | tuple[str, ...],
) -> dict[str, dict[int, dict[str, object]]]:
    dataset = Dataset(dataset_root)
    metadata: dict[str, dict[int, dict[str, object]]] = {}
    for camera in cameras:
        camera_frames: dict[int, dict[str, object]] = {}
        for frame in dataset.frames(str(camera)):
            camera_frames[int(frame.index)] = {
                "timestamp_ns": int(frame.timestamp_ns),
                "image_filename": frame.path.name,
                "image_path": str(frame.path),
            }
        metadata[str(camera)] = camera_frames
    return metadata


def _export_image_camera_to_world_csvs(
    *,
    output: Path,
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    initial_sensor_to_car: dict[str, np.ndarray],
    optimized_sensor_to_car: dict[str, np.ndarray],
    navigation_offset_ns: int,
    frame_rotvecs: dict[tuple[str, int], np.ndarray],
    body_frame_disturbances: dict[int, np.ndarray],
    frame_metadata: dict[str, dict[int, dict[str, object]]] | None = None,
) -> None:
    deliverable_root = Path(output) / "deliverables" / "image_extrinsics"
    before_root = deliverable_root / "before"
    after_root = deliverable_root / "after"
    before_root.mkdir(parents=True, exist_ok=True)
    after_root.mkdir(parents=True, exist_ok=True)

    metadata = _frame_metadata_from_matches(matches)
    if frame_metadata is not None:
        for camera, frames in frame_metadata.items():
            metadata.setdefault(str(camera), {}).update(
                {int(frame): dict(values) for frame, values in frames.items()}
            )

    before_rows: list[dict[str, object]] = []
    after_rows: list[dict[str, object]] = []
    cameras = sorted(set(initial_sensor_to_car) | set(optimized_sensor_to_car))
    for camera in cameras:
        camera_before_rows: list[dict[str, object]] = []
        camera_after_rows: list[dict[str, object]] = []
        for frame, values in sorted(metadata.get(camera, {}).items()):
            timestamp_ns = int(values["timestamp_ns"])
            navigation_timestamp_ns = timestamp_ns + int(navigation_offset_ns)
            outside_end = navigation_timestamp_ns > int(navigation.timestamps_ns[-1])
            end_extrapolation_allowed = (
                bool(getattr(navigation, "allow_end_extrapolation", False))
                and len(navigation.timestamps_ns) >= 2
                and navigation_timestamp_ns
                <= int(navigation.timestamps_ns[-1])
                + int(navigation.timestamps_ns[-1] - navigation.timestamps_ns[-2])
            )
            if (
                navigation_timestamp_ns < int(navigation.timestamps_ns[0])
                or (outside_end and not end_extrapolation_allowed)
            ):
                continue
            base_record = {
                "camera": camera,
                "frame": int(frame),
                "timestamp_ns": timestamp_ns,
                "image_filename": str(values.get("image_filename", "")),
                "image_path": str(values.get("image_path", "")),
            }
            before_transform = _compose_camera_to_world(
                navigation=navigation,
                timestamp_ns=timestamp_ns,
                sensor_to_car=initial_sensor_to_car[camera],
                navigation_offset_ns=navigation_offset_ns,
            )
            after_transform = _compose_camera_to_world(
                navigation=navigation,
                timestamp_ns=timestamp_ns,
                sensor_to_car=optimized_sensor_to_car[camera],
                navigation_offset_ns=navigation_offset_ns,
                body_disturbance=body_frame_disturbances.get(int(frame)),
                camera_disturbance=frame_rotvecs.get((camera, int(frame))),
            )
            before_record = {**base_record, **_matrix_fields(before_transform)}
            after_record = {**base_record, **_matrix_fields(after_transform)}
            before_rows.append(before_record)
            after_rows.append(after_record)
            camera_before_rows.append(before_record)
            camera_after_rows.append(after_record)
        pd.DataFrame(camera_before_rows).to_csv(
            before_root / f"{camera}_image_camera_to_world_before.csv",
            index=False,
        )
        pd.DataFrame(camera_after_rows).to_csv(
            after_root / f"{camera}_image_camera_to_world_after.csv",
            index=False,
        )
    pd.DataFrame(before_rows).to_csv(
        deliverable_root / "image_camera_to_world_before.csv",
        index=False,
    )
    pd.DataFrame(after_rows).to_csv(
        deliverable_root / "image_camera_to_world_after.csv",
        index=False,
    )


def _normalize_multicamera_columns(matches: pd.DataFrame) -> pd.DataFrame:
    result = matches.copy()
    if "camera1" not in result.columns:
        if "camera" not in result.columns:
            raise ValueError("matches must contain camera1/camera2 or camera")
        result["camera1"] = result["camera"]
    if "camera2" not in result.columns:
        if "camera" not in result.columns:
            raise ValueError("matches must contain camera1/camera2 or camera")
        result["camera2"] = result["camera"]
    if "match_scope" not in result.columns:
        result["match_scope"] = np.where(
            result["camera1"].astype(str) == result["camera2"].astype(str),
            "intra_camera",
            "inter_camera",
        )
    return result


def _navigation_coverage_mask(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    navigation_offset_ns: int,
) -> pd.Series:
    first = pd.to_numeric(matches["timestamp1_ns"], errors="coerce").to_numpy(dtype=np.float64)
    second = pd.to_numeric(matches["timestamp2_ns"], errors="coerce").to_numpy(dtype=np.float64)
    lower = float(navigation.timestamps_ns[0])
    upper = float(navigation.timestamps_ns[-1])
    offset = float(navigation_offset_ns)
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first + offset >= lower)
        & (first + offset <= upper)
        & (second + offset >= lower)
        & (second + offset <= upper)
    )
    return pd.Series(valid, index=matches.index)


def _evaluate_multicamera_group(
    group: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    navigation_offset_ns: int,
    frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    body_frame_disturbances: dict[int, np.ndarray] | None = None,
    body_pose_cache: dict[tuple[int, int], np.ndarray] | None = None,
    sensor_pose_cache: dict[tuple[str, int], np.ndarray] | None = None,
    camera_pose_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    first_camera_name = str(group["camera1"].iloc[0])
    second_camera_name = str(group["camera2"].iloc[0])
    first_calibration = calibrations[first_camera_name]
    second_calibration = calibrations[second_camera_name]
    first_frame = int(group["frame1"].iloc[0])
    second_frame = int(group["frame2"].iloc[0])

    def resolve_body_to_world(frame: int, timestamp_ns: int) -> np.ndarray:
        key = (int(frame), int(timestamp_ns))
        if body_pose_cache is not None and key in body_pose_cache:
            return body_pose_cache[key]
        body_to_world = navigation.pose_at(int(timestamp_ns) + int(navigation_offset_ns))
        if body_frame_disturbances is not None:
            body_to_world = body_to_world @ _rotation_delta_transform(
                body_frame_disturbances.get(int(frame), np.zeros(3, dtype=np.float64))
            )
        if body_pose_cache is not None:
            body_pose_cache[key] = body_to_world
        return body_to_world

    def resolve_sensor_to_car(camera: str, frame: int) -> np.ndarray:
        key = (str(camera), int(frame))
        if sensor_pose_cache is not None and key in sensor_pose_cache:
            return sensor_pose_cache[key]
        sensor_to_car = sensor_to_car_by_camera[str(camera)]
        if frame_rotvecs is not None:
            sensor_to_car = sensor_to_car @ _rotation_delta_transform(
                frame_rotvecs.get((str(camera), int(frame)), np.zeros(3, dtype=np.float64))
            )
        if sensor_pose_cache is not None:
            sensor_pose_cache[key] = sensor_to_car
        return sensor_to_car

    def resolve_camera_pose(
        camera: str,
        frame: int,
        timestamp_ns: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (str(camera), int(frame), int(timestamp_ns))
        if camera_pose_cache is not None and key in camera_pose_cache:
            return camera_pose_cache[key]
        camera_to_world = resolve_body_to_world(frame, timestamp_ns) @ resolve_sensor_to_car(
            camera,
            frame,
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        if camera_pose_cache is not None:
            camera_pose_cache[key] = (camera_to_world, world_to_camera)
        return camera_to_world, world_to_camera

    first_timestamp_ns = int(group["timestamp1_ns"].iloc[0])
    second_timestamp_ns = int(group["timestamp2_ns"].iloc[0])
    first_camera_to_world, first_world_to_camera = resolve_camera_pose(
        first_camera_name,
        first_frame,
        first_timestamp_ns,
    )
    second_camera_to_world, second_world_to_camera = resolve_camera_pose(
        second_camera_name,
        second_frame,
        second_timestamp_ns,
    )
    first_to_second = second_world_to_camera @ first_camera_to_world
    essential = essential_from_relative_pose(first_to_second)
    fundamental = fundamental_from_essential(
        essential,
        first_calibration.intrinsic,
        second_calibration.intrinsic,
    )
    result = group.copy()
    pixel_errors = epipolar_errors(
        result[["x1", "y1"]].to_numpy(dtype=np.float64),
        result[["x2", "y2"]].to_numpy(dtype=np.float64),
        fundamental,
    )
    result["nav_sampson_px2"] = pixel_errors["sampson"]
    result["symmetric_epipolar_px"] = pixel_errors["symmetric_epipolar"]

    first_normalized = result[["nx1", "ny1"]].to_numpy(dtype=np.float64)
    second_normalized = result[["nx2", "ny2"]].to_numpy(dtype=np.float64)
    triangulation = triangulate_and_reproject(
        first_normalized,
        second_normalized,
        first_world_to_camera,
        second_world_to_camera,
    )
    first_camera_points = triangulation["first_camera_points"]
    second_camera_points = triangulation["second_camera_points"]
    first_pixels = project_points(first_camera_points, first_calibration)
    second_pixels = project_points(second_camera_points, second_calibration)
    first_reprojection = np.linalg.norm(
        first_pixels - result[["x1", "y1"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    second_reprojection = np.linalg.norm(
        second_pixels - result[["x2", "y2"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    ray_metrics = ray_geometry_metrics(
        first_normalized,
        second_normalized,
        first_camera_to_world,
        second_camera_to_world,
    )
    result["triangulation_valid_depth"] = triangulation["valid_depth"]
    result["triangulation_angle_deg"] = ray_metrics["triangulation_angle_deg"]
    result["ray_closest_distance_m"] = ray_metrics["ray_closest_distance_m"]
    result["reprojection1_px"] = first_reprojection
    result["reprojection2_px"] = second_reprojection
    result["reprojection_rms_px"] = np.sqrt(
        (first_reprojection**2 + second_reprojection**2) / 2.0
    )
    result[["reprojected1_x", "reprojected1_y"]] = first_pixels
    result[["reprojected2_x", "reprojected2_y"]] = second_pixels
    return result


def _evaluate_multicamera_matches_legacy(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    navigation_offset_ns: int = 0,
    frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    body_frame_disturbances: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    normalized = _normalize_multicamera_columns(matches)
    groups: list[pd.DataFrame] = []
    keys = ["camera1", "camera2", "timestamp1_ns", "timestamp2_ns"]
    body_pose_cache: dict[tuple[int, int], np.ndarray] = {}
    sensor_pose_cache: dict[tuple[str, int], np.ndarray] = {}
    camera_pose_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for _, group in normalized.groupby(keys, sort=False):
        groups.append(
            _evaluate_multicamera_group(
                group,
                navigation,
                calibrations,
                sensor_to_car_by_camera,
                navigation_offset_ns,
                frame_rotvecs,
                body_frame_disturbances,
                body_pose_cache,
                sensor_pose_cache,
                camera_pose_cache,
            )
        )
    result = pd.concat(groups, ignore_index=True) if groups else normalized.iloc[0:0].copy()
    if frame_rotvecs is not None and not result.empty:
        first_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                frame_rotvecs.get((str(row.camera1), int(row.frame1)), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        second_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                frame_rotvecs.get((str(row.camera2), int(row.frame2)), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        result["after_frame_delta1_norm_deg"] = first_norms
        result["after_frame_delta2_norm_deg"] = second_norms
        result["after_frame_delta_norm_deg"] = np.maximum(first_norms, second_norms)
    if body_frame_disturbances is not None and not result.empty:
        first_body_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                body_frame_disturbances.get(int(row.frame1), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        second_body_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                body_frame_disturbances.get(int(row.frame2), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        result["after_body_delta1_norm_deg"] = first_body_norms
        result["after_body_delta2_norm_deg"] = second_body_norms
        result["after_body_delta_norm_deg"] = np.maximum(
            first_body_norms, second_body_norms
        )
    return result


def _torch_device_available(device: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    requested = str(device).lower()
    if requested.startswith("cuda"):
        return bool(torch.cuda.is_available())
    return True


def _torch_sampson_sqrt_from_essential(essential, first_points, second_points):
    import torch

    lines_second = (essential @ first_points[..., None]).squeeze(-1)
    lines_first = (essential.transpose(-1, -2) @ second_points[..., None]).squeeze(-1)
    residual = torch.sum(second_points * lines_second, dim=1)
    denom = (
        torch.sum(lines_first[:, :2] ** 2, dim=1)
        + torch.sum(lines_second[:, :2] ** 2, dim=1)
    ).clamp_min(1e-12)
    return torch.sqrt((residual**2 / denom).clamp_min(0.0) + 1e-12)


def _can_use_torch_multicamera_evaluation(
    calibrations: dict[str, CameraCalibration],
) -> bool:
    return all(bool(calibration.images_undistorted) for calibration in calibrations.values())


def _resolve_multicamera_evaluation_backend(
    evaluation_backend: str,
    *,
    evaluation_device: str,
    calibrations: dict[str, CameraCalibration],
) -> str:
    requested = str(evaluation_backend).lower()
    if requested not in {"auto", "legacy", "torch"}:
        raise ValueError(f"unsupported evaluation_backend: {evaluation_backend}")
    torch_supported = _can_use_torch_multicamera_evaluation(
        calibrations
    ) and _torch_device_available(evaluation_device)
    if requested == "auto":
        return (
            "torch"
            if str(evaluation_device).lower().startswith("cuda") and torch_supported
            else "legacy"
        )
    if requested == "torch" and not torch_supported:
        raise ValueError(
            "torch evaluation backend requires available torch device and undistorted images"
        )
    return requested


def _torch_batch_project_undistorted_pixels(
    points_camera,
    intrinsic,
):
    normalized = points_camera[:, :2] / points_camera[:, 2:3].clamp_min(1e-12)
    fx = intrinsic[:, 0, 0]
    fy = intrinsic[:, 1, 1]
    cx = intrinsic[:, 0, 2]
    cy = intrinsic[:, 1, 2]
    x = normalized[:, 0] * fx + cx
    y = normalized[:, 1] * fy + cy
    return normalized, __import__("torch").stack((x, y), dim=1)


def _triangulate_points_dlt_torch(first_points, second_points, p1, p2):
    import torch

    x1 = first_points[:, 0]
    y1 = first_points[:, 1]
    x2 = second_points[:, 0]
    y2 = second_points[:, 1]
    a0 = x1[:, None] * p1[:, 2, :] - p1[:, 0, :]
    a1 = y1[:, None] * p1[:, 2, :] - p1[:, 1, :]
    a2 = x2[:, None] * p2[:, 2, :] - p2[:, 0, :]
    a3 = y2[:, None] * p2[:, 2, :] - p2[:, 1, :]
    a = torch.stack((a0, a1, a2, a3), dim=1)
    _u, _s, vh = torch.linalg.svd(a, full_matrices=False)
    homogeneous = vh[:, -1, :]
    scale = homogeneous[:, 3:4]
    eps = torch.finfo(homogeneous.dtype).eps
    world = homogeneous[:, :3] / torch.where(scale.abs() > eps, scale, torch.full_like(scale, eps))
    return world


def _ray_geometry_metrics_torch(first_points, second_points, first_camera_to_world, second_camera_to_world):
    import torch

    first_dirs_camera = torch.cat(
        (first_points, torch.ones((len(first_points), 1), dtype=first_points.dtype, device=first_points.device)),
        dim=1,
    )
    second_dirs_camera = torch.cat(
        (second_points, torch.ones((len(second_points), 1), dtype=second_points.dtype, device=second_points.device)),
        dim=1,
    )
    first_dirs_camera = first_dirs_camera / first_dirs_camera.norm(dim=1, keepdim=True).clamp_min(1e-12)
    second_dirs_camera = second_dirs_camera / second_dirs_camera.norm(dim=1, keepdim=True).clamp_min(1e-12)
    first_dirs_world = torch.bmm(first_camera_to_world[:, :3, :3], first_dirs_camera.unsqueeze(-1)).squeeze(-1)
    second_dirs_world = torch.bmm(second_camera_to_world[:, :3, :3], second_dirs_camera.unsqueeze(-1)).squeeze(-1)
    first_dirs_world = first_dirs_world / first_dirs_world.norm(dim=1, keepdim=True).clamp_min(1e-12)
    second_dirs_world = second_dirs_world / second_dirs_world.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dot = (first_dirs_world * second_dirs_world).sum(dim=1).clamp(-1.0, 1.0)
    triangulation_angle_deg = torch.rad2deg(torch.acos(dot))
    first_center = first_camera_to_world[:, :3, 3]
    second_center = second_camera_to_world[:, :3, 3]
    baseline = second_center - first_center
    cross = torch.cross(first_dirs_world, second_dirs_world, dim=1)
    cross_norm = cross.norm(dim=1)
    baseline_dot_cross = (cross * baseline).sum(dim=1).abs()
    eps = torch.finfo(first_points.dtype).eps
    ray_gap = torch.empty_like(cross_norm)
    non_parallel = cross_norm > eps
    ray_gap[non_parallel] = baseline_dot_cross[non_parallel] / cross_norm[non_parallel]
    if torch.any(~non_parallel):
        repeated_baseline = baseline[~non_parallel]
        ray_gap[~non_parallel] = torch.cross(repeated_baseline, first_dirs_world[~non_parallel], dim=1).norm(dim=1)
    return {
        "triangulation_angle_deg": triangulation_angle_deg,
        "ray_closest_distance_m": ray_gap,
    }


def _evaluate_multicamera_matches_torch(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    navigation_offset_ns: int = 0,
    frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    body_frame_disturbances: dict[int, np.ndarray] | None = None,
    *,
    evaluation_device: str = "cpu",
    evaluation_batch_size: int = 50_000,
) -> pd.DataFrame:
    import torch

    normalized = _normalize_multicamera_columns(matches).reset_index(drop=True)
    if normalized.empty:
        return normalized.copy()
    torch_device = torch.device(
        evaluation_device if _torch_device_available(evaluation_device) else "cpu"
    )
    dtype = torch.float64
    camera_names = sorted(set(normalized["camera1"].astype(str)) | set(normalized["camera2"].astype(str)))
    camera_to_index = {camera: index for index, camera in enumerate(camera_names)}
    intrinsic = torch.as_tensor(
        np.stack([calibrations[camera].intrinsic for camera in camera_names]),
        dtype=dtype,
        device=torch_device,
    )
    sensor_to_car = torch.as_tensor(
        np.stack([sensor_to_car_by_camera[camera] for camera in camera_names]),
        dtype=dtype,
        device=torch_device,
    )

    first_camera_idx_np = np.array(
        [camera_to_index[str(value)] for value in normalized["camera1"].to_numpy()],
        dtype=np.int64,
    )
    second_camera_idx_np = np.array(
        [camera_to_index[str(value)] for value in normalized["camera2"].to_numpy()],
        dtype=np.int64,
    )

    first_pose_map: dict[tuple[int, int], np.ndarray] = {}
    second_pose_map: dict[tuple[int, int], np.ndarray] = {}
    first_frame_keys = list(
        zip(
            normalized["camera1"].astype(str).map(camera_to_index).to_numpy(dtype=np.int64),
            normalized["frame1"].to_numpy(dtype=np.int64),
            normalized["timestamp1_ns"].to_numpy(dtype=np.int64),
        )
    )
    second_frame_keys = list(
        zip(
            normalized["camera2"].astype(str).map(camera_to_index).to_numpy(dtype=np.int64),
            normalized["frame2"].to_numpy(dtype=np.int64),
            normalized["timestamp2_ns"].to_numpy(dtype=np.int64),
        )
    )
    unique_body_timestamps = sorted(
        set(normalized["timestamp1_ns"].to_numpy(dtype=np.int64).tolist())
        | set(normalized["timestamp2_ns"].to_numpy(dtype=np.int64).tolist())
    )
    body_pose_by_timestamp = {
        int(timestamp): navigation.pose_at(int(timestamp) + int(navigation_offset_ns))
        for timestamp in unique_body_timestamps
    }

    def _resolve_frame_pose(camera_idx: int, frame: int, timestamp_ns: int) -> np.ndarray:
        body_pose = np.asarray(body_pose_by_timestamp[int(timestamp_ns)], dtype=np.float64)
        if body_frame_disturbances is not None:
            body_pose = body_pose @ _rotation_delta_transform(
                body_frame_disturbances.get(int(frame), np.zeros(3, dtype=np.float64))
            )
        sensor_pose = np.asarray(sensor_to_car_by_camera[camera_names[int(camera_idx)]], dtype=np.float64)
        if frame_rotvecs is not None:
            sensor_pose = sensor_pose @ _rotation_delta_transform(
                frame_rotvecs.get((camera_names[int(camera_idx)], int(frame)), np.zeros(3, dtype=np.float64))
            )
        return body_pose @ sensor_pose

    for camera_idx, frame, timestamp_ns in set(first_frame_keys):
        first_pose_map[(int(camera_idx), int(frame), int(timestamp_ns))] = _resolve_frame_pose(
            int(camera_idx), int(frame), int(timestamp_ns)
        )
    for camera_idx, frame, timestamp_ns in set(second_frame_keys):
        second_pose_map[(int(camera_idx), int(frame), int(timestamp_ns))] = _resolve_frame_pose(
            int(camera_idx), int(frame), int(timestamp_ns)
        )

    first_points_all = normalized[["x1", "y1"]].to_numpy(dtype=np.float64)
    second_points_all = normalized[["x2", "y2"]].to_numpy(dtype=np.float64)
    first_normalized_all = normalized[["nx1", "ny1"]].to_numpy(dtype=np.float64)
    second_normalized_all = normalized[["nx2", "ny2"]].to_numpy(dtype=np.float64)

    result = normalized.copy()
    result["nav_sampson_px2"] = np.nan
    result["symmetric_epipolar_px"] = np.nan
    result["triangulation_valid_depth"] = False
    result["triangulation_angle_deg"] = np.nan
    result["ray_closest_distance_m"] = np.nan
    result["reprojection1_px"] = np.nan
    result["reprojection2_px"] = np.nan
    result["reprojection_rms_px"] = np.nan
    result["reprojected1_x"] = np.nan
    result["reprojected1_y"] = np.nan
    result["reprojected2_x"] = np.nan
    result["reprojected2_y"] = np.nan

    batch_size = max(1, int(evaluation_batch_size))
    for start in range(0, len(normalized), batch_size):
        stop = min(len(normalized), start + batch_size)
        batch_slice = slice(start, stop)
        first_camera_idx = torch.as_tensor(first_camera_idx_np[batch_slice], dtype=torch.long, device=torch_device)
        second_camera_idx = torch.as_tensor(second_camera_idx_np[batch_slice], dtype=torch.long, device=torch_device)
        first_camera_to_world = torch.as_tensor(
            np.stack(
                [
                    first_pose_map[(int(camera_idx), int(frame), int(timestamp_ns))]
                    for camera_idx, frame, timestamp_ns in first_frame_keys[batch_slice]
                ]
            ),
            dtype=dtype,
            device=torch_device,
        )
        second_camera_to_world = torch.as_tensor(
            np.stack(
                [
                    second_pose_map[(int(camera_idx), int(frame), int(timestamp_ns))]
                    for camera_idx, frame, timestamp_ns in second_frame_keys[batch_slice]
                ]
            ),
            dtype=dtype,
            device=torch_device,
        )
        first_world_to_camera = _torch_transform_inverse(first_camera_to_world)
        second_world_to_camera = _torch_transform_inverse(second_camera_to_world)
        first_to_second = second_world_to_camera @ first_camera_to_world
        essential = _torch_skew(first_to_second[:, :3, 3]) @ first_to_second[:, :3, :3]
        k1_inv = torch.linalg.inv(intrinsic[first_camera_idx])
        k2_inv = torch.linalg.inv(intrinsic[second_camera_idx])
        fundamental = k2_inv.transpose(-1, -2) @ essential @ k1_inv

        first_points = torch.as_tensor(np.array(first_points_all[batch_slice], copy=True), dtype=dtype, device=torch_device)
        second_points = torch.as_tensor(np.array(second_points_all[batch_slice], copy=True), dtype=dtype, device=torch_device)
        first_h = torch.cat((first_points, torch.ones((len(first_points), 1), dtype=dtype, device=torch_device)), dim=1)
        second_h = torch.cat((second_points, torch.ones((len(second_points), 1), dtype=dtype, device=torch_device)), dim=1)
        lines_second = torch.bmm(fundamental, first_h.unsqueeze(-1)).squeeze(-1)
        lines_first = torch.bmm(fundamental.transpose(-1, -2), second_h.unsqueeze(-1)).squeeze(-1)
        residual = (second_h * lines_second).sum(dim=1)
        norm_second = lines_second[:, :2].norm(dim=1).clamp_min(1e-12)
        norm_first = lines_first[:, :2].norm(dim=1).clamp_min(1e-12)
        symmetric = torch.sqrt((residual.abs() / norm_first) ** 2 + (residual.abs() / norm_second) ** 2)
        sampson = residual.square() / (norm_first.square() + norm_second.square()).clamp_min(1e-12)

        first_normalized = torch.as_tensor(np.array(first_normalized_all[batch_slice], copy=True), dtype=dtype, device=torch_device)
        second_normalized = torch.as_tensor(np.array(second_normalized_all[batch_slice], copy=True), dtype=dtype, device=torch_device)
        world_points = _triangulate_points_dlt_torch(
            first_normalized,
            second_normalized,
            first_world_to_camera[:, :3, :],
            second_world_to_camera[:, :3, :],
        )
        world_h = torch.cat((world_points, torch.ones((len(world_points), 1), dtype=dtype, device=torch_device)), dim=1)
        first_camera_points = torch.bmm(first_world_to_camera, world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        second_camera_points = torch.bmm(second_world_to_camera, world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        _first_reprojected_norm, first_pixels = _torch_batch_project_undistorted_pixels(
            first_camera_points, intrinsic[first_camera_idx]
        )
        _second_reprojected_norm, second_pixels = _torch_batch_project_undistorted_pixels(
            second_camera_points, intrinsic[second_camera_idx]
        )
        first_reprojection = (first_pixels - first_points).norm(dim=1)
        second_reprojection = (second_pixels - second_points).norm(dim=1)
        ray_metrics = _ray_geometry_metrics_torch(
            first_normalized,
            second_normalized,
            first_camera_to_world,
            second_camera_to_world,
        )
        valid_depth = (first_camera_points[:, 2] > 0.0) & (second_camera_points[:, 2] > 0.0)

        result.loc[result.index[batch_slice], "nav_sampson_px2"] = sampson.detach().cpu().numpy()
        result.loc[result.index[batch_slice], "symmetric_epipolar_px"] = symmetric.detach().cpu().numpy()
        result.loc[result.index[batch_slice], "triangulation_valid_depth"] = valid_depth.detach().cpu().numpy()
        result.loc[result.index[batch_slice], "triangulation_angle_deg"] = (
            ray_metrics["triangulation_angle_deg"].detach().cpu().numpy()
        )
        result.loc[result.index[batch_slice], "ray_closest_distance_m"] = (
            ray_metrics["ray_closest_distance_m"].detach().cpu().numpy()
        )
        reproj1_np = first_reprojection.detach().cpu().numpy()
        reproj2_np = second_reprojection.detach().cpu().numpy()
        result.loc[result.index[batch_slice], "reprojection1_px"] = reproj1_np
        result.loc[result.index[batch_slice], "reprojection2_px"] = reproj2_np
        result.loc[result.index[batch_slice], "reprojection_rms_px"] = np.sqrt(
            (reproj1_np**2 + reproj2_np**2) / 2.0
        )
        first_pixels_np = first_pixels.detach().cpu().numpy()
        second_pixels_np = second_pixels.detach().cpu().numpy()
        result.loc[result.index[batch_slice], "reprojected1_x"] = first_pixels_np[:, 0]
        result.loc[result.index[batch_slice], "reprojected1_y"] = first_pixels_np[:, 1]
        result.loc[result.index[batch_slice], "reprojected2_x"] = second_pixels_np[:, 0]
        result.loc[result.index[batch_slice], "reprojected2_y"] = second_pixels_np[:, 1]

    if frame_rotvecs is not None and not result.empty:
        first_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                frame_rotvecs.get((str(row.camera1), int(row.frame1)), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        second_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                frame_rotvecs.get((str(row.camera2), int(row.frame2)), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        result["after_frame_delta1_norm_deg"] = first_norms
        result["after_frame_delta2_norm_deg"] = second_norms
        result["after_frame_delta_norm_deg"] = np.maximum(first_norms, second_norms)
    if body_frame_disturbances is not None and not result.empty:
        first_body_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                body_frame_disturbances.get(int(row.frame1), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        second_body_norms = [
            float(np.rad2deg(np.linalg.norm(np.asarray(
                body_frame_disturbances.get(int(row.frame2), np.zeros(3)),
                dtype=np.float64,
            )[:3])))
            for row in result.itertuples(index=False)
        ]
        result["after_body_delta1_norm_deg"] = first_body_norms
        result["after_body_delta2_norm_deg"] = second_body_norms
        result["after_body_delta_norm_deg"] = np.maximum(first_body_norms, second_body_norms)
    return result


def _evaluate_multicamera_matches(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    navigation_offset_ns: int = 0,
    frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    body_frame_disturbances: dict[int, np.ndarray] | None = None,
    *,
    evaluation_backend: str = "auto",
    evaluation_device: str = "cpu",
    evaluation_batch_size: int = 50_000,
) -> pd.DataFrame:
    backend = _resolve_multicamera_evaluation_backend(
        evaluation_backend,
        evaluation_device=evaluation_device,
        calibrations=calibrations,
    )
    if backend == "legacy":
        return _evaluate_multicamera_matches_legacy(
            matches,
            navigation,
            calibrations,
            sensor_to_car_by_camera,
            navigation_offset_ns=navigation_offset_ns,
            frame_rotvecs=frame_rotvecs,
            body_frame_disturbances=body_frame_disturbances,
        )
    if backend == "torch":
        evaluated = _evaluate_multicamera_matches_torch(
            matches,
            navigation,
            calibrations,
            sensor_to_car_by_camera,
            navigation_offset_ns=navigation_offset_ns,
            frame_rotvecs=frame_rotvecs,
            body_frame_disturbances=body_frame_disturbances,
            evaluation_device=evaluation_device,
            evaluation_batch_size=evaluation_batch_size,
        )
        evaluated.attrs["legacy_refinement_count"] = 0
        if str(evaluation_backend).lower() != "auto" or evaluated.empty:
            return evaluated

        reprojection = evaluated["reprojection_rms_px"].to_numpy(dtype=np.float64)
        refine_mask = ~np.isfinite(reprojection) | (reprojection > 10.0)
        positions = np.flatnonzero(refine_mask)
        if not len(positions):
            return evaluated

        normalized_source = _normalize_multicamera_columns(matches).reset_index(drop=True)
        refined = _evaluate_multicamera_matches_legacy(
            normalized_source.iloc[positions].reset_index(drop=True),
            navigation,
            calibrations,
            sensor_to_car_by_camera,
            navigation_offset_ns=navigation_offset_ns,
            frame_rotvecs=frame_rotvecs,
            body_frame_disturbances=body_frame_disturbances,
        )
        evaluation_columns = [
            "nav_sampson_px2",
            "symmetric_epipolar_px",
            "triangulation_valid_depth",
            "triangulation_angle_deg",
            "ray_closest_distance_m",
            "reprojection1_px",
            "reprojection2_px",
            "reprojection_rms_px",
            "reprojected1_x",
            "reprojected1_y",
            "reprojected2_x",
            "reprojected2_y",
            "after_frame_delta1_norm_deg",
            "after_frame_delta2_norm_deg",
            "after_frame_delta_norm_deg",
            "after_body_delta1_norm_deg",
            "after_body_delta2_norm_deg",
            "after_body_delta_norm_deg",
        ]
        for column in evaluation_columns:
            if column not in refined.columns or column not in evaluated.columns:
                continue
            evaluated.loc[evaluated.index[positions], column] = refined[
                column
            ].to_numpy()
        evaluated.attrs["legacy_refinement_count"] = int(len(positions))
        return evaluated
    raise AssertionError("unreachable evaluation backend state")


def _sample_multicamera_matches(
    matches: pd.DataFrame,
    max_observations: int | None,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    if max_observations is None or len(matches) <= max_observations:
        return matches.reset_index(drop=True)
    group_keys = ["match_scope", "camera1", "camera2", "frame_offset"]
    group_sizes = matches.groupby(group_keys, sort=False).size()
    proportions = (group_sizes / group_sizes.sum()).to_dict()
    pieces = []
    remaining = int(max_observations)
    grouped = list(matches.groupby(group_keys, sort=False))
    for index, (key, group) in enumerate(grouped):
        if remaining <= 0:
            break
        if index == len(grouped) - 1:
            take = min(len(group), remaining)
        else:
            take = min(
                len(group),
                remaining,
                max(1, int(round(max_observations * proportions[key]))),
            )
        remaining -= take
        group_seed_material = "|".join(str(value) for value in key)
        group_seed_digest = hashlib.blake2s(
            group_seed_material.encode("utf-8"),
            digest_size=4,
            person=b"dfseed",
        ).digest()
        group_seed = (int(seed) + int.from_bytes(group_seed_digest, "little")) % (2**32)
        pieces.append(group.sample(n=take, random_state=group_seed))
    return pd.concat(pieces, ignore_index=True) if pieces else matches.iloc[0:0].copy()


def _sample_matches_per_image_pair(
    matches: pd.DataFrame,
    *,
    max_matches_per_pair: int | None,
    seed: int = 0,
    strategy: str = "spatial_grid_random",
    grid_cols: int = 3,
    grid_rows: int = 3,
) -> pd.DataFrame:
    if (
        max_matches_per_pair is None
        or int(max_matches_per_pair) <= 0
        or len(matches) <= int(max_matches_per_pair)
    ):
        return matches.reset_index(drop=True)
    group_keys = [
        "match_scope",
        "camera1",
        "camera2",
        "pair_relation",
        "baseline_target_m",
        "frame1",
        "frame2",
        "timestamp1_ns",
        "timestamp2_ns",
        "frame_offset",
    ]
    available_keys = [key for key in group_keys if key in matches.columns]
    if not available_keys:
        if str(strategy) == "spatial_grid_random":
            return _sample_image_pair_spatial_grid_random(
                matches,
                max_matches=int(max_matches_per_pair),
                seed=int(seed),
                grid_cols=int(grid_cols),
                grid_rows=int(grid_rows),
            ).reset_index(drop=True)
        if str(strategy) != "random":
            raise ValueError(f"unsupported pair sample strategy: {strategy}")
        return matches.sample(
            n=min(len(matches), int(max_matches_per_pair)),
            random_state=int(seed),
        ).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for index, (_key, group) in enumerate(matches.groupby(available_keys, dropna=False, sort=False)):
        if len(group) <= int(max_matches_per_pair):
            pieces.append(group)
            continue
        group_seed = (int(seed) + int(index)) % (2**32)
        if str(strategy) == "random":
            pieces.append(
                group.sample(
                    n=int(max_matches_per_pair),
                    random_state=group_seed,
                )
            )
        elif str(strategy) == "spatial_grid_random":
            pieces.append(
                _sample_image_pair_spatial_grid_random(
                    group,
                    max_matches=int(max_matches_per_pair),
                    seed=group_seed,
                    grid_cols=int(grid_cols),
                    grid_rows=int(grid_rows),
                )
            )
        else:
            raise ValueError(f"unsupported pair sample strategy: {strategy}")
    return pd.concat(pieces, ignore_index=True) if pieces else matches.iloc[0:0].copy()


def _sample_image_pair_spatial_grid_random(
    group: pd.DataFrame,
    *,
    max_matches: int,
    seed: int,
    grid_cols: int,
    grid_rows: int,
) -> pd.DataFrame:
    """Randomly sample one image pair while covering grids in both image views."""

    required = {"x1", "y1", "x2", "y2"}
    if (
        max_matches <= 0
        or len(group) <= max_matches
        or not required <= set(group.columns)
        or grid_cols <= 0
        or grid_rows <= 0
    ):
        return group.sample(
            n=min(len(group), max_matches),
            random_state=int(seed),
        )
    rng = np.random.default_rng(int(seed))
    selected: list[object] = []
    per_view_target = int(np.ceil(max_matches / 2))
    for x_name, y_name in (("x1", "y1"), ("x2", "y2")):
        selected.extend(
            _sample_from_view_grid(
                group,
                x_name=x_name,
                y_name=y_name,
                target=per_view_target,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                rng=rng,
            )
        )
    selected = list(dict.fromkeys(selected))
    if len(selected) > max_matches:
        keep_positions = rng.choice(len(selected), size=max_matches, replace=False)
        selected = [selected[int(position)] for position in keep_positions]
    if len(selected) < max_matches:
        selected_set = set(selected)
        remaining = [index for index in group.index if index not in selected_set]
        if remaining:
            take = min(max_matches - len(selected), len(remaining))
            fill_positions = rng.choice(len(remaining), size=take, replace=False)
            selected.extend(remaining[int(position)] for position in fill_positions)
    return group.loc[selected]


def _sample_from_view_grid(
    group: pd.DataFrame,
    *,
    x_name: str,
    y_name: str,
    target: int,
    grid_cols: int,
    grid_rows: int,
    rng: np.random.Generator,
) -> list[object]:
    points = group[[x_name, y_name]].to_numpy(dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any() or target <= 0:
        return []
    valid_indices = list(group.index[finite])
    valid_points = points[finite]
    x_min, y_min = np.min(valid_points, axis=0)
    x_max, y_max = np.max(valid_points, axis=0)
    if x_max <= x_min:
        cols = np.zeros(len(valid_points), dtype=np.int64)
    else:
        cols = np.floor((valid_points[:, 0] - x_min) / (x_max - x_min) * grid_cols).astype(np.int64)
        cols = np.clip(cols, 0, grid_cols - 1)
    if y_max <= y_min:
        rows = np.zeros(len(valid_points), dtype=np.int64)
    else:
        rows = np.floor((valid_points[:, 1] - y_min) / (y_max - y_min) * grid_rows).astype(np.int64)
        rows = np.clip(rows, 0, grid_rows - 1)
    by_cell: dict[tuple[int, int], list[object]] = {}
    for row, col, original_index in zip(rows, cols, valid_indices):
        by_cell.setdefault((int(row), int(col)), []).append(original_index)
    cells = list(by_cell)
    rng.shuffle(cells)
    quota = max(1, int(np.ceil(target / max(1, len(cells)))))
    selected: list[object] = []
    for cell in cells:
        candidates = by_cell[cell]
        take = min(quota, len(candidates), max(0, target - len(selected)))
        if take <= 0:
            break
        positions = rng.choice(len(candidates), size=take, replace=False)
        selected.extend(candidates[int(position)] for position in positions)
    if len(selected) < target:
        selected_set = set(selected)
        remaining = [index for index in valid_indices if index not in selected_set]
        if remaining:
            take = min(target - len(selected), len(remaining))
            positions = rng.choice(len(remaining), size=take, replace=False)
            selected.extend(remaining[int(position)] for position in positions)
    return selected


def _can_reuse_before_matches_for_adaptive_weights(
    *,
    max_observations: int | None,
    evaluation_max_observations: int | None,
    sampled_count: int,
    evaluation_count: int,
) -> bool:
    """Return True when optimization and evaluation use the same retained rows."""

    return (
        max_observations is None
        and evaluation_max_observations is None
        and int(sampled_count) == int(evaluation_count)
    )


def _select_optimization_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if "precision_filter_valid" in matches.columns:
        return matches.loc[matches["precision_filter_valid"].astype(bool)].copy()
    return matches.loc[
        matches["pose_quality_valid"].astype(bool)
        & matches["reprojection_threshold_inlier"].astype(bool)
    ].copy()


def _summarize_match_quality(matches: pd.DataFrame) -> dict[str, float | int]:
    sampson = summarize_errors(np.sqrt(matches["nav_sampson_px2"].to_numpy(dtype=np.float64)))
    reprojection = summarize_errors(matches["reprojection_rms_px"].to_numpy(dtype=np.float64))
    return {
        "count": int(sampson.get("count", 0)),
        "sampson_mean": float(sampson.get("mean", np.nan)),
        "sampson_median": float(sampson.get("median", np.nan)),
        "sampson_rmse": float(sampson.get("rmse", np.nan)),
        "sampson_p95": float(sampson.get("p95", np.nan)),
        "reprojection_mean_px": float(reprojection.get("mean", np.nan)),
        "reprojection_median_px": float(reprojection.get("median", np.nan)),
        "reprojection_rmse_px": float(reprojection.get("rmse", np.nan)),
        "reprojection_p95_px": float(reprojection.get("p95", np.nan)),
    }


def _write_reprojection_outlier_diagnostics(
    *,
    output: Path,
    before_matches: pd.DataFrame,
    after_matches: pd.DataFrame,
    hard_threshold_px: float = 100.0,
    count_thresholds_px: tuple[float, ...] = (20.0, 50.0, 100.0, 500.0, 1000.0),
) -> dict[str, object]:
    output = Path(output)
    observations_dir = output / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "before_matches.parquet",
        "after_matches.parquet",
        "after_matches_reprojection_inliers.parquet",
        "after_matches_reprojection_outliers.parquet",
    ):
        (observations_dir / legacy_name).unlink(missing_ok=True)
    hard_threshold_px = float(hard_threshold_px)
    after_values = after_matches["reprojection_rms_px"].to_numpy(dtype=np.float64)
    outlier_mask = np.isfinite(after_values) & (after_values > hard_threshold_px)
    inlier_mask = ~outlier_mask
    metric_columns = ["nav_sampson_px2", "reprojection_rms_px"]
    inliers = after_matches.loc[inlier_mask, metric_columns]
    outliers = after_matches.loc[outlier_mask].copy()
    outliers.to_parquet(observations_dir / "reprojection_outliers.parquet", index=False)

    threshold_counts = {
        f">{threshold:g}px": int(np.sum(after_values > float(threshold)))
        for threshold in count_thresholds_px
    }
    group_keys = [
        key
        for key in ["match_scope", "camera1", "camera2", "frame_offset", "frame1", "frame2"]
        if key in after_matches.columns
    ]
    group_summary = pd.DataFrame()
    if group_keys:
        grouped_frame = after_matches[
            [*group_keys, "reprojection_rms_px"]
        ].copy()
        grouped_frame["_is_outlier"] = (
            grouped_frame["reprojection_rms_px"].to_numpy(dtype=np.float64)
            > hard_threshold_px
        )
        grouped = grouped_frame.groupby(group_keys, dropna=False, sort=True)
        group_summary = grouped["reprojection_rms_px"].agg(
            count="count",
            median="median",
            max="max",
        )
        quantiles = grouped["reprojection_rms_px"].quantile([0.90, 0.95]).unstack()
        quantiles.rename(columns={0.90: "p90", 0.95: "p95"}, inplace=True)
        group_outliers = grouped["_is_outlier"].sum().astype(np.int64).rename(
            "outlier_count"
        )
        group_summary = (
            group_summary.join(quantiles).join(group_outliers).reset_index()
        )
        group_summary["outlier_ratio"] = (
            group_summary["outlier_count"] / group_summary["count"].clip(lower=1)
        )
        group_summary = group_summary.sort_values(
            ["outlier_count", "max"], ascending=[False, False]
        )
    group_summary.to_csv(
        output / "metrics" / "reprojection_outlier_groups.csv",
        index=False,
    )
    diagnostics = {
        "hard_threshold_px": hard_threshold_px,
        "count": int(len(after_matches)),
        "inlier_count": int(len(inliers)),
        "outlier_count": int(len(outliers)),
        "outlier_ratio": float(len(outliers) / max(1, len(after_matches))),
        "threshold_counts": threshold_counts,
        "before_all": _summarize_match_quality(before_matches),
        "after_all": _summarize_match_quality(after_matches),
        "after_inliers": _summarize_match_quality(inliers),
        "after_outliers": _summarize_match_quality(outliers)
        if len(outliers)
        else _summarize_match_quality(after_matches.iloc[0:0].copy()),
        "bad_frame_pair_count": int(
            np.sum(group_summary["outlier_count"].to_numpy(dtype=np.int64) > 0)
        )
        if not group_summary.empty
        else 0,
        "bad_frame_pair_count_with_median_over_threshold": int(
            np.sum(
                (
                    group_summary["outlier_count"].to_numpy(dtype=np.int64) > 0
                )
                & (
                    group_summary["median"].to_numpy(dtype=np.float64)
                    > hard_threshold_px
                )
            )
        )
        if not group_summary.empty
        else 0,
    }
    (output / "metrics" / "reprojection_outlier_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    return diagnostics


_REPROJECTION_QC_GROUP_COLUMNS = [
    "match_scope",
    "camera1",
    "camera2",
    "pair_relation",
    "frame1",
    "frame2",
    "timestamp1_ns",
    "timestamp2_ns",
    "frame_offset",
    "baseline_target_m",
]
_REPROJECTION_QC_STRATUM_COLUMNS = [
    "match_scope",
    "camera1",
    "camera2",
    "pair_relation",
    "baseline_target_m",
]


def _with_reprojection_qc_group_columns(matches: pd.DataFrame) -> pd.DataFrame:
    result = matches.copy()
    if "pair_relation" not in result.columns:
        result["pair_relation"] = "unspecified"
    if "baseline_target_m" not in result.columns:
        result["baseline_target_m"] = np.nan
    return result


def _select_reprojection_qc_rows(
    matches: pd.DataFrame,
    *,
    sample_ratio: float,
) -> pd.Index:
    matches = _with_reprojection_qc_group_columns(matches)
    if matches.empty or float(sample_ratio) <= 0.0:
        return pd.Index([], dtype=matches.index.dtype)
    missing = [
        column for column in _REPROJECTION_QC_GROUP_COLUMNS if column not in matches.columns
    ]
    if missing:
        raise KeyError(f"QC sampling requires columns: {', '.join(missing)}")

    selected_indices: list[object] = []
    for _stratum, stratum_frame in matches.groupby(
        _REPROJECTION_QC_STRATUM_COLUMNS,
        dropna=False,
        sort=True,
    ):
        groups = list(
            stratum_frame.groupby(
                _REPROJECTION_QC_GROUP_COLUMNS,
                dropna=False,
                sort=True,
            ).groups.values()
        )
        if not groups:
            continue
        if float(sample_ratio) >= 1.0:
            selected_groups = groups
        else:
            count = max(1, int(np.ceil(len(groups) * float(sample_ratio))))
            positions = np.linspace(0, len(groups) - 1, count, dtype=int)
            selected_groups = [groups[int(position)] for position in positions]
        for indices in selected_groups:
            selected_indices.extend(indices.tolist())
    return pd.Index(selected_indices)


def _build_reprojection_qc_matches(
    before_matches: pd.DataFrame,
    after_matches: pd.DataFrame,
    *,
    sample_ratio: float,
) -> pd.DataFrame:
    if len(before_matches) != len(after_matches):
        raise ValueError(
            f"before/after row count mismatch: {len(before_matches)} vs {len(after_matches)}"
        )
    before_with_groups = _with_reprojection_qc_group_columns(before_matches)
    selected_indices = _select_reprojection_qc_rows(
        before_with_groups,
        sample_ratio=sample_ratio,
    )
    identity_columns = [
        *_REPROJECTION_QC_GROUP_COLUMNS,
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    projection_columns = [
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "reprojection_rms_px",
    ]
    required = [*identity_columns, *projection_columns]
    missing_before = [column for column in required if column not in before_with_groups.columns]
    missing_after = [column for column in projection_columns if column not in after_matches.columns]
    if missing_before:
        raise KeyError(f"before_matches missing QC columns: {missing_before}")
    if missing_after:
        raise KeyError(f"after_matches missing QC columns: {missing_after}")

    compact = before_with_groups.loc[selected_indices, required].copy()
    compact.rename(
        columns={column: f"before_{column}" for column in projection_columns},
        inplace=True,
    )
    aligned_after = after_matches.loc[selected_indices]
    for column in projection_columns:
        compact[f"after_{column}"] = aligned_after[column].to_numpy()
    return compact.reset_index(drop=True)


def _build_boundary_diagnostics(
    parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    threshold_ratio: float = 0.98,
) -> dict[str, object]:
    names = ["rotvec_x_rad", "rotvec_y_rad", "rotvec_z_rad", "tx_m", "ty_m", "tz_m"]
    axes: list[dict[str, object]] = []
    usage_ratios = []
    for name, value, lower, upper in zip(
        names,
        np.asarray(parameters, dtype=np.float64),
        np.asarray(lower_bounds, dtype=np.float64),
        np.asarray(upper_bounds, dtype=np.float64),
    ):
        bound = max(abs(float(lower)), abs(float(upper)), 1e-12)
        usage_ratio = abs(float(value)) / bound
        usage_ratios.append(usage_ratio)
        axes.append(
            {
                "name": name,
                "value": float(value),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "usage_ratio": float(usage_ratio),
                "at_or_near_bound": bool(usage_ratio >= float(threshold_ratio)),
            }
        )
    delta_rotation_deg = Rotation.from_rotvec(
        np.asarray(parameters[:3], dtype=np.float64)
    ).as_euler("xyz", degrees=True)
    return {
        "threshold_ratio": float(threshold_ratio),
        "max_bound_usage_ratio": float(max(usage_ratios) if usage_ratios else 0.0),
        "any_at_or_near_bound": bool(
            any(axis["at_or_near_bound"] for axis in axes)
        ),
        "delta_translation_m": np.asarray(parameters[3:6], dtype=np.float64).tolist(),
        "delta_euler_deg_xyz": np.asarray(delta_rotation_deg, dtype=np.float64).tolist(),
        "axes": axes,
    }


def _summarize_multicamera_edges(
    before_matches: pd.DataFrame,
    after_matches: pd.DataFrame,
) -> pd.DataFrame:
    group_keys = ["match_scope", "camera1", "camera2", "frame_offset"]
    aligned_after = after_matches.loc[before_matches.index]
    frame = before_matches[group_keys].copy()
    frame["_before_sampson"] = np.sqrt(
        before_matches["nav_sampson_px2"].to_numpy(dtype=np.float64)
    )
    frame["_after_sampson"] = np.sqrt(
        aligned_after["nav_sampson_px2"].to_numpy(dtype=np.float64)
    )
    frame["_before_reprojection"] = before_matches[
        "reprojection_rms_px"
    ].to_numpy(dtype=np.float64)
    frame["_after_reprojection"] = aligned_after[
        "reprojection_rms_px"
    ].to_numpy(dtype=np.float64)

    grouped = frame.groupby(group_keys, sort=True, dropna=False)
    summary = grouped.agg(
        count=("_before_sampson", "count"),
        before_sampson_mean=("_before_sampson", "mean"),
        after_sampson_mean=("_after_sampson", "mean"),
        before_reprojection_median_px=("_before_reprojection", "median"),
        after_reprojection_median_px=("_after_reprojection", "median"),
    )
    summary["before_reprojection_p95_px"] = grouped[
        "_before_reprojection"
    ].quantile(0.95)
    summary["after_reprojection_p95_px"] = grouped[
        "_after_reprojection"
    ].quantile(0.95)
    return summary.reset_index()


def _adaptive_group_keys(matches: pd.DataFrame) -> pd.Series:
    scope = matches["match_scope"].astype(str)
    camera1 = matches["camera1"].astype(str)
    camera2 = matches["camera2"].astype(str)
    offset = matches["frame_offset"].astype(int).astype(str)
    return scope + "|" + camera1 + "|" + camera2 + "|" + offset


def _compute_static_adaptive_group_weights(
    evaluated_matches: pd.DataFrame,
    *,
    alpha: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    long_tail_ratio_guard: float = 8.0,
    min_group_count: int = 500,
) -> tuple[np.ndarray, pd.DataFrame]:
    if evaluated_matches.empty:
        return np.zeros((0,), dtype=np.float64), pd.DataFrame()
    frame = evaluated_matches.copy()
    frame["_adaptive_group_key"] = _adaptive_group_keys(frame)
    finite_global = frame["reprojection_rms_px"].to_numpy(dtype=np.float64)
    finite_global = finite_global[np.isfinite(finite_global)]
    global_median = float(np.median(finite_global)) if len(finite_global) else 1.0
    global_median = max(global_median, 1e-9)
    records: list[dict[str, object]] = []
    weight_by_group: dict[str, float] = {}
    for group_key, group in frame.groupby("_adaptive_group_key", sort=False):
        values = group["reprojection_rms_px"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        count = int(len(values))
        if count:
            median = float(np.median(values))
            p95 = float(np.percentile(values, 95))
        else:
            median = float("nan")
            p95 = float("nan")
        ratio = float(p95 / max(median, 1e-9)) if np.isfinite(median) else float("inf")
        guarded = (
            count < int(min_group_count)
            or not np.isfinite(median)
            or ratio > float(long_tail_ratio_guard)
        )
        if guarded:
            weight = 1.0
        else:
            raw = (max(median, 1e-9) / global_median) ** float(alpha)
            weight = float(np.clip(raw, float(min_weight), float(max_weight)))
        weight_by_group[str(group_key)] = weight
        scope, camera1, camera2, frame_offset = str(group_key).split("|", 3)
        records.append(
            {
                "group_key": str(group_key),
                "match_scope": scope,
                "camera1": camera1,
                "camera2": camera2,
                "frame_offset": int(frame_offset),
                "count": count,
                "median_reprojection_px": median,
                "p95_reprojection_px": p95,
                "p95_to_median_ratio": ratio,
                "guarded": bool(guarded),
                "weight": weight,
            }
        )
    weights = frame["_adaptive_group_key"].map(weight_by_group).to_numpy(dtype=np.float64)
    return weights, pd.DataFrame.from_records(records)


def _camera_delta_summary(
    calibrations: dict[str, CameraCalibration],
    optimized_sensor_to_car: dict[str, np.ndarray],
    parameters_by_camera: dict[str, np.ndarray],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, float | str | bool]] = []
    for camera in sorted(optimized_sensor_to_car):
        initial = calibrations[camera].sensor_to_car
        optimized = optimized_sensor_to_car[camera]
        delta = np.linalg.inv(initial) @ optimized
        rotvec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
        euler = Rotation.from_matrix(delta[:3, :3]).as_euler("xyz", degrees=True)
        boundary = _build_boundary_diagnostics(
            parameters_by_camera[camera],
            lower_bounds,
            upper_bounds,
        )
        record: dict[str, float | str | bool] = {
            "camera": camera,
            "delta_tx_m": float(delta[:3, 3][0]),
            "delta_ty_m": float(delta[:3, 3][1]),
            "delta_tz_m": float(delta[:3, 3][2]),
            "delta_translation_norm_m": float(np.linalg.norm(delta[:3, 3])),
            "delta_roll_deg": float(euler[0]),
            "delta_pitch_deg": float(euler[1]),
            "delta_yaw_deg": float(euler[2]),
            "delta_rotation_norm_deg": float(np.rad2deg(np.linalg.norm(rotvec))),
            "max_bound_usage_ratio": float(boundary["max_bound_usage_ratio"]),
            "any_at_or_near_bound": bool(boundary["any_at_or_near_bound"]),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def _frame_disturbance_summary(
    frame_rotvecs: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    records = []
    for (camera, frame), disturbance in sorted(frame_rotvecs.items()):
        values = np.asarray(disturbance, dtype=np.float64).reshape(-1)
        rotation_deg = np.rad2deg(values[:3])
        translation = values[3:6] if len(values) >= 6 else np.zeros(3, dtype=np.float64)
        records.append(
            {
                "camera": camera,
                "frame": int(frame),
                "delta_roll_deg": float(rotation_deg[0]),
                "delta_pitch_deg": float(rotation_deg[1]),
                "delta_yaw_deg": float(rotation_deg[2]),
                "delta_norm_deg": float(np.linalg.norm(rotation_deg)),
                "delta_tx_m": float(translation[0]),
                "delta_ty_m": float(translation[1]),
                "delta_tz_m": float(translation[2]),
                "delta_translation_norm_m": float(np.linalg.norm(translation)),
            }
        )
    return pd.DataFrame.from_records(records)


def _body_disturbance_summary(
    body_frame_disturbances: dict[int, np.ndarray],
) -> pd.DataFrame:
    records = []
    for frame, disturbance in sorted(body_frame_disturbances.items()):
        values = np.asarray(disturbance, dtype=np.float64).reshape(-1)
        rotation_deg = np.rad2deg(values[:3])
        translation = values[3:6] if len(values) >= 6 else np.zeros(3, dtype=np.float64)
        records.append(
            {
                "frame": int(frame),
                "delta_roll_deg": float(rotation_deg[0]),
                "delta_pitch_deg": float(rotation_deg[1]),
                "delta_yaw_deg": float(rotation_deg[2]),
                "delta_norm_deg": float(np.linalg.norm(rotation_deg)),
                "delta_tx_m": float(translation[0]),
                "delta_ty_m": float(translation[1]),
                "delta_tz_m": float(translation[2]),
                "delta_translation_norm_m": float(np.linalg.norm(translation)),
            }
        )
    return pd.DataFrame.from_records(records)


def _rank_cameras_by_intra_reprojection(
    evaluated_matches: pd.DataFrame,
    cameras: tuple[str, ...],
) -> list[str]:
    records: list[tuple[str, float]] = []
    for camera in cameras:
        camera_rows = evaluated_matches.loc[
            (evaluated_matches["match_scope"].astype(str) == "intra_camera")
            & (evaluated_matches["camera1"].astype(str) == camera)
            & (evaluated_matches["camera2"].astype(str) == camera)
        ]
        values = camera_rows.get("reprojection_rms_px", pd.Series(dtype="float64")).to_numpy(
            dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        score = float(np.median(finite)) if len(finite) else float("-inf")
        records.append((camera, score))
    return [
        camera
        for camera, _score in sorted(
            records,
            key=lambda item: (np.nan_to_num(item[1], nan=-np.inf), item[0]),
            reverse=True,
        )
    ]


def _matches_incident_to_camera(matches: pd.DataFrame, camera: str) -> pd.DataFrame:
    return matches.loc[
        (matches["camera1"].astype(str) == camera)
        | (matches["camera2"].astype(str) == camera)
    ].copy()


def _optimize_single_camera_time_varying_block(
    *,
    target_camera: str,
    sampled: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    sensor_to_car_by_camera: dict[str, np.ndarray],
    navigation_offset_ns: int,
    initial_frame_rotvecs: dict[tuple[str, int], np.ndarray],
    max_nfev: int,
    learning_rate: float,
    lr_schedule: str,
    lr_step_size: int,
    lr_decay: float,
    lr_min_factor: float,
    device: str,
    time_varying_rotation_bound_deg: float,
    time_varying_prior_sigma_deg: float,
    time_varying_prior_weight: float,
    time_varying_smooth_sigma_deg: float,
    time_varying_smooth_weight: float,
    intra_weight: float,
    inter_weight: float,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, object]]:
    import torch

    if sampled.empty:
        return dict(initial_frame_rotvecs), {
            "target_camera": target_camera,
            "nfev": 0,
            "cost": float("nan"),
            "message": "no incident matches for target camera",
        }
    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    dtype = torch.float64
    cameras = tuple(sorted(str(camera) for camera in calibrations))
    camera_to_index = {camera: index for index, camera in enumerate(cameras)}
    first_camera_indices = torch.as_tensor(
        [camera_to_index[str(value)] for value in sampled["camera1"].to_numpy()],
        dtype=torch.long,
        device=torch_device,
    )
    second_camera_indices = torch.as_tensor(
        [camera_to_index[str(value)] for value in sampled["camera2"].to_numpy()],
        dtype=torch.long,
        device=torch_device,
    )
    target_frame_keys = sorted(
        {
            (str(row.camera1), int(row.frame1))
            for row in sampled.itertuples(index=False)
            if str(row.camera1) == target_camera
        }
        | {
            (str(row.camera2), int(row.frame2))
            for row in sampled.itertuples(index=False)
            if str(row.camera2) == target_camera
        }
    )
    target_key_to_index = {key: index for index, key in enumerate(target_frame_keys)}
    first_is_target = sampled["camera1"].astype(str).to_numpy() == target_camera
    second_is_target = sampled["camera2"].astype(str).to_numpy() == target_camera
    first_target_indices = torch.as_tensor(
        [
            target_key_to_index[(str(row.camera1), int(row.frame1))]
            if str(row.camera1) == target_camera
            else -1
            for row in sampled.itertuples(index=False)
        ],
        dtype=torch.long,
        device=torch_device,
    )
    second_target_indices = torch.as_tensor(
        [
            target_key_to_index[(str(row.camera2), int(row.frame2))]
            if str(row.camera2) == target_camera
            else -1
            for row in sampled.itertuples(index=False)
        ],
        dtype=torch.long,
        device=torch_device,
    )
    first_body = torch.as_tensor(
        np.stack(
            [
                navigation.pose_at(int(timestamp) + int(navigation_offset_ns))
                for timestamp in sampled["timestamp1_ns"].to_numpy(dtype=np.int64)
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    second_body = torch.as_tensor(
        np.stack(
            [
                navigation.pose_at(int(timestamp) + int(navigation_offset_ns))
                for timestamp in sampled["timestamp2_ns"].to_numpy(dtype=np.int64)
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    sensor_to_car = torch.as_tensor(
        np.stack([sensor_to_car_by_camera[camera] for camera in cameras]),
        dtype=dtype,
        device=torch_device,
    )
    intrinsic_inv = torch.as_tensor(
        np.stack([np.linalg.inv(calibrations[camera].intrinsic) for camera in cameras]),
        dtype=dtype,
        device=torch_device,
    )
    first_points = torch.as_tensor(
        np.column_stack(
            [
                sampled["x1"].to_numpy(dtype=np.float64),
                sampled["y1"].to_numpy(dtype=np.float64),
                np.ones(len(sampled), dtype=np.float64),
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    second_points = torch.as_tensor(
        np.column_stack(
            [
                sampled["x2"].to_numpy(dtype=np.float64),
                sampled["y2"].to_numpy(dtype=np.float64),
                np.ones(len(sampled), dtype=np.float64),
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    base_values = np.stack(
        [
            np.asarray(initial_frame_rotvecs.get(key, np.zeros(3)), dtype=np.float64)
            for key in target_frame_keys
        ]
    ) if target_frame_keys else np.zeros((0, 3), dtype=np.float64)
    frame_parameters = torch.as_tensor(
        base_values,
        dtype=dtype,
        device=torch_device,
    ).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([frame_parameters], lr=float(learning_rate))
    frame_bound_rad = np.deg2rad(float(time_varying_rotation_bound_deg))
    frame_prior_sigma_rad = max(np.deg2rad(float(time_varying_prior_sigma_deg)), 1e-12)
    frame_smooth_sigma_rad = max(np.deg2rad(float(time_varying_smooth_sigma_deg)), 1e-12)
    smooth_pairs = [(i, i + 1) for i in range(max(0, len(target_frame_keys) - 1))]
    smooth_pair_indices = torch.as_tensor(
        smooth_pairs, dtype=torch.long, device=torch_device
    ) if smooth_pairs else None
    weights = np.where(
        sampled["match_scope"].astype(str).to_numpy() == "inter_camera",
        float(inter_weight),
        float(intra_weight),
    )
    weights_t = torch.as_tensor(weights, dtype=dtype, device=torch_device)
    schedule_name = str(lr_schedule).lower()
    if schedule_name not in {"none", "constant", "step"}:
        raise ValueError(f"unsupported torch lr schedule: {lr_schedule}")
    step_size = max(1, int(lr_step_size))
    decay = float(lr_decay)
    min_factor = max(0.0, float(lr_min_factor))
    base_learning_rate = float(learning_rate)
    costs: list[float] = []
    learning_rates: list[float] = []

    fixed_first_delta = [
        np.asarray(initial_frame_rotvecs.get((str(row.camera1), int(row.frame1)), np.zeros(3)), dtype=np.float64)
        for row in sampled.itertuples(index=False)
    ]
    fixed_second_delta = [
        np.asarray(initial_frame_rotvecs.get((str(row.camera2), int(row.frame2)), np.zeros(3)), dtype=np.float64)
        for row in sampled.itertuples(index=False)
    ]
    fixed_first_delta_t = torch.as_tensor(np.stack(fixed_first_delta), dtype=dtype, device=torch_device)
    fixed_second_delta_t = torch.as_tensor(np.stack(fixed_second_delta), dtype=dtype, device=torch_device)

    for step in range(max(1, int(max_nfev))):
        if schedule_name == "step":
            factor = max(min_factor, decay ** (step // step_size))
        else:
            factor = 1.0
        current_lr = base_learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        learning_rates.append(float(current_lr))
        optimizer.zero_grad(set_to_none=True)
        all_first_rotvecs = fixed_first_delta_t.clone()
        all_second_rotvecs = fixed_second_delta_t.clone()
        if len(target_frame_keys):
            first_mask = torch.as_tensor(first_is_target, dtype=torch.bool, device=torch_device)
            second_mask = torch.as_tensor(second_is_target, dtype=torch.bool, device=torch_device)
            all_first_rotvecs[first_mask] = frame_parameters[first_target_indices[first_mask]]
            all_second_rotvecs[second_mask] = frame_parameters[second_target_indices[second_mask]]
        first_delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(len(sampled), 1, 1)
        second_delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(len(sampled), 1, 1)
        first_delta[:, :3, :3] = _torch_rotvec_to_matrix(all_first_rotvecs)
        second_delta[:, :3, :3] = _torch_rotvec_to_matrix(all_second_rotvecs)
        first_camera_to_world = (
            first_body @ sensor_to_car[first_camera_indices] @ first_delta
        )
        second_camera_to_world = (
            second_body @ sensor_to_car[second_camera_indices] @ second_delta
        )
        first_to_second = (
            _torch_transform_inverse(second_camera_to_world) @ first_camera_to_world
        )
        essential = _torch_skew(first_to_second[:, :3, 3]) @ first_to_second[:, :3, :3]
        fundamental = (
            intrinsic_inv[second_camera_indices].transpose(-1, -2)
            @ essential
            @ intrinsic_inv[first_camera_indices]
        )
        lines_second = (fundamental @ first_points[..., None]).squeeze(-1)
        lines_first = (
            fundamental.transpose(-1, -2) @ second_points[..., None]
        ).squeeze(-1)
        residual = torch.sum(second_points * lines_second, dim=1)
        denom = (
            torch.sum(lines_first[:, :2] ** 2, dim=1)
            + torch.sum(lines_second[:, :2] ** 2, dim=1)
        ).clamp_min(1e-12)
        sampson_sqrt = torch.sqrt((residual**2 / denom).clamp_min(0.0) + 1e-12)
        per_match = torch.nn.functional.huber_loss(
            sampson_sqrt,
            torch.zeros_like(sampson_sqrt),
            delta=1.0,
            reduction="none",
        )
        data_loss = torch.sum(weights_t * per_match) / weights_t.sum().clamp_min(1e-12)
        frame_prior_loss = float(time_varying_prior_weight) * torch.mean(
            (frame_parameters / frame_prior_sigma_rad) ** 2
        ) if len(target_frame_keys) else torch.zeros((), dtype=dtype, device=torch_device)
        if smooth_pair_indices is not None and len(smooth_pairs):
            diffs = (
                frame_parameters[smooth_pair_indices[:, 1]]
                - frame_parameters[smooth_pair_indices[:, 0]]
            )
            frame_smooth_loss = float(time_varying_smooth_weight) * torch.mean(
                (diffs / frame_smooth_sigma_rad) ** 2
            )
        else:
            frame_smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
        loss = data_loss + frame_prior_loss + frame_smooth_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            frame_parameters.clamp_(-frame_bound_rad, frame_bound_rad)
        costs.append(float(loss.detach().cpu()))

    result = dict(initial_frame_rotvecs)
    optimized_values = frame_parameters.detach().cpu().numpy()
    for index, key in enumerate(target_frame_keys):
        result[key] = optimized_values[index].copy()
    return result, {
        "target_camera": target_camera,
        "nfev": int(max(1, int(max_nfev))),
        "cost": float(costs[-1] if costs else np.nan),
        "cost_by_step_all": costs,
        "cost_by_step_first": costs[:10],
        "cost_by_step_last": costs[-10:],
        "requested_device": requested_device,
        "resolved_device": str(torch_device),
        "learning_rate": float(learning_rate),
        "lr_schedule": "none" if schedule_name == "constant" else schedule_name,
        "learning_rate_first_steps": learning_rates[:10],
        "learning_rate_last_steps": learning_rates[-10:],
        "learning_rate_by_step_all": learning_rates,
        "target_frame_variable_count": int(len(target_frame_keys)),
        "intra_weight": float(intra_weight),
        "inter_weight": float(inter_weight),
    }


def _torch_optimize_multi_camera_extrinsics(
    *,
    sampled: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    cameras: tuple[str, ...],
    navigation_offset_ns: int,
    lower_one: np.ndarray,
    upper_one: np.ndarray,
    fixed_translation_l2_weight: float,
    fixed_rotation_l2_weight: float,
    max_nfev: int,
    learning_rate: float,
    device: str,
    lr_schedule: str = "step",
    lr_step_size: int = 300,
    lr_decay: float = 0.5,
    lr_min_factor: float = 0.1,
    torch_dtype: str = "float64",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_min_steps: int = 0,
    optimize_time_varying_disturbance: bool = False,
    initial_frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    match_weights: np.ndarray | None = None,
    time_varying_disturbance_dof: int = 6,
    optimize_body_time_varying_disturbance: bool = False,
    body_time_varying_disturbance_dof: int = 6,
    time_varying_rotation_bound_deg: float = 8.0,
    time_varying_translation_bound_m: float = 0.05,
    time_varying_translation_l2_weight: float = 200.0,
    time_varying_rotation_l2_weight: float = 5.0,
    time_varying_translation_smooth_l2_weight: float = 20.0,
    time_varying_rotation_smooth_l2_weight: float = 0.5,
    time_varying_translation_mean_l2_weight: float = 5000.0,
    time_varying_rotation_mean_l2_weight: float = 2500.0,
    time_varying_frame_common_translation_l2_weight: float = 5000.0,
    time_varying_frame_common_rotation_l2_weight: float = 2500.0,
    body_time_varying_rotation_bound_deg: float = 15.0,
    body_time_varying_translation_bound_m: float = 0.15,
    body_time_varying_translation_l2_weight: float = 200.0,
    body_time_varying_rotation_l2_weight: float = 5.0,
    body_time_varying_translation_smooth_l2_weight: float = 20.0,
    body_time_varying_rotation_smooth_l2_weight: float = 0.5,
    mode_projection_enabled: bool = False,
    mode_projection_interval_steps: int = 100,
    mode_projection_absorb_frame_common: bool = True,
    mode_projection_absorb_camera_long_term: bool = True,
    mode_projection_reset_optimizer_state: bool = True,
) -> tuple[
    np.ndarray,
    dict[tuple[str, int], np.ndarray],
    dict[int, np.ndarray],
    dict[str, object],
]:
    import torch

    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    dtype, resolved_dtype_name = _resolve_torch_dtype(torch, torch_dtype)
    time_varying_disturbance_dof = int(time_varying_disturbance_dof)
    if time_varying_disturbance_dof not in (3, 6):
        raise ValueError("time_varying_disturbance_dof must be either 3 or 6")
    body_time_varying_disturbance_dof = int(body_time_varying_disturbance_dof)
    if body_time_varying_disturbance_dof not in (3, 6):
        raise ValueError("body_time_varying_disturbance_dof must be either 3 or 6")
    camera_to_index = {camera: index for index, camera in enumerate(cameras)}
    endpoint_layout = _build_unique_endpoint_layout(sampled)
    frame_keys = sorted(
        {
            (str(row.camera1), int(row.frame1))
            for row in sampled.itertuples(index=False)
        }
        | {
            (str(row.camera2), int(row.frame2))
            for row in sampled.itertuples(index=False)
        }
    )
    frame_key_to_index = {key: index for index, key in enumerate(frame_keys)}
    body_frame_keys = sorted(
        set(sampled["frame1"].astype(int).tolist())
        | set(sampled["frame2"].astype(int).tolist())
    )
    body_frame_to_index = {frame: index for index, frame in enumerate(body_frame_keys)}
    endpoint_camera_indices = torch.as_tensor(
        [camera_to_index[camera] for camera, _frame, _timestamp in endpoint_layout.keys],
        dtype=torch.long,
        device=torch_device,
    )
    endpoint_frame_indices = torch.as_tensor(
        [
            frame_key_to_index[(camera, frame)]
            for camera, frame, _timestamp in endpoint_layout.keys
        ],
        dtype=torch.long,
        device=torch_device,
    )
    endpoint_body_frame_indices = torch.as_tensor(
        [body_frame_to_index[frame] for _camera, frame, _timestamp in endpoint_layout.keys],
        dtype=torch.long,
        device=torch_device,
    )
    first_endpoint_indices = torch.as_tensor(
        endpoint_layout.first_indices,
        dtype=torch.long,
        device=torch_device,
    )
    second_endpoint_indices = torch.as_tensor(
        endpoint_layout.second_indices,
        dtype=torch.long,
        device=torch_device,
    )
    smooth_pairs: list[tuple[int, int]] = []
    frame_mean_groups: list[list[int]] = []
    frame_common_groups: list[list[int]] = []
    if optimize_time_varying_disturbance:
        for camera in cameras:
            indices = [
                frame_key_to_index[key]
                for key in frame_keys
                if key[0] == camera
            ]
            if indices:
                frame_mean_groups.append(indices)
            for first, second in zip(indices[:-1], indices[1:]):
                smooth_pairs.append((first, second))
        for frame in body_frame_keys:
            indices = [
                frame_key_to_index[key]
                for key in frame_keys
                if key[1] == frame
            ]
            if len(indices) >= 2:
                frame_common_groups.append(indices)
    body_smooth_pairs = [
        (first, second)
        for first, second in zip(
            range(0, max(0, len(body_frame_keys) - 1)),
            range(1, len(body_frame_keys)),
        )
    ] if optimize_body_time_varying_disturbance else []
    endpoint_body = torch.as_tensor(
        np.stack(
            [
                navigation.pose_at(int(timestamp) + int(navigation_offset_ns))
                for _camera, _frame, timestamp in endpoint_layout.keys
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    initial_sensor_to_car_numpy = np.stack(
        [calibrations[camera].sensor_to_car for camera in cameras]
    )
    initial_sensor_to_car_by_camera_numpy = {
        camera: np.asarray(calibrations[camera].sensor_to_car, dtype=np.float64)
        for camera in cameras
    }
    initial_sensor_to_car = torch.as_tensor(
        initial_sensor_to_car_numpy,
        dtype=dtype,
        device=torch_device,
    )
    focal_scale = torch.as_tensor(
        np.asarray(
            [
                0.5
                * (
                    0.5
                    * (
                        float(calibrations[str(row.camera1)].intrinsic[0, 0])
                        + float(calibrations[str(row.camera1)].intrinsic[1, 1])
                    )
                    + 0.5
                    * (
                        float(calibrations[str(row.camera2)].intrinsic[0, 0])
                        + float(calibrations[str(row.camera2)].intrinsic[1, 1])
                    )
                )
                for row in sampled.itertuples(index=False)
            ],
            dtype=np.float64,
        ),
        dtype=dtype,
        device=torch_device,
    )
    first_points = torch.as_tensor(
        np.column_stack(
            [
                sampled["nx1"].to_numpy(dtype=np.float64),
                sampled["ny1"].to_numpy(dtype=np.float64),
                np.ones(len(sampled), dtype=np.float64),
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    second_points = torch.as_tensor(
        np.column_stack(
            [
                sampled["nx2"].to_numpy(dtype=np.float64),
                sampled["ny2"].to_numpy(dtype=np.float64),
                np.ones(len(sampled), dtype=np.float64),
            ]
        ),
        dtype=dtype,
        device=torch_device,
    )
    lower = torch.as_tensor(lower_one, dtype=dtype, device=torch_device)
    upper = torch.as_tensor(upper_one, dtype=dtype, device=torch_device)
    fixed_rotation_l2_weight_t = torch.as_tensor(
        max(float(fixed_rotation_l2_weight), 0.0),
        dtype=dtype,
        device=torch_device,
    )
    fixed_translation_l2_weight_t = torch.as_tensor(
        max(float(fixed_translation_l2_weight), 0.0),
        dtype=dtype,
        device=torch_device,
    )
    parameters = torch.zeros(
        (len(cameras), 6),
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    initial_frame_rotvecs = initial_frame_rotvecs or {}
    def _initial_frame_value(key: tuple[str, int]) -> np.ndarray:
        value = np.asarray(initial_frame_rotvecs.get(key, np.zeros(3)), dtype=np.float64).reshape(-1)
        if len(value) >= time_varying_disturbance_dof:
            return value[:time_varying_disturbance_dof].copy()
        padded = np.zeros(time_varying_disturbance_dof, dtype=np.float64)
        padded[: len(value)] = value
        return padded

    initial_frame_values = np.stack(
        [
            _initial_frame_value(key)
            for key in frame_keys
        ]
    )
    frame_parameters = torch.as_tensor(
        initial_frame_values,
        dtype=dtype,
        device=torch_device,
    ).clone().detach().requires_grad_(bool(optimize_time_varying_disturbance))
    body_parameters = torch.zeros(
        (len(body_frame_keys), body_time_varying_disturbance_dof),
        dtype=dtype,
        device=torch_device,
        requires_grad=bool(optimize_body_time_varying_disturbance),
    )
    optimizer_parameters = [parameters]
    if optimize_time_varying_disturbance:
        optimizer_parameters.append(frame_parameters)
    if optimize_body_time_varying_disturbance:
        optimizer_parameters.append(body_parameters)
    optimizer = torch.optim.Adam(optimizer_parameters, lr=float(learning_rate))
    frame_bound_rad = np.deg2rad(float(time_varying_rotation_bound_deg))
    frame_translation_bound = abs(float(time_varying_translation_bound_m))
    frame_lower = np.full(time_varying_disturbance_dof, -frame_bound_rad, dtype=np.float64)
    frame_upper = np.full(time_varying_disturbance_dof, frame_bound_rad, dtype=np.float64)
    if time_varying_disturbance_dof == 6:
        frame_lower[3:6] = -frame_translation_bound
        frame_upper[3:6] = frame_translation_bound
    frame_lower_t = torch.as_tensor(frame_lower, dtype=dtype, device=torch_device)
    frame_upper_t = torch.as_tensor(frame_upper, dtype=dtype, device=torch_device)
    body_bound_rad = np.deg2rad(float(body_time_varying_rotation_bound_deg))
    body_translation_bound = abs(float(body_time_varying_translation_bound_m))
    body_lower = np.full(
        body_time_varying_disturbance_dof, -body_bound_rad, dtype=np.float64
    )
    body_upper = np.full(
        body_time_varying_disturbance_dof, body_bound_rad, dtype=np.float64
    )
    if body_time_varying_disturbance_dof == 6:
        body_lower[3:6] = -body_translation_bound
        body_upper[3:6] = body_translation_bound
    body_lower_t = torch.as_tensor(body_lower, dtype=dtype, device=torch_device)
    body_upper_t = torch.as_tensor(body_upper, dtype=dtype, device=torch_device)
    smooth_pair_indices = torch.as_tensor(
        smooth_pairs,
        dtype=torch.long,
        device=torch_device,
    ) if smooth_pairs else None
    frame_mean_group_layout = _build_padded_group_layout(frame_mean_groups)
    frame_common_group_layout = _build_padded_group_layout(frame_common_groups)

    def _group_layout_tensors(layout: _PaddedGroupLayout):
        if not len(layout.counts):
            return None
        return (
            torch.as_tensor(layout.indices, dtype=torch.long, device=torch_device),
            torch.as_tensor(layout.valid_mask, dtype=torch.bool, device=torch_device),
            torch.as_tensor(layout.counts, dtype=dtype, device=torch_device),
        )

    frame_mean_group_tensors = _group_layout_tensors(frame_mean_group_layout)
    frame_common_group_tensors = _group_layout_tensors(frame_common_group_layout)
    body_smooth_pair_indices = torch.as_tensor(
        body_smooth_pairs,
        dtype=torch.long,
        device=torch_device,
    ) if body_smooth_pairs else None
    match_weights_array = (
        np.array(match_weights, dtype=np.float64, copy=True)
        if match_weights is not None
        else np.ones((len(sampled),), dtype=np.float64)
    )
    if len(match_weights_array) != len(sampled):
        raise ValueError("match_weights length must match sampled observations")
    match_weights_t = torch.as_tensor(match_weights_array, dtype=dtype, device=torch_device)
    costs: list[float] = []
    learning_rates: list[float] = []
    stopped_early = False
    early_stopping_step: int | None = None
    best_cost = float("inf")
    stale_steps = 0
    schedule_name = str(lr_schedule).lower()
    if schedule_name not in {"none", "constant", "step"}:
        raise ValueError(f"unsupported torch lr schedule: {lr_schedule}")
    base_learning_rate = float(learning_rate)
    step_size = max(1, int(lr_step_size))
    decay = float(lr_decay)
    min_factor = max(0.0, float(lr_min_factor))
    mode_projection_interval_steps = max(1, int(mode_projection_interval_steps))
    mode_projection_reports: list[dict[str, object]] = []
    mode_projection_count = 0
    mode_projection_frame_common_count = 0
    mode_projection_camera_long_term_count = 0
    mode_projection_frame_common_norm_sum = 0.0
    mode_projection_camera_long_term_norm_sum = 0.0

    def _weighted_l2(values, *, dof: int, rotation_weight: float, translation_weight: float):
        rotation_loss = max(float(rotation_weight), 0.0) * torch.mean(values[:, :3] ** 2)
        if int(dof) == 6:
            return rotation_loss + max(float(translation_weight), 0.0) * torch.mean(
                values[:, 3:6] ** 2
            )
        return rotation_loss

    def _frame_mean_drift_loss():
        if frame_mean_group_tensors is None:
            return torch.zeros((), dtype=dtype, device=torch_device)
        mean_values = _torch_group_means(
            frame_parameters,
            *frame_mean_group_tensors,
        )
        return _weighted_l2(
            mean_values,
            dof=time_varying_disturbance_dof,
            rotation_weight=time_varying_rotation_mean_l2_weight,
            translation_weight=time_varying_translation_mean_l2_weight,
        )

    def _frame_common_mode_loss():
        if frame_common_group_tensors is None:
            return torch.zeros((), dtype=dtype, device=torch_device)
        mean_values = _torch_group_means(
            frame_parameters,
            *frame_common_group_tensors,
        )
        return _weighted_l2(
            mean_values,
            dof=time_varying_disturbance_dof,
            rotation_weight=time_varying_frame_common_rotation_l2_weight,
            translation_weight=time_varying_frame_common_translation_l2_weight,
        )

    def _project_modes_in_place() -> dict[str, object]:
        fixed_np = parameters.detach().cpu().numpy().copy()
        frame_np = frame_parameters.detach().cpu().numpy().copy()
        body_np = (
            body_parameters.detach().cpu().numpy().copy()
            if optimize_body_time_varying_disturbance
            else np.zeros((0, int(body_time_varying_disturbance_dof)), dtype=np.float64)
        )
        report = _project_time_varying_modes_numpy(
            fixed_parameters=fixed_np,
            frame_parameters=frame_np,
            body_parameters=body_np,
            frame_keys=frame_keys,
            camera_to_index=camera_to_index,
            body_frame_to_index=body_frame_to_index,
            frame_mean_groups=frame_mean_groups,
            frame_common_groups=frame_common_groups,
            frame_dof=time_varying_disturbance_dof,
            body_dof=body_time_varying_disturbance_dof,
            absorb_frame_common=(
                bool(mode_projection_absorb_frame_common)
                and optimize_body_time_varying_disturbance
            ),
            absorb_camera_long_term=bool(mode_projection_absorb_camera_long_term),
            sensor_to_car_by_camera=initial_sensor_to_car_by_camera_numpy,
        )
        parameters.copy_(torch.as_tensor(fixed_np, dtype=dtype, device=torch_device))
        frame_parameters.copy_(torch.as_tensor(frame_np, dtype=dtype, device=torch_device))
        if optimize_body_time_varying_disturbance:
            body_parameters.copy_(
                torch.as_tensor(body_np, dtype=dtype, device=torch_device)
            )
        return report

    for _step in range(max(1, int(max_nfev))):
        if schedule_name == "step":
            factor = max(min_factor, decay ** (_step // step_size))
        else:
            factor = 1.0
        current_learning_rate = base_learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate
        learning_rates.append(float(current_learning_rate))
        optimizer.zero_grad(set_to_none=True)
        delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(len(cameras), 1, 1)
        delta[:, :3, :3] = _torch_rotvec_to_matrix(parameters[:, :3])
        delta[:, :3, 3] = parameters[:, 3:6]
        sensor_to_car = initial_sensor_to_car @ delta
        if optimize_time_varying_disturbance:
            frame_delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(
                len(frame_keys), 1, 1
            )
            frame_delta[:, :3, :3] = _torch_rotvec_to_matrix(frame_parameters[:, :3])
            if time_varying_disturbance_dof == 6:
                frame_delta[:, :3, 3] = frame_parameters[:, 3:6]
        else:
            frame_delta = None
        if optimize_body_time_varying_disturbance:
            body_delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(
                len(body_frame_keys), 1, 1
            )
            body_delta[:, :3, :3] = _torch_rotvec_to_matrix(body_parameters[:, :3])
            if body_time_varying_disturbance_dof == 6:
                body_delta[:, :3, 3] = body_parameters[:, 3:6]
        else:
            body_delta = None
        camera_to_world_by_endpoint = _torch_compose_unique_endpoint_poses(
            endpoint_body,
            body_delta=body_delta,
            endpoint_body_indices=endpoint_body_frame_indices,
            sensor_to_car=sensor_to_car,
            endpoint_camera_indices=endpoint_camera_indices,
            frame_delta=frame_delta,
            endpoint_frame_indices=endpoint_frame_indices,
        )
        first_camera_to_world = camera_to_world_by_endpoint[first_endpoint_indices]
        second_camera_to_world = camera_to_world_by_endpoint[second_endpoint_indices]
        first_to_second = (
            _torch_transform_inverse(second_camera_to_world) @ first_camera_to_world
        )
        essential = (
            _torch_skew(first_to_second[:, :3, 3])
            @ first_to_second[:, :3, :3]
        )
        sampson_sqrt = (
            _torch_sampson_sqrt_from_essential(essential, first_points, second_points)
            * focal_scale
        )
        per_match_loss = torch.nn.functional.huber_loss(
            sampson_sqrt,
            torch.zeros_like(sampson_sqrt),
            delta=1.0,
            reduction="none",
        )
        data_loss = torch.sum(match_weights_t * per_match_loss) / match_weights_t.sum().clamp_min(1e-12)
        prior_loss = (
            fixed_rotation_l2_weight_t * torch.mean(parameters[:, :3] ** 2)
            + fixed_translation_l2_weight_t * torch.mean(parameters[:, 3:6] ** 2)
        )
        if optimize_time_varying_disturbance:
            frame_prior_loss = _weighted_l2(
                frame_parameters,
                dof=time_varying_disturbance_dof,
                rotation_weight=time_varying_rotation_l2_weight,
                translation_weight=time_varying_translation_l2_weight,
            )
            if smooth_pair_indices is not None and len(smooth_pairs):
                diffs = (
                    frame_parameters[smooth_pair_indices[:, 1]]
                    - frame_parameters[smooth_pair_indices[:, 0]]
                )
                frame_smooth_loss = _weighted_l2(
                    diffs,
                    dof=time_varying_disturbance_dof,
                    rotation_weight=time_varying_rotation_smooth_l2_weight,
                    translation_weight=time_varying_translation_smooth_l2_weight,
                )
            else:
                frame_smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
            frame_mean_loss = _frame_mean_drift_loss()
            frame_common_loss = _frame_common_mode_loss()
        else:
            frame_prior_loss = torch.zeros((), dtype=dtype, device=torch_device)
            frame_smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
            frame_mean_loss = torch.zeros((), dtype=dtype, device=torch_device)
            frame_common_loss = torch.zeros((), dtype=dtype, device=torch_device)
        if optimize_body_time_varying_disturbance:
            body_prior_loss = _weighted_l2(
                body_parameters,
                dof=body_time_varying_disturbance_dof,
                rotation_weight=body_time_varying_rotation_l2_weight,
                translation_weight=body_time_varying_translation_l2_weight,
            )
            if body_smooth_pair_indices is not None and len(body_smooth_pairs):
                body_diffs = (
                    body_parameters[body_smooth_pair_indices[:, 1]]
                    - body_parameters[body_smooth_pair_indices[:, 0]]
                )
                body_smooth_loss = _weighted_l2(
                    body_diffs,
                    dof=body_time_varying_disturbance_dof,
                    rotation_weight=body_time_varying_rotation_smooth_l2_weight,
                    translation_weight=body_time_varying_translation_smooth_l2_weight,
                )
            else:
                body_smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
        else:
            body_prior_loss = torch.zeros((), dtype=dtype, device=torch_device)
            body_smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
        loss = (
            data_loss
            + prior_loss
            + frame_prior_loss
            + frame_smooth_loss
            + frame_mean_loss
            + frame_common_loss
            + body_prior_loss
            + body_smooth_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameters.clamp_(lower, upper)
            if optimize_time_varying_disturbance:
                frame_parameters.clamp_(frame_lower_t, frame_upper_t)
            if optimize_body_time_varying_disturbance:
                body_parameters.clamp_(body_lower_t, body_upper_t)
            should_project_modes = (
                bool(mode_projection_enabled)
                and optimize_time_varying_disturbance
                and (_step + 1) % mode_projection_interval_steps == 0
            )
            if should_project_modes:
                projection_report = _project_modes_in_place()
                parameters.clamp_(lower, upper)
                frame_parameters.clamp_(frame_lower_t, frame_upper_t)
                if optimize_body_time_varying_disturbance:
                    body_parameters.clamp_(body_lower_t, body_upper_t)
                mode_projection_reports.append(
                    {"step": int(_step + 1), **projection_report}
                )
                mode_projection_count += 1
                mode_projection_frame_common_count += int(
                    projection_report["frame_common_projection_count"]
                )
                mode_projection_camera_long_term_count += int(
                    projection_report["camera_long_term_projection_count"]
                )
                mode_projection_frame_common_norm_sum += float(
                    projection_report["frame_common_absorbed_norm_sum"]
                )
                mode_projection_camera_long_term_norm_sum += float(
                    projection_report["camera_long_term_absorbed_norm_sum"]
                )
                if bool(mode_projection_reset_optimizer_state):
                    optimizer.state.clear()
        current_cost = float(loss.detach().cpu())
        costs.append(current_cost)
        if early_stopping_patience is not None and int(early_stopping_patience) > 0:
            min_delta = max(0.0, float(early_stopping_min_delta))
            if best_cost - current_cost > min_delta:
                best_cost = current_cost
                stale_steps = 0
            else:
                stale_steps += 1
            if _step + 1 >= max(0, int(early_stopping_min_steps)) and stale_steps >= int(early_stopping_patience):
                stopped_early = True
                early_stopping_step = int(_step + 1)
                break

    frame_values = frame_parameters.detach().cpu().numpy()
    frame_rotvecs = {
        key: frame_values[index].copy()
        for index, key in enumerate(frame_keys)
    } if optimize_time_varying_disturbance else {}
    body_values = body_parameters.detach().cpu().numpy()
    body_frame_disturbances = {
        int(frame): body_values[index].copy()
        for index, frame in enumerate(body_frame_keys)
    } if optimize_body_time_varying_disturbance else {}
    return parameters.detach().cpu().numpy().reshape(-1), frame_rotvecs, body_frame_disturbances, {
        "success": True,
        "status": 0,
        "solver_method": "torch_sparse_graph_full_batch",
        "message": "multi-camera fixed extrinsics and frame disturbances optimized with torch autodiff"
        if optimize_time_varying_disturbance
        else "multi-camera fixed extrinsics optimized with torch autodiff",
        "nfev": int(len(costs)),
        "cost": float(costs[-1] if costs else np.nan),
        "cost_by_step_all": costs,
        "cost_by_step_first": costs[:10],
        "cost_by_step_last": costs[-10:],
        "optimality": float("nan"),
        "requested_device": requested_device,
        "resolved_device": str(torch_device),
        "torch_dtype": resolved_dtype_name,
        "learning_rate": float(learning_rate),
        "lr_schedule": "none" if schedule_name == "constant" else schedule_name,
        "lr_step_size": int(step_size),
        "lr_decay": float(decay),
        "lr_min_factor": float(min_factor),
        "learning_rate_first_steps": learning_rates[:10],
        "learning_rate_last_steps": learning_rates[-10:],
        "learning_rate_by_step_all": learning_rates,
        "optimize_time_varying_disturbance": bool(optimize_time_varying_disturbance),
        "optimize_body_time_varying_disturbance": bool(
            optimize_body_time_varying_disturbance
        ),
        "time_varying_frame_variable_count": int(len(frame_keys))
        if optimize_time_varying_disturbance
        else 0,
        "time_varying_smooth_edge_count": int(len(smooth_pairs)),
        "time_varying_mean_group_count": int(len(frame_mean_groups)),
        "time_varying_frame_common_group_count": int(len(frame_common_groups)),
        "time_varying_disturbance_dof": int(time_varying_disturbance_dof),
        "body_time_varying_frame_variable_count": int(len(body_frame_keys))
        if optimize_body_time_varying_disturbance
        else 0,
        "body_time_varying_smooth_edge_count": int(len(body_smooth_pairs)),
        "body_time_varying_disturbance_dof": int(body_time_varying_disturbance_dof),
        "unique_endpoint_count": int(len(endpoint_layout.keys)),
        "observation_endpoint_count": int(2 * len(sampled)),
        "navigation_pose_evaluation_count": int(len(endpoint_layout.keys)),
        "time_varying_rotation_bound_deg": float(time_varying_rotation_bound_deg),
        "time_varying_translation_bound_m": float(time_varying_translation_bound_m),
        "body_time_varying_rotation_bound_deg": float(
            body_time_varying_rotation_bound_deg
        ),
        "body_time_varying_translation_bound_m": float(
            body_time_varying_translation_bound_m
        ),
        "fixed_translation_l2_weight": float(fixed_translation_l2_weight),
        "fixed_rotation_l2_weight": float(fixed_rotation_l2_weight),
        "time_varying_translation_l2_weight": float(time_varying_translation_l2_weight),
        "time_varying_rotation_l2_weight": float(time_varying_rotation_l2_weight),
        "time_varying_translation_smooth_l2_weight": float(
            time_varying_translation_smooth_l2_weight
        ),
        "time_varying_rotation_smooth_l2_weight": float(
            time_varying_rotation_smooth_l2_weight
        ),
        "time_varying_translation_mean_l2_weight": float(
            time_varying_translation_mean_l2_weight
        ),
        "time_varying_rotation_mean_l2_weight": float(
            time_varying_rotation_mean_l2_weight
        ),
        "time_varying_frame_common_translation_l2_weight": float(
            time_varying_frame_common_translation_l2_weight
        ),
        "time_varying_frame_common_rotation_l2_weight": float(
            time_varying_frame_common_rotation_l2_weight
        ),
        "body_time_varying_translation_l2_weight": float(
            body_time_varying_translation_l2_weight
        ),
        "body_time_varying_rotation_l2_weight": float(
            body_time_varying_rotation_l2_weight
        ),
        "body_time_varying_translation_smooth_l2_weight": float(
            body_time_varying_translation_smooth_l2_weight
        ),
        "body_time_varying_rotation_smooth_l2_weight": float(
            body_time_varying_rotation_smooth_l2_weight
        ),
        "mode_projection_enabled": bool(mode_projection_enabled),
        "mode_projection_interval_steps": int(mode_projection_interval_steps),
        "mode_projection_absorb_frame_common": bool(
            mode_projection_absorb_frame_common
        ),
        "mode_projection_absorb_camera_long_term": bool(
            mode_projection_absorb_camera_long_term
        ),
        "mode_projection_reset_optimizer_state": bool(
            mode_projection_reset_optimizer_state
        ),
        "mode_projection_count": int(mode_projection_count),
        "mode_projection_frame_common_projection_count": int(
            mode_projection_frame_common_count
        ),
        "mode_projection_camera_long_term_projection_count": int(
            mode_projection_camera_long_term_count
        ),
        "mode_projection_frame_common_absorbed_norm_sum": float(
            mode_projection_frame_common_norm_sum
        ),
        "mode_projection_camera_long_term_absorbed_norm_sum": float(
            mode_projection_camera_long_term_norm_sum
        ),
        "mode_projection_reports_first": mode_projection_reports[:10],
        "mode_projection_reports_last": mode_projection_reports[-10:],
        "match_weight_min": float(np.min(match_weights_array)) if len(match_weights_array) else float("nan"),
        "match_weight_max": float(np.max(match_weights_array)) if len(match_weights_array) else float("nan"),
        "match_weight_mean": float(np.mean(match_weights_array)) if len(match_weights_array) else float("nan"),
        "stopped_early": bool(stopped_early),
        "early_stopping_step": early_stopping_step,
        "early_stopping_patience": None if early_stopping_patience is None else int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "early_stopping_min_steps": int(early_stopping_min_steps),
    }


def optimize_multi_camera_extrinsics_from_observations(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    output: Path,
    max_observations: int | None = None,
    pair_sample_max_matches: int | None = None,
    pair_sample_seed: int = 20260704,
    pair_sample_strategy: str = "spatial_grid_random",
    pair_sample_grid_cols: int = 3,
    pair_sample_grid_rows: int = 3,
    translation_bound_m: float = 1.0,
    rotation_bound_deg: float = 10.0,
    fixed_translation_l2_weight: float = 50.0,
    fixed_rotation_l2_weight: float = 300.0,
    navigation_offset_ns: int = 0,
    max_nfev: int = 1500,
    evaluation_max_observations: int | None = None,
    optimizer_method: str = "torch_sparse_graph",
    torch_device: str = "cuda:0",
    torch_learning_rate: float = 0.003,
    torch_lr_schedule: str = "step",
    torch_lr_step_size: int = 100,
    torch_lr_decay: float = 0.5,
    torch_lr_min_factor: float = 0.1,
    torch_dtype: str = "float32",
    torch_early_stopping_patience: int | None = 100,
    torch_early_stopping_min_delta: float = 1e-3,
    torch_early_stopping_min_steps: int = 500,
    evaluation_backend: str = "auto",
    evaluation_device: str | None = None,
    evaluation_batch_size: int = 50_000,
    projection_qc_sample_ratio: float = 0.01,
    adaptive_weight_mode: str = "none",
    adaptive_weight_alpha: float = 0.5,
    adaptive_weight_min: float = 0.5,
    adaptive_weight_max: float = 2.0,
    adaptive_weight_long_tail_ratio_guard: float = 8.0,
    adaptive_weight_min_group_count: int = 500,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
    optimize_time_varying_disturbance: bool = False,
    initial_frame_rotvecs: dict[tuple[str, int], np.ndarray] | None = None,
    time_varying_disturbance_dof: int = 6,
    optimize_body_time_varying_disturbance: bool = False,
    body_time_varying_disturbance_dof: int = 6,
    time_varying_rotation_bound_deg: float = 15.0,
    time_varying_translation_bound_m: float = 0.15,
    time_varying_translation_l2_weight: float = 200.0,
    time_varying_rotation_l2_weight: float = 5.0,
    time_varying_translation_smooth_l2_weight: float = 20.0,
    time_varying_rotation_smooth_l2_weight: float = 0.5,
    time_varying_translation_mean_l2_weight: float = 5000.0,
    time_varying_rotation_mean_l2_weight: float = 2500.0,
    time_varying_frame_common_translation_l2_weight: float = 5000.0,
    time_varying_frame_common_rotation_l2_weight: float = 2500.0,
    body_time_varying_rotation_bound_deg: float = 15.0,
    body_time_varying_translation_bound_m: float = 0.15,
    body_time_varying_translation_l2_weight: float = 200.0,
    body_time_varying_rotation_l2_weight: float = 5.0,
    body_time_varying_translation_smooth_l2_weight: float = 20.0,
    body_time_varying_rotation_smooth_l2_weight: float = 0.5,
    mode_projection_enabled: bool = False,
    mode_projection_interval_steps: int = 100,
    mode_projection_absorb_frame_common: bool = True,
    mode_projection_absorb_camera_long_term: bool = True,
    mode_projection_reset_optimizer_state: bool = True,
    absorb_camera_common_mode: bool = True,
    camera_common_mode_min_cameras: int = 3,
    camera_common_mode_min_median_cos: float = 0.5,
    camera_common_mode_min_positive_projection_fraction: float = 0.8,
    absorb_camera_long_term_mode: bool = True,
    camera_long_term_mode_min_frames: int = 30,
    camera_long_term_mode_min_median_cos: float = 0.5,
    camera_long_term_mode_min_positive_projection_fraction: float = 0.8,
    camera_long_term_mode_min_mean_to_rms: float = 0.35,
    frame_metadata: dict[str, dict[int, dict[str, object]]] | None = None,
    result_interpolation_pkl_path: Path | None = None,
    result_interpolation_source_ego_field: str = "ego2global_transformation_matrix",
    result_interpolation_optimized_ego_output_field: str = (
        "ego2global_transformation_matrix_camera_reference_offset_optimized"
    ),
) -> MultiCameraExtrinsicOptimizationResult:
    total_started = time.perf_counter()
    phase_started = total_started
    phase_timings_seconds: dict[str, float] = {}
    output = Path(output)
    for relative in ("metrics", "observations", "visualizations"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    normalized = _normalize_multicamera_columns(matches)
    retained = _select_optimization_matches(normalized)
    retained = retained.loc[
        _navigation_coverage_mask(retained, navigation, navigation_offset_ns)
    ].copy()
    if retained.empty:
        raise ValueError("no retained matches available for multi-camera optimization")
    cameras = tuple(sorted(str(camera) for camera in calibrations))
    used_cameras = set(retained["camera1"].astype(str)) | set(retained["camera2"].astype(str))
    missing = sorted(used_cameras - set(cameras))
    if missing:
        raise ValueError(f"missing calibration(s): {', '.join(missing)}")

    pair_sampled = _sample_matches_per_image_pair(
        retained,
        max_matches_per_pair=pair_sample_max_matches,
        seed=pair_sample_seed,
        strategy=pair_sample_strategy,
        grid_cols=pair_sample_grid_cols,
        grid_rows=pair_sample_grid_rows,
    )
    sampled = _sample_multicamera_matches(
        pair_sampled, max_observations, seed=pair_sample_seed
    )
    evaluation_matches = _sample_multicamera_matches(retained, evaluation_max_observations)
    rotation_bound_rad = np.deg2rad(rotation_bound_deg)
    resolved_evaluation_device = (
        str(evaluation_device)
        if evaluation_device is not None
        else (str(torch_device) if str(optimizer_method) == "torch_sparse_graph" else "cpu")
    )
    resolved_evaluation_backend = _resolve_multicamera_evaluation_backend(
        evaluation_backend,
        evaluation_device=resolved_evaluation_device,
        calibrations=calibrations,
    )
    phase_timings_seconds["prepare_inputs"] = time.perf_counter() - phase_started

    def unpack(parameters: np.ndarray) -> dict[str, np.ndarray]:
        blocks = np.asarray(parameters, dtype=np.float64).reshape(len(cameras), 6)
        return {
            camera: _apply_sensor_to_car_delta(
                calibrations[camera].sensor_to_car,
                blocks[index],
            )
            for index, camera in enumerate(cameras)
        }

    initial_sensor_to_car = {
        camera: np.asarray(calibrations[camera].sensor_to_car, dtype=np.float64)
        for camera in cameras
    }
    print(
        f"[optimization] evaluating {len(evaluation_matches):,} retained matches before optimization "
        f"with {resolved_evaluation_backend}",
        flush=True,
    )
    before_evaluation_started = time.perf_counter()
    before_matches = _evaluate_multicamera_matches(
        evaluation_matches,
        navigation,
        calibrations,
        initial_sensor_to_car,
        navigation_offset_ns=navigation_offset_ns,
        evaluation_backend=evaluation_backend,
        evaluation_device=resolved_evaluation_device,
        evaluation_batch_size=evaluation_batch_size,
    )
    phase_timings_seconds["before_evaluation"] = time.perf_counter() - before_evaluation_started
    adaptive_mode = str(adaptive_weight_mode).lower()
    if adaptive_mode not in {"none", "group_static"}:
        raise ValueError(f"unsupported adaptive_weight_mode: {adaptive_weight_mode}")
    match_weights: np.ndarray | None = None
    adaptive_weight_summary = pd.DataFrame()
    adaptive_weights_reused_before_evaluation = False
    adaptive_started = time.perf_counter()
    if adaptive_mode == "group_static":
        if _can_reuse_before_matches_for_adaptive_weights(
            max_observations=max_observations,
            evaluation_max_observations=evaluation_max_observations,
            sampled_count=len(sampled),
            evaluation_count=len(evaluation_matches),
        ):
            sampled_before_for_weights = before_matches
            adaptive_weights_reused_before_evaluation = True
        else:
            sampled_before_for_weights = _evaluate_multicamera_matches(
                sampled,
                navigation,
                calibrations,
                initial_sensor_to_car,
                navigation_offset_ns=navigation_offset_ns,
                evaluation_backend=evaluation_backend,
                evaluation_device=resolved_evaluation_device,
                evaluation_batch_size=evaluation_batch_size,
            )
        match_weights, adaptive_weight_summary = _compute_static_adaptive_group_weights(
            sampled_before_for_weights,
            alpha=adaptive_weight_alpha,
            min_weight=adaptive_weight_min,
            max_weight=adaptive_weight_max,
            long_tail_ratio_guard=adaptive_weight_long_tail_ratio_guard,
            min_group_count=adaptive_weight_min_group_count,
        )
    scope_weights = np.where(
        sampled["match_scope"].astype(str).to_numpy() == "inter_camera",
        float(inter_weight),
        float(intra_weight),
    )
    if match_weights is None:
        match_weights = scope_weights.astype(np.float64, copy=False)
    else:
        match_weights = np.asarray(match_weights, dtype=np.float64) * scope_weights
    phase_timings_seconds["adaptive_weights"] = time.perf_counter() - adaptive_started

    def residuals(parameters: np.ndarray) -> np.ndarray:
        evaluated = _evaluate_multicamera_matches(
            sampled,
            navigation,
            calibrations,
            unpack(parameters),
            navigation_offset_ns=navigation_offset_ns,
            evaluation_backend=evaluation_backend,
            evaluation_device=resolved_evaluation_device,
            evaluation_batch_size=evaluation_batch_size,
        )
        values = np.sqrt(evaluated["nav_sampson_px2"].to_numpy(dtype=np.float64))
        values = np.nan_to_num(values, nan=1e3, posinf=1e3, neginf=1e3)
        blocks = np.asarray(parameters, dtype=np.float64).reshape(len(cameras), 6)
        prior_terms = []
        fixed_rotation_weight = max(float(fixed_rotation_l2_weight), 0.0)
        if fixed_rotation_weight > 0:
            prior_terms.append(
                np.sqrt(fixed_rotation_weight) * blocks[:, :3].reshape(-1)
            )
        fixed_translation_weight = max(float(fixed_translation_l2_weight), 0.0)
        if fixed_translation_weight > 0:
            prior_terms.append(
                np.sqrt(fixed_translation_weight) * blocks[:, 3:6].reshape(-1)
            )
        if prior_terms:
            values = np.concatenate([values, *prior_terms])
        return values

    single_lower = np.array(
        [
            -rotation_bound_rad,
            -rotation_bound_rad,
            -rotation_bound_rad,
            -translation_bound_m,
            -translation_bound_m,
            -translation_bound_m,
        ],
        dtype=np.float64,
    )
    single_upper = -single_lower
    lower_bounds = np.tile(single_lower, len(cameras))
    upper_bounds = np.tile(single_upper, len(cameras))
    if optimizer_method == "torch_sparse_graph":
        optimization_started = time.perf_counter()
        (
            parameters,
            frame_rotvecs,
            body_frame_disturbances,
            torch_status,
        ) = _torch_optimize_multi_camera_extrinsics(
            sampled=sampled,
            navigation=navigation,
            calibrations=calibrations,
            cameras=cameras,
            navigation_offset_ns=navigation_offset_ns,
            lower_one=single_lower,
            upper_one=single_upper,
            fixed_translation_l2_weight=fixed_translation_l2_weight,
            fixed_rotation_l2_weight=fixed_rotation_l2_weight,
            max_nfev=max_nfev,
            learning_rate=torch_learning_rate,
            device=torch_device,
            lr_schedule=torch_lr_schedule,
            lr_step_size=torch_lr_step_size,
            lr_decay=torch_lr_decay,
            lr_min_factor=torch_lr_min_factor,
            torch_dtype=torch_dtype,
            early_stopping_patience=torch_early_stopping_patience,
            early_stopping_min_delta=torch_early_stopping_min_delta,
            early_stopping_min_steps=torch_early_stopping_min_steps,
            optimize_time_varying_disturbance=optimize_time_varying_disturbance,
            initial_frame_rotvecs=initial_frame_rotvecs,
            match_weights=match_weights,
            time_varying_disturbance_dof=time_varying_disturbance_dof,
            optimize_body_time_varying_disturbance=optimize_body_time_varying_disturbance,
            body_time_varying_disturbance_dof=body_time_varying_disturbance_dof,
            time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
            time_varying_translation_bound_m=time_varying_translation_bound_m,
            time_varying_translation_l2_weight=time_varying_translation_l2_weight,
            time_varying_rotation_l2_weight=time_varying_rotation_l2_weight,
            time_varying_translation_smooth_l2_weight=time_varying_translation_smooth_l2_weight,
            time_varying_rotation_smooth_l2_weight=time_varying_rotation_smooth_l2_weight,
            time_varying_translation_mean_l2_weight=time_varying_translation_mean_l2_weight,
            time_varying_rotation_mean_l2_weight=time_varying_rotation_mean_l2_weight,
            time_varying_frame_common_translation_l2_weight=time_varying_frame_common_translation_l2_weight,
            time_varying_frame_common_rotation_l2_weight=time_varying_frame_common_rotation_l2_weight,
            body_time_varying_rotation_bound_deg=body_time_varying_rotation_bound_deg,
            body_time_varying_translation_bound_m=body_time_varying_translation_bound_m,
            body_time_varying_translation_l2_weight=body_time_varying_translation_l2_weight,
            body_time_varying_rotation_l2_weight=body_time_varying_rotation_l2_weight,
            body_time_varying_translation_smooth_l2_weight=body_time_varying_translation_smooth_l2_weight,
            body_time_varying_rotation_smooth_l2_weight=body_time_varying_rotation_smooth_l2_weight,
            mode_projection_enabled=mode_projection_enabled,
            mode_projection_interval_steps=mode_projection_interval_steps,
            mode_projection_absorb_frame_common=mode_projection_absorb_frame_common,
            mode_projection_absorb_camera_long_term=mode_projection_absorb_camera_long_term,
            mode_projection_reset_optimizer_state=mode_projection_reset_optimizer_state,
        )
        solver_x = np.asarray(parameters, dtype=np.float64)
        base_solver_status = torch_status
    elif optimizer_method == "scipy_least_squares":
        optimization_started = time.perf_counter()
        if optimize_time_varying_disturbance or optimize_body_time_varying_disturbance:
            raise ValueError(
                "time-varying disturbance optimization requires optimizer_method=torch_sparse_graph"
            )
        solver = least_squares(
            residuals,
            x0=np.zeros(len(cameras) * 6, dtype=np.float64),
            method="dogbox",
            loss="huber",
            f_scale=1.0,
            max_nfev=int(max_nfev),
            bounds=(lower_bounds, upper_bounds),
        )
        solver_x = solver.x
        frame_rotvecs = {}
        body_frame_disturbances = {}
        base_solver_status = {
            "success": bool(solver.success),
            "status": int(solver.status),
            "message": str(solver.message),
            "nfev": int(solver.nfev),
            "cost": float(solver.cost),
            "optimality": float(solver.optimality),
            "solver_method": "scipy_least_squares",
        }
    else:
        raise ValueError(f"unsupported optimizer_method: {optimizer_method}")
    phase_timings_seconds["optimization"] = time.perf_counter() - optimization_started
    optimized_sensor_to_car = unpack(solver_x)
    absorption_report: dict[str, object] = {
        "enabled": bool(absorb_camera_common_mode),
        "applied": False,
        "skip_reason": "",
    }
    if (
        absorb_camera_common_mode
        and optimize_time_varying_disturbance
        and optimize_body_time_varying_disturbance
        and frame_rotvecs
    ):
        absorption_started = time.perf_counter()
        (
            frame_rotvecs,
            body_frame_disturbances,
            absorption_report,
        ) = _absorb_camera_common_frame_disturbance(
            frame_rotvecs=frame_rotvecs,
            body_frame_disturbances=body_frame_disturbances,
            sensor_to_car_by_camera=optimized_sensor_to_car,
            min_cameras=camera_common_mode_min_cameras,
            min_median_cos=camera_common_mode_min_median_cos,
            min_positive_projection_fraction=camera_common_mode_min_positive_projection_fraction,
            body_disturbance_dof=body_time_varying_disturbance_dof,
        )
        absorption_report = {**absorption_report, "applied": True}
        phase_timings_seconds["camera_common_mode_absorption"] = (
            time.perf_counter() - absorption_started
        )
    elif absorb_camera_common_mode:
        absorption_report["skip_reason"] = (
            "requires both camera and body time-varying disturbance variables"
        )
    else:
        absorption_report["skip_reason"] = "disabled"
    long_term_absorption_report: dict[str, object] = {
        "enabled": bool(absorb_camera_long_term_mode),
        "applied": False,
        "skip_reason": "",
    }
    if (
        absorb_camera_long_term_mode
        and optimize_time_varying_disturbance
        and frame_rotvecs
    ):
        long_term_started = time.perf_counter()
        (
            optimized_sensor_to_car,
            frame_rotvecs,
            long_term_absorption_report,
        ) = _absorb_camera_long_term_frame_disturbance(
            frame_rotvecs=frame_rotvecs,
            sensor_to_car_by_camera=optimized_sensor_to_car,
            min_frames=camera_long_term_mode_min_frames,
            min_median_cos=camera_long_term_mode_min_median_cos,
            min_positive_projection_fraction=camera_long_term_mode_min_positive_projection_fraction,
            min_mean_to_rms=camera_long_term_mode_min_mean_to_rms,
        )
        long_term_absorption_report = {
            **long_term_absorption_report,
            "applied": True,
        }
        phase_timings_seconds["camera_long_term_mode_absorption"] = (
            time.perf_counter() - long_term_started
        )
    elif absorb_camera_long_term_mode:
        long_term_absorption_report["skip_reason"] = (
            "requires camera time-varying disturbance variables"
        )
    else:
        long_term_absorption_report["skip_reason"] = "disabled"
    print(
        f"[optimization] evaluating {len(evaluation_matches):,} retained matches after optimization "
        f"with {resolved_evaluation_backend}",
        flush=True,
    )
    after_evaluation_started = time.perf_counter()
    after_matches = _evaluate_multicamera_matches(
        evaluation_matches,
        navigation,
        calibrations,
        optimized_sensor_to_car,
        navigation_offset_ns=navigation_offset_ns,
        frame_rotvecs=frame_rotvecs if optimize_time_varying_disturbance else None,
        body_frame_disturbances=body_frame_disturbances
        if optimize_body_time_varying_disturbance
        else None,
        evaluation_backend=evaluation_backend,
        evaluation_device=resolved_evaluation_device,
        evaluation_batch_size=evaluation_batch_size,
    )
    phase_timings_seconds["after_evaluation"] = time.perf_counter() - after_evaluation_started

    reporting_started = time.perf_counter()
    print("[optimization] computing exact aggregate metrics", flush=True)
    summary_started = time.perf_counter()
    before_summary = _summarize_match_quality(before_matches)
    after_summary = _summarize_match_quality(after_matches)
    edge_summary = _summarize_multicamera_edges(before_matches, after_matches)
    parameters_by_camera = {
        camera: solver_x[index * 6 : (index + 1) * 6]
        for index, camera in enumerate(cameras)
    }
    camera_summary = _camera_delta_summary(
        calibrations,
        optimized_sensor_to_car,
        parameters_by_camera,
        single_lower,
        single_upper,
    )
    phase_timings_seconds["reporting_summaries"] = time.perf_counter() - summary_started

    print("[optimization] exporting outliers and deterministic QC sample", flush=True)
    observation_export_started = time.perf_counter()
    outlier_diagnostics = _write_reprojection_outlier_diagnostics(
        output=output,
        before_matches=before_matches,
        after_matches=after_matches,
        hard_threshold_px=100.0,
    )
    qc_matches = _build_reprojection_qc_matches(
        before_matches,
        after_matches,
        sample_ratio=projection_qc_sample_ratio,
    )
    qc_matches.to_parquet(
        output / "observations" / "qc_matches.parquet",
        index=False,
    )
    qc_group_count = int(
        len(qc_matches[_REPROJECTION_QC_GROUP_COLUMNS].drop_duplicates())
    )
    all_group_count = int(
        len(
            _with_reprojection_qc_group_columns(before_matches)[
                _REPROJECTION_QC_GROUP_COLUMNS
            ].drop_duplicates()
        )
    )
    (output / "observations" / "qc_sample_manifest.json").write_text(
        json.dumps(
            {
                "sample_ratio": float(projection_qc_sample_ratio),
                "total_image_pairs": all_group_count,
                "selected_image_pairs": qc_group_count,
                "selected_matches": int(len(qc_matches)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    phase_timings_seconds["reporting_observation_export"] = (
        time.perf_counter() - observation_export_started
    )
    edge_summary.to_csv(output / "metrics" / "edge_summary.csv", index=False)
    if "cost_by_step_all" in base_solver_status:
        loss_values = list(base_solver_status["cost_by_step_all"])
        learning_rate_values = list(
            base_solver_status.get(
                "learning_rate_by_step_all",
                [np.nan] * len(loss_values),
            )
        )
        pd.DataFrame(
            {
                "step": np.arange(1, len(loss_values) + 1),
                "loss": loss_values,
                "learning_rate": learning_rate_values,
            }
        ).to_csv(output / "metrics" / "loss_history.csv", index=False)
    camera_summary.to_csv(output / "metrics" / "camera_delta_summary.csv", index=False)
    if adaptive_mode == "group_static":
        adaptive_weight_summary.to_csv(
            output / "metrics" / "adaptive_group_weights.csv", index=False
        )
    frame_summary = _frame_disturbance_summary(frame_rotvecs)
    if optimize_time_varying_disturbance:
        frame_summary.to_csv(output / "metrics" / "frame_disturbance_summary.csv", index=False)
    body_summary = _body_disturbance_summary(body_frame_disturbances)
    if optimize_body_time_varying_disturbance:
        body_summary.to_csv(
            output / "metrics" / "body_pose_disturbance_summary.csv", index=False
        )
    absorption_per_frame = pd.DataFrame(absorption_report.get("per_frame", []))
    if not absorption_per_frame.empty:
        absorption_per_frame.to_csv(
            output / "metrics" / "camera_common_mode_absorption.csv",
            index=False,
        )
    long_term_absorption_per_camera = pd.DataFrame(
        long_term_absorption_report.get("per_camera", [])
    )
    if not long_term_absorption_per_camera.empty:
        long_term_absorption_per_camera.to_csv(
            output / "metrics" / "camera_long_term_mode_absorption.csv",
            index=False,
        )
    print("[optimization] exporting per-image camera poses", flush=True)
    image_extrinsics_started = time.perf_counter()
    _export_image_camera_to_world_csvs(
        output=output,
        matches=retained,
        navigation=navigation,
        initial_sensor_to_car=initial_sensor_to_car,
        optimized_sensor_to_car=optimized_sensor_to_car,
        navigation_offset_ns=navigation_offset_ns,
        frame_rotvecs=frame_rotvecs,
        body_frame_disturbances=body_frame_disturbances,
        frame_metadata=frame_metadata,
    )
    phase_timings_seconds["reporting_image_extrinsics"] = (
        time.perf_counter() - image_extrinsics_started
    )
    solver_status = {
        **base_solver_status,
        "cameras": list(cameras),
        "parameters_by_camera": {
            camera: parameters_by_camera[camera].tolist() for camera in cameras
        },
        "retained_observations": int(len(retained)),
        "pair_sample_max_matches": pair_sample_max_matches,
        "pair_sample_seed": int(pair_sample_seed),
        "pair_sample_strategy": str(pair_sample_strategy),
        "pair_sample_grid_cols": int(pair_sample_grid_cols),
        "pair_sample_grid_rows": int(pair_sample_grid_rows),
        "pair_sampled_observations": int(len(pair_sampled)),
        "sampled_observations": int(len(sampled)),
        "evaluated_observations": int(len(evaluation_matches)),
        "max_observations": max_observations,
        "evaluation_max_observations": evaluation_max_observations,
        "max_nfev": int(max_nfev),
        "optimizer_method": str(optimizer_method),
        "torch_device": str(torch_device),
        "torch_learning_rate": float(torch_learning_rate),
        "torch_lr_schedule": str(torch_lr_schedule),
        "torch_lr_step_size": int(torch_lr_step_size),
        "torch_lr_decay": float(torch_lr_decay),
        "torch_lr_min_factor": float(torch_lr_min_factor),
        "torch_dtype": str(torch_dtype),
        "torch_early_stopping_patience": None
        if torch_early_stopping_patience is None
        else int(torch_early_stopping_patience),
        "torch_early_stopping_min_delta": float(torch_early_stopping_min_delta),
        "torch_early_stopping_min_steps": int(torch_early_stopping_min_steps),
        "evaluation_backend": str(evaluation_backend),
        "resolved_evaluation_backend": str(resolved_evaluation_backend),
        "evaluation_device": str(resolved_evaluation_device),
        "evaluation_batch_size": int(evaluation_batch_size),
        "before_legacy_refinement_count": int(
            before_matches.attrs.get("legacy_refinement_count", 0)
        ),
        "after_legacy_refinement_count": int(
            after_matches.attrs.get("legacy_refinement_count", 0)
        ),
        "projection_qc_sample_ratio": float(projection_qc_sample_ratio),
        "projection_qc_total_image_pairs": all_group_count,
        "projection_qc_selected_image_pairs": qc_group_count,
        "projection_qc_selected_matches": int(len(qc_matches)),
        "adaptive_weight_mode": adaptive_mode,
        "adaptive_weight_alpha": float(adaptive_weight_alpha),
        "adaptive_weight_min": float(adaptive_weight_min),
        "adaptive_weight_max": float(adaptive_weight_max),
        "adaptive_weight_long_tail_ratio_guard": float(
            adaptive_weight_long_tail_ratio_guard
        ),
        "adaptive_weight_min_group_count": int(adaptive_weight_min_group_count),
        "intra_weight": float(intra_weight),
        "inter_weight": float(inter_weight),
        "adaptive_weights_reused_before_evaluation": bool(
            adaptive_weights_reused_before_evaluation
        ),
        "optimize_time_varying_disturbance": bool(optimize_time_varying_disturbance),
        "time_varying_disturbance_dof": int(time_varying_disturbance_dof),
        "optimize_body_time_varying_disturbance": bool(
            optimize_body_time_varying_disturbance
        ),
        "body_time_varying_disturbance_dof": int(body_time_varying_disturbance_dof),
        "time_varying_rotation_bound_deg": float(time_varying_rotation_bound_deg),
        "time_varying_translation_bound_m": float(time_varying_translation_bound_m),
        "body_time_varying_rotation_bound_deg": float(
            body_time_varying_rotation_bound_deg
        ),
        "body_time_varying_translation_bound_m": float(
            body_time_varying_translation_bound_m
        ),
        "translation_bound_m": float(translation_bound_m),
        "rotation_bound_deg": float(rotation_bound_deg),
        "fixed_translation_l2_weight": float(fixed_translation_l2_weight),
        "fixed_rotation_l2_weight": float(fixed_rotation_l2_weight),
        "time_varying_translation_l2_weight": float(time_varying_translation_l2_weight),
        "time_varying_rotation_l2_weight": float(time_varying_rotation_l2_weight),
        "time_varying_translation_smooth_l2_weight": float(
            time_varying_translation_smooth_l2_weight
        ),
        "time_varying_rotation_smooth_l2_weight": float(
            time_varying_rotation_smooth_l2_weight
        ),
        "body_time_varying_translation_l2_weight": float(
            body_time_varying_translation_l2_weight
        ),
        "body_time_varying_rotation_l2_weight": float(
            body_time_varying_rotation_l2_weight
        ),
        "body_time_varying_translation_smooth_l2_weight": float(
            body_time_varying_translation_smooth_l2_weight
        ),
        "body_time_varying_rotation_smooth_l2_weight": float(
            body_time_varying_rotation_smooth_l2_weight
        ),
        "time_varying_translation_mean_l2_weight": float(
            time_varying_translation_mean_l2_weight
        ),
        "time_varying_rotation_mean_l2_weight": float(
            time_varying_rotation_mean_l2_weight
        ),
        "time_varying_frame_common_translation_l2_weight": float(
            time_varying_frame_common_translation_l2_weight
        ),
        "time_varying_frame_common_rotation_l2_weight": float(
            time_varying_frame_common_rotation_l2_weight
        ),
        "mode_projection_enabled": bool(mode_projection_enabled),
        "mode_projection_interval_steps": int(mode_projection_interval_steps),
        "mode_projection_absorb_frame_common": bool(mode_projection_absorb_frame_common),
        "mode_projection_absorb_camera_long_term": bool(
            mode_projection_absorb_camera_long_term
        ),
        "mode_projection_reset_optimizer_state": bool(
            mode_projection_reset_optimizer_state
        ),
        "absorb_camera_common_mode": bool(absorb_camera_common_mode),
        "camera_common_mode_min_cameras": int(camera_common_mode_min_cameras),
        "camera_common_mode_min_median_cos": float(camera_common_mode_min_median_cos),
        "camera_common_mode_min_positive_projection_fraction": float(
            camera_common_mode_min_positive_projection_fraction
        ),
        "camera_common_mode_absorption": {
            key: value
            for key, value in absorption_report.items()
            if key != "per_frame"
        },
        "absorb_camera_long_term_mode": bool(absorb_camera_long_term_mode),
        "camera_long_term_mode_min_frames": int(camera_long_term_mode_min_frames),
        "camera_long_term_mode_min_median_cos": float(camera_long_term_mode_min_median_cos),
        "camera_long_term_mode_min_positive_projection_fraction": float(
            camera_long_term_mode_min_positive_projection_fraction
        ),
        "camera_long_term_mode_min_mean_to_rms": float(
            camera_long_term_mode_min_mean_to_rms
        ),
        "camera_long_term_mode_absorption": {
            key: value
            for key, value in long_term_absorption_report.items()
            if key != "per_camera"
        },
        "navigation_time_offset_ns": int(navigation_offset_ns),
    }
    phase_timings_seconds["reporting_and_export"] = time.perf_counter() - reporting_started
    phase_timings_seconds["total"] = time.perf_counter() - total_started
    solver_status["phase_timings_seconds"] = {
        key: float(value) for key, value in phase_timings_seconds.items()
    }
    optimized_yaml_path = output / "optimized_multi_camera_extrinsics.yaml"
    optimized_yaml_payload = {
                "initial_sensor_to_car": {
                    camera: calibrations[camera].sensor_to_car.tolist()
                    for camera in cameras
                },
                "optimized_sensor_to_car": {
                    camera: optimized_sensor_to_car[camera].tolist()
                    for camera in cameras
                },
                "optimized_frame_rotvecs": {
                    f"{camera}:{frame}": np.asarray(rotvec, dtype=np.float64).tolist()
                    for (camera, frame), rotvec in sorted(frame_rotvecs.items())
                },
                "optimized_body_frame_disturbances": {
                    int(frame): np.asarray(value, dtype=np.float64).tolist()
                    for frame, value in sorted(body_frame_disturbances.items())
                },
                "solver": solver_status,
            }
    optimized_yaml_path.write_text(
        yaml.safe_dump(
            optimized_yaml_payload,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if result_interpolation_pkl_path is not None:
        print("[optimization] exporting optimized result_interpolation pkl", flush=True)
        pkl_export_started = time.perf_counter()
        deliverables = output / "deliverables"
        generate_optimized_result_interpolation_pkl(
            input_pkl=Path(result_interpolation_pkl_path),
            optimized_yaml=optimized_yaml_path,
            image_extrinsics_csv=deliverables
            / "image_extrinsics"
            / "image_camera_to_world_after.csv",
            output_pkl=deliverables / "result_interpolation_optimized.pkl",
            manifest_json=deliverables
            / "result_interpolation_optimized_manifest.json",
            validation_json=deliverables
            / "result_interpolation_optimized_validation.json",
            source_ego_field=result_interpolation_source_ego_field,
            optimized_ego_output_field=result_interpolation_optimized_ego_output_field,
        )
        phase_timings_seconds["reporting_optimized_pkl"] = (
            time.perf_counter() - pkl_export_started
        )
    phase_timings_seconds["reporting_and_export"] = time.perf_counter() - reporting_started
    phase_timings_seconds["total"] = time.perf_counter() - total_started
    solver_status["phase_timings_seconds"] = {
        key: float(value) for key, value in phase_timings_seconds.items()
    }
    optimized_yaml_payload["solver"] = solver_status
    optimized_yaml_path.write_text(
        yaml.safe_dump(
            optimized_yaml_payload,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "before": before_summary,
                "after": after_summary,
                "after_reprojection_inliers": outlier_diagnostics["after_inliers"],
                "after_reprojection_outlier_diagnostics": outlier_diagnostics,
                "solver": solver_status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MultiCameraExtrinsicOptimizationResult(
        output=output,
        optimized_sensor_to_car=optimized_sensor_to_car,
        before_matches=before_matches,
        after_matches=after_matches,
        before_summary=before_summary,
        after_summary=after_summary,
        edge_summary=edge_summary,
        solver_status=solver_status,
    )


def optimize_multi_camera_alternating_from_observations(
    *,
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibrations: dict[str, CameraCalibration],
    output: Path,
    block_rounds: int = 1,
    block_max_nfev: int = 500,
    global_max_nfev: int = 1500,
    max_observations: int | None = None,
    evaluation_max_observations: int | None = None,
    navigation_offset_ns: int = 0,
    torch_device: str = "cuda:0",
    torch_learning_rate: float = 0.003,
    torch_lr_schedule: str = "step",
    torch_lr_step_size: int = 100,
    torch_lr_decay: float = 0.5,
    torch_lr_min_factor: float = 0.1,
    torch_dtype: str = "float32",
    torch_early_stopping_patience: int | None = 100,
    torch_early_stopping_min_delta: float = 1e-3,
    torch_early_stopping_min_steps: int = 500,
    time_varying_rotation_bound_deg: float = 3.0,
    time_varying_prior_sigma_deg: float = 0.3,
    time_varying_prior_weight: float = 10.0,
    time_varying_smooth_sigma_deg: float = 0.3,
    time_varying_smooth_weight: float = 1.0,
    block_intra_weight: float = 1.0,
    block_inter_weight: float = 0.5,
    run_global_refine: bool = True,
    translation_bound_m: float = 0.3,
    rotation_bound_deg: float = 2.0,
    translation_prior_sigma_m: float = 0.03,
    rotation_prior_sigma_deg: float = 0.3,
    translation_prior_weight: float = 100.0,
    rotation_prior_weight: float = 100.0,
    prior_normalization_reference_observations: int = 100_000,
) -> MultiCameraAlternatingOptimizationResult:
    output = Path(output)
    for relative in ("metrics", "observations"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    normalized = _normalize_multicamera_columns(matches)
    retained = _select_optimization_matches(normalized)
    retained = retained.loc[
        _navigation_coverage_mask(retained, navigation, navigation_offset_ns)
    ].copy()
    if retained.empty:
        raise ValueError("no retained matches available for alternating optimization")
    cameras = tuple(sorted(str(camera) for camera in calibrations))
    sensor_to_car_by_camera = {
        camera: np.asarray(calibrations[camera].sensor_to_car, dtype=np.float64)
        for camera in cameras
    }
    sampled_all = _sample_multicamera_matches(retained, max_observations)
    evaluation_matches = _sample_multicamera_matches(retained, evaluation_max_observations)
    current_frame_rotvecs: dict[tuple[str, int], np.ndarray] = {}
    before_matches = _evaluate_multicamera_matches(
        evaluation_matches,
        navigation,
        calibrations,
        sensor_to_car_by_camera,
        navigation_offset_ns=navigation_offset_ns,
    )
    before_summary = _summarize_match_quality(before_matches)
    block_records: list[dict[str, object]] = []
    block_order: list[str] = []
    block_statuses: list[dict[str, object]] = []

    for round_index in range(max(1, int(block_rounds))):
        current_eval = _evaluate_multicamera_matches(
            evaluation_matches,
            navigation,
            calibrations,
            sensor_to_car_by_camera,
            navigation_offset_ns=navigation_offset_ns,
            frame_rotvecs=current_frame_rotvecs or None,
        )
        round_order = _rank_cameras_by_intra_reprojection(current_eval, cameras)
        for camera in round_order:
            incident = _matches_incident_to_camera(sampled_all, camera)
            before_block_eval = _evaluate_multicamera_matches(
                _matches_incident_to_camera(evaluation_matches, camera),
                navigation,
                calibrations,
                sensor_to_car_by_camera,
                navigation_offset_ns=navigation_offset_ns,
                frame_rotvecs=current_frame_rotvecs or None,
            )
            current_frame_rotvecs, status = _optimize_single_camera_time_varying_block(
                target_camera=camera,
                sampled=incident,
                navigation=navigation,
                calibrations=calibrations,
                sensor_to_car_by_camera=sensor_to_car_by_camera,
                navigation_offset_ns=navigation_offset_ns,
                initial_frame_rotvecs=current_frame_rotvecs,
                max_nfev=block_max_nfev,
                learning_rate=torch_learning_rate,
                lr_schedule=torch_lr_schedule,
                lr_step_size=torch_lr_step_size,
                lr_decay=torch_lr_decay,
                lr_min_factor=torch_lr_min_factor,
                device=torch_device,
                time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
                time_varying_prior_sigma_deg=time_varying_prior_sigma_deg,
                time_varying_prior_weight=time_varying_prior_weight,
                time_varying_smooth_sigma_deg=time_varying_smooth_sigma_deg,
                time_varying_smooth_weight=time_varying_smooth_weight,
                intra_weight=block_intra_weight,
                inter_weight=block_inter_weight,
            )
            after_block_eval = _evaluate_multicamera_matches(
                _matches_incident_to_camera(evaluation_matches, camera),
                navigation,
                calibrations,
                sensor_to_car_by_camera,
                navigation_offset_ns=navigation_offset_ns,
                frame_rotvecs=current_frame_rotvecs or None,
            )
            before_block_summary = _summarize_match_quality(before_block_eval)
            after_block_summary = _summarize_match_quality(after_block_eval)
            block_order.append(camera)
            block_statuses.append(status)
            block_records.append(
                {
                    "round": int(round_index),
                    "camera": camera,
                    "incident_observations": int(len(incident)),
                    "before_reprojection_median_px": before_block_summary[
                        "reprojection_median_px"
                    ],
                    "after_reprojection_median_px": after_block_summary[
                        "reprojection_median_px"
                    ],
                    "before_reprojection_p95_px": before_block_summary[
                        "reprojection_p95_px"
                    ],
                    "after_reprojection_p95_px": after_block_summary[
                        "reprojection_p95_px"
                    ],
                    "cost": status.get("cost", np.nan),
                    "target_frame_variable_count": status.get(
                        "target_frame_variable_count", 0
                    ),
                }
            )

    after_block_matches = _evaluate_multicamera_matches(
        evaluation_matches,
        navigation,
        calibrations,
        sensor_to_car_by_camera,
        navigation_offset_ns=navigation_offset_ns,
        frame_rotvecs=current_frame_rotvecs or None,
    )
    after_block_summary = _summarize_match_quality(after_block_matches)
    block_summary = pd.DataFrame.from_records(block_records)
    before_matches.to_parquet(output / "observations" / "before_matches.parquet", index=False)
    after_block_matches.to_parquet(
        output / "observations" / "after_block_matches.parquet", index=False
    )
    block_summary.to_csv(output / "metrics" / "block_summary.csv", index=False)
    _frame_disturbance_summary(current_frame_rotvecs).to_csv(
        output / "metrics" / "frame_disturbance_summary_after_blocks.csv",
        index=False,
    )
    global_refine_output: str | None = None
    if run_global_refine:
        global_output = output / "global_refine"
        optimize_multi_camera_extrinsics_from_observations(
            matches=retained,
            navigation=navigation,
            calibrations=calibrations,
            output=global_output,
            max_observations=max_observations,
            translation_bound_m=translation_bound_m,
            rotation_bound_deg=rotation_bound_deg,
            translation_prior_sigma_m=translation_prior_sigma_m,
            rotation_prior_sigma_deg=rotation_prior_sigma_deg,
            translation_prior_weight=translation_prior_weight,
            rotation_prior_weight=rotation_prior_weight,
            prior_normalization_reference_observations=prior_normalization_reference_observations,
            navigation_offset_ns=navigation_offset_ns,
            max_nfev=global_max_nfev,
            evaluation_max_observations=evaluation_max_observations,
            optimizer_method="torch_sparse_graph",
            torch_device=torch_device,
            torch_learning_rate=torch_learning_rate,
            torch_lr_schedule=torch_lr_schedule,
            torch_lr_step_size=torch_lr_step_size,
            torch_lr_decay=torch_lr_decay,
            torch_lr_min_factor=torch_lr_min_factor,
            torch_dtype=torch_dtype,
            torch_early_stopping_patience=torch_early_stopping_patience,
            torch_early_stopping_min_delta=torch_early_stopping_min_delta,
            torch_early_stopping_min_steps=torch_early_stopping_min_steps,
            optimize_time_varying_disturbance=True,
            initial_frame_rotvecs=current_frame_rotvecs,
            time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
            time_varying_prior_sigma_deg=time_varying_prior_sigma_deg,
            time_varying_prior_weight=time_varying_prior_weight,
            time_varying_smooth_sigma_deg=time_varying_smooth_sigma_deg,
            time_varying_smooth_weight=time_varying_smooth_weight,
        )
        global_refine_output = str(global_output)

    solver_status = {
        "block_rounds": int(block_rounds),
        "block_order": block_order,
        "block_statuses": block_statuses,
        "block_max_nfev": int(block_max_nfev),
        "global_max_nfev": int(global_max_nfev),
        "global_refine_output": global_refine_output,
        "block_intra_weight": float(block_intra_weight),
        "block_inter_weight": float(block_inter_weight),
        "sampled_observations": int(len(sampled_all)),
        "evaluated_observations": int(len(evaluation_matches)),
        "torch_device": str(torch_device),
        "torch_learning_rate": float(torch_learning_rate),
        "torch_lr_schedule": str(torch_lr_schedule),
        "torch_dtype": str(torch_dtype),
        "torch_early_stopping_patience": None
        if torch_early_stopping_patience is None
        else int(torch_early_stopping_patience),
        "torch_early_stopping_min_delta": float(torch_early_stopping_min_delta),
        "torch_early_stopping_min_steps": int(torch_early_stopping_min_steps),
        "time_varying_rotation_bound_deg": float(time_varying_rotation_bound_deg),
    }
    (output / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "before": before_summary,
                "after_blocks": after_block_summary,
                "solver": solver_status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MultiCameraAlternatingOptimizationResult(
        output=output,
        frame_rotvecs=current_frame_rotvecs,
        before_matches=before_matches,
        after_block_matches=after_block_matches,
        before_summary=before_summary,
        after_block_summary=after_block_summary,
        block_summary=block_summary,
        solver_status=solver_status,
    )


def _load_retained_match_file(path: Path, match_scope: str, camera1: str, camera2: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame = _normalize_multicamera_columns(frame)
    frame["match_scope"] = match_scope
    expected_cameras = {str(camera1), str(camera2)}
    actual_cameras = set(frame["camera1"].astype(str)) | set(frame["camera2"].astype(str))
    if actual_cameras - expected_cameras:
        raise ValueError(
            f"match file {path} contains cameras {sorted(actual_cameras)} "
            f"outside directory pair {sorted(expected_cameras)}"
        )
    if "camera1" not in frame.columns:
        frame["camera1"] = camera1
    if "camera2" not in frame.columns:
        frame["camera2"] = camera2
    return frame


def _normalize_match_cache_roots(
    match_cache_root: Path | str | list[Path | str] | tuple[Path | str, ...],
) -> tuple[Path, ...]:
    if isinstance(match_cache_root, (str, Path)):
        return (Path(match_cache_root),)
    roots = tuple(Path(root) for root in match_cache_root)
    if not roots:
        raise ValueError("at least one match cache root is required")
    return roots


def _first_existing_retained_path(
    match_cache_roots: tuple[Path, ...],
    relative_dir: Path,
) -> Path | None:
    for root in match_cache_roots:
        path = root / relative_dir / "retained_matches.parquet"
        if path.exists():
            return path
    return None


def _collect_multicamera_match_cache(
    match_cache_root: Path | str | list[Path | str] | tuple[Path | str, ...],
    cameras: list[str] | tuple[str, ...],
    intra_offsets: list[int] | tuple[int, ...] | None,
    intra_baselines_m: list[float] | tuple[float, ...] | None,
    inter_pairs: list[str] | tuple[str, ...] | None,
    inter_relations: list[str] | tuple[str, ...] | None = None,
    inter_relations_by_pair: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    match_cache_roots = _normalize_match_cache_roots(match_cache_root)
    pieces: list[pd.DataFrame] = []
    for camera in cameras:
        intra_dirs: list[Path] = []
        if intra_offsets is not None:
            intra_dirs.extend(
                Path("intra_camera") / camera / f"offsets_{int(offset)}"
                for offset in intra_offsets
            )
        if intra_baselines_m is not None:
            for baseline in intra_baselines_m:
                label = f"{float(baseline):g}m".replace(".", "p")
                intra_dirs.append(Path("intra_camera") / camera / f"baselines_{label}")
        if intra_offsets is None and intra_baselines_m is None:
            seen_dirs: set[str] = set()
            for root in match_cache_roots:
                camera_root = root / "intra_camera" / camera
                for directory in [
                    *sorted(camera_root.glob("offsets_*")),
                    *sorted(camera_root.glob("baselines_*")),
                ]:
                    relative = directory.relative_to(root)
                    key = relative.as_posix()
                    if key not in seen_dirs:
                        intra_dirs.append(relative)
                        seen_dirs.add(key)
        for intra_dir in intra_dirs:
            path = _first_existing_retained_path(match_cache_roots, intra_dir)
            if path is not None:
                pieces.append(
                    _load_retained_match_file(path, "intra_camera", camera, camera)
                )
    if inter_pairs is not None:
        pair_names = [str(pair) for pair in inter_pairs]
    else:
        pair_names = []
        seen_pairs: set[str] = set()
        for root in match_cache_roots:
            inter_root = root / "inter_camera"
            if not inter_root.exists():
                continue
            for path in sorted(inter_root.iterdir()):
                if path.is_dir() and path.name not in seen_pairs:
                    pair_names.append(path.name)
                    seen_pairs.add(path.name)
    for pair in pair_names:
        if "__" not in pair:
            raise ValueError(f"inter-camera pair must use camera_a__camera_b naming: {pair}")
        camera1, camera2 = pair.split("__", 1)
        selected_relations = (
            tuple(str(relation) for relation in inter_relations_by_pair[pair])
            if inter_relations_by_pair is not None and pair in inter_relations_by_pair
            else None if inter_relations is None
            else tuple(str(relation) for relation in inter_relations)
        )
        relation_dirs = (
            [Path("inter_camera") / pair / relation for relation in selected_relations]
            if selected_relations is not None
            else []
        )
        if selected_relations is not None:
            seen_relation_dirs = {relation_dir.as_posix() for relation_dir in relation_dirs}
            for root in match_cache_roots:
                pair_root = root / "inter_camera" / pair
                if not pair_root.exists():
                    continue
                for relation in selected_relations:
                    prefix = f"{relation}_"
                    for path in sorted(pair_root.glob(f"{prefix}*")):
                        if not path.is_dir():
                            continue
                        relative = path.relative_to(root)
                        key = relative.as_posix()
                        if key not in seen_relation_dirs:
                            relation_dirs.append(relative)
                            seen_relation_dirs.add(key)
        if selected_relations is None:
            seen_relations: set[str] = set()
            for root in match_cache_roots:
                pair_root = root / "inter_camera" / pair
                if not pair_root.exists():
                    continue
                for path in sorted(pair_root.iterdir()):
                    if path.is_dir() and path.name not in seen_relations:
                        relation_dirs.append(path.relative_to(root))
                        seen_relations.add(path.name)
        for relation_dir in relation_dirs:
            path = _first_existing_retained_path(match_cache_roots, relation_dir)
            if path is not None:
                pieces.append(
                    _load_retained_match_file(path, "inter_camera", camera1, camera2)
                )
    pieces = [frame for frame in pieces if not frame.empty]
    if not pieces:
        roots_text = ", ".join(str(root) for root in match_cache_roots)
        raise FileNotFoundError(f"no retained matches found under {roots_text}")
    return pd.concat(pieces, ignore_index=True)


def optimize_multi_camera_extrinsics_from_match_cache(
    config_path: Path,
    match_cache_root: Path | str | list[Path | str] | tuple[Path | str, ...],
    output: Path,
    cameras: list[str] | tuple[str, ...],
    intra_offsets: list[int] | tuple[int, ...] | None = None,
    intra_baselines_m: list[float] | tuple[float, ...] | None = None,
    inter_pairs: list[str] | tuple[str, ...] | None = None,
    inter_relations: list[str] | tuple[str, ...] | None = None,
    inter_relations_by_pair: Mapping[str, Sequence[str]] | None = None,
    max_observations: int | None = None,
    pair_sample_max_matches: int | None = None,
    pair_sample_seed: int = 20260704,
    pair_sample_strategy: str = "spatial_grid_random",
    pair_sample_grid_cols: int = 3,
    pair_sample_grid_rows: int = 3,
    translation_bound_m: float = 1.0,
    rotation_bound_deg: float = 10.0,
    fixed_translation_l2_weight: float = 50.0,
    fixed_rotation_l2_weight: float = 300.0,
    max_nfev: int = 1500,
    evaluation_max_observations: int | None = None,
    optimizer_method: str = "torch_sparse_graph",
    torch_device: str = "cuda:0",
    torch_learning_rate: float = 0.003,
    torch_lr_schedule: str = "step",
    torch_lr_step_size: int = 100,
    torch_lr_decay: float = 0.5,
    torch_lr_min_factor: float = 0.1,
    torch_dtype: str = "float32",
    torch_early_stopping_patience: int | None = 100,
    torch_early_stopping_min_delta: float = 1e-3,
    torch_early_stopping_min_steps: int = 500,
    evaluation_backend: str = "auto",
    evaluation_device: str | None = None,
    evaluation_batch_size: int = 50_000,
    projection_qc_sample_ratio: float = 0.01,
    adaptive_weight_mode: str = "none",
    adaptive_weight_alpha: float = 0.5,
    adaptive_weight_min: float = 0.5,
    adaptive_weight_max: float = 2.0,
    adaptive_weight_long_tail_ratio_guard: float = 8.0,
    adaptive_weight_min_group_count: int = 500,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
    optimize_time_varying_disturbance: bool = False,
    time_varying_disturbance_dof: int = 6,
    optimize_body_time_varying_disturbance: bool = False,
    body_time_varying_disturbance_dof: int = 6,
    time_varying_rotation_bound_deg: float = 15.0,
    time_varying_translation_bound_m: float = 0.15,
    time_varying_translation_l2_weight: float = 200.0,
    time_varying_rotation_l2_weight: float = 5.0,
    time_varying_translation_smooth_l2_weight: float = 20.0,
    time_varying_rotation_smooth_l2_weight: float = 0.5,
    time_varying_translation_mean_l2_weight: float = 5000.0,
    time_varying_rotation_mean_l2_weight: float = 2500.0,
    time_varying_frame_common_translation_l2_weight: float = 5000.0,
    time_varying_frame_common_rotation_l2_weight: float = 2500.0,
    body_time_varying_rotation_bound_deg: float = 15.0,
    body_time_varying_translation_bound_m: float = 0.15,
    body_time_varying_translation_l2_weight: float = 200.0,
    body_time_varying_rotation_l2_weight: float = 5.0,
    body_time_varying_translation_smooth_l2_weight: float = 20.0,
    body_time_varying_rotation_smooth_l2_weight: float = 0.5,
    mode_projection_enabled: bool = False,
    mode_projection_interval_steps: int = 100,
    mode_projection_absorb_frame_common: bool = True,
    mode_projection_absorb_camera_long_term: bool = True,
    mode_projection_reset_optimizer_state: bool = True,
    absorb_camera_common_mode: bool = True,
    camera_common_mode_min_cameras: int = 3,
    camera_common_mode_min_median_cos: float = 0.5,
    camera_common_mode_min_positive_projection_fraction: float = 0.8,
    absorb_camera_long_term_mode: bool = True,
    camera_long_term_mode_min_frames: int = 30,
    camera_long_term_mode_min_median_cos: float = 0.5,
    camera_long_term_mode_min_positive_projection_fraction: float = 0.8,
    camera_long_term_mode_min_mean_to_rms: float = 0.35,
) -> Path:
    config = load_config(Path(config_path))
    result_interpolation_path = resolve_result_interpolation_path(
        config.assumptions,
        config.dataset_root,
    )
    if result_interpolation_path is not None:
        result_interpolation_ego_field = str(
            config.assumptions.get(
                "result_interpolation_ego_field",
                "ego2global_transformation_matrix",
            )
        )
        result_interpolation_optimized_ego_output_field = str(
            config.assumptions.get(
                "result_interpolation_optimized_ego_output_field",
                "ego2global_transformation_matrix",
            )
        )
        result_interpolation_extrinsic_direction = str(
            config.assumptions.get(
                "result_interpolation_extrinsic_direction",
                "camera_to_car",
            )
        )
        calibrations = calibrations_from_result_interpolation(
            result_interpolation_path,
            tuple(cameras),
            extrinsic_direction=result_interpolation_extrinsic_direction,
            images_undistorted=bool(
                config.assumptions.get(
                    "result_interpolation_images_undistorted",
                    False,
                )
            ),
        )
        navigation = navigation_from_result_interpolation(
            result_interpolation_path,
            ego_field=result_interpolation_ego_field,
            allow_end_extrapolation=bool(
                config.assumptions.get("extrapolate_navigation", False)
            ),
        )
    else:
        result_interpolation_ego_field = "ego2global_transformation_matrix"
        result_interpolation_optimized_ego_output_field = (
            "ego2global_transformation_matrix"
        )
        calibrations = {
            camera: CameraCalibration.from_dataset(config.dataset_root, camera)
            for camera in cameras
        }
        navigation = NavigationTrajectory.from_csv(
            config.dataset_root / config.assumptions["navigation_source"]
        )
    navigation_offset_ns = navigation_time_offset_ns(config.assumptions)
    if result_interpolation_path is not None:
        image_root = resolve_result_interpolation_image_root(
            config.assumptions,
            config.dataset_root,
        )
        if image_root is not None:
            frame_metadata = frame_metadata_from_image_root(image_root, tuple(cameras))
        else:
            frame_metadata = frame_metadata_from_result_interpolation(
                result_interpolation_path,
                tuple(cameras),
                image_root=image_root,
            )
    else:
        try:
            frame_metadata = _frame_metadata_from_dataset(config.dataset_root, tuple(cameras))
        except FileNotFoundError:
            frame_metadata = None
    matches = _collect_multicamera_match_cache(
        match_cache_root,
        tuple(cameras),
        None if intra_offsets is None else tuple(intra_offsets),
        None if intra_baselines_m is None else tuple(intra_baselines_m),
        None if inter_pairs is None else tuple(inter_pairs),
        None if inter_relations is None else tuple(inter_relations),
        inter_relations_by_pair=inter_relations_by_pair,
    )
    optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=Path(output),
        max_observations=max_observations,
        pair_sample_max_matches=pair_sample_max_matches,
        pair_sample_seed=pair_sample_seed,
        pair_sample_strategy=pair_sample_strategy,
        pair_sample_grid_cols=pair_sample_grid_cols,
        pair_sample_grid_rows=pair_sample_grid_rows,
        translation_bound_m=translation_bound_m,
        rotation_bound_deg=rotation_bound_deg,
        fixed_translation_l2_weight=fixed_translation_l2_weight,
        fixed_rotation_l2_weight=fixed_rotation_l2_weight,
        navigation_offset_ns=navigation_offset_ns,
        max_nfev=max_nfev,
        evaluation_max_observations=evaluation_max_observations,
        optimizer_method=optimizer_method,
        torch_device=torch_device,
        torch_learning_rate=torch_learning_rate,
        torch_lr_schedule=torch_lr_schedule,
        torch_lr_step_size=torch_lr_step_size,
        torch_lr_decay=torch_lr_decay,
        torch_lr_min_factor=torch_lr_min_factor,
        torch_dtype=torch_dtype,
        torch_early_stopping_patience=torch_early_stopping_patience,
        torch_early_stopping_min_delta=torch_early_stopping_min_delta,
        torch_early_stopping_min_steps=torch_early_stopping_min_steps,
        evaluation_backend=evaluation_backend,
        evaluation_device=evaluation_device,
        evaluation_batch_size=evaluation_batch_size,
        projection_qc_sample_ratio=projection_qc_sample_ratio,
        adaptive_weight_mode=adaptive_weight_mode,
        adaptive_weight_alpha=adaptive_weight_alpha,
        adaptive_weight_min=adaptive_weight_min,
        adaptive_weight_max=adaptive_weight_max,
        adaptive_weight_long_tail_ratio_guard=adaptive_weight_long_tail_ratio_guard,
        adaptive_weight_min_group_count=adaptive_weight_min_group_count,
        intra_weight=intra_weight,
        inter_weight=inter_weight,
        optimize_time_varying_disturbance=optimize_time_varying_disturbance,
        time_varying_disturbance_dof=time_varying_disturbance_dof,
        optimize_body_time_varying_disturbance=optimize_body_time_varying_disturbance,
        body_time_varying_disturbance_dof=body_time_varying_disturbance_dof,
        time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
        time_varying_translation_bound_m=time_varying_translation_bound_m,
        time_varying_translation_l2_weight=time_varying_translation_l2_weight,
        time_varying_rotation_l2_weight=time_varying_rotation_l2_weight,
        time_varying_translation_smooth_l2_weight=time_varying_translation_smooth_l2_weight,
        time_varying_rotation_smooth_l2_weight=time_varying_rotation_smooth_l2_weight,
        time_varying_translation_mean_l2_weight=time_varying_translation_mean_l2_weight,
        time_varying_rotation_mean_l2_weight=time_varying_rotation_mean_l2_weight,
        time_varying_frame_common_translation_l2_weight=time_varying_frame_common_translation_l2_weight,
        time_varying_frame_common_rotation_l2_weight=time_varying_frame_common_rotation_l2_weight,
        body_time_varying_rotation_bound_deg=body_time_varying_rotation_bound_deg,
        body_time_varying_translation_bound_m=body_time_varying_translation_bound_m,
        body_time_varying_translation_l2_weight=body_time_varying_translation_l2_weight,
        body_time_varying_rotation_l2_weight=body_time_varying_rotation_l2_weight,
        body_time_varying_translation_smooth_l2_weight=body_time_varying_translation_smooth_l2_weight,
        body_time_varying_rotation_smooth_l2_weight=body_time_varying_rotation_smooth_l2_weight,
        mode_projection_enabled=mode_projection_enabled,
        mode_projection_interval_steps=mode_projection_interval_steps,
        mode_projection_absorb_frame_common=mode_projection_absorb_frame_common,
        mode_projection_absorb_camera_long_term=mode_projection_absorb_camera_long_term,
        mode_projection_reset_optimizer_state=mode_projection_reset_optimizer_state,
        absorb_camera_common_mode=absorb_camera_common_mode,
        camera_common_mode_min_cameras=camera_common_mode_min_cameras,
        camera_common_mode_min_median_cos=camera_common_mode_min_median_cos,
        camera_common_mode_min_positive_projection_fraction=camera_common_mode_min_positive_projection_fraction,
        absorb_camera_long_term_mode=absorb_camera_long_term_mode,
        camera_long_term_mode_min_frames=camera_long_term_mode_min_frames,
        camera_long_term_mode_min_median_cos=camera_long_term_mode_min_median_cos,
        camera_long_term_mode_min_positive_projection_fraction=camera_long_term_mode_min_positive_projection_fraction,
        camera_long_term_mode_min_mean_to_rms=camera_long_term_mode_min_mean_to_rms,
        frame_metadata=frame_metadata,
        result_interpolation_pkl_path=result_interpolation_path,
        result_interpolation_source_ego_field=result_interpolation_ego_field,
        result_interpolation_optimized_ego_output_field=(
            result_interpolation_optimized_ego_output_field
        ),
    )
    return Path(output)


def optimize_multi_camera_alternating_from_match_cache(
    config_path: Path,
    match_cache_root: Path | str | list[Path | str] | tuple[Path | str, ...],
    output: Path,
    cameras: list[str] | tuple[str, ...],
    intra_offsets: list[int] | tuple[int, ...] | None = None,
    intra_baselines_m: list[float] | tuple[float, ...] | None = None,
    inter_pairs: list[str] | tuple[str, ...] | None = None,
    inter_relations: list[str] | tuple[str, ...] | None = None,
    block_rounds: int = 1,
    block_max_nfev: int = 500,
    global_max_nfev: int = 1500,
    max_observations: int | None = None,
    evaluation_max_observations: int | None = None,
    torch_device: str = "cuda:0",
    torch_learning_rate: float = 0.003,
    torch_lr_schedule: str = "step",
    torch_lr_step_size: int = 100,
    torch_lr_decay: float = 0.5,
    torch_lr_min_factor: float = 0.1,
    torch_dtype: str = "float32",
    torch_early_stopping_patience: int | None = 100,
    torch_early_stopping_min_delta: float = 1e-3,
    torch_early_stopping_min_steps: int = 500,
    time_varying_rotation_bound_deg: float = 3.0,
    time_varying_prior_sigma_deg: float = 0.3,
    time_varying_prior_weight: float = 10.0,
    time_varying_smooth_sigma_deg: float = 0.3,
    time_varying_smooth_weight: float = 1.0,
    block_intra_weight: float = 1.0,
    block_inter_weight: float = 0.5,
    run_global_refine: bool = True,
    translation_bound_m: float = 1.0,
    rotation_bound_deg: float = 10.0,
    translation_prior_sigma_m: float = 0.03,
    rotation_prior_sigma_deg: float = 0.3,
    translation_prior_weight: float = 100.0,
    rotation_prior_weight: float = 100.0,
    prior_normalization_reference_observations: int = 100_000,
) -> Path:
    config = load_config(Path(config_path))
    calibrations = {
        camera: CameraCalibration.from_dataset(config.dataset_root, camera)
        for camera in cameras
    }
    navigation = NavigationTrajectory.from_csv(
        config.dataset_root / config.assumptions["navigation_source"]
    )
    navigation_offset_ns = navigation_time_offset_ns(config.assumptions)
    matches = _collect_multicamera_match_cache(
        match_cache_root,
        tuple(cameras),
        None if intra_offsets is None else tuple(intra_offsets),
        None if intra_baselines_m is None else tuple(intra_baselines_m),
        None if inter_pairs is None else tuple(inter_pairs),
        None if inter_relations is None else tuple(inter_relations),
    )
    optimize_multi_camera_alternating_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=Path(output),
        block_rounds=block_rounds,
        block_max_nfev=block_max_nfev,
        global_max_nfev=global_max_nfev,
        max_observations=max_observations,
        evaluation_max_observations=evaluation_max_observations,
        navigation_offset_ns=navigation_offset_ns,
        torch_device=torch_device,
        torch_learning_rate=torch_learning_rate,
        torch_lr_schedule=torch_lr_schedule,
        torch_lr_step_size=torch_lr_step_size,
        torch_lr_decay=torch_lr_decay,
        torch_lr_min_factor=torch_lr_min_factor,
        torch_dtype=torch_dtype,
        torch_early_stopping_patience=torch_early_stopping_patience,
        torch_early_stopping_min_delta=torch_early_stopping_min_delta,
        torch_early_stopping_min_steps=torch_early_stopping_min_steps,
        time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
        time_varying_prior_sigma_deg=time_varying_prior_sigma_deg,
        time_varying_prior_weight=time_varying_prior_weight,
        time_varying_smooth_sigma_deg=time_varying_smooth_sigma_deg,
        time_varying_smooth_weight=time_varying_smooth_weight,
        block_intra_weight=block_intra_weight,
        block_inter_weight=block_inter_weight,
        run_global_refine=run_global_refine,
        translation_bound_m=translation_bound_m,
        rotation_bound_deg=rotation_bound_deg,
        translation_prior_sigma_m=translation_prior_sigma_m,
        rotation_prior_sigma_deg=rotation_prior_sigma_deg,
        translation_prior_weight=translation_prior_weight,
        rotation_prior_weight=rotation_prior_weight,
        prior_normalization_reference_observations=prior_normalization_reference_observations,
    )
    return Path(output)


