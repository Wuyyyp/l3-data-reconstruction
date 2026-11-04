from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .camera_models import CameraCalibration, project_points
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
from .match_paths import resolve_retained_matches_path
from .navigation import NavigationTrajectory, navigation_time_offset_ns
from .visualization import draw_tiled_before_after_reprojection_overlay


def build_frame_variable_index(matches: pd.DataFrame) -> dict[int, int]:
    frames = sorted(
        set(matches["frame1"].astype(int).tolist())
        | set(matches["frame2"].astype(int).tolist())
    )
    return {frame: index for index, frame in enumerate(frames)}


@dataclass(frozen=True)
class KnotFrameInterpolator:
    frames: tuple[int, ...]
    knots: tuple[int, ...]
    frame_weights: dict[int, tuple[int, int, float]]
    dimension: int = 3

    @property
    def variable_count(self) -> int:
        return len(self.knots) * int(self.dimension)

    def unpack(self, parameters: np.ndarray) -> dict[int, np.ndarray]:
        knot_values = np.asarray(parameters, dtype=np.float64).reshape(
            len(self.knots), int(self.dimension)
        )
        return self.frame_values(knot_values)

    def frame_values(self, knot_values: np.ndarray) -> dict[int, np.ndarray]:
        knot_values = np.asarray(knot_values, dtype=np.float64).reshape(
            len(self.knots), int(self.dimension)
        )
        result: dict[int, np.ndarray] = {}
        for frame in self.frames:
            left, right, alpha = self.frame_weights[frame]
            if left == right:
                result[frame] = knot_values[left].copy()
            else:
                result[frame] = (
                    (1.0 - alpha) * knot_values[left] + alpha * knot_values[right]
                )
        return result


def build_knot_frame_interpolator(
    frames: list[int] | tuple[int, ...],
    knot_stride: int,
    dimension: int = 3,
) -> KnotFrameInterpolator:
    dimension = int(dimension)
    if dimension < 3:
        raise ValueError("dimension must be at least 3")
    unique_frames = tuple(sorted({int(frame) for frame in frames}))
    if not unique_frames:
        return KnotFrameInterpolator((), (), {}, dimension=dimension)
    stride = max(1, int(knot_stride))
    first = unique_frames[0]
    last = unique_frames[-1]
    knots = list(range(first, last + 1, stride))
    if knots[-1] != last:
        knots.append(last)
    knot_array = np.asarray(knots, dtype=np.float64)
    frame_weights: dict[int, tuple[int, int, float]] = {}
    for frame in unique_frames:
        right = int(np.searchsorted(knot_array, frame, side="left"))
        if right <= 0:
            frame_weights[frame] = (0, 0, 0.0)
        elif right >= len(knots):
            frame_weights[frame] = (len(knots) - 1, len(knots) - 1, 0.0)
        elif knots[right] == frame:
            frame_weights[frame] = (right, right, 0.0)
        else:
            left = right - 1
            alpha = (frame - knots[left]) / float(knots[right] - knots[left])
            frame_weights[frame] = (left, right, float(alpha))
    return KnotFrameInterpolator(
        tuple(unique_frames), tuple(knots), frame_weights, dimension=dimension
    )


def build_jacobian_sparsity(
    matches: pd.DataFrame,
    frame_to_block: dict[int, int],
) -> lil_matrix:
    match_count = len(matches)
    frame_count = len(frame_to_block)
    smooth_count = max(0, frame_count - 1) * 3
    residual_count = match_count + frame_count * 3 + smooth_count
    variable_count = frame_count * 3
    sparsity = lil_matrix((residual_count, variable_count), dtype=int)

    for row_index, row in enumerate(matches.itertuples(index=False)):
        for frame in (int(row.frame1), int(row.frame2)):
            start = frame_to_block[frame] * 3
            sparsity[row_index, start : start + 3] = 1

    row_offset = match_count
    for frame_index in range(frame_count):
        for axis in range(3):
            sparsity[row_offset + frame_index * 3 + axis, frame_index * 3 + axis] = 1

    row_offset += frame_count * 3
    for frame_index in range(frame_count - 1):
        for axis in range(3):
            row = row_offset + frame_index * 3 + axis
            sparsity[row, frame_index * 3 + axis] = 1
            sparsity[row, (frame_index + 1) * 3 + axis] = 1
    return sparsity


def _frame_knot_blocks(
    interpolator: KnotFrameInterpolator,
    frame: int,
) -> tuple[int, ...]:
    left, right, _alpha = interpolator.frame_weights[int(frame)]
    return (left,) if left == right else (left, right)


def build_knot_jacobian_sparsity(
    matches: pd.DataFrame,
    interpolator: KnotFrameInterpolator,
) -> lil_matrix:
    match_count = len(matches)
    knot_count = len(interpolator.knots)
    dimension = int(interpolator.dimension)
    smooth_count = max(0, knot_count - 1) * dimension
    residual_count = match_count + knot_count * dimension + smooth_count
    variable_count = interpolator.variable_count
    sparsity = lil_matrix((residual_count, variable_count), dtype=int)

    for row_index, row in enumerate(matches.itertuples(index=False)):
        blocks = set(_frame_knot_blocks(interpolator, int(row.frame1))) | set(
            _frame_knot_blocks(interpolator, int(row.frame2))
        )
        for block in blocks:
            start = block * dimension
            sparsity[row_index, start : start + dimension] = 1

    row_offset = match_count
    for knot_index in range(knot_count):
        for axis in range(dimension):
            sparsity[
                row_offset + knot_index * dimension + axis,
                knot_index * dimension + axis,
            ] = 1

    row_offset += knot_count * dimension
    for knot_index in range(knot_count - 1):
        for axis in range(dimension):
            row = row_offset + knot_index * dimension + axis
            sparsity[row, knot_index * dimension + axis] = 1
            sparsity[row, (knot_index + 1) * dimension + axis] = 1
    return sparsity


def _delta_transform(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        raise ValueError("delta transform requires at least 3 rotation values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    if len(values) >= 6:
        transform[:3, 3] = values[3:6]
    return transform


def _camera_to_world(
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    timestamp_ns: int,
    rotvec: np.ndarray,
    navigation_offset_ns: int,
) -> np.ndarray:
    body_to_world = navigation.pose_at(int(timestamp_ns) + int(navigation_offset_ns))
    return body_to_world @ calibration.sensor_to_car @ _delta_transform(rotvec)


def _disturbance_dimension(frame_rotvecs: dict[int, np.ndarray]) -> int:
    for value in frame_rotvecs.values():
        size = int(np.asarray(value, dtype=np.float64).reshape(-1).size)
        if size:
            return max(3, size)
    return 3


def _evaluate_group(
    group: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    first_rotvec: np.ndarray,
    second_rotvec: np.ndarray,
    navigation_offset_ns: int,
) -> pd.DataFrame:
    first_camera = _camera_to_world(
        navigation,
        calibration,
        int(group["timestamp1_ns"].iloc[0]),
        first_rotvec,
        navigation_offset_ns,
    )
    second_camera = _camera_to_world(
        navigation,
        calibration,
        int(group["timestamp2_ns"].iloc[0]),
        second_rotvec,
        navigation_offset_ns,
    )
    first_to_second = np.linalg.inv(second_camera) @ first_camera
    essential = essential_from_relative_pose(first_to_second)
    fundamental = fundamental_from_essential(
        essential,
        calibration.intrinsic,
        calibration.intrinsic,
    )
    pixel_errors = epipolar_errors(
        group[["x1", "y1"]].to_numpy(dtype=np.float64),
        group[["x2", "y2"]].to_numpy(dtype=np.float64),
        fundamental,
    )
    result = group.copy()
    result["tv_sampson_px"] = np.sqrt(pixel_errors["sampson"])
    result["tv_symmetric_epipolar_px"] = pixel_errors["symmetric_epipolar"]
    return result


def evaluate_time_varying_rotation(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    frame_rotvecs: dict[int, np.ndarray],
    navigation_offset_ns: int = 0,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    zero = np.zeros(_disturbance_dimension(frame_rotvecs), dtype=np.float64)
    for (_, _), group in matches.groupby(["frame1", "frame2"], sort=False):
        first = int(group["frame1"].iloc[0])
        second = int(group["frame2"].iloc[0])
        groups.append(
            _evaluate_group(
                group,
                navigation,
                calibration,
                frame_rotvecs.get(first, zero),
                frame_rotvecs.get(second, zero),
                navigation_offset_ns,
            )
        )
    return pd.concat(groups, ignore_index=True) if groups else matches.iloc[0:0].copy()


def _evaluate_reprojection_group(
    group: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    first_rotvec: np.ndarray,
    second_rotvec: np.ndarray,
    navigation_offset_ns: int,
) -> pd.DataFrame:
    result = group.copy()
    first_camera_to_world = _camera_to_world(
        navigation,
        calibration,
        int(group["timestamp1_ns"].iloc[0]),
        first_rotvec,
        navigation_offset_ns,
    )
    second_camera_to_world = _camera_to_world(
        navigation,
        calibration,
        int(group["timestamp2_ns"].iloc[0]),
        second_rotvec,
        navigation_offset_ns,
    )
    first_world_to_camera = np.linalg.inv(first_camera_to_world)
    second_world_to_camera = np.linalg.inv(second_camera_to_world)
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
    first_pixels = project_points(first_camera_points, calibration)
    second_pixels = project_points(second_camera_points, calibration)
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
    result["tv_triangulation_valid_depth"] = triangulation["valid_depth"]
    result["tv_triangulation_angle_deg"] = ray_metrics["triangulation_angle_deg"]
    result["tv_ray_closest_distance_m"] = ray_metrics["ray_closest_distance_m"]
    result["tv_reprojection1_px"] = first_reprojection
    result["tv_reprojection2_px"] = second_reprojection
    result["tv_reprojection_rms_px"] = np.sqrt(
        (first_reprojection**2 + second_reprojection**2) / 2.0
    )
    result[["tv_reprojected1_x", "tv_reprojected1_y"]] = first_pixels
    result[["tv_reprojected2_x", "tv_reprojected2_y"]] = second_pixels
    return result


def evaluate_time_varying_reprojection(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    frame_rotvecs: dict[int, np.ndarray],
    navigation_offset_ns: int = 0,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    zero = np.zeros(_disturbance_dimension(frame_rotvecs), dtype=np.float64)
    for (_, _), group in matches.groupby(["frame1", "frame2"], sort=False):
        first = int(group["frame1"].iloc[0])
        second = int(group["frame2"].iloc[0])
        groups.append(
            _evaluate_reprojection_group(
                group,
                navigation,
                calibration,
                frame_rotvecs.get(first, zero),
                frame_rotvecs.get(second, zero),
                navigation_offset_ns,
            )
        )
    return pd.concat(groups, ignore_index=True) if groups else matches.iloc[0:0].copy()


def sample_matches(matches: pd.DataFrame, max_observations: int | None) -> pd.DataFrame:
    if max_observations is None or len(matches) <= max_observations:
        return matches.reset_index(drop=True)
    if max_observations <= 0:
        return matches.iloc[0:0].copy().reset_index(drop=True)
    group_sizes = matches.groupby(["frame1", "frame2"], sort=False).size()
    proportions = (group_sizes / group_sizes.sum()).to_dict()
    pieces = []
    remaining = int(max_observations)
    grouped = list(matches.groupby(["frame1", "frame2"], sort=False))
    for index, ((frame1, frame2), group) in enumerate(grouped):
        if remaining <= 0:
            break
        if index == len(grouped) - 1:
            take = min(len(group), remaining)
        else:
            take = min(
                len(group),
                remaining,
                max(1, int(round(max_observations * proportions[(frame1, frame2)]))),
            )
        remaining -= take
        pieces.append(group.sample(n=take, random_state=int(frame1) * 10007 + int(frame2)))
    return pd.concat(pieces, ignore_index=True) if pieces else matches.iloc[0:0].copy()


def summarize_frame_disturbance(
    matches: pd.DataFrame,
    frame_rotvecs: dict[int, np.ndarray],
) -> pd.DataFrame:
    records = []
    frames = sorted(
        set(matches["frame1"].astype(int).tolist())
        | set(matches["frame2"].astype(int).tolist())
    )
    for frame in frames:
        incident = matches[
            (matches["frame1"].astype(int) == frame)
            | (matches["frame2"].astype(int) == frame)
        ]
        disturbance = np.asarray(frame_rotvecs.get(frame, np.zeros(3)), dtype=np.float64).reshape(-1)
        rot = disturbance[:3]
        trans = disturbance[3:6] if len(disturbance) >= 6 else np.zeros(3, dtype=np.float64)
        rot_deg = np.rad2deg(rot)
        records.append(
            {
                "frame": int(frame),
                "delta_roll_deg": float(rot_deg[0]),
                "delta_pitch_deg": float(rot_deg[1]),
                "delta_yaw_deg": float(rot_deg[2]),
                "delta_norm_deg": float(np.linalg.norm(rot_deg)),
                "delta_tx_m": float(trans[0]),
                "delta_ty_m": float(trans[1]),
                "delta_tz_m": float(trans[2]),
                "delta_translation_norm_m": float(np.linalg.norm(trans)),
                "incident_pair_count": int(
                    incident[["frame1", "frame2"]].drop_duplicates().shape[0]
                ),
                "incident_match_count": int(len(incident)),
                "before_incident_sampson_mean_px": float(
                    incident["before_sampson_px"].mean()
                ),
                "after_incident_sampson_mean_px": float(
                    incident["after_sampson_px"].mean()
                ),
                "before_incident_sampson_median_px": float(
                    incident["before_sampson_px"].median()
                ),
                "after_incident_sampson_median_px": float(
                    incident["after_sampson_px"].median()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _export_framewise_equivalent_extrinsics(
    output: Path,
    camera: str,
    base_extrinsic_source: str,
    calibration: CameraCalibration,
    frame_rotvecs: dict[int, np.ndarray],
    frame_timestamps_ns: dict[int, int] | None = None,
    frame_image_paths: dict[int, Path] | None = None,
) -> tuple[Path, Path, Path]:
    output = Path(output)
    base = np.asarray(calibration.sensor_to_car, dtype=np.float64)
    records: list[dict[str, float | int | str]] = []
    for frame in sorted(int(value) for value in frame_rotvecs):
        disturbance = np.asarray(frame_rotvecs[frame], dtype=np.float64).reshape(-1)
        rotvec = disturbance[:3]
        trans = disturbance[3:6] if len(disturbance) >= 6 else np.zeros(3, dtype=np.float64)
        equivalent = base @ _delta_transform(disturbance)
        image_path = None if frame_image_paths is None else frame_image_paths.get(frame)
        record: dict[str, float | int | str] = {
            "camera": camera,
            "frame": int(frame),
            "timestamp_ns": int(frame_timestamps_ns.get(frame, -1)) if frame_timestamps_ns else -1,
            "image_path": "" if image_path is None else str(image_path),
            "image_filename": "" if image_path is None else image_path.name,
            "base_extrinsic_source": str(base_extrinsic_source),
            "sensor_to_car_tx_m": float(equivalent[0, 3]),
            "sensor_to_car_ty_m": float(equivalent[1, 3]),
            "sensor_to_car_tz_m": float(equivalent[2, 3]),
            "delta_roll_deg": float(np.rad2deg(rotvec[0])),
            "delta_pitch_deg": float(np.rad2deg(rotvec[1])),
            "delta_yaw_deg": float(np.rad2deg(rotvec[2])),
            "delta_tx_m": float(trans[0]),
            "delta_ty_m": float(trans[1]),
            "delta_tz_m": float(trans[2]),
            "delta_translation_norm_m": float(np.linalg.norm(trans)),
        }
        for row in range(4):
            for column in range(4):
                record[f"sensor_to_car_{row}{column}"] = float(equivalent[row, column])
        records.append(record)
    exported = pd.DataFrame.from_records(records)
    csv_path = output / "observations" / "framewise_equivalent_extrinsics.csv"
    parquet_path = output / "observations" / "framewise_equivalent_extrinsics.parquet"
    summary_path = output / "metrics" / "framewise_equivalent_extrinsics_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(csv_path, index=False)
    exported.to_parquet(parquet_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "camera": camera,
                "base_extrinsic_source": str(base_extrinsic_source),
                "frame_count": int(len(exported)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    dictionary_path = output / "framewise_equivalent_extrinsics_README.md"
    dictionary_lines = [
        f"# {camera} framewise equivalent extrinsics",
        "",
        "一行对应一张图片，可直接按 `image_path` 或 `image_filename` 对照核验。",
        "",
        "## 字段说明",
        "",
        "- `camera`: 相机名称",
        "- `frame`: 序列内帧号",
        "- `timestamp_ns`: 图像纳秒时间戳",
        "- `image_path`: 图像绝对路径",
        "- `image_filename`: 图像文件名",
        "- `base_extrinsic_source`: 原始安装外参文件路径",
        "- `delta_roll_deg`, `delta_pitch_deg`, `delta_yaw_deg`: 在原始安装外参基础上的每帧小扰动角",
        "- `sensor_to_car_tx_m`, `sensor_to_car_ty_m`, `sensor_to_car_tz_m`: 每帧等效外参平移",
        "- `sensor_to_car_00` ... `sensor_to_car_33`: 每帧等效 `sensor_to_car` 4x4 齐次变换矩阵，按行展开",
        "",
        "## 关系说明",
        "",
        "- 原始固定安装外参来自 `base_extrinsic_source`。",
        "- 每帧等效外参 = 原始固定安装外参 × 当前帧小扰动旋转。",
        "- 当前导出用于甲方按图片逐帧核验，不替换原始标定文件定义。",
        "",
        "## 文件",
        "",
        "- `observations/framewise_equivalent_extrinsics.csv`",
        "- `observations/framewise_equivalent_extrinsics.parquet`",
        "- `metrics/framewise_equivalent_extrinsics_summary.json`",
    ]
    dictionary_path.write_text("\n".join(dictionary_lines) + "\n", encoding="utf-8")
    deliverable_root = output / "deliverables"
    deliverable_root.mkdir(parents=True, exist_ok=True)
    client_columns = ["image_filename"] + [
        f"sensor_to_car_{row}{column}" for row in range(4) for column in range(4)
    ]
    client_exported = exported[client_columns].copy()
    client_exported.to_csv(deliverable_root / "image_camera_extrinsics.csv", index=False)
    client_exported.to_parquet(
        deliverable_root / "image_camera_extrinsics.parquet", index=False
    )
    (deliverable_root / "README.md").write_text(
        "\n".join(
            [
                f"# {camera} image camera extrinsics",
                "",
                "交付内容为逐图片外参表。",
                "",
                "- 一行对应一张图片。",
                "- `image_filename` 是图片文件名。",
                "- `sensor_to_car_00` ... `sensor_to_car_33` 是该图片对应的 4x4 相机外参矩阵，按行展开。",
                "",
                "文件：",
                "",
                "- `image_camera_extrinsics.csv`",
                "- `image_camera_extrinsics.parquet`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, parquet_path, summary_path


def _export_image_camera_to_world_deliverable(
    output: Path,
    frame_image_paths: dict[int, Path],
    camera_to_world_by_frame: dict[int, np.ndarray],
) -> Path:
    output = Path(output)
    deliverable_root = output / "deliverables"
    deliverable_root.mkdir(parents=True, exist_ok=True)
    columns = ["image_filename"] + [
        f"camera_to_world_{row}{column}" for row in range(4) for column in range(4)
    ]
    rows: list[dict[str, float | str]] = []
    for frame in sorted(int(value) for value in camera_to_world_by_frame):
        image_path = frame_image_paths.get(frame)
        if image_path is None:
            continue
        transform = np.asarray(camera_to_world_by_frame[frame], dtype=np.float64)
        record: dict[str, float | str] = {"image_filename": image_path.name}
        for row in range(4):
            for column in range(4):
                record[f"camera_to_world_{row}{column}"] = float(transform[row, column])
        rows.append(record)
    csv_path = deliverable_root / "image_camera_to_world.csv"
    pd.DataFrame.from_records(rows, columns=columns).to_csv(csv_path, index=False)
    for stale_name in (
        "README.md",
        "image_camera_extrinsics.csv",
        "image_camera_extrinsics.parquet",
    ):
        stale_path = deliverable_root / stale_name
        if stale_path.exists():
            stale_path.unlink()
    return csv_path


def _export_image_camera_to_world_origin_deliverable(
    output: Path,
    frame_image_paths: dict[int, Path],
    camera_to_world_origin_by_frame: dict[int, np.ndarray],
) -> Path:
    output = Path(output)
    deliverable_root = output / "deliverables"
    deliverable_root.mkdir(parents=True, exist_ok=True)
    columns = ["image_filename"] + [
        f"camera_to_world_{row}{column}" for row in range(4) for column in range(4)
    ]
    rows: list[dict[str, float | str]] = []
    for frame in sorted(int(value) for value in camera_to_world_origin_by_frame):
        image_path = frame_image_paths.get(frame)
        if image_path is None:
            continue
        transform = np.asarray(camera_to_world_origin_by_frame[frame], dtype=np.float64)
        record: dict[str, float | str] = {"image_filename": image_path.name}
        for row in range(4):
            for column in range(4):
                record[f"camera_to_world_{row}{column}"] = float(transform[row, column])
        rows.append(record)
    csv_path = deliverable_root / "image_camera_to_world_origin.csv"
    pd.DataFrame.from_records(rows, columns=columns).to_csv(csv_path, index=False)
    return csv_path


def _export_angle_change_deliverable(
    output: Path,
    frame_image_paths: dict[int, Path],
    angle_change_deg_by_frame: dict[int, float],
) -> Path:
    output = Path(output)
    deliverable_root = output / "deliverables"
    deliverable_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for frame in sorted(int(value) for value in angle_change_deg_by_frame):
        image_path = frame_image_paths.get(frame)
        if image_path is None:
            continue
        rows.append(
            {
                "image_filename": image_path.name,
                "angle_change_deg": float(angle_change_deg_by_frame[frame]),
            }
        )
    csv_path = deliverable_root / "angle_change_deg.csv"
    pd.DataFrame.from_records(
        rows, columns=["image_filename", "angle_change_deg"]
    ).to_csv(csv_path, index=False)
    return csv_path


def _summarize_matches(matches: pd.DataFrame, column: str) -> dict[str, float | int]:
    summary = summarize_errors(matches[column].to_numpy(dtype=np.float64))
    return {
        "count": int(summary.get("count", 0)),
        "mean": float(summary.get("mean", np.nan)),
        "median": float(summary.get("median", np.nan)),
        "rmse": float(summary.get("rmse", np.nan)),
        "p90": float(summary.get("p90", np.nan)),
        "p95": float(summary.get("p95", np.nan)),
        "max": float(summary.get("max", np.nan)),
    }


def _summarize_offset_split(
    comparison: pd.DataFrame,
    *,
    train_offsets: set[int],
    eval_offsets: set[int],
) -> pd.DataFrame:
    groups: list[tuple[str, pd.DataFrame]] = [("all", comparison)]
    if train_offsets:
        groups.append(
            (
                "train",
                comparison.loc[
                    comparison["frame_offset"].astype(int).isin(sorted(train_offsets))
                ].copy(),
            )
        )
    if eval_offsets:
        groups.append(
            (
                "eval",
                comparison.loc[
                    comparison["frame_offset"].astype(int).isin(sorted(eval_offsets))
                ].copy(),
            )
        )
    rows: list[dict[str, float | int | str]] = []
    for group_name, group in groups:
        before_reproj = _summarize_matches(group, "before_reprojection_rms_px")
        after_reproj = _summarize_matches(group, "after_reprojection_rms_px")
        before_sampson = _summarize_matches(group, "before_sampson_px")
        after_sampson = _summarize_matches(group, "after_sampson_px")
        rows.append(
            {
                "split_group": group_name,
                "count": int(before_reproj["count"]),
                "before_reprojection_median_px": float(before_reproj["median"]),
                "after_reprojection_median_px": float(after_reproj["median"]),
                "before_reprojection_p95_px": float(before_reproj["p95"]),
                "after_reprojection_p95_px": float(after_reproj["p95"]),
                "before_reprojection_rmse_px": float(before_reproj["rmse"]),
                "after_reprojection_rmse_px": float(after_reproj["rmse"]),
                "before_sampson_median_px": float(before_sampson["median"]),
                "after_sampson_median_px": float(after_sampson["median"]),
                "before_sampson_p95_px": float(before_sampson["p95"]),
                "after_sampson_p95_px": float(after_sampson["p95"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _resolve_smooth_sigma_axes_deg(
    *,
    smooth_sigma_deg: float,
    roll_smooth_sigma_deg: float | None,
    pitch_smooth_sigma_deg: float | None,
    yaw_smooth_sigma_deg: float | None,
) -> np.ndarray:
    base = float(smooth_sigma_deg)
    return np.array(
        [
            base if roll_smooth_sigma_deg is None else float(roll_smooth_sigma_deg),
            base if pitch_smooth_sigma_deg is None else float(pitch_smooth_sigma_deg),
            base if yaw_smooth_sigma_deg is None else float(yaw_smooth_sigma_deg),
        ],
        dtype=np.float64,
    )


def _coordinate_descent_optimize_knots(
    *,
    sampled: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    interpolator: KnotFrameInterpolator,
    navigation_offset_ns: int,
    lower_one: np.ndarray,
    upper_one: np.ndarray,
    prior_sigma: np.ndarray,
    smooth_sigma: np.ndarray,
    sweeps: int,
    local_max_nfev: int,
) -> tuple[np.ndarray, dict[str, object]]:
    knot_count = len(interpolator.knots)
    knot_values = np.zeros((knot_count, 3), dtype=np.float64)
    frame_to_knots = {
        frame: set(_frame_knot_blocks(interpolator, frame)) for frame in interpolator.frames
    }
    affected_matches: list[pd.DataFrame] = []
    for knot_index in range(knot_count):
        affected_frames = {
            frame for frame, blocks in frame_to_knots.items() if knot_index in blocks
        }
        mask = sampled["frame1"].astype(int).isin(affected_frames) | sampled[
            "frame2"
        ].astype(int).isin(affected_frames)
        affected_matches.append(sampled.loc[mask].copy())

    costs: list[float] = []
    total_nfev = 0
    local_success_count = 0
    local_failure_count = 0
    for _sweep in range(max(1, int(sweeps))):
        sweep_cost = 0.0
        for knot_index in range(knot_count):
            local_matches = affected_matches[knot_index]
            if local_matches.empty:
                continue

            def local_residual(value: np.ndarray) -> np.ndarray:
                trial_values = knot_values.copy()
                trial_values[knot_index] = value
                frame_rotvecs = interpolator.unpack(trial_values.reshape(-1))
                evaluated = evaluate_time_varying_rotation(
                    local_matches,
                    navigation,
                    calibration,
                    frame_rotvecs,
                    navigation_offset_ns=navigation_offset_ns,
                )
                data = evaluated["tv_sampson_px"].to_numpy(dtype=np.float64)
                data = np.nan_to_num(data, nan=1e3, posinf=1e3, neginf=1e3)
                prior = value / prior_sigma
                regularizers = [data, prior]
                if np.any(smooth_sigma > 0):
                    if knot_index > 0:
                        regularizers.append(
                            (value - trial_values[knot_index - 1]) / smooth_sigma
                        )
                    if knot_index + 1 < knot_count:
                        regularizers.append(
                            (value - trial_values[knot_index + 1]) / smooth_sigma
                        )
                return np.concatenate(regularizers)

            solver = least_squares(
                local_residual,
                x0=knot_values[knot_index],
                bounds=(lower_one, upper_one),
                method="trf",
                loss="huber",
                f_scale=1.0,
                max_nfev=max(1, int(local_max_nfev)),
            )
            total_nfev += int(solver.nfev)
            sweep_cost += float(solver.cost)
            if solver.success:
                local_success_count += 1
            else:
                local_failure_count += 1
            knot_values[knot_index] = solver.x
        costs.append(sweep_cost)
    return knot_values, {
        "success": local_failure_count == 0,
        "status": 0 if local_failure_count == 0 else 1,
        "solver_method": "coordinate_descent",
        "message": "coordinate descent over time-varying rotation knots",
        "nfev": int(total_nfev),
        "cost": float(costs[-1] if costs else np.nan),
        "cost_by_sweep": costs,
        "optimality": float("nan"),
        "local_success_count": int(local_success_count),
        "local_failure_count": int(local_failure_count),
    }


def sparse_graph_optimize_knots(
    *,
    sampled: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    interpolator: KnotFrameInterpolator,
    navigation_offset_ns: int,
    lower_one: np.ndarray,
    upper_one: np.ndarray,
    prior_sigma: np.ndarray,
    smooth_sigma: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, object]]:
    knot_count = len(interpolator.knots)
    dimension = int(interpolator.dimension)

    def unpack(parameters: np.ndarray) -> dict[int, np.ndarray]:
        return interpolator.unpack(parameters)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        frame_rotvecs = unpack(parameters)
        evaluated = evaluate_time_varying_rotation(
            sampled,
            navigation,
            calibration,
            frame_rotvecs,
            navigation_offset_ns=navigation_offset_ns,
        )
        data_values = evaluated["tv_sampson_px"].to_numpy(dtype=np.float64)
        data_values = np.nan_to_num(data_values, nan=1e3, posinf=1e3, neginf=1e3)
        knot_values = parameters.reshape(knot_count, dimension)
        prior = (knot_values / prior_sigma).reshape(-1)
        if knot_count > 1 and np.any(smooth_sigma > 0):
            smooth = (np.diff(knot_values, axis=0) / smooth_sigma).reshape(-1)
            return np.concatenate([data_values, prior, smooth])
        return np.concatenate([data_values, prior])

    lower = np.tile(lower_one, knot_count)
    upper = np.tile(upper_one, knot_count)
    sparsity = build_knot_jacobian_sparsity(sampled, interpolator).tocsr()
    solver = least_squares(
        residuals,
        x0=np.zeros(interpolator.variable_count, dtype=np.float64),
        bounds=(lower, upper),
        jac_sparsity=sparsity,
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        verbose=0,
    )
    knot_values = solver.x.reshape(knot_count, dimension)
    return knot_values, {
        "success": bool(solver.success),
        "status": int(solver.status),
        "solver_method": "sparse_graph",
        "message": str(solver.message),
        "nfev": int(solver.nfev),
        "cost": float(solver.cost),
        "optimality": float(solver.optimality),
        "jac_sparsity_shape": list(sparsity.shape),
        "jac_sparsity_nnz": int(sparsity.nnz),
    }


def _torch_skew(vector):
    import torch

    zeros = torch.zeros_like(vector[..., 0])
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _torch_rotvec_to_matrix(rotvec):
    import torch

    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec / theta
    skew_axis = _torch_skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(
        rotvec.shape[:-1] + (3, 3)
    )
    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    return eye + sin_theta * skew_axis + (1.0 - cos_theta) * (skew_axis @ skew_axis)


def _torch_transform_inverse(transform):
    import torch

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device).expand_as(
        transform
    ).clone()
    rotation_t = rotation.transpose(-1, -2)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -(rotation_t @ translation[..., None]).squeeze(-1)
    return inverse


def _torch_project_world_to_pixels(world_points, world_to_camera, intrinsic):
    camera_points = (
        world_to_camera[:, :3, :3] @ world_points[..., None]
        + world_to_camera[:, :3, 3:4]
    ).squeeze(-1)
    z = camera_points[:, 2].clamp_min(1e-12)
    normalized = camera_points[:, :2] / z[:, None]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    pixels = normalized.clone()
    pixels[:, 0] = fx * normalized[:, 0] + cx
    pixels[:, 1] = fy * normalized[:, 1] + cy
    return pixels


def _torch_two_view_reprojection_rms(
    first_normalized,
    second_normalized,
    first_world_to_camera,
    second_world_to_camera,
    intrinsic,
    first_pixels,
    second_pixels,
):
    import torch

    p1 = first_world_to_camera[:, :3, :]
    p2 = second_world_to_camera[:, :3, :]
    a = torch.stack(
        [
            first_normalized[:, 0:1] * p1[:, 2, :] - p1[:, 0, :],
            first_normalized[:, 1:2] * p1[:, 2, :] - p1[:, 1, :],
            second_normalized[:, 0:1] * p2[:, 2, :] - p2[:, 0, :],
            second_normalized[:, 1:2] * p2[:, 2, :] - p2[:, 1, :],
        ],
        dim=1,
    )
    _u, _s, vh = torch.linalg.svd(a)
    world_h = vh[:, -1, :]
    denom = world_h[:, 3:4]
    denom = torch.where(
        torch.abs(denom) < 1e-12,
        torch.where(denom >= 0, torch.full_like(denom, 1e-12), torch.full_like(denom, -1e-12)),
        denom,
    )
    world_points = world_h[:, :3] / denom
    first_reprojected = _torch_project_world_to_pixels(
        world_points, first_world_to_camera, intrinsic
    )
    second_reprojected = _torch_project_world_to_pixels(
        world_points, second_world_to_camera, intrinsic
    )
    first_error = torch.linalg.norm(first_reprojected - first_pixels, dim=1)
    second_error = torch.linalg.norm(second_reprojected - second_pixels, dim=1)
    return torch.sqrt((first_error**2 + second_error**2) / 2.0)


def _torch_temporal_regularizer(
    diff,
    smooth_sigma_t,
    *,
    mode: str,
    huber_delta: float,
):
    import torch

    normalized = diff / smooth_sigma_t
    if mode == "none":
        return torch.zeros((), dtype=normalized.dtype, device=normalized.device)
    if mode == "l2":
        return torch.sum(normalized**2)
    if mode == "huber":
        return torch.nn.functional.huber_loss(
            normalized,
            torch.zeros_like(normalized),
            delta=float(huber_delta),
            reduction="sum",
        )
    raise ValueError(f"unsupported temporal regularizer mode: {mode}")


def _build_frame_timestamp_map(matches: pd.DataFrame) -> dict[int, int]:
    timestamps: dict[int, int] = {}
    for row in matches.itertuples(index=False):
        frame1 = int(row.frame1)
        frame2 = int(row.frame2)
        timestamps.setdefault(frame1, int(row.timestamp1_ns))
        timestamps.setdefault(frame2, int(row.timestamp2_ns))
    return timestamps


def _load_raw_imu_frame_shock_scores(
    dataset_root: Path,
    frame_timestamps_ns: dict[int, int],
    *,
    window_ms: float,
) -> dict[int, float]:
    imu_path = Path(dataset_root) / "ved" / "bynav_rawimu.csv"
    frame = pd.read_csv(imu_path)
    imu_timestamps_ns = np.round(
        frame["gps_seconds"].to_numpy(dtype=np.float64) * 1_000_000_000.0
    ).astype(np.int64)
    ax = frame["ax(m/s2)"].to_numpy(dtype=np.float64)
    ay = frame["ay(m/s2)"].to_numpy(dtype=np.float64)
    az = frame["az(m/s2)"].to_numpy(dtype=np.float64)
    wx = frame["wx(rad/s)"].to_numpy(dtype=np.float64)
    wy = frame["wy(rad/s)"].to_numpy(dtype=np.float64)
    wz = frame["wz(rad/s)"].to_numpy(dtype=np.float64)
    gravity = float(np.median(az))
    half_window_ns = int(round(float(window_ms) * 1_000_000.0 / 2.0))
    raw_scores: dict[int, float] = {}
    for frame_index, timestamp_ns in frame_timestamps_ns.items():
        left = int(timestamp_ns) - half_window_ns
        right = int(timestamp_ns) + half_window_ns
        mask = (imu_timestamps_ns >= left) & (imu_timestamps_ns <= right)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(imu_timestamps_ns - int(timestamp_ns))))
            mask = np.zeros_like(imu_timestamps_ns, dtype=bool)
            mask[nearest] = True
        angular_rms = float(
            np.sqrt(np.mean(wx[mask] ** 2 + wy[mask] ** 2 + wz[mask] ** 2))
        )
        linear_rms = float(
            np.sqrt(np.mean(ax[mask] ** 2 + ay[mask] ** 2 + (az[mask] - gravity) ** 2))
        )
        raw_scores[int(frame_index)] = angular_rms + 0.2 * linear_rms
    values = np.array(list(raw_scores.values()), dtype=np.float64)
    if len(values) == 0:
        return {}
    lo = float(np.quantile(values, 0.1))
    hi = float(np.quantile(values, 0.9))
    if hi <= lo:
        return {frame: 0.0 for frame in raw_scores}
    return {
        frame: float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))
        for frame, score in raw_scores.items()
    }


def _build_shock_smooth_scale_per_segment(
    *,
    frame_scores: dict[int, float],
    frame_sequence: list[int] | tuple[int, ...],
    relax_factor: float,
) -> np.ndarray:
    frames = [int(frame) for frame in frame_sequence]
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float64)
    factor = max(1.0, float(relax_factor))
    scales = []
    for first, second in zip(frames[:-1], frames[1:]):
        endpoint_score = max(
            float(frame_scores.get(int(first), 0.0)),
            float(frame_scores.get(int(second), 0.0)),
        )
        scales.append(1.0 + (factor - 1.0) * endpoint_score)
    return np.asarray(scales, dtype=np.float64)


def torch_sparse_graph_optimize_knots(
    *,
    sampled: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    interpolator: KnotFrameInterpolator,
    navigation_offset_ns: int,
    lower_one: np.ndarray,
    upper_one: np.ndarray,
    prior_sigma: np.ndarray,
    smooth_sigma: np.ndarray,
    max_nfev: int,
    learning_rate: float = 0.03,
    device: str = "cuda:0",
    temporal_regularizer: str = "l2",
    temporal_huber_delta: float = 1.0,
    use_prior_regularizer: bool = True,
    smooth_segment_scales: np.ndarray | None = None,
    use_reprojection_term: bool = False,
    reprojection_loss_weight: float = 0.25,
    lr_schedule: str = "constant",
    lr_step_size: int = 300,
    lr_decay: float = 0.5,
    lr_min_factor: float = 0.1,
) -> tuple[np.ndarray, dict[str, object]]:
    import torch

    requested_device = device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    dtype = torch.float64
    dimension = int(interpolator.dimension)
    frames = list(interpolator.frames)
    frame_position = {frame: index for index, frame in enumerate(frames)}
    base_transforms = []
    for frame in frames:
        rows1 = sampled.loc[sampled["frame1"].astype(int) == frame, "timestamp1_ns"]
        if not rows1.empty:
            timestamp_ns = int(rows1.iloc[0])
        else:
            rows2 = sampled.loc[sampled["frame2"].astype(int) == frame, "timestamp2_ns"]
            timestamp_ns = int(rows2.iloc[0])
        body_to_world = navigation.pose_at(timestamp_ns + int(navigation_offset_ns))
        base_transforms.append(body_to_world @ calibration.sensor_to_car)
    base_transforms_t = torch.as_tensor(
        np.stack(base_transforms),
        dtype=dtype,
        device=torch_device,
    )
    first_indices = torch.as_tensor(
        [frame_position[int(frame)] for frame in sampled["frame1"].to_numpy()],
        dtype=torch.long,
        device=torch_device,
    )
    second_indices = torch.as_tensor(
        [frame_position[int(frame)] for frame in sampled["frame2"].to_numpy()],
        dtype=torch.long,
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
    first_pixels_xy = first_points[:, :2]
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
    second_pixels_xy = second_points[:, :2]
    intrinsic_t = torch.as_tensor(
        np.array(calibration.intrinsic, dtype=np.float64, copy=True),
        dtype=dtype,
        device=torch_device,
    )
    first_normalized_points = torch.as_tensor(
        np.array(sampled[["nx1", "ny1"]].to_numpy(dtype=np.float64), copy=True),
        dtype=dtype,
        device=torch_device,
    )
    second_normalized_points = torch.as_tensor(
        np.array(sampled[["nx2", "ny2"]].to_numpy(dtype=np.float64), copy=True),
        dtype=dtype,
        device=torch_device,
    )
    intrinsic_inv = torch.as_tensor(
        np.linalg.inv(calibration.intrinsic),
        dtype=dtype,
        device=torch_device,
    )
    intrinsic_inv_t = intrinsic_inv.transpose(0, 1)
    left_indices = torch.as_tensor(
        [interpolator.frame_weights[frame][0] for frame in frames],
        dtype=torch.long,
        device=torch_device,
    )
    right_indices = torch.as_tensor(
        [interpolator.frame_weights[frame][1] for frame in frames],
        dtype=torch.long,
        device=torch_device,
    )
    alphas = torch.as_tensor(
        [interpolator.frame_weights[frame][2] for frame in frames],
        dtype=dtype,
        device=torch_device,
    )[:, None]
    lower = torch.as_tensor(lower_one, dtype=dtype, device=torch_device)
    upper = torch.as_tensor(upper_one, dtype=dtype, device=torch_device)
    prior_sigma_t = torch.as_tensor(prior_sigma, dtype=dtype, device=torch_device)
    if smooth_segment_scales is not None and len(interpolator.knots) > 1:
        smooth_sigma_effective = (
            np.asarray(smooth_segment_scales, dtype=np.float64)[:, None]
            * np.asarray(smooth_sigma, dtype=np.float64)[None, :]
        )
    else:
        smooth_sigma_effective = np.asarray(smooth_sigma, dtype=np.float64)
    smooth_sigma_t = torch.as_tensor(
        smooth_sigma_effective, dtype=dtype, device=torch_device
    )
    knots = torch.zeros(
        (len(interpolator.knots), dimension),
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([knots], lr=float(learning_rate))
    initial_learning_rate = float(learning_rate)
    schedule = str(lr_schedule)
    step_size = max(1, int(lr_step_size))
    decay = float(lr_decay)
    min_factor = float(lr_min_factor)
    costs: list[float] = []
    learning_rates: list[float] = []

    def current_frame_values():
        return (1.0 - alphas) * knots[left_indices] + alphas * knots[right_indices]

    def apply_scheduler(step_index: int) -> None:
        if schedule == "constant":
            factor = 1.0
        elif schedule == "step":
            factor = decay ** (step_index // step_size)
        elif schedule == "cosine":
            if max_nfev <= 1:
                factor = min_factor
            else:
                progress = step_index / float(max_nfev - 1)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                factor = min_factor + (1.0 - min_factor) * cosine
        else:
            raise ValueError(f"unsupported lr_schedule: {schedule}")
        factor = max(min_factor, float(factor))
        current_lr = initial_learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        learning_rates.append(float(current_lr))

    for _step in range(max(1, int(max_nfev))):
        apply_scheduler(_step)
        optimizer.zero_grad(set_to_none=True)
        frame_values = current_frame_values()
        frame_rotations = _torch_rotvec_to_matrix(frame_values[:, :3])
        delta = torch.eye(4, dtype=dtype, device=torch_device).repeat(len(frames), 1, 1)
        delta[:, :3, :3] = frame_rotations
        if dimension >= 6:
            delta[:, :3, 3] = frame_values[:, 3:6]
        camera_to_world = base_transforms_t @ delta
        first_camera = camera_to_world[first_indices]
        second_camera = camera_to_world[second_indices]
        first_to_second = _torch_transform_inverse(second_camera) @ first_camera
        first_world_to_camera = _torch_transform_inverse(first_camera)
        second_world_to_camera = _torch_transform_inverse(second_camera)
        essential = _torch_skew(first_to_second[:, :3, 3]) @ first_to_second[:, :3, :3]
        fundamental = intrinsic_inv_t[None, :, :] @ essential @ intrinsic_inv[None, :, :]
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
        data_loss = torch.nn.functional.huber_loss(
            sampson_sqrt,
            torch.zeros_like(sampson_sqrt),
            delta=1.0,
            reduction="mean",
        )
        if bool(use_reprojection_term):
            reprojection_rms = _torch_two_view_reprojection_rms(
                first_normalized_points,
                second_normalized_points,
                first_world_to_camera,
                second_world_to_camera,
                intrinsic_t,
                first_pixels_xy,
                second_pixels_xy,
            )
            reprojection_loss = torch.nn.functional.huber_loss(
                reprojection_rms,
                torch.zeros_like(reprojection_rms),
                delta=1.0,
                reduction="mean",
            ) * float(reprojection_loss_weight)
        else:
            reprojection_loss = torch.zeros((), dtype=dtype, device=torch_device)
        observation_count = max(1, int(len(sampled)))
        if use_prior_regularizer:
            prior_loss = torch.sum((knots / prior_sigma_t) ** 2) / observation_count
        else:
            prior_loss = torch.zeros((), dtype=dtype, device=torch_device)
        if (
            temporal_regularizer != "none"
            and len(interpolator.knots) > 1
            and bool(np.any(smooth_sigma > 0))
        ):
            smooth_loss = (
                _torch_temporal_regularizer(
                    torch.diff(knots, dim=0),
                    smooth_sigma_t,
                    mode=temporal_regularizer,
                    huber_delta=float(temporal_huber_delta),
                )
                / observation_count
            )
        else:
            smooth_loss = torch.zeros((), dtype=dtype, device=torch_device)
        loss = data_loss + reprojection_loss + prior_loss + smooth_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            knots.clamp_(lower, upper)
        costs.append(float(loss.detach().cpu()))

    knot_values = knots.detach().cpu().numpy()
    return knot_values, {
        "success": True,
        "status": 0,
        "solver_method": "torch_sparse_graph_full_batch",
        "message": "full retained-match sparse graph optimized with torch autodiff",
        "nfev": int(max(1, int(max_nfev))),
        "cost": float(costs[-1] if costs else np.nan),
        "cost_by_step_first": costs[:10],
        "cost_by_step_last": costs[-10:],
        "optimality": float("nan"),
        "requested_device": str(requested_device),
        "resolved_device": str(torch_device),
        "learning_rate": float(learning_rate),
        "temporal_regularizer": str(temporal_regularizer),
        "temporal_huber_delta": float(temporal_huber_delta),
        "use_prior_regularizer": bool(use_prior_regularizer),
        "use_reprojection_term": bool(use_reprojection_term),
        "reprojection_loss_weight": float(reprojection_loss_weight),
        "lr_schedule": schedule,
        "lr_step_size": int(step_size),
        "lr_decay": float(decay),
        "lr_min_factor": float(min_factor),
        "learning_rate_first_steps": learning_rates[:10],
        "learning_rate_last_steps": learning_rates[-10:],
        "disturbance_dof": int(dimension),
    }


def run_time_varying_rotation_diagnostic(
    config_path: Path,
    retained_matches_input: Path,
    output: Path,
    camera: str = "center_camera_fov30",
    max_observations: int | None = 12_000,
    rotation_bound_deg: float = 0.5,
    yaw_bound_deg: float = 0.2,
    rotation_prior_sigma_deg: float = 0.12,
    yaw_prior_sigma_deg: float = 0.06,
    disturbance_dof: int = 3,
    translation_bound_m: float = 0.05,
    translation_prior_sigma_m: float = 0.03,
    translation_smooth_sigma_m: float = 0.03,
    smooth_sigma_deg: float = 0.08,
    roll_smooth_sigma_deg: float | None = None,
    pitch_smooth_sigma_deg: float | None = None,
    yaw_smooth_sigma_deg: float | None = None,
    max_nfev: int = 100,
    knot_stride_frames: int = 10,
    export_pair_visualizations: bool = True,
    visualization_max_pairs: int | None = None,
    visualization_max_per_row: int = 150,
    coordinate_sweeps: int = 6,
    optimizer_method: str = "torch_sparse_graph",
    torch_learning_rate: float = 0.03,
    torch_device: str = "cuda:0",
    temporal_regularizer: str = "l2",
    temporal_huber_delta: float = 1.0,
    use_prior_regularizer: bool = True,
    imu_adaptive_temporal: bool = False,
    imu_shock_window_ms: float = 120.0,
    imu_shock_relax_factor: float = 3.0,
    use_reprojection_term: bool = False,
    reprojection_loss_weight: float = 0.25,
    lr_schedule: str = "constant",
    lr_step_size: int = 300,
    lr_decay: float = 0.5,
    lr_min_factor: float = 0.1,
    train_offsets: tuple[int, ...] | None = None,
    eval_offsets: tuple[int, ...] | None = None,
) -> Path:
    disturbance_dof = int(disturbance_dof)
    if disturbance_dof not in (3, 6):
        raise ValueError("disturbance_dof must be either 3 or 6")
    config = load_config(Path(config_path))
    calibration = CameraCalibration.from_dataset(config.dataset_root, camera)
    base_extrinsic_source = (
        config.dataset_root / "calib" / camera / f"{camera}-to-car_center-extrinsic.json"
    )
    navigation = NavigationTrajectory.from_csv(
        config.dataset_root / config.assumptions["navigation_source"]
    )
    navigation_offset_ns = navigation_time_offset_ns(config.assumptions)
    matches = pd.read_parquet(
        resolve_retained_matches_path(Path(retained_matches_input))
    )
    retained = matches[
        matches["pose_quality_valid"].astype(bool)
        & matches["reprojection_threshold_inlier"].astype(bool)
    ].copy()
    available_offsets = sorted(retained["frame_offset"].astype(int).unique().tolist())
    resolved_train_offsets = (
        set(int(value) for value in train_offsets)
        if train_offsets is not None
        else set(available_offsets)
    )
    if not resolved_train_offsets:
        raise ValueError("train_offsets must not be empty")
    unknown_train_offsets = sorted(resolved_train_offsets.difference(available_offsets))
    if unknown_train_offsets:
        raise ValueError(f"unknown train_offsets: {unknown_train_offsets}")
    resolved_eval_offsets = (
        set(int(value) for value in eval_offsets)
        if eval_offsets is not None
        else set(available_offsets).difference(resolved_train_offsets)
    )
    unknown_eval_offsets = sorted(resolved_eval_offsets.difference(available_offsets))
    if unknown_eval_offsets:
        raise ValueError(f"unknown eval_offsets: {unknown_eval_offsets}")
    train_retained = retained.loc[
        retained["frame_offset"].astype(int).isin(sorted(resolved_train_offsets))
    ].copy()
    sampled = sample_matches(train_retained, max_observations=max_observations)
    output = Path(output)
    for relative in ("metrics", "observations", "visualizations"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    retained_frames = sorted(
        set(retained["frame1"].astype(int).tolist())
        | set(retained["frame2"].astype(int).tolist())
    )
    train_frames = sorted(
        set(sampled["frame1"].astype(int).tolist())
        | set(sampled["frame2"].astype(int).tolist())
    )
    dataset = Dataset(config.dataset_root)
    frame_image_paths = {
        int(frame.index): frame.path for frame in dataset.frames(camera)
    }
    interpolator = build_knot_frame_interpolator(
        train_frames,
        knot_stride=max(1, int(knot_stride_frames)),
        dimension=disturbance_dof,
    )
    knot_count = len(interpolator.knots)
    zero_disturbances = {
        frame: np.zeros(disturbance_dof, dtype=np.float64) for frame in retained_frames
    }
    before_sampled = evaluate_time_varying_rotation(
        sampled,
        navigation,
        calibration,
        zero_disturbances,
        navigation_offset_ns=navigation_offset_ns,
    )

    prior_sigma = np.deg2rad(
        np.array(
            [rotation_prior_sigma_deg, rotation_prior_sigma_deg, yaw_prior_sigma_deg],
            dtype=np.float64,
        )
    )
    smooth_sigma_axes_deg = _resolve_smooth_sigma_axes_deg(
        smooth_sigma_deg=float(smooth_sigma_deg),
        roll_smooth_sigma_deg=roll_smooth_sigma_deg,
        pitch_smooth_sigma_deg=pitch_smooth_sigma_deg,
        yaw_smooth_sigma_deg=yaw_smooth_sigma_deg,
    )
    smooth_sigma = np.deg2rad(smooth_sigma_axes_deg)
    if disturbance_dof == 6:
        prior_sigma = np.concatenate(
            [
                prior_sigma,
                np.full(3, float(translation_prior_sigma_m), dtype=np.float64),
            ]
        )
        smooth_sigma = np.concatenate(
            [
                smooth_sigma,
                np.full(3, float(translation_smooth_sigma_m), dtype=np.float64),
            ]
        )
    smooth_segment_scales = None
    if bool(imu_adaptive_temporal) and len(interpolator.knots) > 1:
        frame_timestamps_ns = _build_frame_timestamp_map(retained)
        frame_scores = _load_raw_imu_frame_shock_scores(
            config.dataset_root,
            frame_timestamps_ns,
            window_ms=float(imu_shock_window_ms),
        )
        smooth_segment_scales = _build_shock_smooth_scale_per_segment(
            frame_scores=frame_scores,
            frame_sequence=list(interpolator.knots),
            relax_factor=float(imu_shock_relax_factor),
        )

    bound = np.deg2rad(float(rotation_bound_deg))
    yaw_bound = np.deg2rad(float(yaw_bound_deg))
    lower_one = np.array([-bound, -bound, -yaw_bound], dtype=np.float64)
    upper_one = np.array([bound, bound, yaw_bound], dtype=np.float64)
    if disturbance_dof == 6:
        trans_bound = abs(float(translation_bound_m))
        lower_one = np.concatenate([lower_one, np.full(3, -trans_bound)])
        upper_one = np.concatenate([upper_one, np.full(3, trans_bound)])
    if optimizer_method == "coordinate_descent" and disturbance_dof != 3:
        raise ValueError("coordinate_descent currently supports only 3-DOF disturbances")
    if optimizer_method == "sparse_graph":
        knot_values, solver_status = sparse_graph_optimize_knots(
            sampled=sampled,
            navigation=navigation,
            calibration=calibration,
            interpolator=interpolator,
            navigation_offset_ns=navigation_offset_ns,
            lower_one=lower_one,
            upper_one=upper_one,
            prior_sigma=prior_sigma,
            smooth_sigma=smooth_sigma,
            max_nfev=int(max_nfev),
        )
    elif optimizer_method == "torch_sparse_graph":
        knot_values, solver_status = torch_sparse_graph_optimize_knots(
            sampled=sampled,
            navigation=navigation,
            calibration=calibration,
            interpolator=interpolator,
            navigation_offset_ns=navigation_offset_ns,
            lower_one=lower_one,
            upper_one=upper_one,
            prior_sigma=prior_sigma,
            smooth_sigma=smooth_sigma,
            max_nfev=int(max_nfev),
            learning_rate=float(torch_learning_rate),
            device=str(torch_device),
            temporal_regularizer=str(temporal_regularizer),
            temporal_huber_delta=float(temporal_huber_delta),
            use_prior_regularizer=bool(use_prior_regularizer),
            smooth_segment_scales=smooth_segment_scales,
            use_reprojection_term=bool(use_reprojection_term),
            reprojection_loss_weight=float(reprojection_loss_weight),
            lr_schedule=str(lr_schedule),
            lr_step_size=int(lr_step_size),
            lr_decay=float(lr_decay),
            lr_min_factor=float(lr_min_factor),
        )
    elif optimizer_method == "coordinate_descent":
        knot_values, solver_status = _coordinate_descent_optimize_knots(
            sampled=sampled,
            navigation=navigation,
            calibration=calibration,
            interpolator=interpolator,
            navigation_offset_ns=navigation_offset_ns,
            lower_one=lower_one,
            upper_one=upper_one,
            prior_sigma=prior_sigma,
            smooth_sigma=smooth_sigma,
            sweeps=int(coordinate_sweeps),
            local_max_nfev=int(max_nfev),
        )
    else:
        raise ValueError(f"unsupported optimizer_method: {optimizer_method}")
    optimized_rotvecs = interpolator.unpack(knot_values.reshape(-1))
    before_all = evaluate_time_varying_rotation(
        retained,
        navigation,
        calibration,
        zero_disturbances,
        navigation_offset_ns=navigation_offset_ns,
    ).rename(columns={"tv_sampson_px": "before_sampson_px"})
    after_all = evaluate_time_varying_rotation(
        retained,
        navigation,
        calibration,
        optimized_rotvecs,
        navigation_offset_ns=navigation_offset_ns,
    ).rename(columns={"tv_sampson_px": "after_sampson_px"})
    comparison = before_all.copy()
    comparison["after_sampson_px"] = after_all["after_sampson_px"].to_numpy()
    before_reprojection = evaluate_time_varying_reprojection(
        retained,
        navigation,
        calibration,
        zero_disturbances,
        navigation_offset_ns=navigation_offset_ns,
    )
    after_reprojection = evaluate_time_varying_reprojection(
        retained,
        navigation,
        calibration,
        optimized_rotvecs,
        navigation_offset_ns=navigation_offset_ns,
    )
    comparison["before_reprojection1_px"] = before_reprojection[
        "tv_reprojection1_px"
    ].to_numpy()
    comparison["before_reprojection2_px"] = before_reprojection[
        "tv_reprojection2_px"
    ].to_numpy()
    comparison["before_reprojection_rms_px"] = before_reprojection[
        "tv_reprojection_rms_px"
    ].to_numpy()
    comparison[["before_reprojected1_x", "before_reprojected1_y"]] = before_reprojection[
        ["tv_reprojected1_x", "tv_reprojected1_y"]
    ].to_numpy()
    comparison[["before_reprojected2_x", "before_reprojected2_y"]] = before_reprojection[
        ["tv_reprojected2_x", "tv_reprojected2_y"]
    ].to_numpy()
    comparison["after_reprojection1_px"] = after_reprojection[
        "tv_reprojection1_px"
    ].to_numpy()
    comparison["after_reprojection2_px"] = after_reprojection[
        "tv_reprojection2_px"
    ].to_numpy()
    comparison["after_reprojection_rms_px"] = after_reprojection[
        "tv_reprojection_rms_px"
    ].to_numpy()
    comparison[["after_reprojected1_x", "after_reprojected1_y"]] = after_reprojection[
        ["tv_reprojected1_x", "tv_reprojected1_y"]
    ].to_numpy()
    comparison[["after_reprojected2_x", "after_reprojected2_y"]] = after_reprojection[
        ["tv_reprojected2_x", "tv_reprojected2_y"]
    ].to_numpy()
    frame_summary = summarize_frame_disturbance(comparison, optimized_rotvecs)

    offset_rows = []
    for offset, group in comparison.groupby("frame_offset", sort=True):
        before = _summarize_matches(group, "before_sampson_px")
        after = _summarize_matches(group, "after_sampson_px")
        before_reproj = _summarize_matches(group, "before_reprojection_rms_px")
        after_reproj = _summarize_matches(group, "after_reprojection_rms_px")
        offset_rows.append(
            {
                "frame_offset": int(offset),
                "count": before["count"],
                "before_sampson_mean_px": before["mean"],
                "after_sampson_mean_px": after["mean"],
                "before_sampson_median_px": before["median"],
                "after_sampson_median_px": after["median"],
                "before_sampson_p95_px": before["p95"],
                "after_sampson_p95_px": after["p95"],
                "before_reprojection_median_px": before_reproj["median"],
                "after_reprojection_median_px": after_reproj["median"],
                "before_reprojection_p95_px": before_reproj["p95"],
                "after_reprojection_p95_px": after_reproj["p95"],
                "before_reprojection_rmse_px": before_reproj["rmse"],
                "after_reprojection_rmse_px": after_reproj["rmse"],
            }
        )
    offset_summary = pd.DataFrame.from_records(offset_rows)
    offset_split_summary = _summarize_offset_split(
        comparison,
        train_offsets=resolved_train_offsets,
        eval_offsets=resolved_eval_offsets,
    )
    before_summary = _summarize_matches(comparison, "before_sampson_px")
    after_summary = _summarize_matches(comparison, "after_sampson_px")
    before_reprojection_summary = _summarize_matches(
        comparison, "before_reprojection_rms_px"
    )
    after_reprojection_summary = _summarize_matches(
        comparison, "after_reprojection_rms_px"
    )

    comparison.to_parquet(output / "observations" / "time_varying_rotation_matches.parquet", index=False)
    frame_summary.to_csv(output / "metrics" / "frame_disturbance_summary.csv", index=False)
    offset_summary.to_csv(output / "metrics" / "offset_summary.csv", index=False)
    offset_split_summary.to_csv(output / "metrics" / "offset_split_summary.csv", index=False)
    frame_timestamps_ns = _build_frame_timestamp_map(retained)
    _export_framewise_equivalent_extrinsics(
        output=output,
        camera=camera,
        base_extrinsic_source=str(base_extrinsic_source),
        calibration=calibration,
        frame_rotvecs=optimized_rotvecs,
        frame_timestamps_ns=frame_timestamps_ns,
        frame_image_paths=frame_image_paths,
    )
    camera_to_world_by_frame = {
        int(frame.index): _camera_to_world(
            navigation,
            calibration,
            int(frame.timestamp_ns),
            np.asarray(optimized_rotvecs.get(int(frame.index), np.zeros(3, dtype=np.float64)), dtype=np.float64),
            navigation_offset_ns,
        )
        for frame in dataset.frames(camera)
    }
    _export_image_camera_to_world_deliverable(
        output=output,
        frame_image_paths=frame_image_paths,
        camera_to_world_by_frame=camera_to_world_by_frame,
    )
    camera_to_world_origin_by_frame = {
        int(frame.index): _camera_to_world(
            navigation,
            calibration,
            int(frame.timestamp_ns),
            np.zeros(3, dtype=np.float64),
            navigation_offset_ns,
        )
        for frame in dataset.frames(camera)
    }
    angle_change_deg_by_frame = {
        int(frame): float(np.rad2deg(np.linalg.norm(np.asarray(rotvec, dtype=np.float64)[:3])))
        for frame, rotvec in optimized_rotvecs.items()
    }
    _export_image_camera_to_world_origin_deliverable(
        output=output,
        frame_image_paths=frame_image_paths,
        camera_to_world_origin_by_frame=camera_to_world_origin_by_frame,
    )
    _export_angle_change_deliverable(
        output=output,
        frame_image_paths=frame_image_paths,
        angle_change_deg_by_frame=angle_change_deg_by_frame,
    )
    solver_status = {
        **solver_status,
        "max_observations": max_observations,
        "retained_observations": int(len(retained)),
        "sampled_observations": int(len(sampled)),
        "train_frame_count": int(len(train_frames)),
        "train_offsets": sorted(int(value) for value in resolved_train_offsets),
        "eval_offsets": sorted(int(value) for value in resolved_eval_offsets),
        "frame_count": int(len(retained_frames)),
        "knot_count": int(knot_count),
        "knot_stride_frames": int(knot_stride_frames),
        "knot_frames": list(interpolator.knots),
        "rotation_bound_deg": float(rotation_bound_deg),
        "yaw_bound_deg": float(yaw_bound_deg),
        "disturbance_dof": int(disturbance_dof),
        "translation_bound_m": float(translation_bound_m),
        "rotation_prior_sigma_deg": float(rotation_prior_sigma_deg),
        "yaw_prior_sigma_deg": float(yaw_prior_sigma_deg),
        "translation_prior_sigma_m": float(translation_prior_sigma_m),
        "translation_smooth_sigma_m": float(translation_smooth_sigma_m),
        "smooth_sigma_deg": float(smooth_sigma_deg),
        "roll_smooth_sigma_deg": float(smooth_sigma_axes_deg[0]),
        "pitch_smooth_sigma_deg": float(smooth_sigma_axes_deg[1]),
        "yaw_smooth_sigma_deg": float(smooth_sigma_axes_deg[2]),
        "imu_adaptive_temporal": bool(imu_adaptive_temporal),
        "imu_shock_window_ms": float(imu_shock_window_ms),
        "imu_shock_relax_factor": float(imu_shock_relax_factor),
        "use_reprojection_term": bool(use_reprojection_term),
        "reprojection_loss_weight": float(reprojection_loss_weight),
        "lr_schedule": str(lr_schedule),
        "lr_step_size": int(lr_step_size),
        "lr_decay": float(lr_decay),
        "lr_min_factor": float(lr_min_factor),
        "navigation_time_offset_ns": int(navigation_offset_ns),
        "export_pair_visualizations": bool(export_pair_visualizations),
        "visualization_max_pairs": visualization_max_pairs,
        "visualization_max_per_row": int(visualization_max_per_row),
        "coordinate_sweeps": int(coordinate_sweeps),
        "optimizer_method": str(optimizer_method),
        "torch_learning_rate": float(torch_learning_rate),
        "torch_device": str(torch_device),
        "temporal_regularizer": str(temporal_regularizer),
        "temporal_huber_delta": float(temporal_huber_delta),
        "use_prior_regularizer": bool(use_prior_regularizer),
    }
    (output / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "before_sampson": before_summary,
                "after_sampson": after_summary,
                "before_reprojection": before_reprojection_summary,
                "after_reprojection": after_reprojection_summary,
                "solver": solver_status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "resolved_time_varying_rotation.yaml").write_text(
        yaml.safe_dump(
            {
                "camera": camera,
                "retained_matches_input": str(retained_matches_input),
                **solver_status,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_plots(frame_summary, offset_summary, output)
    if export_pair_visualizations:
        _write_pair_reprojection_visualizations(
            config.dataset_root,
            camera,
            comparison,
            output,
            max_pairs=visualization_max_pairs,
            max_per_row=visualization_max_per_row,
        )
    _write_report(
        output,
        camera,
        before_summary,
        after_summary,
        before_reprojection_summary,
        after_reprojection_summary,
        offset_summary,
        frame_summary,
        solver_status,
    )
    return output


def _write_plots(frame_summary: pd.DataFrame, offset_summary: pd.DataFrame, output: Path) -> None:
    visual = output / "visualizations"
    figure, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(frame_summary["frame"], frame_summary["before_incident_sampson_mean_px"], label="before")
    axes[0].plot(frame_summary["frame"], frame_summary["after_incident_sampson_mean_px"], label="after")
    axes[0].set_ylabel("sqrt Sampson px")
    axes[0].legend()
    axes[1].plot(frame_summary["frame"], frame_summary["delta_roll_deg"], label="delta x/roll")
    axes[1].plot(frame_summary["frame"], frame_summary["delta_pitch_deg"], label="delta y/pitch")
    axes[1].plot(frame_summary["frame"], frame_summary["delta_yaw_deg"], label="delta z/yaw")
    axes[1].set_ylabel("deg")
    axes[1].legend()
    axes[2].plot(frame_summary["frame"], frame_summary["delta_norm_deg"], color="black")
    axes[2].set_ylabel("norm deg")
    axes[3].plot(frame_summary["frame"], frame_summary["incident_match_count"], color="gray")
    axes[3].set_ylabel("matches")
    axes[3].set_xlabel("frame")
    figure.suptitle("Time-varying rotation diagnostic by frame")
    figure.tight_layout()
    figure.savefig(visual / "frame_disturbance_timeline.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(offset_summary))
    width = 0.35
    axis.bar(x - width / 2, offset_summary["before_sampson_median_px"], width, label="before")
    axis.bar(x + width / 2, offset_summary["after_sampson_median_px"], width, label="after")
    axis.set_xticks(x, offset_summary["frame_offset"].astype(str).tolist())
    axis.set_xlabel("frame offset")
    axis.set_ylabel("median sqrt Sampson px")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(visual / "sampson_median_by_offset.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(offset_summary))
    width = 0.35
    axis.bar(
        x - width / 2,
        offset_summary["before_reprojection_median_px"],
        width,
        label="before",
    )
    axis.bar(
        x + width / 2,
        offset_summary["after_reprojection_median_px"],
        width,
        label="after",
    )
    axis.set_xticks(x, offset_summary["frame_offset"].astype(str).tolist())
    axis.set_xlabel("frame offset")
    axis.set_ylabel("median reprojection RMS (px)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(visual / "reprojection_median_by_offset.png", dpi=160)
    plt.close(figure)


def _write_pair_metric_plots(comparison: pd.DataFrame, output: Path) -> None:
    visual = output / "visualizations"
    pair_metrics = (
        comparison.groupby(["frame_offset", "frame1", "frame2", "timestamp1_ns"], sort=True)
        .agg(
            before_reprojection_median_px=("before_reprojection_rms_px", "median"),
            after_reprojection_median_px=("after_reprojection_rms_px", "median"),
            before_reprojection_p95_px=("before_reprojection_rms_px", lambda x: np.percentile(x, 95)),
            after_reprojection_p95_px=("after_reprojection_rms_px", lambda x: np.percentile(x, 95)),
        )
        .reset_index()
    )
    pair_metrics.to_csv(output / "metrics" / "pair_reprojection_comparison.csv", index=False)
    figure, axis = plt.subplots(figsize=(14, 5))
    for offset, group in pair_metrics.groupby("frame_offset", sort=True):
        axis.plot(
            group["frame1"],
            group["before_reprojection_median_px"],
            alpha=0.35,
            linestyle="--",
            label=f"before offset {offset}",
        )
        axis.plot(
            group["frame1"],
            group["after_reprojection_median_px"],
            alpha=0.9,
            label=f"after offset {offset}",
        )
    axis.set_xlabel("frame1")
    axis.set_ylabel("pair median reprojection RMS (px)")
    axis.set_title("Time-varying rotation compensation: reprojection over frame")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(visual / "reprojection_median_over_frame_before_after.png", dpi=160)
    plt.close(figure)


def _write_pair_reprojection_visualizations(
    dataset_root: Path,
    camera: str,
    comparison: pd.DataFrame,
    output: Path,
    max_pairs: int | None,
    max_per_row: int,
) -> None:
    _write_pair_metric_plots(comparison, output)
    frames = {frame.index: frame for frame in Dataset(dataset_root).frames(camera)}
    grouped = list(comparison.groupby(["frame_offset", "frame1", "frame2"], sort=True))
    if max_pairs is not None:
        grouped = grouped[: int(max_pairs)]
    layout_records = []
    for (offset, frame1, frame2), group in grouped:
        frame = frames.get(int(frame2))
        if frame is None:
            continue
        image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        observed = group[["x2", "y2"]].to_numpy(dtype=np.float64)
        before = group[["before_reprojected2_x", "before_reprojected2_y"]].to_numpy(
            dtype=np.float64
        )
        after = group[["after_reprojected2_x", "after_reprojected2_y"]].to_numpy(
            dtype=np.float64
        )
        stem = f"offset_{int(offset)}_{int(frame1):04d}_{int(frame2):04d}"
        layout = draw_tiled_before_after_reprojection_overlay(
            image,
            observed,
            before,
            after,
            output
            / "visualizations"
            / "reprojection_qc"
            / "before_after_reprojection"
            / f"{stem}_before_after_reprojection_full.jpg",
            max_per_row=max_per_row,
        )
        layout_records.append(
            {
                "frame_offset": int(offset),
                "frame1": int(frame1),
                "frame2": int(frame2),
                "match_count": int(len(group)),
                "rows": int(len(layout)),
            }
        )
    pd.DataFrame.from_records(layout_records).to_csv(
        output / "visualizations" / "reprojection_qc" / "reprojection_visualization_index.csv",
        index=False,
    )


def _write_report(
    output: Path,
    camera: str,
    before_summary: dict[str, float | int],
    after_summary: dict[str, float | int],
    before_reprojection_summary: dict[str, float | int],
    after_reprojection_summary: dict[str, float | int],
    offset_summary: pd.DataFrame,
    frame_summary: pd.DataFrame,
    solver_status: dict[str, object],
) -> None:
    top_frames = frame_summary.sort_values("delta_norm_deg", ascending=False).head(20)
    lines = [
        f"# {camera} time-varying rotation diagnostic",
        "",
        "This is diagnostic 5.5: fixed navigation, fixed installation extrinsic, per-frame small rotation disturbance only.",
        "",
        "## Overall",
        "",
        f"- Before mean sqrt(Sampson): {before_summary['mean']:.6f} px",
        f"- After mean sqrt(Sampson): {after_summary['mean']:.6f} px",
        f"- Before median sqrt(Sampson): {before_summary['median']:.6f} px",
        f"- After median sqrt(Sampson): {after_summary['median']:.6f} px",
        f"- Before median reprojection RMS: {before_reprojection_summary['median']:.6f} px",
        f"- After median reprojection RMS: {after_reprojection_summary['median']:.6f} px",
        f"- Before P95 reprojection RMS: {before_reprojection_summary['p95']:.6f} px",
        f"- After P95 reprojection RMS: {after_reprojection_summary['p95']:.6f} px",
        f"- Solver success: {solver_status['success']}",
        f"- Solver message: {solver_status['message']}",
        f"- Solver nfev: {solver_status['nfev']}",
        "",
        "## By Frame Offset",
        "",
        "| frame offset | count | before Sampson median | after Sampson median | before reproj median | after reproj median | before reproj P95 | after reproj P95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in offset_summary.itertuples(index=False):
        lines.append(
            f"| {row.frame_offset} | {row.count} | {row.before_sampson_median_px:.6f} | "
            f"{row.after_sampson_median_px:.6f} | {row.before_reprojection_median_px:.6f} | "
            f"{row.after_reprojection_median_px:.6f} | {row.before_reprojection_p95_px:.6f} | "
            f"{row.after_reprojection_p95_px:.6f} |"
        )
    lines += [
        "",
        "## Largest Estimated Frame Disturbances",
        "",
        "| frame | norm deg | dx/roll deg | dy/pitch deg | dz/yaw deg | before incident mean | after incident mean | incident matches |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_frames.itertuples(index=False):
        lines.append(
            f"| {row.frame} | {row.delta_norm_deg:.6f} | {row.delta_roll_deg:.6f} | "
            f"{row.delta_pitch_deg:.6f} | {row.delta_yaw_deg:.6f} | "
            f"{row.before_incident_sampson_mean_px:.6f} | "
            f"{row.after_incident_sampson_mean_px:.6f} | {row.incident_match_count} |"
        )
    lines += [
        "",
        "## Visualizations",
        "",
        "- `visualizations/frame_disturbance_timeline.png`",
        "- `visualizations/sampson_median_by_offset.png`",
        "- `visualizations/reprojection_median_by_offset.png`",
        "- `visualizations/reprojection_median_over_frame_before_after.png`",
        "- `visualizations/reprojection_qc/before_after_reprojection/`",
    ]
    (output / f"{camera}_time_varying_rotation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
