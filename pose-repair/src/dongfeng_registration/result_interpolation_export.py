from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation


PKL_TO_PROJECT_CAMERA = {
    "center_camera_fov30": "front_camera_fov30",
    "center_camera_fov120": "front_camera_fov120",
}


def delta_transform(parameters: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if values.size < 3:
        raise ValueError("disturbance transform requires at least 3 rotation values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    if values.size >= 6:
        transform[:3, 3] = values[3:6]
    return transform


def _as_transform(value: object) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got shape {transform.shape}")
    return transform


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> list[np.float64]:
    quat_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return [np.float64(value) for value in np.roll(quat_xyzw, 1)]


def _project_camera_name(pkl_camera: str) -> str:
    return PKL_TO_PROJECT_CAMERA.get(str(pkl_camera), str(pkl_camera))


def _project_camera_candidates(
    pkl_camera: str, available: set[str] | None = None
) -> list[str]:
    """Return canonical name first, with legacy center/front aliases as fallback."""
    name = str(pkl_camera)
    candidates = [name]
    mapped = PKL_TO_PROJECT_CAMERA.get(name)
    if mapped and mapped not in candidates:
        candidates.append(mapped)
    if available is not None:
        present = [candidate for candidate in candidates if candidate in available]
        if present:
            return present
    return candidates


def _timestamp_ns_from_camera_timestamp(value: object) -> int:
    timestamp = int(value)
    return timestamp if timestamp >= 1_000_000_000_000_000_000 else timestamp * 1000


def _frame_index_for_info(
    info: dict[str, Any],
    *,
    optimized_sensor_to_car: dict[str, np.ndarray],
    timestamp_to_frame: dict[int, int],
) -> int | None:
    cams = info.get("sensors", {}).get("cams", {})
    for pkl_camera, cam in cams.items():
        candidates = _project_camera_candidates(
            str(pkl_camera), set(optimized_sensor_to_car)
        )
        project_camera = next(
            (candidate for candidate in candidates if candidate in optimized_sensor_to_car),
            None,
        )
        if project_camera is None:
            continue
        if not isinstance(cam, dict) or "timestamp" not in cam:
            continue
        frame_index = timestamp_to_frame.get(
            _timestamp_ns_from_camera_timestamp(cam["timestamp"])
        )
        if frame_index is not None:
            return int(frame_index)
    return timestamp_to_frame.get(int(info["timestamp"]))


def _lidar_to_ego_for_info(info: dict[str, Any]) -> np.ndarray | None:
    lidar = info.get("sensors", {}).get("lidar", {})
    for name in ("top_center_lidar", "perception"):
        item = lidar.get(name)
        if isinstance(item, dict) and "extrinsic" in item:
            return _as_transform(item["extrinsic"])
    return None


def _timestamp_to_frame_from_image_extrinsics(path: Path) -> dict[int, int]:
    frame = pd.read_csv(path, usecols=["timestamp_ns", "frame"])
    result: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        result.setdefault(int(row.timestamp_ns), int(row.frame))
    return result


def _load_optimized_yaml(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], np.ndarray], dict[int, np.ndarray]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    optimized_sensor_to_car = {
        str(camera): _as_transform(value)
        for camera, value in dict(data.get("optimized_sensor_to_car", {})).items()
    }
    camera_frame_disturbances: dict[tuple[str, int], np.ndarray] = {}
    for key, value in dict(data.get("optimized_frame_rotvecs", {})).items():
        camera, frame_text = str(key).rsplit(":", 1)
        camera_frame_disturbances[(camera, int(frame_text))] = np.asarray(
            value,
            dtype=np.float64,
        )
    body_frame_disturbances = {
        int(frame): np.asarray(value, dtype=np.float64)
        for frame, value in dict(data.get("optimized_body_frame_disturbances", {})).items()
    }
    return optimized_sensor_to_car, camera_frame_disturbances, body_frame_disturbances


def apply_optimized_transforms_to_payload(
    payload: dict[str, Any],
    *,
    optimized_sensor_to_car: dict[str, np.ndarray],
    camera_frame_disturbances: dict[tuple[str, int], np.ndarray],
    body_frame_disturbances: dict[int, np.ndarray],
    timestamp_to_frame: dict[int, int],
    camera_world_pose_lookup: dict[tuple[str, int], np.ndarray] | None = None,
    source_ego_field: str = "ego2global_transformation_matrix",
    optimized_ego_output_field: str = "ego2global_transformation_matrix",
) -> dict[str, int]:
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError("payload must contain list field 'infos'")
    report = {
        "infos_total": len(infos),
        "infos_updated": 0,
        "infos_without_frame": 0,
        "cameras_total": 0,
        "cameras_updated": 0,
        "cameras_skipped_not_optimized": 0,
        "sensor2lidar_updated": 0,
    }
    for info in infos:
        frame_index = _frame_index_for_info(
            info,
            optimized_sensor_to_car=optimized_sensor_to_car,
            timestamp_to_frame=timestamp_to_frame,
        )
        if frame_index is None:
            info[str(optimized_ego_output_field)] = _as_transform(
                info[str(source_ego_field)]
            ).tolist()
            report["infos_without_frame"] += 1
            continue

        body_delta = delta_transform(
            body_frame_disturbances.get(frame_index, np.zeros(6, dtype=np.float64))
        )
        original_ego_to_global = _as_transform(info[str(source_ego_field)])
        optimized_ego_to_global = original_ego_to_global @ body_delta
        info[str(optimized_ego_output_field)] = optimized_ego_to_global.tolist()
        report["infos_updated"] += 1

        lidar_to_ego = _lidar_to_ego_for_info(info)
        ego_to_lidar = None if lidar_to_ego is None else np.linalg.inv(lidar_to_ego)
        cams = info.get("sensors", {}).get("cams", {})
        for pkl_camera, cam in cams.items():
            report["cameras_total"] += 1
            project_camera = next(
                (
                    candidate
                    for candidate in _project_camera_candidates(
                        str(pkl_camera), set(optimized_sensor_to_car)
                    )
                    if candidate in optimized_sensor_to_car
                ),
                _project_camera_name(str(pkl_camera)),
            )
            fixed_sensor_to_ego = optimized_sensor_to_car.get(project_camera)
            if fixed_sensor_to_ego is None:
                report["cameras_skipped_not_optimized"] += 1
                continue
            camera_delta = delta_transform(
                camera_frame_disturbances.get(
                    (project_camera, frame_index),
                    np.zeros(6, dtype=np.float64),
                )
            )
            sensor_to_ego = fixed_sensor_to_ego @ camera_delta
            ego_to_sensor = np.linalg.inv(sensor_to_ego)
            cam["extrinsic"] = ego_to_sensor.tolist()
            cam["ego2sensor_translation"] = ego_to_sensor[:3, 3].astype(np.float32)
            cam["ego2sensor_rotation"] = _quat_wxyz_from_matrix(ego_to_sensor[:3, :3])
            if ego_to_lidar is not None:
                sensor_to_lidar = ego_to_lidar @ sensor_to_ego
                cam["sensor2lidar_rotation"] = sensor_to_lidar[:3, :3].astype(np.float32)
                cam["sensor2lidar_translation"] = sensor_to_lidar[:3, 3].astype(np.float32)
                report["sensor2lidar_updated"] += 1
            if camera_world_pose_lookup is not None:
                timestamp_ns = _timestamp_ns_from_camera_timestamp(cam["timestamp"])
                world_pose = next(
                    (
                        camera_world_pose_lookup.get((candidate, timestamp_ns))
                        for candidate in _project_camera_candidates(str(pkl_camera))
                        if camera_world_pose_lookup.get((candidate, timestamp_ns)) is not None
                    ),
                    None,
                )
                if world_pose is not None:
                    cam["camera_to_world_optimized"] = _as_transform(world_pose).tolist()
            report["cameras_updated"] += 1
    return report


def _image_extrinsics_lookup(path: Path) -> dict[tuple[str, int], np.ndarray]:
    frame = pd.read_csv(path)
    required = {"camera", "timestamp_ns"} | {f"T_cw_{r}{c}" for r in range(4) for c in range(4)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"image extrinsics CSV is missing columns: {sorted(missing)}")
    lookup: dict[tuple[str, int], np.ndarray] = {}
    matrix_columns = [f"T_cw_{r}{c}" for r in range(4) for c in range(4)]
    for row in frame.itertuples(index=False):
        values = [getattr(row, column) for column in matrix_columns]
        lookup[(str(row.camera), int(row.timestamp_ns))] = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(4, 4)
    return lookup


def validate_optimized_result_interpolation_pkl(
    *,
    optimized_pkl: Path,
    image_extrinsics_csv: Path,
    validation_json: Path | None = None,
    ego_field: str = "ego2global_transformation_matrix",
) -> dict[str, object]:
    with optimized_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    csv_lookup = _image_extrinsics_lookup(image_extrinsics_csv)
    camera_pose_errors: list[float] = []
    missing_csv_count = 0
    skipped_unoptimized_camera_count = 0
    for info in payload.get("infos", []):
        if not isinstance(info, dict):
            continue
        ego_to_global = _as_transform(info[str(ego_field)])
        cams = info.get("sensors", {}).get("cams", {})
        for pkl_camera, cam in cams.items():
            candidates = _project_camera_candidates(str(pkl_camera))
            if not isinstance(cam, dict) or "timestamp" not in cam or "extrinsic" not in cam:
                skipped_unoptimized_camera_count += 1
                continue
            timestamp_ns = _timestamp_ns_from_camera_timestamp(cam["timestamp"])
            expected = next(
                (
                    csv_lookup.get((candidate, timestamp_ns))
                    for candidate in candidates
                    if csv_lookup.get((candidate, timestamp_ns)) is not None
                ),
                None,
            )
            direct_pose = cam.get("camera_to_world_optimized")
            if direct_pose is not None and expected is not None:
                camera_pose_errors.append(float(np.linalg.norm(_as_transform(direct_pose) - expected)))
                continue
            if expected is None:
                if not any(
                    camera == candidate for candidate in candidates for camera, _timestamp in csv_lookup
                ):
                    skipped_unoptimized_camera_count += 1
                else:
                    missing_csv_count += 1
                continue
            ego_to_sensor = _as_transform(cam["extrinsic"])
            sensor_to_global = ego_to_global @ np.linalg.inv(ego_to_sensor)
            camera_pose_errors.append(float(np.linalg.norm(sensor_to_global - expected)))
    errors = np.asarray(camera_pose_errors, dtype=np.float64)
    validation = {
        "optimized_pkl": str(optimized_pkl),
        "image_extrinsics_csv": str(image_extrinsics_csv),
        "checked_camera_pose_count": int(errors.size),
        "missing_csv_count": int(missing_csv_count),
        "skipped_unoptimized_camera_count": int(skipped_unoptimized_camera_count),
        "max_frobenius_error": float(np.max(errors)) if errors.size else None,
        "median_frobenius_error": float(np.median(errors)) if errors.size else None,
        "p95_frobenius_error": float(np.percentile(errors, 95)) if errors.size else None,
    }
    if validation_json is not None:
        validation_json.parent.mkdir(parents=True, exist_ok=True)
        validation_json.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return validation


def generate_optimized_result_interpolation_pkl(
    *,
    input_pkl: Path,
    optimized_yaml: Path,
    image_extrinsics_csv: Path,
    output_pkl: Path,
    manifest_json: Path | None = None,
    validation_json: Path | None = None,
    source_ego_field: str = "ego2global_transformation_matrix",
    optimized_ego_output_field: str = (
        "ego2global_transformation_matrix_camera_reference_offset_optimized"
    ),
) -> dict[str, object]:
    with input_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    (
        optimized_sensor_to_car,
        camera_frame_disturbances,
        body_frame_disturbances,
    ) = _load_optimized_yaml(optimized_yaml)
    timestamp_to_frame = _timestamp_to_frame_from_image_extrinsics(image_extrinsics_csv)
    camera_world_pose_lookup = _image_extrinsics_lookup(image_extrinsics_csv)
    report = apply_optimized_transforms_to_payload(
        payload,
        optimized_sensor_to_car=optimized_sensor_to_car,
        camera_frame_disturbances=camera_frame_disturbances,
        body_frame_disturbances=body_frame_disturbances,
        timestamp_to_frame=timestamp_to_frame,
        camera_world_pose_lookup=camera_world_pose_lookup,
        source_ego_field=source_ego_field,
        optimized_ego_output_field=optimized_ego_output_field,
    )
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "input_pkl": str(input_pkl),
        "optimized_yaml": str(optimized_yaml),
        "image_extrinsics_csv": str(image_extrinsics_csv),
        "output_pkl": str(output_pkl),
        "validation_json": None if validation_json is None else str(validation_json),
        "updated_fields": [
            f"infos[*].{optimized_ego_output_field}",
            "infos[*].sensors.cams.<optimized_camera>.extrinsic",
            "infos[*].sensors.cams.<optimized_camera>.camera_to_world_optimized",
            "infos[*].sensors.cams.<optimized_camera>.ego2sensor_translation",
            "infos[*].sensors.cams.<optimized_camera>.ego2sensor_rotation",
            "infos[*].sensors.cams.<optimized_camera>.sensor2lidar_rotation",
            "infos[*].sensors.cams.<optimized_camera>.sensor2lidar_translation",
        ],
        "camera_name_mapping": dict(PKL_TO_PROJECT_CAMERA),
        "source_ego_field": str(source_ego_field),
        "optimized_ego_output_field": str(optimized_ego_output_field),
        **report,
    }
    if manifest_json is not None:
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if validation_json is not None:
        validate_optimized_result_interpolation_pkl(
            optimized_pkl=output_pkl,
            image_extrinsics_csv=image_extrinsics_csv,
            validation_json=validation_json,
            ego_field=optimized_ego_output_field,
        )
    return manifest
