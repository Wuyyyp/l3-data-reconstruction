from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np

from .camera_models import CameraCalibration
from .dataset import Dataset
from .navigation import NavigationTrajectory


STANDARD_CAMERA_NAMES = (
    "front_camera_fov30",
    "front_camera_fov120",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_front_camera",
    "right_rear_camera",
)


@dataclass(frozen=True)
class NormalizationResult:
    output_root: Path
    standard_pkl_path: Path
    report_path: Path
    frames_csv_path: Path
    cameras_json_path: Path
    lidars_json_path: Path


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _load_pickle(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), list):
        raise ValueError(f"{path} must contain a dict with list field 'infos'")
    return payload


def _write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _copy_file(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return True
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return True
        except OSError:
            pass
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return True
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def _resolve_relative_or_dataset_path(dataset_root: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return dataset_root / path


def _camera_timestamp_to_filename(timestamp: int) -> str:
    return f"{int(timestamp)}.jpg"


def _resolve_camera_image(
    *,
    dataset_root: Path,
    image_root: Path,
    camera_payload: dict[str, Any],
    source_camera: str,
    target_camera: str,
) -> Path:
    data_path = camera_payload.get("data_path") or camera_payload.get("image_path")
    candidates: list[Path] = []
    if data_path:
        candidates.append(_resolve_relative_or_dataset_path(dataset_root, str(data_path)))
        candidates.append(image_root / Path(str(data_path)).name)
        candidates.append(image_root / source_camera / Path(str(data_path)).name)
        candidates.append(image_root / target_camera / Path(str(data_path)).name)
    timestamp = camera_payload.get("timestamp")
    if timestamp is not None:
        candidates.append(image_root / source_camera / _camera_timestamp_to_filename(int(timestamp)))
        candidates.append(image_root / target_camera / _camera_timestamp_to_filename(int(timestamp)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else image_root / source_camera / "missing.jpg"


def _resolve_lidar_file(
    *,
    dataset_root: Path,
    lidar_payload: dict[str, Any],
    lidar_name: str,
) -> Path:
    data_path = lidar_payload.get("data_path") or lidar_payload.get("aws_path")
    candidates: list[Path] = []
    if data_path:
        raw = Path(str(data_path))
        if str(lidar_name) != "perception":
            candidates.append(dataset_root / "lidar" / f"{lidar_name}_compensation" / raw.name)
            candidates.append(dataset_root / f"{lidar_name}_compensation" / raw.name)
        candidates.append(_resolve_relative_or_dataset_path(dataset_root, raw))
        candidates.append(dataset_root / raw.name)
        candidates.append(dataset_root / lidar_name / raw.name)
        candidates.append(dataset_root / "lidar" / lidar_name / raw.name)
        if str(lidar_name) != "perception":
            candidates.append(dataset_root / "lidar" / f"{lidar_name}_compensation" / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else dataset_root / lidar_name / "missing.pcd"


def _normalize_extrinsic(extrinsic: Any, direction: str) -> list[list[float]]:
    matrix = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    normalized_direction = str(direction).lower()
    if normalized_direction in {"car_to_camera", "ego_to_camera", "car_to_sensor"}:
        output = matrix
    elif normalized_direction in {"camera_to_car", "camera_to_ego", "sensor_to_car"}:
        output = np.linalg.inv(matrix)
    else:
        raise ValueError(
            "extrinsic_direction must be one of car_to_camera or camera_to_car"
        )
    return output.tolist()


def _relative_to(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_t3_annotation(dataset_root: Path, timestamp_ns: int) -> dict[str, Any] | None:
    path = dataset_root / "format_output" / "annotations" / "NV" / f"{int(timestamp_ns)}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _copy_t3_annotation_fields(info: dict[str, Any], annotation: dict[str, Any]) -> None:
    field_names = (
        "gt_boxes",
        "gt_names",
        "track_id",
        "num_lidar_pts",
        "valid_flag",
        "gt_qualities",
    )
    for field_name in field_names:
        if field_name in annotation:
            info[field_name] = _as_jsonable(annotation[field_name])
    if "track_id" in annotation:
        info["ID"] = _as_jsonable(annotation["track_id"])
    if "num_lidar_pts" in annotation:
        info["num_points"] = _as_jsonable(annotation["num_lidar_pts"])
    if "velocity" in annotation:
        info["vels"] = _as_jsonable(annotation["velocity"])


def _load_t3_temporal_alignment(dataset_root: Path) -> list[tuple[int, dict[str, str]]]:
    path = dataset_root / "sensor_temporal_alignment.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return []
    rows: list[tuple[int, dict[str, str]]] = []
    for item in payload:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        timestamp_text, sensors = next(iter(item.items()))
        if not isinstance(sensors, dict):
            continue
        rows.append((int(timestamp_text), {str(k): str(v) for k, v in sensors.items()}))
    rows.sort(key=lambda row: row[0])
    return rows


def _load_t3_sensor_to_car_extrinsic(dataset_root: Path, sensor_name: str) -> list[list[float]] | None:
    path = dataset_root / "calib" / sensor_name / f"{sensor_name}-to-car_center-extrinsic.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def _find_sensor_calib(value: Any) -> Any:
        if isinstance(value, dict):
            if "sensor_calib" in value:
                return value["sensor_calib"]
            for child in value.values():
                found = _find_sensor_calib(child)
                if found is not None:
                    return found
        return None

    sensor_calib = _find_sensor_calib(payload)
    if isinstance(sensor_calib, dict) and "data" in sensor_calib:
        matrix = np.asarray(sensor_calib["data"], dtype=np.float64).reshape(4, 4)
        return matrix.tolist()
    return None


def _aligned_sensor_timestamp_us(path_text: str) -> int:
    return int(Path(str(path_text)).stem) // 1000


def normalize_result_interpolation_dataset(
    *,
    dataset_root: Path | str,
    pkl_path: Path | str,
    output_root: Path | str,
    vehicle_model: str,
    image_root: Path | str | None = None,
    cameras: tuple[str, ...] | list[str] | None = None,
    camera_name_map: dict[str, str] | None = None,
    extrinsic_direction: str = "car_to_camera",
) -> NormalizationResult:
    """Normalize a result_interpolation.pkl dataset into the project canonical layout.

    The canonical pkl keeps the source ego pose untouched, renames cameras to project
    names, converts camera extrinsics to car_to_camera, and rewrites image/lidar
    paths to be relative to the canonical dataset directory.
    """

    dataset_root = Path(dataset_root).resolve()
    pkl_path = Path(pkl_path)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    pkl_path = pkl_path.resolve()
    output_root = Path(output_root).resolve()
    image_root = (Path(image_root) if image_root is not None else dataset_root).resolve()
    camera_name_map = dict(camera_name_map or {})

    payload = _load_pickle(pkl_path)
    infos = payload["infos"]
    selected_cameras = tuple(cameras or STANDARD_CAMERA_NAMES)

    normalized_infos: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    missing_lidar_files: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    camera_metadata: dict[str, Any] = {}
    lidar_metadata: dict[str, Any] = {}

    for frame_index, source_info in enumerate(infos):
        info = dict(source_info)
        source_sensors = dict(source_info.get("sensors", {}))
        source_cams = dict(source_sensors.get("cams", {}))
        source_lidars = dict(source_sensors.get("lidar", {}))

        normalized_cams: dict[str, Any] = {}
        for target_camera in selected_cameras:
            source_camera = camera_name_map.get(target_camera, target_camera)
            if source_camera not in source_cams:
                missing_images.append(
                    {
                        "frame_index": frame_index,
                        "camera": target_camera,
                        "source_camera": source_camera,
                        "reason": "missing camera payload",
                    }
                )
                continue
            cam_payload = dict(source_cams[source_camera])
            source_image = _resolve_camera_image(
                dataset_root=dataset_root,
                image_root=image_root,
                camera_payload=cam_payload,
                source_camera=source_camera,
                target_camera=target_camera,
            )
            target_image = output_root / "images" / target_camera / source_image.name
            copied = _copy_file(source_image, target_image)
            if not copied:
                missing_images.append(
                    {
                        "frame_index": frame_index,
                        "camera": target_camera,
                        "source_camera": source_camera,
                        "path": str(source_image),
                    }
                )
            cam_payload["data_path"] = _relative_to(output_root, target_image)
            if "extrinsic" in cam_payload:
                cam_payload["extrinsic"] = _normalize_extrinsic(
                    cam_payload["extrinsic"],
                    extrinsic_direction,
                )
            normalized_cams[target_camera] = cam_payload

            timestamp = int(cam_payload.get("timestamp", source_info.get("timestamp", 0)))
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "camera": target_camera,
                    "source_camera": source_camera,
                    "timestamp": timestamp,
                    "timestamp_ns": timestamp * 1000
                    if timestamp < 1_000_000_000_000_000_000
                    else timestamp,
                    "image_path": cam_payload["data_path"],
                    "image_exists": bool(copied),
                }
            )
            camera_metadata.setdefault(
                target_camera,
                {
                    "source_camera": source_camera,
                    "cam_intrinsic": _as_jsonable(cam_payload.get("cam_intrinsic")),
                    "distortion": _as_jsonable(cam_payload.get("d")),
                    "extrinsic_direction": "car_to_camera",
                },
            )

        normalized_lidars: dict[str, Any] = {}
        for lidar_name, lidar_payload_raw in source_lidars.items():
            if not isinstance(lidar_payload_raw, dict):
                normalized_lidars[str(lidar_name)] = lidar_payload_raw
                continue
            lidar_payload = dict(lidar_payload_raw)
            source_lidar = _resolve_lidar_file(
                dataset_root=dataset_root,
                lidar_payload=lidar_payload,
                lidar_name=str(lidar_name),
            )
            if str(lidar_name) == "perception":
                target_lidar = output_root / "lidar" / "perception" / source_lidar.name
            elif source_lidar.parent.name == f"{lidar_name}_compensation":
                target_lidar = (
                    output_root / "lidar" / f"{lidar_name}_compensation" / source_lidar.name
                )
                lidar_payload["point_coordinate_frame"] = "ego"
            else:
                target_lidar = (
                    output_root / "lidar" / "raw_lidars" / str(lidar_name) / source_lidar.name
                )
                lidar_payload.setdefault("point_coordinate_frame", "lidar")
            copied = _copy_file(source_lidar, target_lidar)
            if not copied:
                missing_lidar_files.append(
                    {
                        "frame_index": frame_index,
                        "lidar": str(lidar_name),
                        "path": str(source_lidar),
                    }
                )
            lidar_payload["data_path"] = _relative_to(output_root, target_lidar)
            normalized_lidars[str(lidar_name)] = lidar_payload
            lidar_metadata.setdefault(
                str(lidar_name),
                {
                    "canonical_folder": str(Path(lidar_payload["data_path"]).parent),
                    "has_extrinsic": "extrinsic" in lidar_payload,
                },
            )

        source_sensors["cams"] = normalized_cams
        source_sensors["lidar"] = normalized_lidars
        info["sensors"] = source_sensors
        normalized_infos.append(info)

    normalized_payload = dict(payload)
    normalized_payload["infos"] = normalized_infos
    normalized_payload.setdefault("metadata", {})
    normalized_payload["metadata"] = {
        **dict(normalized_payload.get("metadata", {})),
        "canonical_dataset": True,
        "vehicle_model": str(vehicle_model),
        "source_dataset_root": str(dataset_root),
        "source_pkl_path": str(pkl_path),
        "standardized_source": {
            "pose_backend": "result_interpolation_pkl",
            "dataset_root": str(dataset_root),
            "pkl_path": str(pkl_path),
            "image_root": str(image_root),
        },
        "extrinsic_direction": "car_to_camera",
        "camera_name_map": dict(camera_name_map),
    }

    metadata_root = output_root / "metadata"
    standard_pkl_path = output_root / "result_interpolation_standard.pkl"
    report_path = metadata_root / "normalization_report.json"
    frames_csv_path = metadata_root / "frames.csv"
    cameras_json_path = metadata_root / "cameras.json"
    lidars_json_path = metadata_root / "lidars.json"

    _write_pickle(standard_pkl_path, normalized_payload)
    metadata_root.mkdir(parents=True, exist_ok=True)
    with frames_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "camera",
                "source_camera",
                "timestamp",
                "timestamp_ns",
                "image_path",
                "image_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    cameras_json_path.write_text(
        json.dumps(_as_jsonable(camera_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lidars_json_path.write_text(
        json.dumps(_as_jsonable(lidar_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "canonical_dataset": True,
        "vehicle_model": str(vehicle_model),
        "source_dataset_root": str(dataset_root),
        "source_pkl_path": str(pkl_path),
        "output_root": str(output_root),
        "standard_pkl_path": str(standard_pkl_path),
        "frame_count": len(normalized_infos),
        "cameras": list(selected_cameras),
        "camera_name_map": dict(camera_name_map),
        "lidars": sorted(lidar_metadata),
        "timestamp_unit": "nanoseconds",
        "extrinsic_input_direction": str(extrinsic_direction),
        "extrinsic_output_direction": "car_to_camera",
        "missing_images": missing_images,
        "missing_lidar_files": missing_lidar_files,
    }
    report_path.write_text(
        json.dumps(_as_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return NormalizationResult(
        output_root=output_root,
        standard_pkl_path=standard_pkl_path,
        report_path=report_path,
        frames_csv_path=frames_csv_path,
        cameras_json_path=cameras_json_path,
        lidars_json_path=lidars_json_path,
    )


def normalize_ved_localization_dataset(
    *,
    dataset_root: Path | str,
    output_root: Path | str,
    vehicle_model: str,
    cameras: tuple[str, ...] | list[str] | None = None,
    camera_name_map: dict[str, str] | None = None,
    navigation_csv: Path | str | None = None,
) -> NormalizationResult:
    """Normalize the legacy T3 camera/calib/LocalLocalization layout.

    The output is a result_interpolation-style canonical pkl so downstream
    modules can use the same pkl pose backend as L29/M18.
    """

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    camera_name_map = dict(camera_name_map or {})
    selected_cameras = tuple(cameras or STANDARD_CAMERA_NAMES)
    navigation_path = (
        Path(navigation_csv)
        if navigation_csv is not None
        else dataset_root / "ved" / "LocalLocalization.csv"
    )
    if not navigation_path.is_absolute():
        navigation_path = dataset_root / navigation_path
    navigation_path = navigation_path.resolve()

    dataset = Dataset(dataset_root)
    navigation = NavigationTrajectory.from_csv(navigation_path)
    temporal_alignment = _load_t3_temporal_alignment(dataset_root)
    front_lidar_extrinsic = _load_t3_sensor_to_car_extrinsic(dataset_root, "front_lidar")

    source_frames_by_target: dict[str, Any] = {}
    calibrations_by_target: dict[str, CameraCalibration] = {}
    missing_images: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    camera_metadata: dict[str, Any] = {}

    for target_camera in selected_cameras:
        source_camera = camera_name_map.get(target_camera, target_camera)
        try:
            source_frames_by_target[target_camera] = dataset.frames(source_camera)
        except FileNotFoundError as exc:
            missing_images.append(
                {
                    "frame_index": None,
                    "camera": target_camera,
                    "source_camera": source_camera,
                    "path": str(exc),
                    "reason": "missing camera timestamp file",
                }
            )
            continue
        calibrations_by_target[target_camera] = CameraCalibration.from_dataset(
            dataset_root,
            source_camera,
        )

    if not source_frames_by_target:
        raise ValueError("no camera frames were found for normalization")

    frame_count = (
        len(temporal_alignment)
        if temporal_alignment
        else min(len(frames) for frames in source_frames_by_target.values())
    )
    normalized_infos: list[dict[str, Any]] = []
    missing_lidar_files: list[dict[str, Any]] = []
    missing_annotations: list[dict[str, Any]] = []
    lidar_metadata: dict[str, Any] = {}

    for frame_index in range(frame_count):
        aligned_sensors: dict[str, str] = {}
        if temporal_alignment:
            reference_timestamp_ns, aligned_sensors = temporal_alignment[frame_index]
        else:
            reference_camera = next(iter(source_frames_by_target))
            reference_timestamp_ns = int(
                source_frames_by_target[reference_camera][frame_index].timestamp_ns
            )
        ego_to_global = navigation.pose_at(reference_timestamp_ns)
        cams_payload: dict[str, Any] = {}

        for target_camera, frames in source_frames_by_target.items():
            source_camera = camera_name_map.get(target_camera, target_camera)
            if source_camera in aligned_sensors:
                source_image = dataset_root / aligned_sensors[source_camera]
                timestamp_ns = int(Path(aligned_sensors[source_camera]).stem)
                timestamp_us = timestamp_ns // 1000
            else:
                frame = frames[frame_index]
                source_image = Path(frame.path)
                timestamp_ns = int(frame.timestamp_ns)
                timestamp_us = int(frame.timestamp_ns // 1000)
            target_image = (
                output_root
                / "images"
                / target_camera
                / _camera_timestamp_to_filename(timestamp_us)
            )
            copied = _copy_file(source_image, target_image)
            if not copied:
                missing_images.append(
                    {
                        "frame_index": frame_index,
                        "camera": target_camera,
                        "source_camera": source_camera,
                        "path": str(source_image),
                    }
                )

            calibration = calibrations_by_target[target_camera]
            car_to_camera = np.linalg.inv(
                np.asarray(calibration.sensor_to_car, dtype=np.float64).reshape(4, 4)
            )
            cams_payload[target_camera] = {
                "timestamp": timestamp_us,
                "data_path": _relative_to(output_root, target_image),
                "cam_intrinsic": calibration.intrinsic.tolist(),
                "d": calibration.distortion.tolist(),
                "extrinsic": car_to_camera.tolist(),
                "images_undistorted": bool(calibration.images_undistorted),
                "source_camera": source_camera,
            }
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "camera": target_camera,
                        "source_camera": source_camera,
                        "timestamp": timestamp_us,
                        "timestamp_ns": int(timestamp_ns),
                        "image_path": _relative_to(output_root, target_image),
                        "image_exists": bool(copied),
                    }
            )
            camera_metadata.setdefault(
                target_camera,
                {
                    "source_camera": source_camera,
                    "image_size": list(calibration.image_size),
                    "model": calibration.model,
                    "cam_intrinsic": calibration.intrinsic.tolist(),
                    "distortion": calibration.distortion.tolist(),
                    "input_extrinsic_direction": "camera_to_car",
                    "extrinsic_direction": "car_to_camera",
                    "time_lag": float(calibration.time_lag),
                    "images_undistorted": bool(calibration.images_undistorted),
                },
            )

        lidar_payload: dict[str, Any] = {}
        front_lidar_path_text = aligned_sensors.get("front_lidar")
        if front_lidar_path_text:
            source_lidar = dataset_root / front_lidar_path_text
            target_lidar = (
                output_root / "lidar" / "front_lidar_compensation" / source_lidar.name
            )
            copied = _copy_file(source_lidar, target_lidar)
            if not copied:
                missing_lidar_files.append(
                    {
                        "frame_index": frame_index,
                        "lidar": "front_lidar",
                        "path": str(source_lidar),
                    }
                )
            front_payload = {
                "timestamp": int(reference_timestamp_ns),
                "data_path": _relative_to(output_root, target_lidar),
                "point_coordinate_frame": "ego",
            }
            if front_lidar_extrinsic is not None:
                front_payload["extrinsic"] = front_lidar_extrinsic
            lidar_payload["front_lidar"] = front_payload
            lidar_payload["perception"] = {
                "timestamp": int(reference_timestamp_ns),
                "data_path": _relative_to(output_root, target_lidar),
                "point_coordinate_frame": "ego",
            }
            lidar_metadata.setdefault(
                "front_lidar",
                {
                    "canonical_folder": str(Path(front_payload["data_path"]).parent),
                    "has_extrinsic": front_lidar_extrinsic is not None,
                    "point_coordinate_frame": "lidar",
                },
            )
            lidar_metadata.setdefault(
                "perception",
                {
                    "canonical_folder": str(Path(front_payload["data_path"]).parent),
                    "source_lidar": "front_lidar",
                    "point_coordinate_frame": "lidar",
                },
            )

        normalized_info = {
            "timestamp": reference_timestamp_ns,
            "ego2global_transformation_matrix": ego_to_global.tolist(),
            "sensors": {
                "cams": cams_payload,
                "lidar": lidar_payload,
            },
        }
        annotation = _load_t3_annotation(dataset_root, reference_timestamp_ns)
        if annotation is None:
            missing_annotations.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": reference_timestamp_ns,
                    "path": str(
                        dataset_root
                        / "format_output"
                        / "annotations"
                        / "NV"
                        / f"{reference_timestamp_ns}.json"
                    ),
                }
            )
        else:
            _copy_t3_annotation_fields(normalized_info, annotation)
        normalized_infos.append(normalized_info)

    normalized_payload: dict[str, Any] = {
        "infos": normalized_infos,
        "metadata": {
            "canonical_dataset": True,
            "vehicle_model": str(vehicle_model),
            "pose_backend": "ved_localization",
            "source_dataset_root": str(dataset_root),
            "navigation_csv": str(navigation_path),
            "standardized_source": {
                "pose_backend": "ved_localization",
                "dataset_root": str(dataset_root),
                "navigation_csv": str(navigation_path),
            },
            "extrinsic_direction": "car_to_camera",
            "camera_name_map": dict(camera_name_map),
        },
    }

    metadata_root = output_root / "metadata"
    standard_pkl_path = output_root / "result_interpolation_standard.pkl"
    report_path = metadata_root / "normalization_report.json"
    frames_csv_path = metadata_root / "frames.csv"
    cameras_json_path = metadata_root / "cameras.json"
    lidars_json_path = metadata_root / "lidars.json"

    _write_pickle(standard_pkl_path, normalized_payload)
    metadata_root.mkdir(parents=True, exist_ok=True)
    with frames_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "camera",
                "source_camera",
                "timestamp",
                "timestamp_ns",
                "image_path",
                "image_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    cameras_json_path.write_text(
        json.dumps(_as_jsonable(camera_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lidars_json_path.write_text(
        json.dumps(_as_jsonable(lidar_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "canonical_dataset": True,
        "vehicle_model": str(vehicle_model),
        "pose_backend": "ved_localization",
        "source_dataset_root": str(dataset_root),
        "navigation_csv": str(navigation_path),
        "output_root": str(output_root),
        "standard_pkl_path": str(standard_pkl_path),
        "frame_count": len(normalized_infos),
        "cameras": list(source_frames_by_target),
        "camera_name_map": dict(camera_name_map),
        "lidars": sorted(lidar_metadata),
        "timestamp_unit": "nanoseconds",
        "extrinsic_input_direction": "camera_to_car",
        "extrinsic_output_direction": "car_to_camera",
        "missing_images": missing_images,
        "missing_lidar_files": missing_lidar_files,
        "annotation_source": "format_output/annotations/NV",
        "temporal_alignment_source": "sensor_temporal_alignment.json"
        if temporal_alignment
        else None,
        "missing_annotations": missing_annotations,
    }
    report_path.write_text(
        json.dumps(_as_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return NormalizationResult(
        output_root=output_root,
        standard_pkl_path=standard_pkl_path,
        report_path=report_path,
        frames_csv_path=frames_csv_path,
        cameras_json_path=cameras_json_path,
        lidars_json_path=lidars_json_path,
    )
