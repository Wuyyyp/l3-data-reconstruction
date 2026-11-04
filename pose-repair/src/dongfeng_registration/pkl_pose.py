from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_models import CameraCalibration
from .dataset import Frame
from .navigation import NavigationTrajectory


PROJECT_TO_PKL_CAMERA = {
    "front_camera_fov30": "center_camera_fov30",
    "front_camera_fov120": "center_camera_fov120",
}


def pose_backend_is_result_interpolation(assumptions: dict[str, object] | None) -> bool:
    assumptions = assumptions or {}
    return str(assumptions.get("pose_backend", "")).lower() == "result_interpolation_pkl"


def resolve_result_interpolation_path(
    assumptions: dict[str, object],
    dataset_root: Path,
) -> Path | None:
    if not pose_backend_is_result_interpolation(assumptions):
        return None
    value = assumptions.get("result_interpolation_pkl_path")
    if value in (None, ""):
        value = assumptions.get("pose_pkl_path")
    if value in (None, ""):
        raise KeyError(
            "pose_backend=result_interpolation_pkl requires result_interpolation_pkl_path"
        )
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(dataset_root) / path
    return path


def resolve_result_interpolation_image_root(
    assumptions: dict[str, object],
    dataset_root: Path,
) -> Path | None:
    value = assumptions.get("result_interpolation_image_root")
    if value in (None, ""):
        value = assumptions.get("pose_pkl_image_root")
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(dataset_root) / path
    return path


@lru_cache(maxsize=8)
def _load_infos_cached(path_text: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_text)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"]
    return tuple(infos)


def load_result_interpolation_infos(path: Path | str) -> tuple[dict[str, Any], ...]:
    return _load_infos_cached(str(Path(path).resolve()))


def _pkl_camera_name(camera: str) -> str:
    return PROJECT_TO_PKL_CAMERA.get(str(camera), str(camera))


def _camera_payload(cameras: dict[str, Any], camera: str) -> dict[str, Any]:
    """Resolve raw result_interpolation and canonical camera names."""
    mapped_name = _pkl_camera_name(camera)
    if mapped_name in cameras:
        return cameras[mapped_name]
    if str(camera) in cameras:
        return cameras[str(camera)]
    raise KeyError(
        f"camera {camera!r} is unavailable; expected {mapped_name!r} or "
        f"{str(camera)!r}, found {sorted(cameras)}"
    )


def _timestamp_ns_from_image_stem(stem: str) -> int:
    value = int(str(stem))
    # L29 result_interpolation image names are microsecond timestamps
    # (e.g. 1753002123700312.jpg).  Existing compat data used ns names.
    return value if value >= 1_000_000_000_000_000_000 else value * 1000


def frames_from_image_root(image_root: Path | str, camera: str) -> list[Frame]:
    camera_dir = Path(image_root) / str(camera)
    if not camera_dir.exists():
        raise FileNotFoundError(camera_dir)
    frames: list[Frame] = []
    for index, path in enumerate(sorted(camera_dir.glob("*.jpg"))):
        frames.append(
            Frame(
                index=int(index),
                timestamp_ns=_timestamp_ns_from_image_stem(path.stem),
                path=path,
            )
        )
    return frames


def frame_metadata_from_image_root(
    image_root: Path | str,
    cameras: tuple[str, ...] | list[str],
) -> dict[str, dict[int, dict[str, object]]]:
    metadata: dict[str, dict[int, dict[str, object]]] = {}
    for camera in cameras:
        by_frame: dict[int, dict[str, object]] = {}
        for frame in frames_from_image_root(image_root, str(camera)):
            by_frame[int(frame.index)] = {
                "timestamp_ns": int(frame.timestamp_ns),
                "image_filename": frame.path.name,
                "image_path": str(frame.path),
            }
        metadata[str(camera)] = by_frame
    return metadata


def navigation_from_result_interpolation(
    path: Path | str,
    *,
    ego_field: str = "ego2global_transformation_matrix",
    allow_end_extrapolation: bool = False,
) -> NavigationTrajectory:
    infos = load_result_interpolation_infos(path)
    timestamps = np.asarray([int(info["timestamp"]) for info in infos], dtype=np.int64)
    transforms = np.asarray(
        [info[str(ego_field)] for info in infos],
        dtype=np.float64,
    )
    positions = transforms[:, :3, 3].copy()
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    order = np.argsort(timestamps)
    return NavigationTrajectory(
        timestamps_ns=timestamps[order],
        positions=positions[order],
        quaternions_xyzw=quaternions[order],
        allow_end_extrapolation=bool(allow_end_extrapolation),
    )


def calibration_from_result_interpolation(
    path: Path | str,
    camera: str,
    *,
    frame_index: int = 0,
    extrinsic_direction: str = "camera_to_car",
    images_undistorted: bool = False,
) -> CameraCalibration:
    infos = load_result_interpolation_infos(path)
    cam = _camera_payload(
        infos[int(frame_index)]["sensors"]["cams"],
        str(camera),
    )
    intrinsic = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(cam["d"], dtype=np.float64).reshape(-1)
    extrinsic = np.asarray(cam["extrinsic"], dtype=np.float64)
    direction = str(extrinsic_direction).lower()
    if direction in {"camera_to_car", "camera_to_ego", "sensor_to_car"}:
        sensor_to_car = extrinsic
    elif direction in {"car_to_camera", "ego_to_camera", "car_to_sensor"}:
        sensor_to_car = np.linalg.inv(extrinsic)
    else:
        raise ValueError(
            "result_interpolation extrinsic_direction must be one of "
            "camera_to_car or car_to_camera"
        )
    if distortion.size not in (4, 8):
        raise ValueError(
            f"unsupported distortion count for {camera} in result_interpolation.pkl: "
            f"{distortion.size}"
        )
    model = "fisheye" if distortion.size == 4 else "rational"
    width = int(round(float(intrinsic[0, 2]) * 2.0))
    height = int(round(float(intrinsic[1, 2]) * 2.0))
    return CameraCalibration(
        camera=str(camera),
        model=model,
        image_size=(width, height),
        intrinsic=intrinsic,
        distortion=distortion,
        sensor_to_car=sensor_to_car,
        time_lag=0.0,
        images_undistorted=bool(images_undistorted),
    )


def calibrations_from_result_interpolation(
    path: Path | str,
    cameras: tuple[str, ...] | list[str],
    *,
    extrinsic_direction: str = "camera_to_car",
    images_undistorted: bool = False,
) -> dict[str, CameraCalibration]:
    return {
        str(camera): calibration_from_result_interpolation(
            path,
            str(camera),
            extrinsic_direction=extrinsic_direction,
            images_undistorted=images_undistorted,
        )
        for camera in cameras
    }


def frame_metadata_from_result_interpolation(
    path: Path | str,
    cameras: tuple[str, ...] | list[str],
    *,
    image_root: Path | str | None = None,
) -> dict[str, dict[int, dict[str, object]]]:
    infos = load_result_interpolation_infos(path)
    root = None if image_root is None else Path(image_root)
    metadata: dict[str, dict[int, dict[str, object]]] = {}
    for camera in cameras:
        camera = str(camera)
        by_frame: dict[int, dict[str, object]] = {}
        for frame_index, info in enumerate(infos):
            cam = _camera_payload(info["sensors"]["cams"], camera)
            raw_timestamp = int(cam["timestamp"])
            timestamp_ns = raw_timestamp * 1000
            image_filename = f"{raw_timestamp}.jpg"
            image_path = (
                str(root / camera / image_filename)
                if root is not None
                else image_filename
            )
            by_frame[int(frame_index)] = {
                "timestamp_ns": int(timestamp_ns),
                "image_filename": image_filename,
                "image_path": image_path,
            }
        metadata[camera] = by_frame
    return metadata
