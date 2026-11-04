from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera: str
    model: str
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    distortion: np.ndarray
    sensor_to_car: np.ndarray
    time_lag: float
    images_undistorted: bool

    @classmethod
    def from_dataset(cls, root: Path, camera: str) -> "CameraCalibration":
        calib_dir = Path(root) / "calib" / camera
        intrinsic_raw = json.loads(
            (calib_dir / f"{camera}-intrinsic.json").read_text(encoding="utf-8")
        )["value0"]["param"]
        extrinsic_doc = json.loads(
            (calib_dir / f"{camera}-to-car_center-extrinsic.json").read_text(encoding="utf-8")
        )
        extrinsic_raw = next(iter(extrinsic_doc.values()))["param"]
        status_path = calib_dir / f"{camera}-undistortion_status.json"
        images_undistorted = False
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status_param = next(iter(status.values()))["param"]
            images_undistorted = bool(status_param.get("undistorsion_result", 0))
        distortion = np.asarray(intrinsic_raw["cam_dist"]["data"][0], dtype=np.float64)
        model = "fisheye" if distortion.size == 4 else "rational"
        if distortion.size not in (4, 8):
            raise ValueError(f"unsupported distortion count for {camera}: {distortion.size}")
        return cls(
            camera=camera,
            model=model,
            image_size=(int(intrinsic_raw["img_dist_w"]), int(intrinsic_raw["img_dist_h"])),
            intrinsic=np.asarray(intrinsic_raw["cam_K_new"]["data"], dtype=np.float64),
            distortion=distortion,
            sensor_to_car=np.asarray(extrinsic_raw["sensor_calib"]["data"], dtype=np.float64),
            time_lag=float(extrinsic_raw["time_lag"]),
            images_undistorted=images_undistorted,
        )


def undistort_normalized(points: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if calibration.images_undistorted:
        result = cv2.undistortPoints(
            shaped, calibration.intrinsic, np.zeros(5, dtype=np.float64)
        )
    elif calibration.model == "fisheye":
        result = cv2.fisheye.undistortPoints(
            shaped, calibration.intrinsic, calibration.distortion.reshape(-1, 1)
        )
    else:
        result = cv2.undistortPoints(shaped, calibration.intrinsic, calibration.distortion)
    return result.reshape(-1, 2)


def project_points(points_camera: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    raw_points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if calibration.images_undistorted:
        normalized = raw_points[:, :2] / raw_points[:, 2, None]
        return normalized @ calibration.intrinsic[:2, :2].T + calibration.intrinsic[:2, 2]
    points = raw_points.reshape(-1, 1, 3)
    zero = np.zeros((3, 1), dtype=np.float64)
    if calibration.model == "fisheye":
        pixels, _ = cv2.fisheye.projectPoints(
            points, zero, zero, calibration.intrinsic, calibration.distortion.reshape(-1, 1)
        )
    else:
        pixels, _ = cv2.projectPoints(
            points, zero, zero, calibration.intrinsic, calibration.distortion
        )
    return pixels.reshape(-1, 2)
