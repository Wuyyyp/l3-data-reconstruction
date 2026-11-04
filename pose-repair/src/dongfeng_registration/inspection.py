from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .camera_models import CameraCalibration
from .dataset import Dataset


CAMERAS = [
    "center_camera_fov120", "center_camera_fov30", "front_camera_fov195",
    "left_camera_fov195", "left_front_camera", "left_rear_camera", "rear_camera",
    "rear_camera_fov195", "right_camera_fov195", "right_front_camera", "right_rear_camera",
]


def inspect_dataset(root: Path, cameras: list[str] | None = None) -> pd.DataFrame:
    dataset = Dataset(root)
    records = []
    for camera in cameras or CAMERAS:
        frames = dataset.frames(camera)
        calibration = CameraCalibration.from_dataset(root, camera)
        image = cv2.imread(str(frames[0].path))
        actual_size = (image.shape[1], image.shape[0]) if image is not None else (0, 0)
        rotation = calibration.sensor_to_car[:3, :3]
        records.append({
            "camera": camera,
            "frame_count": len(frames),
            "timestamps_monotonic": all(a.timestamp_ns < b.timestamp_ns for a, b in zip(frames, frames[1:])),
            "all_images_exist": all(frame.path.exists() for frame in frames),
            "image_width": actual_size[0], "image_height": actual_size[1],
            "image_size_matches": actual_size == calibration.image_size,
            "model": calibration.model,
            "calibration_valid": bool(
                calibration.intrinsic.shape == (3, 3)
                and calibration.sensor_to_car.shape == (4, 4)
                and np.isfinite(calibration.intrinsic).all()
                and abs(np.linalg.det(rotation) - 1.0) < 5e-4
            ),
        })
    return pd.DataFrame(records)
