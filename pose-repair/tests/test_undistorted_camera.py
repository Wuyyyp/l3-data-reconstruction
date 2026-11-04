from pathlib import Path

import cv2
import numpy as np
import pytest

from dongfeng_registration.camera_models import (
    CameraCalibration,
    project_points,
    undistort_normalized,
)
from tests.conftest import real_t3_data_available


DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)

pytestmark = pytest.mark.skipif(
    not real_t3_data_available(DATA_ROOT),
    reason=f"optional real T3 dataset not available: {DATA_ROOT}",
)


def test_center_camera_images_are_marked_undistorted():
    calibration = CameraCalibration.from_dataset(DATA_ROOT, "center_camera_fov30")
    assert calibration.images_undistorted is True


def test_undistorted_camera_uses_new_intrinsic_without_distortion():
    calibration = CameraCalibration.from_dataset(DATA_ROOT, "center_camera_fov30")
    point = np.array([[0.1, -0.05, 1.0]])
    expected_pixel = np.array(
        [[
            calibration.intrinsic[0, 0] * 0.1 + calibration.intrinsic[0, 2],
            calibration.intrinsic[1, 1] * -0.05 + calibration.intrinsic[1, 2],
        ]]
    )
    pixels = project_points(point, calibration)
    normalized = undistort_normalized(expected_pixel, calibration)
    np.testing.assert_allclose(pixels, expected_pixel, atol=1e-9)
    np.testing.assert_allclose(normalized, [[0.1, -0.05]], atol=1e-9)
