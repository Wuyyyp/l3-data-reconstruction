from pathlib import Path

import cv2
import numpy as np
import pytest

from dongfeng_registration.camera_models import CameraCalibration
from dongfeng_registration.config import MatchingConfig
from dongfeng_registration.dataset import Dataset
from dongfeng_registration.evaluation import summarize_errors
from dongfeng_registration.features import match_frame_pair
from dongfeng_registration.inspection import inspect_dataset
from dongfeng_registration.tracks import build_tracks
from tests.conftest import real_t3_data_available


DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)

requires_real_t3_data = pytest.mark.skipif(
    not real_t3_data_available(DATA_ROOT),
    reason=f"optional real T3 dataset not available: {DATA_ROOT}",
)


@requires_real_t3_data
def test_real_pinhole_pair_produces_lk_verified_matches():
    dataset = Dataset(DATA_ROOT)
    calibration = CameraCalibration.from_dataset(DATA_ROOT, "center_camera_fov30")
    frames = dataset.frames("center_camera_fov30")
    first = cv2.imread(str(frames[285].path))
    second = cv2.imread(str(frames[286].path))
    matches = match_frame_pair(
        first,
        second,
        calibration,
        MatchingConfig(max_features=1000, max_image_dimension=1200),
    )
    assert len(matches) > 20
    assert matches["lk_valid"].all()
    assert {"x1", "y1", "x2", "y2", "ransac_inlier", "ratio", "lk_fb_error"} <= set(
        matches.columns
    )


@requires_real_t3_data
def test_real_fisheye_pair_produces_finite_normalized_coordinates():
    dataset = Dataset(DATA_ROOT)
    calibration = CameraCalibration.from_dataset(DATA_ROOT, "front_camera_fov195")
    frames = dataset.frames("front_camera_fov195")
    first = cv2.imread(str(frames[285].path))
    second = cv2.imread(str(frames[286].path))
    matches = match_frame_pair(
        first,
        second,
        calibration,
        MatchingConfig(max_features=1000, max_image_dimension=1200),
    )
    assert len(matches) > 20
    assert np.isfinite(matches[["nx1", "ny1", "nx2", "ny2"]].to_numpy()).all()


def test_tracks_link_shared_observations_and_reject_conflicts():
    import pandas as pd

    matches = pd.DataFrame(
        [
            {"frame1": 0, "frame2": 1, "kp1": 10, "kp2": 20, "x1": 1, "y1": 2, "x2": 3, "y2": 4},
            {"frame1": 1, "frame2": 2, "kp1": 20, "kp2": 30, "x1": 3, "y1": 4, "x2": 5, "y2": 6},
        ]
    )
    observations = build_tracks(matches, "camera")
    assert observations["track_id"].nunique() == 1
    assert len(observations) == 3


def test_error_summary_contains_required_statistics():
    summary = summarize_errors(np.array([0.5, 1.5, 3.0, 12.0]))
    assert summary["count"] == 4
    assert summary["median"] == 2.25
    assert summary["over_1px_ratio"] == 0.75
    assert {"mean", "rmse", "p90", "p95", "max", "over_10px_ratio"} <= summary.keys()


@requires_real_t3_data
def test_inspection_checks_all_eleven_real_cameras():
    report = inspect_dataset(DATA_ROOT)
    assert len(report) == 11
    assert report["frame_count"].eq(600).all()
    assert report["image_size_matches"].all()
    assert report["calibration_valid"].all()
