from pathlib import Path

import cv2
import numpy as np
import pytest

from dongfeng_registration.camera_models import CameraCalibration
from dongfeng_registration.config import MatchingConfig
from dongfeng_registration.dataset import Dataset
from dongfeng_registration.features import match_frame_pair
from dongfeng_registration.match_backends import run_match_backend


DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)
MASK_ROOT = Path(
    "data_cache/semantic_masks/backends/mask2former"
    "/mask2former-swin-large-mapillary-vistas-semantic-af445a831fc47800"
    "/center_camera_fov30/masks"
)


def _dummy_calibration(image: np.ndarray) -> CameraCalibration:
    height, width = image.shape[:2]
    kwargs = {
        "camera": "test",
        "model": "rational",
        "image_size": (width, height),
        "intrinsic": np.eye(3, dtype=np.float64),
        "distortion": np.zeros(8, dtype=np.float64),
        "sensor_to_car": np.eye(4, dtype=np.float64),
        "time_lag": 0.0,
    }
    if "images_undistorted" in getattr(CameraCalibration, "__dataclass_fields__", {}):
        kwargs["images_undistorted"] = False
    return CameraCalibration(
        **kwargs,
    )


def _load_real_pair_or_skip():
    if not DATA_ROOT.exists():
        pytest.skip(f"dataset root not available: {DATA_ROOT}")
    if not MASK_ROOT.exists():
        pytest.skip(f"semantic mask root not available: {MASK_ROOT}")

    frames = Dataset(DATA_ROOT).frames("center_camera_fov30")
    if len(frames) <= 286:
        pytest.skip("not enough frames for semantic matching test pair")

    first, second = frames[285], frames[286]
    first_mask_path = MASK_ROOT / f"{first.timestamp_ns}.png"
    second_mask_path = MASK_ROOT / f"{second.timestamp_ns}.png"
    required_paths = [first.path, second.path, first_mask_path, second_mask_path]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        pytest.skip(f"required real-data assets missing: {missing}")

    first_image = cv2.imread(str(first.path))
    second_image = cv2.imread(str(second.path))
    first_mask = cv2.imread(str(first_mask_path), 0)
    second_mask = cv2.imread(str(second_mask_path), 0)
    if any(item is None for item in (first_image, second_image, first_mask, second_mask)):
        pytest.skip("required real-data images or masks could not be read")

    calibration = CameraCalibration.from_dataset(DATA_ROOT, "center_camera_fov30")
    return first_image, second_image, first_mask, second_mask, calibration


def _assert_mask_rejection_with_boundary_tolerance(
    matches,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> None:
    first_xy = np.rint(matches[["x1", "y1"]].to_numpy()).astype(int)
    second_xy = np.rint(matches[["x2", "y2"]].to_numpy()).astype(int)
    first_hits = int(np.count_nonzero(first_mask[first_xy[:, 1], first_xy[:, 0]]))
    second_hits = int(np.count_nonzero(second_mask[second_xy[:, 1], second_xy[:, 0]]))
    tolerance = max(3, int(np.ceil(len(matches) * 0.01)))
    assert first_hits <= tolerance
    assert second_hits <= tolerance


def _textured_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    image = np.full((160, 200, 3), 128, dtype=np.uint8)
    for _ in range(40):
        x = int(rng.integers(10, 190))
        y = int(rng.integers(10, 150))
        color = int(rng.integers(180, 255))
        cv2.circle(image, (x, y), int(rng.integers(2, 6)), (color, color, color), -1)
    for _ in range(20):
        x = int(rng.integers(10, 180))
        y = int(rng.integers(10, 140))
        width = int(rng.integers(4, 10))
        height = int(rng.integers(4, 10))
        cv2.rectangle(image, (x, y), (x + width, y + height), (20, 20, 20), -1)
    shifted = np.roll(image, 3, axis=1)
    return image, shifted


def _match_synthetic_pair(
    *,
    min_texture_strength: float = 0.0,
    min_corner_response: float = 0.0,
):
    image, shifted = _textured_pair()
    calibration = _dummy_calibration(image)
    return match_frame_pair(
        image,
        shifted,
        calibration,
        MatchingConfig(
            max_features=300,
            max_image_dimension=160,
            min_texture_strength=min_texture_strength,
            min_corner_response=min_corner_response,
        ),
    )


def test_semantic_matching_rejects_dynamic_source_and_destination_points():
    first_image, second_image, first_mask, second_mask, calibration = _load_real_pair_or_skip()

    matches = match_frame_pair(
        first_image,
        second_image,
        calibration,
        MatchingConfig(max_features=1500, max_image_dimension=1200),
        first_dynamic_mask=first_mask,
        second_dynamic_mask=second_mask,
    )

    assert len(matches) > 20
    _assert_mask_rejection_with_boundary_tolerance(matches, first_mask, second_mask)
    assert matches.attrs["semantic_destination_rejected"] > 0


def test_run_match_backend_dispatches_sift_lk_and_sets_backend_name():
    first_image, second_image, first_mask, second_mask, calibration = _load_real_pair_or_skip()

    matches = run_match_backend(
        first_image,
        second_image,
        calibration,
        MatchingConfig(
            backend="sift_lk",
            max_features=1500,
            max_image_dimension=1200,
        ),
        first_semantic_mask=first_mask,
        second_semantic_mask=second_mask,
    )

    assert len(matches) > 20
    assert matches["backend"].eq("sift_lk").all()
    assert matches["backend_score"].isna().all()


def test_run_match_backend_preserves_semantic_filtering():
    first_image, second_image, first_mask, second_mask, calibration = _load_real_pair_or_skip()

    matches = run_match_backend(
        first_image,
        second_image,
        calibration,
        MatchingConfig(
            backend="sift_lk",
            max_features=1500,
            max_image_dimension=1200,
        ),
        first_semantic_mask=first_mask,
        second_semantic_mask=second_mask,
    )

    assert len(matches) > 20
    _assert_mask_rejection_with_boundary_tolerance(matches, first_mask, second_mask)
    assert matches.attrs["semantic_destination_rejected"] > 0


@pytest.mark.parametrize("failure_stage", ["forward", "backward"])
def test_match_frame_pair_returns_empty_when_lk_returns_none(monkeypatch, failure_stage):
    image, shifted = _textured_pair()
    calibration = _dummy_calibration(image)
    calls = {"count": 0}
    original = cv2.calcOpticalFlowPyrLK

    def fake_calc_optical_flow(*args, **kwargs):
        calls["count"] += 1
        if failure_stage == "forward" and calls["count"] == 1:
            return None, None, None
        if failure_stage == "backward" and calls["count"] == 2:
            return None, None, None
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_calc_optical_flow)

    matches = match_frame_pair(
        image,
        shifted,
        calibration,
        MatchingConfig(
            max_features=300,
            max_image_dimension=200,
            min_texture_strength=0.0,
            min_corner_response=0.0,
        ),
    )

    assert matches.empty
    assert matches.attrs["pre_lk_matches"] > 0
    assert matches.attrs["lk_failure_count"] == 1


def test_texture_gate_rejects_low_texture_candidates_independently():
    loose = _match_synthetic_pair(
        min_texture_strength=0.0,
        min_corner_response=0.0,
    )
    texture_strict = _match_synthetic_pair(
        min_texture_strength=100.0,
        min_corner_response=0.0,
    )

    assert texture_strict.attrs["pre_lk_matches"] < loose.attrs["pre_lk_matches"]
    assert texture_strict.attrs["texture_corner_rejected"] > loose.attrs["texture_corner_rejected"]


def test_corner_gate_rejects_low_corner_candidates_independently():
    loose = _match_synthetic_pair(
        min_texture_strength=0.0,
        min_corner_response=0.0,
    )
    corner_strict = _match_synthetic_pair(
        min_texture_strength=0.0,
        min_corner_response=1e-6,
    )

    assert corner_strict.attrs["pre_lk_matches"] < loose.attrs["pre_lk_matches"]
    assert corner_strict.attrs["texture_corner_rejected"] > loose.attrs["texture_corner_rejected"]
