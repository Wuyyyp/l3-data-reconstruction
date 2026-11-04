from pathlib import Path

import numpy as np
import pytest

from dongfeng_registration.camera_models import CameraCalibration, undistort_normalized
from dongfeng_registration.config import (
    ImageCacheConfig,
    MatchingConfig,
    PipelineConfig,
    SemanticSegmentationConfig,
    load_config,
)
from dongfeng_registration.dataset import Dataset
from dongfeng_registration.geometry import (
    epipolar_errors,
    essential_from_relative_pose,
    relative_pose,
)
from dongfeng_registration.navigation import NavigationTrajectory
from tests.conftest import real_t3_data_available


DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)

requires_real_t3_data = pytest.mark.skipif(
    not real_t3_data_available(DATA_ROOT),
    reason=f"optional real T3 dataset not available: {DATA_ROOT}",
)


def test_load_config_resolves_dataset_path():
    config = load_config(Path("configs/t3.yaml"))
    assert config.dataset_root.is_absolute()
    assert config.frame_offsets == [1, 2, 5]
    assert config.matching.backend == "loma"
    assert config.matching.deep_batch_size == 4
    assert config.cameras == [
        "center_camera_fov30",
        "center_camera_fov120",
        "left_front_camera",
        "left_rear_camera",
        "rear_camera",
        "right_front_camera",
        "right_rear_camera",
    ]


def test_t3_config_includes_current_mainline_temporal_overlap_pairs():
    config = load_config(Path("configs/t3.yaml"))
    assumptions = config.assumptions

    assert assumptions["inter_camera_temporal_overlap_allowed_pairs"] == [
        "center_camera_fov120__center_camera_fov30",
        "center_camera_fov120__left_front_camera",
        "center_camera_fov120__right_front_camera",
        "center_camera_fov30__left_front_camera",
        "center_camera_fov30__right_front_camera",
        "rear_camera__left_rear_camera",
        "rear_camera__right_rear_camera",
        "left_front_camera__left_rear_camera",
        "right_front_camera__right_rear_camera",
    ]
    assert assumptions["inter_camera_temporal_overlap_target_baselines_by_pair"][
        "center_camera_fov30__left_front_camera"
    ] == [12.0]
    assert assumptions["inter_camera_temporal_overlap_target_baselines_by_pair"][
        "center_camera_fov30__right_front_camera"
    ] == [12.0]
    assert assumptions["inter_camera_temporal_overlap_target_baselines_by_pair"][
        "left_front_camera__left_rear_camera"
    ] == [8.0]
    assert assumptions["inter_camera_temporal_overlap_target_baselines_by_pair"][
        "right_front_camera__right_rear_camera"
    ] == [8.0]
    assert config.matching.deep_score_threshold_inter == 0.1
    assert "inter_camera_deep_score_threshold_by_pair" not in assumptions


def test_image_cache_resolution_covers_semantic_and_loma_requirements():
    with pytest.raises(ValueError, match="image cache resolution"):
        PipelineConfig(
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            cameras=["camera"],
            matching=MatchingConfig(deep_image_size=1600),
            semantic_segmentation=SemanticSegmentationConfig(
                processor_image_size=1024
            ),
            image_cache=ImageCacheConfig(max_long_side=1200),
        )


def test_semantic_segmentation_defaults_use_fast_fp16_batch_four():
    config = SemanticSegmentationConfig()

    assert config.inference_dtype == "fp16"
    assert config.use_fast_processor is True
    assert config.batch_size == 4


@pytest.mark.parametrize(
    ("camera", "model", "distortion_count", "size"),
    [
        ("center_camera_fov30", "rational", 8, (3840, 2160)),
        ("front_camera_fov195", "fisheye", 4, (1920, 1536)),
    ],
)
@requires_real_t3_data
def test_parse_real_calibration(camera, model, distortion_count, size):
    calibration = CameraCalibration.from_dataset(DATA_ROOT, camera)
    assert calibration.model == model
    assert calibration.distortion.shape == (distortion_count,)
    assert calibration.image_size == size
    assert np.linalg.det(calibration.sensor_to_car[:3, :3]) == pytest.approx(1, abs=2e-5)


@pytest.mark.parametrize("camera", ["center_camera_fov30", "front_camera_fov195"])
@requires_real_t3_data
def test_principal_point_undistorts_to_origin(camera):
    calibration = CameraCalibration.from_dataset(DATA_ROOT, camera)
    principal_point = calibration.intrinsic[[0, 1], [2, 2]][None, :]
    normalized = undistort_normalized(principal_point, calibration)
    assert normalized == pytest.approx([[0.0, 0.0]], abs=2e-3)


@requires_real_t3_data
def test_dataset_reads_all_camera_frames_and_timestamps():
    dataset = Dataset(DATA_ROOT)
    frames = dataset.frames("center_camera_fov30")
    assert len(frames) == 600
    assert frames[0].path.name == f"{frames[0].timestamp_ns}.jpg"
    assert frames[1].timestamp_ns > frames[0].timestamp_ns


@requires_real_t3_data
def test_navigation_interpolates_position_and_rejects_extrapolation():
    trajectory = NavigationTrajectory.from_csv(DATA_ROOT / "ved/LocalLocalization.csv")
    middle = (trajectory.timestamps_ns[10] + trajectory.timestamps_ns[11]) // 2
    pose = trajectory.pose_at(middle)
    expected = (trajectory.positions[10] + trajectory.positions[11]) / 2
    assert pose[:3, 3] == pytest.approx(expected)
    with pytest.raises(ValueError, match="outside navigation range"):
        trajectory.pose_at(int(trajectory.timestamps_ns[0] - 1))


def test_relative_pose_and_essential_matrix():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 2.0
    transform = relative_pose(first, second)
    essential = essential_from_relative_pose(transform)
    assert transform[:3, 3] == pytest.approx([2.0, 0.0, 0.0])
    np.testing.assert_allclose(
        essential,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]]),
    )


def test_epipolar_errors_are_zero_for_consistent_points_and_grow_when_perturbed():
    essential = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    first = np.array([[0.1, 0.2], [-0.2, 0.3]])
    second = np.array([[0.15, 0.2], [-0.1, 0.3]])
    ideal = epipolar_errors(first, second, essential)
    perturbed = epipolar_errors(first, second + np.array([0.0, 0.05]), essential)
    assert np.max(ideal["sampson"]) < 1e-12
    assert np.mean(perturbed["sampson"]) > np.mean(ideal["sampson"])


def test_epipolar_errors_accept_object_dtype_inputs():
    essential = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=object,
    )
    first = np.array([[0.1, 0.2], [-0.2, 0.3]], dtype=object)
    second = np.array([[0.15, 0.2], [-0.1, 0.3]], dtype=object)

    errors = epipolar_errors(first, second, essential)

    assert np.max(errors["sampson"]) < 1e-12

