from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from visualize_perception_projection_offset_sweep import (  # noqa: E402
    camera_time_with_offset_seconds,
    select_frame_indices,
    sample_points_for_visualization,
    points_in_camera_time_ego,
)


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def test_camera_time_with_offset_seconds_adds_offset_to_camera_timestamp():
    info = {"sensors": {"cams": {"front": {"timestamp": 1_000_000_000_000_000}}}}

    assert camera_time_with_offset_seconds(info, "front", 0.010) == pytest.approx(
        1_000_000_000.010
    )


def test_points_in_camera_time_ego_keeps_lidar_source_pose_fixed_and_moves_target_ego():
    points_source_ego = np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32)
    source_ego_to_world = _translation(0.0)
    target_ego_to_world = _translation(1.0)

    shifted = points_in_camera_time_ego(
        points_source_ego,
        source_ego_to_world=source_ego_to_world,
        target_ego_to_world=target_ego_to_world,
    )

    assert shifted == pytest.approx(np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32))


def test_sample_points_for_visualization_keeps_deterministic_ratio():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    sampled_once = sample_points_for_visualization(points, sample_ratio=0.2, seed=7)
    sampled_twice = sample_points_for_visualization(points, sample_ratio=0.2, seed=7)

    assert sampled_once.shape == (2, 3)
    np.testing.assert_array_equal(sampled_once, sampled_twice)


def test_select_frame_indices_samples_frames_deterministically_without_point_sampling():
    selected_once = select_frame_indices(start=0, end=10, stride=1, sample_ratio=0.2, seed=3)
    selected_twice = select_frame_indices(start=0, end=10, stride=1, sample_ratio=0.2, seed=3)

    assert len(selected_once) == 2
    assert selected_once == selected_twice
    assert all(0 <= index < 10 for index in selected_once)
