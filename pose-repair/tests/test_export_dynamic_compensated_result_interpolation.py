from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_dynamic_compensated_result_interpolation_pkl import (  # noqa: E402
    interpolate_ego_from_precomputed,
    box_center_from_corners,
    update_info_for_dynamic_compensation,
)


def _corners(center=(1.0, 2.0, 3.0), length=4.0, width=2.0, height=1.5):
    cx, cy, cz = center
    dx, dy, dz = length / 2, width / 2, height / 2
    return [
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ]


def test_box_center_from_corners_uses_corner_mean():
    assert box_center_from_corners(_corners(center=(3.0, 4.0, 5.0))) == [3.0, 4.0, 5.0]


def test_precomputed_ego_interpolation_matches_expected_midpoint():
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = 10.0

    interpolated = interpolate_ego_from_precomputed(
        np.asarray([10.0, 20.0]),
        np.stack([first, second]),
        15.0,
    )

    assert interpolated is not None
    np.testing.assert_allclose(interpolated[:3, 3], [5.0, 0.0, 0.0])


def test_update_info_keeps_original_timestamp_and_ego_and_adds_offset_ego():
    original_ego = np.eye(4).tolist()
    offset_ego = np.eye(4)
    offset_ego[0, 3] = 7.0
    info = {
        "timestamp": 123,
        "ego2global_transformation_matrix": original_ego,
        "sensors": {
            "lidar": {
                "perception": {
                    "timestamp": 123,
                    "data_path": "perception/original.pcd",
                    "aws_path": "perception/original.pcd",
                }
            }
        },
        "objects": [
            {
                "id": 42,
                "type": "Car",
                "location": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0],
                "velocity": [1.0, 0.0, 0.0],
            },
            {"id": 99, "type": "Trafficcone", "location": [9.0, 9.0, 9.0]},
        ],
    }
    before = copy.deepcopy(info)
    box_payload = {
        "reference_time_s": 1000.125,
        "reference_time_offset_s": 0.007,
        "after": [
            {
                "object_id": "42",
                "corners": _corners(center=(3.0, 4.0, 5.0), length=4.0, width=2.0, height=1.5),
                "length": 4.0,
                "width": 2.0,
                "height": 1.5,
                "yaw": 1.25,
                "status": "after_comp",
                "point_count": 1234,
                "timestamp_s": 1000.125,
            }
        ],
    }

    updated = update_info_for_dynamic_compensation(
        info,
        box_payload=box_payload,
        compensated_pcd_relative_path="compensated_perception/updated.pcd",
        offset_ego_to_world=offset_ego,
        offset_timestamp_ns=1_000_125_000_000,
        offset_field_name="ego2global_transformation_matrix_camera_reference_offset",
        offset_timestamp_field_name="camera_reference_offset_timestamp",
    )

    assert updated["timestamp"] == before["timestamp"]
    assert updated["ego2global_transformation_matrix"] == before["ego2global_transformation_matrix"]
    assert updated["camera_reference_offset_timestamp"] == 1_000_125_000_000
    assert updated["ego2global_transformation_matrix_camera_reference_offset"] == offset_ego.tolist()
    assert updated["sensors"]["lidar"]["perception"]["data_path"] == "compensated_perception/updated.pcd"
    assert updated["objects"][0]["location"] == [3.0, 4.0, 5.0]
    assert updated["objects"][0]["size"] == [4.0, 2.0, 1.5]
    assert updated["objects"][0]["rotation"] == [0.0, 0.0, 1.25]
    assert updated["objects"][0]["dynamic_compensation"]["point_count"] == 1234
    assert updated["objects"][1] == before["objects"][1]


def test_update_info_updates_matching_gt_box_from_compensated_box_payload():
    info = {
        "sensors": {"lidar": {"perception": {"data_path": "perception/original.pcd"}}},
        "gt_boxes": [
            [1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1],
            [9.0, 8.0, 7.0, 5.0, 3.0, 2.0, -0.2],
        ],
        "ID": [42, 99],
        "track_id": [42, 99],
    }
    box_payload = {
        "reference_time_s": 1000.125,
        "after": [
            {
                "object_id": "42",
                "corners": _corners(center=(3.0, 4.0, 5.0), length=6.0, width=2.5, height=2.0),
                "length": 6.0,
                "width": 2.5,
                "height": 2.0,
                "yaw": 1.25,
                "status": "after_comp",
            }
        ],
    }

    updated = update_info_for_dynamic_compensation(
        info,
        box_payload=box_payload,
        compensated_pcd_relative_path="compensated_perception/updated.pcd",
        offset_ego_to_world=np.eye(4),
        offset_timestamp_ns=1_000_125_000_000,
    )

    assert updated["gt_boxes"][0] == [3.0, 4.0, 5.0, 6.0, 2.5, 2.0, 1.25]
    assert updated["gt_boxes"][1] == info["gt_boxes"][1]
