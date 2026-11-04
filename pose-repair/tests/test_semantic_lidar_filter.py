from __future__ import annotations

import numpy as np

from dongfeng_registration.semantic_lidar_filter import (
    SemanticClassMask,
    dynamic_point_mask_from_semantic_views,
)


def _ego_to_camera() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_dynamic_point_mask_removes_points_in_dynamic_class_pixels():
    class_mask = np.zeros((8, 8), dtype=np.uint16)
    class_mask[2, 3] = 7
    semantic_view = SemanticClassMask(
        labels=class_mask,
        dynamic_label_ids=frozenset({7}),
    )
    points_ego = np.asarray(
        [
            [3.0, 2.0, 1.0],  # (u, v) = (3, 2), dynamic.
            [1.0, 1.0, 1.0],  # static.
        ],
        dtype=np.float32,
    )

    dynamic, details = dynamic_point_mask_from_semantic_views(
        points_ego,
        camera_views={
            "front": (
                _ego_to_camera(),
                np.eye(3, dtype=np.float64),
                semantic_view,
            )
        },
    )

    np.testing.assert_array_equal(dynamic, np.asarray([True, False]))
    assert details["front"]["projected_points"] == 2
    assert details["front"]["dynamic_points"] == 1


def test_dynamic_point_mask_keeps_points_outside_all_semantic_views():
    class_mask = np.full((4, 4), 3, dtype=np.uint16)
    semantic_view = SemanticClassMask(
        labels=class_mask,
        dynamic_label_ids=frozenset({3}),
    )
    points_ego = np.asarray(
        [
            [10.0, 10.0, 1.0],  # outside image.
            [0.0, 0.0, -1.0],  # behind camera.
        ],
        dtype=np.float32,
    )

    dynamic, details = dynamic_point_mask_from_semantic_views(
        points_ego,
        camera_views={
            "front": (
                _ego_to_camera(),
                np.eye(3, dtype=np.float64),
                semantic_view,
            )
        },
    )

    np.testing.assert_array_equal(dynamic, np.asarray([False, False]))
    assert details["front"]["projected_points"] == 0
    assert details["front"]["dynamic_points"] == 0
