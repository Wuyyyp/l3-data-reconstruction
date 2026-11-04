from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dongfeng_registration.result_interpolation_export import (
    apply_optimized_transforms_to_payload,
    delta_transform,
)


def test_apply_optimized_transforms_updates_ego_camera_and_sensor2lidar():
    payload = {
        "metadata": {"version": "test"},
        "infos": [
            {
                "timestamp": 999,
                "ego2global_transformation_matrix": np.eye(4).tolist(),
                "sensors": {
                    "lidar": {
                        "top_center_lidar": {
                            "extrinsic": np.eye(4).tolist(),
                        }
                    },
                    "cams": {
                        "center_camera_fov30": {
                            "timestamp": 1,
                            "extrinsic": np.eye(4).tolist(),
                            "ego2sensor_translation": np.zeros(3, dtype=np.float32),
                            "ego2sensor_rotation": [1.0, 0.0, 0.0, 0.0],
                            "sensor2lidar_rotation": np.eye(3, dtype=np.float32),
                            "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
                        }
                    },
                },
            }
        ],
    }
    optimized_sensor_to_car = {
        "front_camera_fov30": np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    }
    camera_frame_disturbances = {
        ("front_camera_fov30", 2): np.array([0.0, 0.0, np.pi / 2.0, 0.0, 2.0, 0.0])
    }
    body_frame_disturbances = {
        2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=np.float64)
    }

    report = apply_optimized_transforms_to_payload(
        payload,
        optimized_sensor_to_car=optimized_sensor_to_car,
        camera_frame_disturbances=camera_frame_disturbances,
        body_frame_disturbances=body_frame_disturbances,
        timestamp_to_frame={1000: 2},
    )

    info = payload["infos"][0]
    expected_ego = delta_transform(body_frame_disturbances[2])
    expected_sensor_to_ego = optimized_sensor_to_car["front_camera_fov30"] @ delta_transform(
        camera_frame_disturbances[("front_camera_fov30", 2)]
    )
    expected_ego_to_sensor = np.linalg.inv(expected_sensor_to_ego)

    assert report["infos_updated"] == 1
    assert report["cameras_updated"] == 1
    np.testing.assert_allclose(
        np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64),
        expected_ego,
    )
    np.testing.assert_allclose(
        np.asarray(info["sensors"]["cams"]["center_camera_fov30"]["extrinsic"], dtype=np.float64),
        expected_ego_to_sensor,
    )
    np.testing.assert_allclose(
        info["sensors"]["cams"]["center_camera_fov30"]["ego2sensor_translation"],
        expected_ego_to_sensor[:3, 3].astype(np.float32),
    )
    expected_quat_wxyz = np.roll(
        Rotation.from_matrix(expected_ego_to_sensor[:3, :3]).as_quat(),
        1,
    )
    np.testing.assert_allclose(
        info["sensors"]["cams"]["center_camera_fov30"]["ego2sensor_rotation"],
        expected_quat_wxyz,
    )
    np.testing.assert_allclose(
        info["sensors"]["cams"]["center_camera_fov30"]["sensor2lidar_rotation"],
        expected_sensor_to_ego[:3, :3].astype(np.float32),
    )
    np.testing.assert_allclose(
        info["sensors"]["cams"]["center_camera_fov30"]["sensor2lidar_translation"],
        expected_sensor_to_ego[:3, 3].astype(np.float32),
    )


def test_apply_optimized_transforms_can_write_custom_optimized_ego_field():
    original_ego = np.eye(4, dtype=np.float64)
    offset_ego = np.eye(4, dtype=np.float64)
    offset_ego[0, 3] = 10.0
    payload = {
        "infos": [
            {
                "timestamp": 999,
                "ego2global_transformation_matrix": original_ego.tolist(),
                "ego2global_transformation_matrix_camera_reference_offset": offset_ego.tolist(),
                "sensors": {"cams": {}},
            }
        ],
    }
    body_frame_disturbances = {
        2: np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0], dtype=np.float64)
    }

    report = apply_optimized_transforms_to_payload(
        payload,
        optimized_sensor_to_car={},
        camera_frame_disturbances={},
        body_frame_disturbances=body_frame_disturbances,
        timestamp_to_frame={999: 2},
        source_ego_field="ego2global_transformation_matrix_camera_reference_offset",
        optimized_ego_output_field=(
            "ego2global_transformation_matrix_camera_reference_offset_optimized"
        ),
    )

    info = payload["infos"][0]
    expected = offset_ego @ delta_transform(body_frame_disturbances[2])
    assert report["infos_updated"] == 1
    np.testing.assert_allclose(
        np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64),
        original_ego,
    )
    np.testing.assert_allclose(
        np.asarray(
            info["ego2global_transformation_matrix_camera_reference_offset"],
            dtype=np.float64,
        ),
        offset_ego,
    )
    np.testing.assert_allclose(
        np.asarray(
            info["ego2global_transformation_matrix_camera_reference_offset_optimized"],
            dtype=np.float64,
        ),
        expected,
    )


def test_apply_optimized_transforms_copies_custom_ego_field_when_frame_is_missing():
    offset_ego = np.eye(4, dtype=np.float64)
    offset_ego[0, 3] = 4.0
    payload = {
        "infos": [
            {
                "timestamp": 999,
                "ego2global_transformation_matrix_camera_reference_offset": offset_ego.tolist(),
                "sensors": {"cams": {}},
            }
        ],
    }

    report = apply_optimized_transforms_to_payload(
        payload,
        optimized_sensor_to_car={},
        camera_frame_disturbances={},
        body_frame_disturbances={},
        timestamp_to_frame={},
        source_ego_field="ego2global_transformation_matrix_camera_reference_offset",
        optimized_ego_output_field=(
            "ego2global_transformation_matrix_camera_reference_offset_optimized"
        ),
    )

    assert report["infos_updated"] == 0
    assert report["infos_without_frame"] == 1
    np.testing.assert_allclose(
        np.asarray(
            payload["infos"][0][
                "ego2global_transformation_matrix_camera_reference_offset_optimized"
            ],
            dtype=np.float64,
        ),
        offset_ego,
    )
