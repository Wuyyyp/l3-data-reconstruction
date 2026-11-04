from pathlib import Path
import pickle

import numpy as np
import pytest

from dongfeng_registration.pkl_pose import (
    calibration_from_result_interpolation,
    frame_metadata_from_image_root,
    frame_metadata_from_result_interpolation,
    frames_from_image_root,
    navigation_from_result_interpolation,
    resolve_result_interpolation_path,
)


def _write_sample_pkl(path: Path) -> None:
    infos = []
    for index in range(2):
        timestamp_ns = 1_000_000_000 + index * 100_000_000
        ego = np.eye(4, dtype=float)
        ego[0, 3] = float(index)
        infos.append(
            {
                "timestamp": timestamp_ns,
                "ego2global_transformation_matrix": ego.tolist(),
                "sensors": {
                    "cams": {
                        "center_camera_fov30": {
                            "timestamp": timestamp_ns // 1000 + 12,
                            "cam_intrinsic": [
                                [100.0, 0.0, 320.0],
                                [0.0, 110.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "d": [0.1, -0.2, 0.001, 0.002, 0.3, 0.4, 0.5, 0.6],
                            "extrinsic": [
                                [1.0, 0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0, 2.0],
                                [0.0, 0.0, 1.0, 3.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        }
                    }
                },
            }
        )
    with path.open("wb") as handle:
        pickle.dump({"infos": infos}, handle)


def test_resolve_result_interpolation_path_uses_dataset_relative_value(tmp_path: Path):
    config_root = tmp_path / "compat_dataset"
    config_root.mkdir()
    expected = config_root / "result_interpolation.pkl"

    resolved = resolve_result_interpolation_path(
        {
            "pose_backend": "result_interpolation_pkl",
            "result_interpolation_pkl_path": "result_interpolation.pkl",
        },
        config_root,
    )

    assert resolved == expected


def test_result_interpolation_builds_navigation_calibration_and_frame_metadata(
    tmp_path: Path,
):
    pkl_path = tmp_path / "result_interpolation.pkl"
    image_root = tmp_path / "camera_distorted"
    _write_sample_pkl(pkl_path)

    navigation = navigation_from_result_interpolation(pkl_path)
    assert navigation.timestamps_ns.tolist() == [1_000_000_000, 1_100_000_000]
    assert navigation.pose_at(1_050_000_000)[:3, 3] == pytest.approx([0.5, 0.0, 0.0])

    calibration = calibration_from_result_interpolation(
        pkl_path,
        "front_camera_fov30",
    )
    assert calibration.camera == "front_camera_fov30"
    assert calibration.model == "rational"
    assert calibration.image_size == (640, 480)
    assert calibration.intrinsic[0, 0] == pytest.approx(100.0)
    assert calibration.distortion.shape == (8,)
    np.testing.assert_allclose(calibration.sensor_to_car[:3, 3], [1.0, 2.0, 3.0])
    assert calibration.images_undistorted is False

    metadata = frame_metadata_from_result_interpolation(
        pkl_path,
        ("front_camera_fov30",),
        image_root=image_root,
    )
    assert metadata["front_camera_fov30"][0]["timestamp_ns"] == 1_000_012_000
    assert metadata["front_camera_fov30"][0]["image_filename"] == "1000012.jpg"
    assert metadata["front_camera_fov30"][0]["image_path"] == str(
        image_root / "front_camera_fov30" / "1000012.jpg"
    )


def test_result_interpolation_uses_canonical_camera_name_when_present(tmp_path: Path):
    pkl_path = tmp_path / "result_interpolation_standard.pkl"
    _write_sample_pkl(pkl_path)
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    for info in payload["infos"]:
        cameras = info["sensors"]["cams"]
        cameras["front_camera_fov30"] = cameras.pop("center_camera_fov30")
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    calibration = calibration_from_result_interpolation(pkl_path, "front_camera_fov30")
    metadata = frame_metadata_from_result_interpolation(
        pkl_path,
        ("front_camera_fov30",),
    )

    assert calibration.camera == "front_camera_fov30"
    assert metadata["front_camera_fov30"][0]["image_filename"] == "1000012.jpg"


def test_result_interpolation_navigation_can_use_custom_ego_field(tmp_path: Path):
    pkl_path = tmp_path / "result_interpolation.pkl"
    _write_sample_pkl(pkl_path)
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    for index, info in enumerate(payload["infos"]):
        offset_ego = np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64)
        offset_ego = offset_ego.copy()
        offset_ego[1, 3] = 10.0 + float(index)
        info["ego2global_transformation_matrix_camera_reference_offset"] = (
            offset_ego.tolist()
        )
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    navigation = navigation_from_result_interpolation(
        pkl_path,
        ego_field="ego2global_transformation_matrix_camera_reference_offset",
    )

    assert navigation.pose_at(1_050_000_000)[:3, 3] == pytest.approx(
        [0.5, 10.5, 0.0]
    )


def test_result_interpolation_calibration_can_invert_car_to_camera_extrinsic(
    tmp_path: Path,
):
    pkl_path = tmp_path / "result_interpolation.pkl"
    _write_sample_pkl(pkl_path)

    calibration = calibration_from_result_interpolation(
        pkl_path,
        "front_camera_fov30",
        extrinsic_direction="car_to_camera",
    )

    expected = np.linalg.inv(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_allclose(calibration.sensor_to_car, expected)


def test_result_interpolation_calibration_can_mark_images_undistorted(
    tmp_path: Path,
):
    pkl_path = tmp_path / "result_interpolation.pkl"
    _write_sample_pkl(pkl_path)

    calibration = calibration_from_result_interpolation(
        pkl_path,
        "front_camera_fov30",
        images_undistorted=True,
    )

    assert calibration.images_undistorted is True


def test_image_root_metadata_uses_timestamp_named_images_without_compat_dataset(
    tmp_path: Path,
):
    image_root = tmp_path / "camera_distorted"
    camera_dir = image_root / "front_camera_fov30"
    camera_dir.mkdir(parents=True)
    (camera_dir / "1753002123700312.jpg").write_bytes(b"fake")
    (camera_dir / "1753002123750318.jpg").write_bytes(b"fake")

    frames = frames_from_image_root(image_root, "front_camera_fov30")
    metadata = frame_metadata_from_image_root(image_root, ("front_camera_fov30",))

    assert [frame.index for frame in frames] == [0, 1]
    assert [frame.timestamp_ns for frame in frames] == [
        1753002123700312000,
        1753002123750318000,
    ]
    assert frames[0].path == camera_dir / "1753002123700312.jpg"
    assert metadata["front_camera_fov30"][1]["image_filename"] == (
        "1753002123750318.jpg"
    )
    assert metadata["front_camera_fov30"][1]["image_path"] == str(
        camera_dir / "1753002123750318.jpg"
    )
