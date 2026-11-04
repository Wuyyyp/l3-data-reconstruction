from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from dongfeng_registration.cli import app
from dongfeng_registration.dataset_normalization import (
    _copy_file,
    normalize_ved_localization_dataset,
    normalize_result_interpolation_dataset,
)


runner = CliRunner()


def test_normalization_materializes_unchanged_files_as_hardlinks(tmp_path: Path):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "canonical" / "source.jpg"
    source.write_bytes(b"immutable-source")

    assert _copy_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino == source.stat().st_ino

    assert _copy_file(source, destination)
    assert destination.stat().st_ino == source.stat().st_ino


def test_normalization_file_materialization_falls_back_to_copy(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "canonical" / "source.jpg"
    source.write_bytes(b"copy-fallback")
    monkeypatch.setattr(
        os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    assert _copy_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def _write_pcd(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# .PCD v0.7 - Point Cloud Data file format",
                "VERSION 0.7",
                "FIELDS x y z timestamp",
                "SIZE 4 4 4 8",
                "TYPE F F F F",
                "COUNT 1 1 1 1",
                "WIDTH 1",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                "POINTS 1",
                "DATA ascii",
                "1 2 3 1000.0",
            ]
        ),
        encoding="utf-8",
    )


def _write_sample_result_interpolation_dataset(root: Path) -> Path:
    timestamp_ns = 1_753_002_123_700_000_000
    image_timestamp_us = timestamp_ns // 1000
    _write_image(root / "camera_distorted" / "center_camera_fov30" / f"{image_timestamp_us}.jpg")
    _write_image(root / "camera_distorted" / "left_front_camera" / f"{image_timestamp_us}.jpg")
    _write_pcd(root / "perception" / f"{timestamp_ns}.pcd")
    _write_pcd(root / "top_center_lidar" / f"{timestamp_ns}.pcd")
    _write_pcd(root / "top_center_lidar_compensation" / f"{timestamp_ns}.pcd")

    ego = np.eye(4, dtype=float)
    car_to_camera = np.eye(4, dtype=float)
    car_to_camera[0, 3] = 1.5
    camera_to_car = np.linalg.inv(car_to_camera)
    payload = {
        "infos": [
            {
                "timestamp": timestamp_ns,
                "ego2global_transformation_matrix": ego.tolist(),
                "sensors": {
                    "cams": {
                        "center_camera_fov30": {
                            "timestamp": image_timestamp_us,
                            "data_path": f"camera_distorted/center_camera_fov30/{image_timestamp_us}.jpg",
                            "cam_intrinsic": [
                                [100.0, 0.0, 320.0],
                                [0.0, 100.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "d": [0.0, 0.0, 0.0, 0.0],
                            "extrinsic": camera_to_car.tolist(),
                        },
                        "left_front_camera": {
                            "timestamp": image_timestamp_us,
                            "data_path": f"camera_distorted/left_front_camera/{image_timestamp_us}.jpg",
                            "cam_intrinsic": [
                                [100.0, 0.0, 320.0],
                                [0.0, 100.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "d": [0.0, 0.0, 0.0, 0.0],
                            "extrinsic": car_to_camera.tolist(),
                        },
                    },
                    "lidar": {
                        "perception": {
                            "timestamp": timestamp_ns,
                            "data_path": f"perception/{timestamp_ns}.pcd",
                        },
                        "top_center_lidar": {
                            "timestamp": timestamp_ns,
                            "data_path": f"top_center_lidar/{timestamp_ns}.pcd",
                            "extrinsic": np.eye(4, dtype=float).tolist(),
                        },
                    },
                },
                "gt_boxes": [
                    {
                        "object_id": "car_1",
                        "category": "car",
                        "location": [1.0, 2.0, 3.0],
                        "size": [4.0, 2.0, 1.5],
                        "rotation": [0.0, 0.0, 0.0],
                    }
                ],
            }
        ]
    }
    pkl_path = root / "result_interpolation.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return pkl_path


def _write_t3_camera_calibration(root: Path, camera: str, sensor_to_car: np.ndarray) -> None:
    calib_dir = root / "calib" / camera
    calib_dir.mkdir(parents=True, exist_ok=True)
    intrinsic = {
        "value0": {
            "param": {
                "img_dist_w": 640,
                "img_dist_h": 480,
                "cam_K_new": {
                    "data": [
                        [100.0, 0.0, 320.0],
                        [0.0, 100.0, 240.0],
                        [0.0, 0.0, 1.0],
                    ]
                },
                "cam_dist": {"data": [[0.0, 0.0, 0.0, 0.0]]},
            }
        }
    }
    extrinsic = {
        "value0": {
            "param": {
                "sensor_calib": {"data": sensor_to_car.tolist()},
                "time_lag": 0.0,
            }
        }
    }
    status = {"value0": {"param": {"undistorsion_result": 1}}}
    (calib_dir / f"{camera}-intrinsic.json").write_text(
        json.dumps(intrinsic),
        encoding="utf-8",
    )
    (calib_dir / f"{camera}-to-car_center-extrinsic.json").write_text(
        json.dumps(extrinsic),
        encoding="utf-8",
    )
    (calib_dir / f"{camera}-undistortion_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )


def _write_sample_t3_dataset(root: Path) -> None:
    timestamps_ns = [1_000_000_000, 1_100_000_000]
    alignment_rows = []
    for camera in ("center_camera_fov30", "left_front_camera"):
        camera_dir = root / "camera" / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for index, timestamp_ns in enumerate(timestamps_ns):
            lines.append(f"{index}, {timestamp_ns}")
            _write_image(camera_dir / f"{timestamp_ns}.jpg")
        (root / "camera" / f"{camera}.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        sensor_to_car = np.eye(4, dtype=float)
        sensor_to_car[0, 3] = 2.0 if camera == "center_camera_fov30" else 3.0
        _write_t3_camera_calibration(root, camera, sensor_to_car)

    lidar_dir = root / "lidar" / "front_lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    lidar_extrinsic_dir = root / "calib" / "front_lidar"
    lidar_extrinsic_dir.mkdir(parents=True, exist_ok=True)
    lidar_sensor_to_car = np.eye(4, dtype=float)
    lidar_sensor_to_car[0, 3] = 1.0
    (lidar_extrinsic_dir / "front_lidar-to-car_center-extrinsic.json").write_text(
        json.dumps(
            {
                "front_lidar-to-car_center-extrinsic": {
                    "param": {
                        "sensor_calib": {
                            "data": lidar_sensor_to_car.tolist(),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    for index, timestamp_ns in enumerate(timestamps_ns):
        _write_pcd(lidar_dir / f"{timestamp_ns}.pcd")
        alignment_rows.append(
            {
                str(timestamp_ns): {
                    "front_lidar": f"lidar/front_lidar/{timestamp_ns}.pcd",
                    "center_camera_fov30": f"camera/center_camera_fov30/{timestamp_ns}.jpg",
                    "left_front_camera": f"camera/left_front_camera/{timestamp_ns}.jpg",
                }
            }
        )
    (root / "sensor_temporal_alignment.json").write_text(
        json.dumps(alignment_rows),
        encoding="utf-8",
    )

    ved_dir = root / "ved"
    ved_dir.mkdir(parents=True, exist_ok=True)
    (ved_dir / "LocalLocalization.csv").write_text(
        "\n".join(
            [
                "timestamp.IsPlane,timestamp.sec,timestamp.nsec,local_pose.translation.x,local_pose.translation.y,local_pose.translation.z,local_pose.rotation.x,local_pose.rotation.y,local_pose.rotation.z,local_pose.rotation.w",
                "1,1,0,10.0,20.0,30.0,0.0,0.0,0.0,1.0",
                "1,1,100000000,11.0,20.0,30.0,0.0,0.0,0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    annotation_dir = root / "format_output" / "annotations" / "NV"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for index, timestamp_ns in enumerate(timestamps_ns):
        annotation = {
            "gt_boxes": [
                [1.0 + index, 2.0, 3.0, 2.2, 4.5, 1.6, 0.1],
            ],
            "gt_names": ["VEHICLE_CAR"],
            "track_id": [42],
            "num_lidar_pts": [17],
            "valid_flag": [True],
        }
        (annotation_dir / f"{timestamp_ns}.json").write_text(
            json.dumps(annotation),
            encoding="utf-8",
        )


def test_normalize_result_interpolation_dataset_writes_canonical_outputs(tmp_path: Path):
    source_root = tmp_path / "source"
    pkl_path = _write_sample_result_interpolation_dataset(source_root)
    output_root = tmp_path / "canonical"

    result = normalize_result_interpolation_dataset(
        dataset_root=source_root,
        pkl_path=pkl_path,
        output_root=output_root,
        vehicle_model="TEST",
        image_root=source_root / "camera_distorted",
        cameras=("front_camera_fov30", "left_front_camera"),
        camera_name_map={"front_camera_fov30": "center_camera_fov30"},
        extrinsic_direction="camera_to_car",
    )

    standard_pkl = output_root / "result_interpolation_standard.pkl"
    report_path = output_root / "metadata" / "normalization_report.json"
    frames_path = output_root / "metadata" / "frames.csv"
    cameras_path = output_root / "metadata" / "cameras.json"

    assert result.standard_pkl_path == standard_pkl
    assert standard_pkl.exists()
    assert report_path.exists()
    assert frames_path.exists()
    assert cameras_path.exists()
    assert (output_root / "images" / "front_camera_fov30" / "1753002123700000.jpg").exists()
    assert (output_root / "lidar" / "perception" / "1753002123700000000.pcd").exists()

    with standard_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    info = payload["infos"][0]
    cams = info["sensors"]["cams"]
    assert set(cams) == {"front_camera_fov30", "left_front_camera"}
    assert cams["front_camera_fov30"]["data_path"] == (
        "images/front_camera_fov30/1753002123700000.jpg"
    )
    np.testing.assert_allclose(
        np.asarray(cams["front_camera_fov30"]["extrinsic"], dtype=float)[:3, 3],
        [1.5, 0.0, 0.0],
    )
    assert info["sensors"]["lidar"]["perception"]["data_path"] == (
        "lidar/perception/1753002123700000000.pcd"
    )
    assert info["sensors"]["lidar"]["top_center_lidar"]["data_path"] == (
        "lidar/top_center_lidar_compensation/1753002123700000000.pcd"
    )
    assert (
        output_root
        / "lidar"
        / "top_center_lidar_compensation"
        / "1753002123700000000.pcd"
    ).exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["vehicle_model"] == "TEST"
    assert report["frame_count"] == 1
    assert report["camera_name_map"]["front_camera_fov30"] == "center_camera_fov30"
    assert report["missing_images"] == []
    assert report["missing_lidar_files"] == []
    assert report["extrinsic_output_direction"] == "car_to_camera"
    assert payload["metadata"]["standardized_source"]["pose_backend"] == (
        "result_interpolation_pkl"
    )


def test_normalize_result_interpolation_dataset_falls_back_to_target_image_folder_for_stale_absolute_paths(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    timestamp_ns = 1_753_002_123_700_000_000
    image_timestamp_us = timestamp_ns // 1000
    _write_image(
        source_root
        / "camera_distorted"
        / "front_camera_fov30"
        / f"{image_timestamp_us}.jpg"
    )
    pkl_path = source_root / "result_interpolation.pkl"
    payload = {
        "infos": [
            {
                "timestamp": timestamp_ns,
                "ego2global_transformation_matrix": np.eye(4, dtype=float).tolist(),
                "sensors": {
                    "cams": {
                        "center_camera_fov30": {
                            "timestamp": image_timestamp_us,
                            "data_path": (
                                "D:/stale/vendor/path/camera/front_camera_fov30/"
                                f"{image_timestamp_us}.jpg"
                            ),
                            "cam_intrinsic": np.eye(3, dtype=float).tolist(),
                            "extrinsic": np.eye(4, dtype=float).tolist(),
                        }
                    },
                    "lidar": {},
                },
            }
        ]
    }
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    output_root = tmp_path / "canonical"
    result = normalize_result_interpolation_dataset(
        dataset_root=source_root,
        pkl_path=pkl_path,
        output_root=output_root,
        vehicle_model="L29",
        image_root=source_root / "camera_distorted",
        cameras=("front_camera_fov30",),
        camera_name_map={"front_camera_fov30": "center_camera_fov30"},
    )

    assert (
        output_root
        / "images"
        / "front_camera_fov30"
        / f"{image_timestamp_us}.jpg"
    ).exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["missing_images"] == []


def test_normalize_dataset_cli_result_interpolation(tmp_path: Path):
    source_root = tmp_path / "source"
    pkl_path = _write_sample_result_interpolation_dataset(source_root)
    output_root = tmp_path / "canonical"

    cli_result = runner.invoke(
        app,
        [
            "normalize-dataset",
            "--pose-backend",
            "result_interpolation_pkl",
            "--dataset-root",
            str(source_root),
            "--pkl",
            str(pkl_path),
            "--output-root",
            str(output_root),
            "--vehicle-model",
            "TEST",
            "--image-root",
            str(source_root / "camera_distorted"),
            "--camera",
            "front_camera_fov30=center_camera_fov30",
            "--camera",
            "left_front_camera",
            "--extrinsic-direction",
            "camera_to_car",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert "result_interpolation_standard.pkl" in cli_result.output
    assert (output_root / "result_interpolation_standard.pkl").exists()
    assert (output_root / "metadata" / "normalization_report.json").exists()


def test_normalize_ved_localization_dataset_writes_result_interpolation_standard_pkl(
    tmp_path: Path,
):
    source_root = tmp_path / "t3_source"
    _write_sample_t3_dataset(source_root)
    output_root = tmp_path / "canonical_t3"

    result = normalize_ved_localization_dataset(
        dataset_root=source_root,
        output_root=output_root,
        vehicle_model="T3",
        cameras=("front_camera_fov30", "left_front_camera"),
        camera_name_map={"front_camera_fov30": "center_camera_fov30"},
        navigation_csv=source_root / "ved" / "LocalLocalization.csv",
    )

    assert result.standard_pkl_path.exists()
    assert (output_root / "images" / "front_camera_fov30" / "1000000.jpg").exists()
    assert (
        output_root / "lidar" / "front_lidar_compensation" / "1000000000.pcd"
    ).exists()
    assert (output_root / "metadata" / "normalization_report.json").exists()

    with result.standard_pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["metadata"]["pose_backend"] == "ved_localization"
    assert len(payload["infos"]) == 2
    first_info = payload["infos"][0]
    assert first_info["gt_boxes"] == [[1.0, 2.0, 3.0, 2.2, 4.5, 1.6, 0.1]]
    assert first_info["gt_names"] == ["VEHICLE_CAR"]
    assert first_info["ID"] == [42]
    assert first_info["track_id"] == [42]
    assert first_info["num_lidar_pts"] == [17]
    assert first_info["valid_flag"] == [True]
    assert payload["metadata"]["standardized_source"]["pose_backend"] == "ved_localization"
    np.testing.assert_allclose(
        np.asarray(first_info["ego2global_transformation_matrix"], dtype=float)[:3, 3],
        [10.0, 20.0, 30.0],
    )
    cams = first_info["sensors"]["cams"]
    assert set(cams) == {"front_camera_fov30", "left_front_camera"}
    assert cams["front_camera_fov30"]["timestamp"] == 1_000_000
    assert cams["front_camera_fov30"]["data_path"] == (
        "images/front_camera_fov30/1000000.jpg"
    )
    np.testing.assert_allclose(
        np.asarray(cams["front_camera_fov30"]["extrinsic"], dtype=float)[:3, 3],
        [-2.0, 0.0, 0.0],
    )
    lidars = first_info["sensors"]["lidar"]
    assert set(lidars) == {"front_lidar", "perception"}
    assert lidars["front_lidar"]["data_path"] == (
        "lidar/front_lidar_compensation/1000000000.pcd"
    )
    assert lidars["front_lidar"]["point_coordinate_frame"] == "ego"
    assert lidars["perception"]["point_coordinate_frame"] == "ego"
    np.testing.assert_allclose(
        np.asarray(lidars["front_lidar"]["extrinsic"], dtype=float)[:3, 3],
        [1.0, 0.0, 0.0],
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["pose_backend"] == "ved_localization"
    assert report["camera_name_map"]["front_camera_fov30"] == "center_camera_fov30"
    assert report["missing_images"] == []
    assert report["lidars"] == ["front_lidar", "perception"]
    assert report["annotation_source"] == "format_output/annotations/NV"
    assert report["temporal_alignment_source"] == "sensor_temporal_alignment.json"
    assert report["missing_annotations"] == []


def test_normalize_dataset_cli_ved_localization(tmp_path: Path):
    source_root = tmp_path / "t3_source"
    _write_sample_t3_dataset(source_root)
    output_root = tmp_path / "canonical_t3"

    cli_result = runner.invoke(
        app,
        [
            "normalize-dataset",
            "--pose-backend",
            "ved_localization",
            "--dataset-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--vehicle-model",
            "T3",
            "--camera",
            "front_camera_fov30=center_camera_fov30",
            "--camera",
            "left_front_camera",
            "--navigation-csv",
            str(source_root / "ved" / "LocalLocalization.csv"),
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert "result_interpolation_standard.pkl" in cli_result.output
    assert (output_root / "result_interpolation_standard.pkl").exists()
