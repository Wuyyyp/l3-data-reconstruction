from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from dongfeng_registration.camera_models import CameraCalibration, project_points
from dongfeng_registration.navigation import NavigationTrajectory
from dongfeng_registration.time_offset_scan import (
    run_time_offset_scan_from_retained_matches,
    scan_navigation_time_offsets,
    summarize_time_offset_scan,
)


def _transform_from_rt(translation_xyz, rotvec_xyz) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotvec_xyz).as_matrix()
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return transform


def _make_calibration(sensor_to_car: np.ndarray) -> CameraCalibration:
    return CameraCalibration(
        camera="center_camera_fov30",
        model="rational",
        image_size=(640, 480),
        intrinsic=np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=sensor_to_car,
        time_lag=0.0,
        images_undistorted=True,
    )


def _make_navigation() -> NavigationTrajectory:
    timestamps_ns = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0], [1.9, 0.9, 0.0], [3.4, 2.1, 0.0]],
        dtype=np.float64,
    )
    quaternions_xyzw = Rotation.from_euler(
        "z", np.array([[0.0], [4.0], [12.0], [20.0]], dtype=np.float64), degrees=True
    ).as_quat()
    return NavigationTrajectory(timestamps_ns, positions, quaternions_xyzw)


def _make_shifted_matches(
    calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    time_offset_ns: int,
) -> pd.DataFrame:
    world_points = np.array(
        [
            [2.0, -0.8, 8.0],
            [2.5, -0.2, 10.0],
            [3.0, 0.5, 9.0],
            [3.5, 0.8, 11.0],
            [4.0, -0.5, 12.0],
            [4.5, 0.2, 13.0],
        ],
        dtype=np.float64,
    )
    records: list[dict[str, float | int | str | bool]] = []
    pair_specs = [(0, 1, 5), (1, 2, 10), (0, 2, 30)]
    for first_idx, second_idx, frame_offset in pair_specs:
        stored_t1 = int(navigation.timestamps_ns[first_idx])
        stored_t2 = int(navigation.timestamps_ns[second_idx])
        actual_t1 = stored_t1 + time_offset_ns
        actual_t2 = stored_t2 + time_offset_ns
        first_body = navigation.pose_at(actual_t1)
        second_body = navigation.pose_at(actual_t2)
        first_world_to_camera = np.linalg.inv(first_body @ calibration.sensor_to_car)
        second_world_to_camera = np.linalg.inv(second_body @ calibration.sensor_to_car)
        world_h = np.column_stack([world_points, np.ones(len(world_points))])
        first_points_camera = (first_world_to_camera @ world_h.T).T[:, :3]
        second_points_camera = (second_world_to_camera @ world_h.T).T[:, :3]
        first_pixels = project_points(first_points_camera, calibration)
        second_pixels = project_points(second_points_camera, calibration)
        first_normalized = first_points_camera[:, :2] / first_points_camera[:, 2, None]
        second_normalized = second_points_camera[:, :2] / second_points_camera[:, 2, None]
        for point_idx in range(len(world_points)):
            records.append(
                {
                    "camera": calibration.camera,
                    "frame_offset": frame_offset,
                    "frame1": first_idx,
                    "frame2": second_idx,
                    "timestamp1_ns": stored_t1,
                    "timestamp2_ns": stored_t2,
                    "x1": float(first_pixels[point_idx, 0]),
                    "y1": float(first_pixels[point_idx, 1]),
                    "x2": float(second_pixels[point_idx, 0]),
                    "y2": float(second_pixels[point_idx, 1]),
                    "nx1": float(first_normalized[point_idx, 0]),
                    "ny1": float(first_normalized[point_idx, 1]),
                    "nx2": float(second_normalized[point_idx, 0]),
                    "ny2": float(second_normalized[point_idx, 1]),
                    "pose_quality_valid": True,
                    "reprojection_threshold_inlier": True,
                }
            )
    return pd.DataFrame.from_records(records)


def test_scan_navigation_time_offsets_recovers_synthetic_best_offset():
    navigation = _make_navigation()
    calibration = _make_calibration(_transform_from_rt([0.2, 0.0, 0.1], [0.0, 0.0, 0.0]))
    matches = _make_shifted_matches(calibration, navigation, time_offset_ns=100_000_000)
    offsets_ns = np.array(
        [-200_000_000, -100_000_000, 0, 100_000_000, 200_000_000], dtype=np.int64
    )

    results = scan_navigation_time_offsets(matches, navigation, calibration, offsets_ns)
    summary = summarize_time_offset_scan(results)

    best = summary.sort_values("reprojection_median_px", ascending=True).iloc[0]
    assert int(best["time_offset_ms"]) == 100
    assert float(best["reprojection_median_px"]) < float(
        summary.loc[summary["time_offset_ms"] == 0, "reprojection_median_px"].iloc[0]
    )


def test_run_time_offset_scan_accepts_new_match_cache_dir(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "data_cache" / "matches" / "cache" / "intra_camera" / "center_camera_fov30" / "offsets_5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.DataFrame(
        {
            "camera": ["center_camera_fov30"],
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [1],
            "timestamp1_ns": [100],
            "timestamp2_ns": [200],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    )
    matches.to_parquet(cache_dir / "retained_matches.parquet", index=False)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    seen = {}

    def fake_scan(matches, _navigation, _calibration, offsets_ns):
        seen["match_count"] = len(matches)
        seen["offsets_ns"] = offsets_ns.tolist()
        result = matches.copy()
        result["navigation_time_offset_ns"] = offsets_ns[0]
        result["nav_sampson_px2"] = 1.0
        result["reprojection_rms_px"] = 2.0
        return result

    monkeypatch.setattr("dongfeng_registration.time_offset_scan.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.time_offset_scan.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.time_offset_scan.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr("dongfeng_registration.time_offset_scan.scan_navigation_time_offsets", fake_scan)

    run_time_offset_scan_from_retained_matches(
        config_path=tmp_path / "config.yaml",
        retained_matches_input=cache_dir,
        output=tmp_path / "out",
        camera="center_camera_fov30",
        scan_from_ms=0,
        scan_to_ms=0,
        step_ms=10,
    )

    assert seen["match_count"] == 1


def test_run_time_offset_scan_rejects_legacy_pose_quality_dir(monkeypatch, tmp_path: Path):
    retained_matches_input = tmp_path / "pose_quality"
    retained_root = retained_matches_input / "retained_matches"
    retained_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "camera": ["center_camera_fov30"],
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [1],
            "timestamp1_ns": [100],
            "timestamp2_ns": [200],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    ).to_parquet(retained_root / "retained_semantic_matches.parquet", index=False)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    monkeypatch.setattr("dongfeng_registration.time_offset_scan.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.time_offset_scan.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.time_offset_scan.NavigationTrajectory", StubNavigation)

    with pytest.raises(FileNotFoundError):
        run_time_offset_scan_from_retained_matches(
            config_path=tmp_path / "config.yaml",
            retained_matches_input=retained_matches_input,
            output=tmp_path / "out",
            camera="center_camera_fov30",
            scan_from_ms=0,
            scan_to_ms=0,
            step_ms=10,
        )
