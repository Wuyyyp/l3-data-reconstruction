from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dongfeng_registration.camera_models import CameraCalibration, project_points
from dongfeng_registration.time_varying_rotation import (
    _delta_transform,
    _export_angle_change_deliverable,
    _summarize_offset_split,
    _write_pair_reprojection_visualizations,
    _export_image_camera_to_world_deliverable,
    _export_image_camera_to_world_origin_deliverable,
    _export_framewise_equivalent_extrinsics,
    _build_shock_smooth_scale_per_segment,
    _resolve_smooth_sigma_axes_deg,
    _torch_two_view_reprojection_rms,
    _torch_temporal_regularizer,
    build_frame_variable_index,
    build_jacobian_sparsity,
    build_knot_jacobian_sparsity,
    build_knot_frame_interpolator,
    evaluate_time_varying_reprojection,
    sample_matches,
    sparse_graph_optimize_knots,
    summarize_frame_disturbance,
    torch_sparse_graph_optimize_knots,
    run_time_varying_rotation_diagnostic,
)


def test_pair_reprojection_visualizations_use_reprojection_qc_directory(
    monkeypatch, tmp_path: Path
):
    image_path = tmp_path / "frame_0001.jpg"
    image_path.write_bytes(b"stub")
    frame = type("Frame", (), {"index": 1, "path": image_path})()
    comparison = pd.DataFrame(
        {
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [1],
            "timestamp1_ns": [100],
            "x2": [10.0],
            "y2": [20.0],
            "before_reprojection_rms_px": [1.0],
            "after_reprojection_rms_px": [0.5],
            "before_reprojected2_x": [11.0],
            "before_reprojected2_y": [21.0],
            "after_reprojected2_x": [10.5],
            "after_reprojected2_y": [20.5],
        }
    )
    seen_paths = []
    (tmp_path / "output" / "metrics").mkdir(parents=True)
    (tmp_path / "output" / "visualizations").mkdir()

    monkeypatch.setattr(
        "dongfeng_registration.time_varying_rotation.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "dongfeng_registration.time_varying_rotation.Dataset",
        lambda _root: type("DatasetStub", (), {"frames": lambda self, _camera: [frame]})(),
    )

    def fake_draw(_image, _observed, _before, _after, path, max_per_row=60):
        seen_paths.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")
        return [{"row": 0}]

    monkeypatch.setattr(
        "dongfeng_registration.time_varying_rotation.draw_tiled_before_after_reprojection_overlay",
        fake_draw,
    )

    _write_pair_reprojection_visualizations(
        tmp_path / "dataset",
        "center_camera_fov30",
        comparison,
        tmp_path / "output",
        max_pairs=None,
        max_per_row=60,
    )

    assert seen_paths
    assert "reprojection_qc" in seen_paths[0].parts
    assert "full_pairs" not in seen_paths[0].parts
    assert (
        tmp_path
        / "output"
        / "visualizations"
        / "reprojection_qc"
        / "reprojection_visualization_index.csv"
    ).exists()


def test_knot_frame_interpolator_can_interpolate_six_dof_disturbances():
    interpolator = build_knot_frame_interpolator([0, 5, 10], knot_stride=10, dimension=6)
    assert interpolator.variable_count == 12

    knot_values = np.array(
        [
            [0.0, 0.0, 0.0, 0.00, 0.00, 0.00],
            [0.0, 0.0, 0.2, 0.10, 0.20, 0.30],
        ],
        dtype=np.float64,
    )

    frame_values = interpolator.frame_values(knot_values)

    np.testing.assert_allclose(frame_values[0], knot_values[0])
    np.testing.assert_allclose(frame_values[5], [0.0, 0.0, 0.1, 0.05, 0.10, 0.15])
    np.testing.assert_allclose(frame_values[10], knot_values[1])


def test_delta_transform_accepts_six_dof_rotation_and_translation():
    transform = _delta_transform(np.array([0.0, 0.0, np.pi / 2.0, 1.0, 2.0, 3.0]))

    np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        transform[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_jacobian_sparsity_connects_pair_residuals_to_two_endpoint_frames():
    matches = pd.DataFrame(
        {
            "frame1": [10, 11],
            "frame2": [15, 16],
        }
    )
    frame_to_block = build_frame_variable_index(matches)

    sparsity = build_jacobian_sparsity(matches, frame_to_block)

    assert sparsity.shape == (2 + 4 * 3 + 3 * 3, 4 * 3)
    first_match_columns = set(sparsity.getrow(0).indices.tolist())
    assert first_match_columns == set(range(0, 3)) | set(range(2 * 3, 3 * 3))
    second_match_columns = set(sparsity.getrow(1).indices.tolist())
    assert second_match_columns == set(range(1 * 3, 2 * 3)) | set(range(3 * 3, 4 * 3))


def test_summarize_frame_disturbance_aggregates_incident_pair_errors():
    matches = pd.DataFrame(
        {
            "frame1": [0, 0, 1],
            "frame2": [5, 6, 6],
            "before_sampson_px": [10.0, 20.0, 30.0],
            "after_sampson_px": [8.0, 12.0, 15.0],
            "frame_offset": [5, 6, 5],
        }
    )
    frame_rotvecs = {
        0: np.deg2rad([0.1, 0.2, 0.3]),
        1: np.deg2rad([0.0, 0.0, 0.1]),
        5: np.zeros(3),
        6: np.deg2rad([0.4, 0.0, 0.0]),
    }

    summary = summarize_frame_disturbance(matches, frame_rotvecs)

    frame0 = summary.loc[summary["frame"] == 0].iloc[0]
    assert frame0["incident_pair_count"] == 2
    assert frame0["before_incident_sampson_mean_px"] == 15.0
    assert frame0["after_incident_sampson_mean_px"] == 10.0
    assert frame0["delta_pitch_deg"] == 0.2


def test_sample_matches_never_requests_negative_rows_when_many_groups():
    matches = pd.DataFrame(
        {
            "frame1": list(range(10)),
            "frame2": list(range(5, 15)),
            "value": list(range(10)),
        }
    )

    sampled = sample_matches(matches, max_observations=3)

    assert len(sampled) == 3


def test_knot_frame_interpolator_linearly_interpolates_frame_rotations():
    frames = [0, 5, 10]
    interpolator = build_knot_frame_interpolator(frames, knot_stride=10)
    parameters = np.deg2rad(np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])).reshape(-1)

    frame_rotvecs = interpolator.unpack(parameters)

    assert np.rad2deg(frame_rotvecs[0]).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert np.rad2deg(frame_rotvecs[5]).tolist() == pytest.approx([5.0, 10.0, 15.0])
    assert np.rad2deg(frame_rotvecs[10]).tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert interpolator.variable_count == 6


def test_knot_jacobian_sparsity_connects_pair_to_endpoint_knots():
    matches = pd.DataFrame({"frame1": [5], "frame2": [15]})
    interpolator = build_knot_frame_interpolator([0, 5, 10, 15], knot_stride=10)

    sparsity = build_knot_jacobian_sparsity(matches, interpolator)

    first_row_columns = set(sparsity.getrow(0).indices.tolist())
    # frame 5 interpolates between knots 0 and 1; frame 15 uses knot 2.
    assert first_row_columns == set(range(0, 3)) | set(range(3, 6)) | set(range(6, 9))
    assert sparsity.shape[1] == interpolator.variable_count


def test_sparse_graph_optimizer_passes_jacobian_sparsity(monkeypatch):
    matches = pd.DataFrame(
        {
            "frame1": [0, 5],
            "frame2": [5, 10],
            "timestamp1_ns": [0, 5],
            "timestamp2_ns": [5, 10],
        }
    )
    interpolator = build_knot_frame_interpolator([0, 5, 10], knot_stride=10)
    captured = {}

    class _Solver:
        success = True
        status = 1
        message = "ok"
        nfev = 3
        cost = 2.5
        optimality = 0.1
        x = np.zeros(interpolator.variable_count)

    def fake_least_squares(*args, **kwargs):
        captured.update(kwargs)
        return _Solver()

    monkeypatch.setattr(
        "dongfeng_registration.time_varying_rotation.least_squares",
        fake_least_squares,
    )
    monkeypatch.setattr(
        "dongfeng_registration.time_varying_rotation.evaluate_time_varying_rotation",
        lambda *args, **kwargs: pd.DataFrame({"tv_sampson_px": [1.0, 2.0]}),
    )

    values, status = sparse_graph_optimize_knots(
        sampled=matches,
        navigation=None,
        calibration=None,
        interpolator=interpolator,
        navigation_offset_ns=0,
        lower_one=np.deg2rad(np.array([-0.5, -0.5, -0.2])),
        upper_one=np.deg2rad(np.array([0.5, 0.5, 0.2])),
        prior_sigma=np.deg2rad(np.array([0.12, 0.12, 0.06])),
        smooth_sigma=np.deg2rad(0.08),
        max_nfev=20,
    )

    assert status["solver_method"] == "sparse_graph"
    assert values.shape == (len(interpolator.knots), 3)
    assert captured["jac_sparsity"].shape[1] == interpolator.variable_count


def test_torch_temporal_regularizer_huber_is_less_aggressive_than_l2_for_large_jumps():
    torch = pytest.importorskip("torch")
    diff = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)

    l2 = _torch_temporal_regularizer(diff, sigma, mode="l2", huber_delta=1.0)
    huber = _torch_temporal_regularizer(diff, sigma, mode="huber", huber_delta=1.0)

    assert huber.item() < l2.item()


def test_torch_temporal_regularizer_none_disables_temporal_penalty():
    torch = pytest.importorskip("torch")
    diff = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)

    penalty = _torch_temporal_regularizer(diff, sigma, mode="none", huber_delta=1.0)

    assert penalty.item() == pytest.approx(0.0)


def test_torch_temporal_regularizer_supports_per_axis_sigma():
    torch = pytest.importorskip("torch")
    diff = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64)
    sigma = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    penalty = _torch_temporal_regularizer(diff, sigma, mode="l2", huber_delta=1.0)

    assert penalty.item() == pytest.approx(5.25)


def test_torch_temporal_regularizer_supports_per_segment_sigma():
    torch = pytest.importorskip("torch")
    diff = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    sigma = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64)

    penalty = _torch_temporal_regularizer(diff, sigma, mode="l2", huber_delta=1.0)

    assert penalty.item() == pytest.approx(5.0)


def test_resolve_smooth_sigma_axes_deg_supports_axis_overrides():
    resolved = _resolve_smooth_sigma_axes_deg(
        smooth_sigma_deg=0.08,
        roll_smooth_sigma_deg=0.20,
        pitch_smooth_sigma_deg=None,
        yaw_smooth_sigma_deg=0.05,
    )

    assert resolved.tolist() == pytest.approx([0.20, 0.08, 0.05])


def test_build_shock_smooth_scale_per_segment_relaxes_high_shock_segments():
    frame_scores = {0: 0.1, 1: 0.2, 2: 0.9, 3: 0.3}

    scales = _build_shock_smooth_scale_per_segment(
        frame_scores=frame_scores,
        frame_sequence=[0, 1, 2, 3],
        relax_factor=3.0,
    )

    assert scales.shape == (3,)
    assert scales.tolist() == pytest.approx([1.4, 2.8, 2.8])


def test_export_framewise_equivalent_extrinsics_writes_client_deliverables(tmp_path: Path):
    calibration = CameraCalibration(
        camera="left_front_camera",
        model="rational",
        image_size=(640, 480),
        intrinsic=np.eye(3, dtype=np.float64),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=np.array(
            [
                [1.0, 0.0, 0.0, 1.2],
                [0.0, 1.0, 0.0, -0.3],
                [0.0, 0.0, 1.0, 1.6],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        time_lag=0.0,
        images_undistorted=True,
    )
    frame_rotvecs = {
        12: np.deg2rad(np.array([0.1, -0.2, 0.3], dtype=np.float64)),
        13: np.deg2rad(np.array([0.0, 0.0, 0.0], dtype=np.float64)),
    }
    frame_timestamps_ns = {12: 1_200_000_000, 13: 1_300_000_000}

    csv_path, parquet_path, summary_json = _export_framewise_equivalent_extrinsics(
        output=tmp_path,
        camera="left_front_camera",
        base_extrinsic_source="calib/left_front_camera/left_front_camera-to-car_center-extrinsic.json",
        calibration=calibration,
        frame_rotvecs=frame_rotvecs,
        frame_timestamps_ns=frame_timestamps_ns,
    )

    assert csv_path == tmp_path / "observations" / "framewise_equivalent_extrinsics.csv"
    assert parquet_path == tmp_path / "observations" / "framewise_equivalent_extrinsics.parquet"
    assert summary_json == tmp_path / "metrics" / "framewise_equivalent_extrinsics_summary.json"
    exported = pd.read_csv(csv_path)
    assert exported["frame"].tolist() == [12, 13]
    assert exported["base_extrinsic_source"].nunique() == 1
    assert "sensor_to_car_00" in exported.columns
    assert "delta_roll_deg" in exported.columns
    assert "delta_pitch_deg" in exported.columns
    assert "delta_yaw_deg" in exported.columns
    assert exported.loc[0, "sensor_to_car_tx_m"] == pytest.approx(1.2)
    assert parquet_path.exists()
    assert summary_json.exists()


def test_export_framewise_equivalent_extrinsics_includes_image_mapping_and_data_dictionary(
    tmp_path: Path,
):
    calibration = CameraCalibration(
        camera="center_camera_fov30",
        model="rational",
        image_size=(640, 480),
        intrinsic=np.eye(3, dtype=np.float64),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=np.eye(4, dtype=np.float64),
        time_lag=0.0,
        images_undistorted=True,
    )
    frame_rotvecs = {0: np.zeros(3, dtype=np.float64)}
    frame_timestamps_ns = {0: 1772699910107617000}
    frame_image_paths = {
        0: Path(
            r"D:\DongFeng\data\camera\center_camera_fov30\1772699910107617000.jpg"
        )
    }

    _export_framewise_equivalent_extrinsics(
        output=tmp_path,
        camera="center_camera_fov30",
        base_extrinsic_source="calib/center_camera_fov30/center_camera_fov30-to-car_center-extrinsic.json",
        calibration=calibration,
        frame_rotvecs=frame_rotvecs,
        frame_timestamps_ns=frame_timestamps_ns,
        frame_image_paths=frame_image_paths,
    )

    exported = pd.read_csv(tmp_path / "observations" / "framewise_equivalent_extrinsics.csv")
    assert exported.loc[0, "image_filename"] == "1772699910107617000.jpg"
    assert exported.loc[0, "image_path"].endswith(r"1772699910107617000.jpg")
    dictionary_path = tmp_path / "framewise_equivalent_extrinsics_README.md"
    assert dictionary_path.exists()
    dictionary_text = dictionary_path.read_text(encoding="utf-8")
    assert "image_path" in dictionary_text
    assert "sensor_to_car_00" in dictionary_text
    camera_to_world_by_frame = {0: np.eye(4, dtype=np.float64)}
    _export_image_camera_to_world_deliverable(
        output=tmp_path,
        frame_image_paths=frame_image_paths,
        camera_to_world_by_frame=camera_to_world_by_frame,
    )

    client_csv = tmp_path / "deliverables" / "image_camera_to_world.csv"
    assert client_csv.exists()
    client_exported = pd.read_csv(client_csv)
    assert client_exported.columns.tolist() == [
        "image_filename",
        "camera_to_world_00",
        "camera_to_world_01",
        "camera_to_world_02",
        "camera_to_world_03",
        "camera_to_world_10",
        "camera_to_world_11",
        "camera_to_world_12",
        "camera_to_world_13",
        "camera_to_world_20",
        "camera_to_world_21",
        "camera_to_world_22",
        "camera_to_world_23",
        "camera_to_world_30",
        "camera_to_world_31",
        "camera_to_world_32",
        "camera_to_world_33",
    ]
    assert not (tmp_path / "deliverables" / "README.md").exists()

    origin_csv = _export_image_camera_to_world_origin_deliverable(
        output=tmp_path,
        frame_image_paths=frame_image_paths,
        camera_to_world_origin_by_frame={0: np.eye(4, dtype=np.float64)},
    )
    assert origin_csv == tmp_path / "deliverables" / "image_camera_to_world_origin.csv"
    origin_exported = pd.read_csv(origin_csv)
    assert origin_exported.columns.tolist() == [
        "image_filename",
        "camera_to_world_00",
        "camera_to_world_01",
        "camera_to_world_02",
        "camera_to_world_03",
        "camera_to_world_10",
        "camera_to_world_11",
        "camera_to_world_12",
        "camera_to_world_13",
        "camera_to_world_20",
        "camera_to_world_21",
        "camera_to_world_22",
        "camera_to_world_23",
        "camera_to_world_30",
        "camera_to_world_31",
        "camera_to_world_32",
        "camera_to_world_33",
    ]
    angle_csv = _export_angle_change_deliverable(
        output=tmp_path,
        frame_image_paths=frame_image_paths,
        angle_change_deg_by_frame={0: 1.234},
    )
    assert angle_csv == tmp_path / "deliverables" / "angle_change_deg.csv"
    angle_exported = pd.read_csv(angle_csv)
    assert angle_exported.columns.tolist() == ["image_filename", "angle_change_deg"]
    assert angle_exported.loc[0, "angle_change_deg"] == pytest.approx(1.234)


class _SyntheticNavigation:
    def __init__(self):
        self.poses = {
            0: np.eye(4, dtype=np.float64),
            1: np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        }

    def pose_at(self, timestamp_ns):
        return self.poses[int(timestamp_ns)]


def _calibration():
    return CameraCalibration(
        camera="synthetic",
        model="rational",
        image_size=(640, 480),
        intrinsic=np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=np.eye(4, dtype=np.float64),
        time_lag=0.0,
        images_undistorted=True,
    )


def test_time_varying_reprojection_uses_frame_disturbances_for_pixel_errors():
    calibration = _calibration()
    navigation = _SyntheticNavigation()
    world_points = np.array(
        [[4.0, 0.0, 12.0], [5.0, 0.5, 14.0], [6.0, -0.3, 16.0]],
        dtype=np.float64,
    )
    first_camera = navigation.pose_at(0) @ calibration.sensor_to_car
    second_camera = navigation.pose_at(1) @ calibration.sensor_to_car
    first_points_camera = (np.linalg.inv(first_camera) @ np.column_stack([world_points, np.ones(3)]).T).T[:, :3]
    second_points_camera = (np.linalg.inv(second_camera) @ np.column_stack([world_points, np.ones(3)]).T).T[:, :3]
    first_pixels = project_points(first_points_camera, calibration)
    second_pixels = project_points(second_points_camera, calibration)
    matches = pd.DataFrame(
        {
            "frame1": [0, 0, 0],
            "frame2": [1, 1, 1],
            "timestamp1_ns": [0, 0, 0],
            "timestamp2_ns": [1, 1, 1],
            "x1": first_pixels[:, 0],
            "y1": first_pixels[:, 1],
            "x2": second_pixels[:, 0],
            "y2": second_pixels[:, 1],
            "nx1": (first_pixels[:, 0] - 320.0) / 500.0,
            "ny1": (first_pixels[:, 1] - 240.0) / 500.0,
            "nx2": (second_pixels[:, 0] - 320.0) / 500.0,
            "ny2": (second_pixels[:, 1] - 240.0) / 500.0,
        }
    )

    before = evaluate_time_varying_reprojection(
        matches, navigation, calibration, {0: np.zeros(3), 1: np.zeros(3)}
    )
    disturbed = evaluate_time_varying_reprojection(
        matches, navigation, calibration, {0: np.deg2rad([0.0, 0.3, 0.0]), 1: np.zeros(3)}
    )

    assert before["tv_reprojection_rms_px"].max() < 1e-6
    assert disturbed["tv_reprojection_rms_px"].median() > before["tv_reprojection_rms_px"].median() + 1e-3
    assert {"tv_reprojected1_x", "tv_reprojected2_x"}.issubset(disturbed.columns)


def test_torch_two_view_reprojection_rms_is_near_zero_for_perfect_geometry():
    torch = pytest.importorskip("torch")
    calibration = _calibration()
    navigation = _SyntheticNavigation()
    world_points = np.array(
        [[4.0, 0.0, 12.0], [5.0, 0.5, 14.0], [6.0, -0.3, 16.0]],
        dtype=np.float64,
    )
    first_camera = navigation.pose_at(0) @ calibration.sensor_to_car
    second_camera = navigation.pose_at(1) @ calibration.sensor_to_car
    first_points_camera = (
        np.linalg.inv(first_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    second_points_camera = (
        np.linalg.inv(second_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    first_pixels = project_points(first_points_camera, calibration)
    second_pixels = project_points(second_points_camera, calibration)
    first_world_to_camera = torch.tensor(
        np.linalg.inv(first_camera), dtype=torch.float64
    ).unsqueeze(0).repeat(3, 1, 1)
    second_world_to_camera = torch.tensor(
        np.linalg.inv(second_camera), dtype=torch.float64
    ).unsqueeze(0).repeat(3, 1, 1)
    intrinsic = torch.tensor(calibration.intrinsic, dtype=torch.float64)
    first_normalized = torch.tensor(
        np.column_stack(
            [
                (first_pixels[:, 0] - 320.0) / 500.0,
                (first_pixels[:, 1] - 240.0) / 500.0,
            ]
        ),
        dtype=torch.float64,
    )
    second_normalized = torch.tensor(
        np.column_stack(
            [
                (second_pixels[:, 0] - 320.0) / 500.0,
                (second_pixels[:, 1] - 240.0) / 500.0,
            ]
        ),
        dtype=torch.float64,
    )
    first_pixels_t = torch.tensor(first_pixels, dtype=torch.float64)
    second_pixels_t = torch.tensor(second_pixels, dtype=torch.float64)

    rms = _torch_two_view_reprojection_rms(
        first_normalized,
        second_normalized,
        first_world_to_camera,
        second_world_to_camera,
        intrinsic,
        first_pixels_t,
        second_pixels_t,
    )

    assert float(rms.max()) < 1e-6


def test_torch_two_view_reprojection_rms_increases_after_pose_disturbance():
    torch = pytest.importorskip("torch")
    calibration = _calibration()
    navigation = _SyntheticNavigation()
    world_points = np.array(
        [[4.0, 0.0, 12.0], [5.0, 0.5, 14.0], [6.0, -0.3, 16.0]],
        dtype=np.float64,
    )
    first_camera = navigation.pose_at(0) @ calibration.sensor_to_car
    second_camera = navigation.pose_at(1) @ calibration.sensor_to_car
    first_points_camera = (
        np.linalg.inv(first_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    second_points_camera = (
        np.linalg.inv(second_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    first_pixels = project_points(first_points_camera, calibration)
    second_pixels = project_points(second_points_camera, calibration)
    disturbed_second = second_camera.copy()
    disturbed_second[:3, :3] = second_camera[:3, :3] @ np.array(
        [
            [0.9998477, 0.0, 0.0174524],
            [0.0, 1.0, 0.0],
            [-0.0174524, 0.0, 0.9998477],
        ],
        dtype=np.float64,
    )
    first_world_to_camera = torch.tensor(
        np.linalg.inv(first_camera), dtype=torch.float64
    ).unsqueeze(0).repeat(3, 1, 1)
    second_world_to_camera = torch.tensor(
        np.linalg.inv(disturbed_second), dtype=torch.float64
    ).unsqueeze(0).repeat(3, 1, 1)
    intrinsic = torch.tensor(calibration.intrinsic, dtype=torch.float64)
    first_normalized = torch.tensor(
        np.column_stack(
            [
                (first_pixels[:, 0] - 320.0) / 500.0,
                (first_pixels[:, 1] - 240.0) / 500.0,
            ]
        ),
        dtype=torch.float64,
    )
    second_normalized = torch.tensor(
        np.column_stack(
            [
                (second_pixels[:, 0] - 320.0) / 500.0,
                (second_pixels[:, 1] - 240.0) / 500.0,
            ]
        ),
        dtype=torch.float64,
    )
    first_pixels_t = torch.tensor(first_pixels, dtype=torch.float64)
    second_pixels_t = torch.tensor(second_pixels, dtype=torch.float64)

    rms = _torch_two_view_reprojection_rms(
        first_normalized,
        second_normalized,
        first_world_to_camera,
        second_world_to_camera,
        intrinsic,
        first_pixels_t,
        second_pixels_t,
    )

    assert float(torch.median(rms)) > 1e-3


def test_torch_sparse_graph_optimizer_accepts_joint_reprojection_term():
    torch = pytest.importorskip("torch")
    calibration = _calibration()
    navigation = _SyntheticNavigation()
    world_points = np.array(
        [[4.0, 0.0, 12.0], [5.0, 0.5, 14.0], [6.0, -0.3, 16.0]],
        dtype=np.float64,
    )
    first_camera = navigation.pose_at(0) @ calibration.sensor_to_car
    second_camera = navigation.pose_at(1) @ calibration.sensor_to_car
    first_points_camera = (
        np.linalg.inv(first_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    second_points_camera = (
        np.linalg.inv(second_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    first_pixels = project_points(first_points_camera, calibration)
    second_pixels = project_points(second_points_camera, calibration)
    matches = pd.DataFrame(
        {
            "frame1": [0, 0, 0],
            "frame2": [1, 1, 1],
            "timestamp1_ns": [0, 0, 0],
            "timestamp2_ns": [1, 1, 1],
            "x1": first_pixels[:, 0],
            "y1": first_pixels[:, 1],
            "x2": second_pixels[:, 0],
            "y2": second_pixels[:, 1],
            "nx1": (first_pixels[:, 0] - 320.0) / 500.0,
            "ny1": (first_pixels[:, 1] - 240.0) / 500.0,
            "nx2": (second_pixels[:, 0] - 320.0) / 500.0,
            "ny2": (second_pixels[:, 1] - 240.0) / 500.0,
        }
    )
    interpolator = build_knot_frame_interpolator([0, 1], knot_stride=1)

    values, status = torch_sparse_graph_optimize_knots(
        sampled=matches,
        navigation=navigation,
        calibration=calibration,
        interpolator=interpolator,
        navigation_offset_ns=0,
        lower_one=np.deg2rad(np.array([-5.0, -5.0, -5.0])),
        upper_one=np.deg2rad(np.array([5.0, 5.0, 5.0])),
        prior_sigma=np.deg2rad(np.array([0.12, 0.12, 0.06])),
        smooth_sigma=np.deg2rad(np.array([0.08, 0.08, 0.08])),
        max_nfev=1,
        learning_rate=0.001,
        device="cpu",
        temporal_regularizer="huber",
        temporal_huber_delta=1.0,
        use_prior_regularizer=True,
        use_reprojection_term=True,
        reprojection_loss_weight=0.25,
    )

    assert values.shape == (len(interpolator.knots), 3)
    assert status["use_reprojection_term"] is True
    assert status["reprojection_loss_weight"] == pytest.approx(0.25)


def test_torch_sparse_graph_optimizer_reports_learning_rate_schedule():
    torch = pytest.importorskip("torch")
    calibration = _calibration()
    navigation = _SyntheticNavigation()
    world_points = np.array(
        [[4.0, 0.0, 12.0], [5.0, 0.5, 14.0], [6.0, -0.3, 16.0]],
        dtype=np.float64,
    )
    first_camera = navigation.pose_at(0) @ calibration.sensor_to_car
    second_camera = navigation.pose_at(1) @ calibration.sensor_to_car
    first_points_camera = (
        np.linalg.inv(first_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    second_points_camera = (
        np.linalg.inv(second_camera) @ np.column_stack([world_points, np.ones(3)]).T
    ).T[:, :3]
    first_pixels = project_points(first_points_camera, calibration)
    second_pixels = project_points(second_points_camera, calibration)
    matches = pd.DataFrame(
        {
            "frame1": [0, 0, 0],
            "frame2": [1, 1, 1],
            "timestamp1_ns": [0, 0, 0],
            "timestamp2_ns": [1, 1, 1],
            "x1": first_pixels[:, 0],
            "y1": first_pixels[:, 1],
            "x2": second_pixels[:, 0],
            "y2": second_pixels[:, 1],
            "nx1": (first_pixels[:, 0] - 320.0) / 500.0,
            "ny1": (first_pixels[:, 1] - 240.0) / 500.0,
            "nx2": (second_pixels[:, 0] - 320.0) / 500.0,
            "ny2": (second_pixels[:, 1] - 240.0) / 500.0,
        }
    )
    interpolator = build_knot_frame_interpolator([0, 1], knot_stride=1)

    values, status = torch_sparse_graph_optimize_knots(
        sampled=matches,
        navigation=navigation,
        calibration=calibration,
        interpolator=interpolator,
        navigation_offset_ns=0,
        lower_one=np.deg2rad(np.array([-5.0, -5.0, -5.0])),
        upper_one=np.deg2rad(np.array([5.0, 5.0, 5.0])),
        prior_sigma=np.deg2rad(np.array([0.12, 0.12, 0.06])),
        smooth_sigma=np.deg2rad(np.array([0.08, 0.08, 0.08])),
        max_nfev=2,
        learning_rate=0.003,
        device="cpu",
        temporal_regularizer="huber",
        temporal_huber_delta=1.0,
        use_prior_regularizer=True,
        lr_schedule="step",
        lr_step_size=1,
        lr_decay=0.5,
        lr_min_factor=0.1,
    )

    assert values.shape == (len(interpolator.knots), 3)
    assert status["lr_schedule"] == "step"
    assert status["lr_step_size"] == 1
    assert status["lr_decay"] == pytest.approx(0.5)
    assert status["lr_min_factor"] == pytest.approx(0.1)


def test_summarize_offset_split_separates_train_and_eval_groups():
    matches = pd.DataFrame(
        {
            "frame_offset": [5, 5, 10, 10, 30, 30],
            "before_reprojection_rms_px": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
            "after_reprojection_rms_px": [4.0, 5.0, 8.0, 9.0, 29.0, 31.0],
            "before_sampson_px": [8.0, 9.0, 18.0, 19.0, 28.0, 29.0],
            "after_sampson_px": [3.0, 4.0, 7.0, 8.0, 27.0, 28.0],
        }
    )

    summary = _summarize_offset_split(
        matches,
        train_offsets={5, 10},
        eval_offsets={30},
    )

    groups = set(summary["split_group"].tolist())
    assert groups == {"all", "train", "eval"}
    train_row = summary.loc[summary["split_group"] == "train"].iloc[0]
    eval_row = summary.loc[summary["split_group"] == "eval"].iloc[0]
    assert train_row["count"] == 4
    assert eval_row["count"] == 2
    assert train_row["after_reprojection_median_px"] < train_row["before_reprojection_median_px"]
    assert eval_row["after_reprojection_median_px"] < eval_row["before_reprojection_median_px"]


def test_run_time_varying_rotation_diagnostic_respects_train_and_eval_offsets(monkeypatch, tmp_path: Path):
    retained_root = tmp_path / "data_cache" / "matches" / "cache" / "intra_camera" / "center_camera_fov30" / "offsets_5"
    retained_root.mkdir(parents=True)
    retained = pd.DataFrame(
        {
            "frame1": [0, 1, 2],
            "frame2": [5, 11, 32],
            "frame_offset": [5, 10, 30],
            "timestamp1_ns": [0, 1, 2],
            "timestamp2_ns": [5, 11, 32],
            "x1": [10.0, 11.0, 12.0],
            "y1": [20.0, 21.0, 22.0],
            "x2": [13.0, 14.0, 15.0],
            "y2": [23.0, 24.0, 25.0],
            "nx1": [0.1, 0.1, 0.1],
            "ny1": [0.1, 0.1, 0.1],
            "nx2": [0.1, 0.1, 0.1],
            "ny2": [0.1, 0.1, 0.1],
            "pose_quality_valid": [True, True, True],
            "reprojection_threshold_inlier": [True, True, True],
        }
    )
    retained.to_parquet(retained_root / "retained_matches.parquet", index=False)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

        def pose_at(self, _timestamp_ns):
            return np.eye(4)

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type("Frame", (), {"index": 0, "timestamp_ns": 0, "path": tmp_path / "0.jpg"})(),
                type("Frame", (), {"index": 1, "timestamp_ns": 1, "path": tmp_path / "1.jpg"})(),
                type("Frame", (), {"index": 2, "timestamp_ns": 2, "path": tmp_path / "2.jpg"})(),
                type("Frame", (), {"index": 5, "timestamp_ns": 5, "path": tmp_path / "5.jpg"})(),
                type("Frame", (), {"index": 11, "timestamp_ns": 11, "path": tmp_path / "11.jpg"})(),
                type("Frame", (), {"index": 32, "timestamp_ns": 32, "path": tmp_path / "32.jpg"})(),
            ]

    captured = {}

    def fake_optimizer(*, sampled, **kwargs):
        captured["train_offsets"] = sorted(sampled["frame_offset"].astype(int).unique().tolist())
        interpolator = kwargs["interpolator"]
        captured["optimizer_frames"] = list(interpolator.frames)
        return np.zeros((len(interpolator.knots), 3), dtype=np.float64), {
            "success": True,
            "status": 0,
            "solver_method": "torch_sparse_graph_full_batch",
            "message": "ok",
            "nfev": 1,
            "cost": 0.0,
            "optimality": 0.0,
        }

    def fake_eval(matches, *_args, **_kwargs):
        result = matches.copy()
        result["tv_sampson_px"] = matches["frame_offset"].astype(float)
        return result

    def fake_reproj(matches, *_args, **_kwargs):
        result = matches.copy()
        base = matches["frame_offset"].astype(float).to_numpy()
        result["tv_reprojection1_px"] = base
        result["tv_reprojection2_px"] = base
        result["tv_reprojection_rms_px"] = base
        result["tv_reprojected1_x"] = result["x1"]
        result["tv_reprojected1_y"] = result["y1"]
        result["tv_reprojected2_x"] = result["x2"]
        result["tv_reprojected2_y"] = result["y2"]
        return result

    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.navigation_time_offset_ns", lambda _assumptions: 0)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.torch_sparse_graph_optimize_knots", fake_optimizer)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.evaluate_time_varying_rotation", fake_eval)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.evaluate_time_varying_reprojection", fake_reproj)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_framewise_equivalent_extrinsics", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_image_camera_to_world_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_image_camera_to_world_origin_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_angle_change_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_pair_reprojection_visualizations", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_report", lambda *args, **kwargs: None)

    output = run_time_varying_rotation_diagnostic(
        config_path=tmp_path / "config.yaml",
        retained_matches_input=retained_root,
        output=tmp_path / "diag",
        camera="center_camera_fov30",
        train_offsets=(5, 10),
        eval_offsets=(30,),
        max_observations=None,
        max_nfev=1,
        knot_stride_frames=1,
        export_pair_visualizations=False,
        optimizer_method="torch_sparse_graph",
        torch_device="cpu",
    )

    assert captured["train_offsets"] == [5, 10]
    assert captured["optimizer_frames"] == [0, 1, 5, 11]
    split_summary = pd.read_csv(output / "metrics" / "offset_split_summary.csv")
    assert set(split_summary["split_group"].tolist()) == {"all", "train", "eval"}


def test_run_time_varying_rotation_diagnostic_accepts_new_match_cache_dir(
    monkeypatch, tmp_path: Path
):
    cache_dir = tmp_path / "data_cache" / "matches" / "cache" / "intra_camera" / "center_camera_fov30" / "offsets_5"
    cache_dir.mkdir(parents=True)
    retained = pd.DataFrame(
        {
            "frame1": [0],
            "frame2": [5],
            "frame_offset": [5],
            "timestamp1_ns": [0],
            "timestamp2_ns": [5],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [13.0],
            "y2": [23.0],
            "nx1": [0.1],
            "ny1": [0.1],
            "nx2": [0.1],
            "ny2": [0.1],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    )
    retained.to_parquet(cache_dir / "retained_matches.parquet", index=False)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

        def pose_at(self, _timestamp_ns):
            return np.eye(4)

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type("Frame", (), {"index": 0, "timestamp_ns": 0, "path": tmp_path / "0.jpg"})(),
                type("Frame", (), {"index": 5, "timestamp_ns": 5, "path": tmp_path / "5.jpg"})(),
            ]

    seen = {}

    def fake_optimizer(*, sampled, **kwargs):
        seen["match_count"] = len(sampled)
        interpolator = kwargs["interpolator"]
        seen["dimension"] = interpolator.dimension
        seen["lower_one"] = kwargs["lower_one"].copy()
        seen["upper_one"] = kwargs["upper_one"].copy()
        seen["prior_sigma"] = kwargs["prior_sigma"].copy()
        seen["smooth_sigma"] = kwargs["smooth_sigma"].copy()
        return np.zeros((len(interpolator.knots), interpolator.dimension), dtype=np.float64), {
            "success": True,
            "status": 0,
            "solver_method": "torch_sparse_graph_full_batch",
            "message": "ok",
            "nfev": 1,
            "cost": 0.0,
            "optimality": 0.0,
        }

    def fake_eval(matches, *_args, **_kwargs):
        result = matches.copy()
        result["tv_sampson_px"] = 1.0
        return result

    def fake_reproj(matches, *_args, **_kwargs):
        result = matches.copy()
        result["tv_reprojection1_px"] = 1.0
        result["tv_reprojection2_px"] = 1.0
        result["tv_reprojection_rms_px"] = 1.0
        result["tv_reprojected1_x"] = result["x1"]
        result["tv_reprojected1_y"] = result["y1"]
        result["tv_reprojected2_x"] = result["x2"]
        result["tv_reprojected2_y"] = result["y2"]
        return result

    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.navigation_time_offset_ns", lambda _assumptions: 0)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.torch_sparse_graph_optimize_knots", fake_optimizer)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.evaluate_time_varying_rotation", fake_eval)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.evaluate_time_varying_reprojection", fake_reproj)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_framewise_equivalent_extrinsics", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_image_camera_to_world_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_image_camera_to_world_origin_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._export_angle_change_deliverable", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_pair_reprojection_visualizations", lambda *args, **kwargs: None)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation._write_report", lambda *args, **kwargs: None)

    run_time_varying_rotation_diagnostic(
        config_path=tmp_path / "config.yaml",
        retained_matches_input=cache_dir,
        output=tmp_path / "diag",
        camera="center_camera_fov30",
        train_offsets=(5,),
        max_observations=None,
        max_nfev=1,
        knot_stride_frames=1,
        export_pair_visualizations=False,
        optimizer_method="torch_sparse_graph",
        torch_device="cpu",
        disturbance_dof=6,
        translation_bound_m=0.2,
        translation_prior_sigma_m=0.03,
        translation_smooth_sigma_m=0.04,
    )

    assert seen["match_count"] == 1
    assert seen["dimension"] == 6
    np.testing.assert_allclose(seen["lower_one"][3:], [-0.2, -0.2, -0.2])
    np.testing.assert_allclose(seen["upper_one"][3:], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(seen["prior_sigma"][3:], [0.03, 0.03, 0.03])
    np.testing.assert_allclose(seen["smooth_sigma"][3:], [0.04, 0.04, 0.04])


def test_run_time_varying_rotation_diagnostic_rejects_legacy_pose_quality_dir(
    monkeypatch, tmp_path: Path
):
    retained_root = tmp_path / "pose_quality" / "retained_matches"
    retained_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "frame1": [0],
            "frame2": [5],
            "frame_offset": [5],
            "timestamp1_ns": [0],
            "timestamp2_ns": [5],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [13.0],
            "y2": [23.0],
            "nx1": [0.1],
            "ny1": [0.1],
            "nx2": [0.1],
            "ny2": [0.1],
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
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr("dongfeng_registration.time_varying_rotation.navigation_time_offset_ns", lambda _assumptions: 0)

    with pytest.raises(FileNotFoundError):
        run_time_varying_rotation_diagnostic(
            config_path=tmp_path / "config.yaml",
            retained_matches_input=tmp_path / "pose_quality",
            output=tmp_path / "diag",
            camera="center_camera_fov30",
            train_offsets=(5,),
            max_observations=None,
            max_nfev=1,
            knot_stride_frames=1,
            export_pair_visualizations=False,
            optimizer_method="torch_sparse_graph",
            torch_device="cpu",
        )
