from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import dongfeng_registration.extrinsic_optimization as extrinsic_optimization
from dongfeng_registration.camera_models import CameraCalibration, project_points
from dongfeng_registration.extrinsic_optimization import (
    _build_padded_group_layout,
    _build_unique_endpoint_layout,
    _compute_static_adaptive_group_weights,
    _build_reprojection_qc_matches,
    _evaluate_multicamera_matches,
    _evaluate_multicamera_matches_legacy,
    _sample_matches_per_image_pair,
    _select_reprojection_qc_rows,
    _absorb_camera_common_frame_disturbance,
    _absorb_camera_long_term_frame_disturbance,
    _camera_frame_common_mode_loss_numpy,
    _project_time_varying_modes_numpy,
    _rotation_delta_transform,
    _resolve_multicamera_evaluation_backend,
    _summarize_multicamera_edges,
    _torch_sampson_sqrt_from_essential,
    _torch_compose_unique_endpoint_poses,
    _torch_group_means,
    _write_reprojection_outlier_diagnostics,
    optimize_multi_camera_alternating_from_observations,
    optimize_multi_camera_extrinsics_from_match_cache,
    optimize_multi_camera_extrinsics_from_observations,
)
from dongfeng_registration.navigation import NavigationTrajectory


def _transform_from_rt(translation_xyz, rotvec_xyz) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotvec_xyz).as_matrix()
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return transform


def _disturbance_from_transform(transform: np.ndarray, dof: int = 6) -> np.ndarray:
    values = np.zeros(int(dof), dtype=np.float64)
    values[:3] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    if int(dof) >= 6:
        values[3:6] = transform[:3, 3]
    return values


def test_unique_endpoint_layout_deduplicates_camera_frame_timestamp_triplets():
    matches = pd.DataFrame(
        {
            "camera1": ["camera_a", "camera_a", "camera_b"],
            "frame1": [10, 10, 10],
            "timestamp1_ns": [100, 100, 110],
            "camera2": ["camera_b", "camera_b", "camera_a"],
            "frame2": [11, 11, 10],
            "timestamp2_ns": [200, 200, 101],
        }
    )

    layout = _build_unique_endpoint_layout(matches)

    assert layout.keys == (
        ("camera_a", 10, 100),
        ("camera_a", 10, 101),
        ("camera_b", 10, 110),
        ("camera_b", 11, 200),
    )
    assert layout.first_indices.tolist() == [0, 0, 2]
    assert layout.second_indices.tolist() == [3, 3, 1]


def test_torch_unique_endpoint_composition_matches_per_observation_formula():
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    endpoint_body = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    endpoint_body[:, :3, 3] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, -0.1, 0.1]], dtype=dtype
    )
    body_delta = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    body_delta[:, 1, 3] = torch.tensor([0.01, -0.02], dtype=dtype)
    sensor_to_car = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    sensor_to_car[:, 0, 3] = torch.tensor([0.3, -0.4], dtype=dtype)
    frame_delta = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    frame_delta[:, 2, 3] = torch.tensor([0.02, 0.03, -0.01], dtype=dtype)
    endpoint_body_indices = torch.tensor([0, 1, 1])
    endpoint_camera_indices = torch.tensor([0, 1, 0])
    endpoint_frame_indices = torch.tensor([0, 1, 2])
    observation_endpoint_indices = torch.tensor([0, 1, 0, 2, 1])

    unique = _torch_compose_unique_endpoint_poses(
        endpoint_body,
        body_delta=body_delta,
        endpoint_body_indices=endpoint_body_indices,
        sensor_to_car=sensor_to_car,
        endpoint_camera_indices=endpoint_camera_indices,
        frame_delta=frame_delta,
        endpoint_frame_indices=endpoint_frame_indices,
    )
    actual = unique[observation_endpoint_indices]
    expected = (
        endpoint_body[observation_endpoint_indices]
        @ body_delta[endpoint_body_indices[observation_endpoint_indices]]
        @ sensor_to_car[endpoint_camera_indices[observation_endpoint_indices]]
        @ frame_delta[endpoint_frame_indices[observation_endpoint_indices]]
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_torch_group_means_match_loop_values_and_gradients_for_ragged_groups():
    torch = pytest.importorskip("torch")
    groups = [[0, 2, 4], [1], [3, 5]]
    layout = _build_padded_group_layout(groups)
    source = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2],
            [0.4, 0.5, 0.6],
            [2.0, 2.1, 2.2],
            [0.7, 0.8, 0.9],
            [3.0, 3.1, 3.2],
        ],
        dtype=torch.float64,
    )
    loop_values = source.clone().requires_grad_(True)
    vector_values = source.clone().requires_grad_(True)
    expected = torch.stack(
        [loop_values[torch.tensor(group)].mean(dim=0) for group in groups]
    )
    actual = _torch_group_means(
        vector_values,
        torch.as_tensor(layout.indices),
        torch.as_tensor(layout.valid_mask),
        torch.as_tensor(layout.counts, dtype=torch.float64),
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(vector_values.grad, loop_values.grad, rtol=1e-14, atol=1e-14)


def test_torch_sampson_data_term_uses_undistorted_normalized_points():
    torch = pytest.importorskip("torch")
    essential = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    normalized_first = torch.tensor([[0.1, 0.2, 1.0]], dtype=torch.float64)
    normalized_second = torch.tensor([[0.3, 0.2, 1.0]], dtype=torch.float64)
    distorted_pixel_first = torch.tensor([[1100.0, 1200.0, 1.0]], dtype=torch.float64)
    distorted_pixel_second = torch.tensor([[1300.0, 1250.0, 1.0]], dtype=torch.float64)

    normalized_error = _torch_sampson_sqrt_from_essential(
        essential,
        normalized_first,
        normalized_second,
    )
    pixel_error = _torch_sampson_sqrt_from_essential(
        essential,
        distorted_pixel_first,
        distorted_pixel_second,
    )

    assert float(normalized_error.item()) < 1e-5
    assert float(pixel_error.item()) > 10.0


def test_write_reprojection_outlier_diagnostics_exports_only_outlier_table(
    tmp_path: Path,
):
    observations = tmp_path / "observations"
    observations.mkdir(parents=True)
    legacy_names = (
        "before_matches.parquet",
        "after_matches.parquet",
        "after_matches_reprojection_inliers.parquet",
        "after_matches_reprojection_outliers.parquet",
    )
    for name in legacy_names:
        (observations / name).write_bytes(b"stale")

    before = pd.DataFrame(
        {
            "match_scope": ["intra_camera"] * 4,
            "camera1": ["camera_a"] * 4,
            "camera2": ["camera_a"] * 4,
            "frame_offset": [5, 5, 5, 5],
            "reprojection_rms_px": [2.0, 3.0, 4.0, 5.0],
            "nav_sampson_px2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    after = before.copy()
    after["reprojection_rms_px"] = [1.0, 9.0, 101.0, 700.0]

    diagnostics = _write_reprojection_outlier_diagnostics(
        output=tmp_path,
        before_matches=before,
        after_matches=after,
        hard_threshold_px=100.0,
    )

    assert diagnostics["hard_threshold_px"] == 100.0
    assert diagnostics["outlier_count"] == 2
    assert diagnostics["inlier_count"] == 2
    assert diagnostics["threshold_counts"][">100px"] == 2
    assert diagnostics["threshold_counts"][">500px"] == 1
    for name in legacy_names:
        assert not (observations / name).exists()
    assert (tmp_path / "observations" / "reprojection_outliers.parquet").exists()
    assert (tmp_path / "metrics" / "reprojection_outlier_groups.csv").exists()
    outliers = pd.read_parquet(tmp_path / "observations" / "reprojection_outliers.parquet")
    assert outliers["reprojection_rms_px"].min() > 100.0
    assert diagnostics["after_inliers"]["reprojection_rmse_px"] < diagnostics["after_all"][
        "reprojection_rmse_px"
    ]


def test_reprojection_qc_sampling_is_deterministic_and_stratified_by_relation():
    records = []
    for relation in ("same_index", "temporal_overlap_8m"):
        for frame in range(100):
            for match_index in range(2):
                records.append(
                    {
                        "match_scope": "inter_camera",
                        "camera1": "camera_a",
                        "camera2": "camera_b",
                        "pair_relation": relation,
                        "baseline_target_m": np.nan,
                        "frame1": frame,
                        "frame2": frame + 1,
                        "timestamp1_ns": frame * 100,
                        "timestamp2_ns": (frame + 1) * 100,
                        "frame_offset": 1,
                    }
                )
    matches = pd.DataFrame.from_records(records)

    first = _select_reprojection_qc_rows(matches, sample_ratio=0.01)
    second = _select_reprojection_qc_rows(matches, sample_ratio=0.01)

    assert first.tolist() == second.tolist()
    selected = matches.loc[first]
    assert set(selected["pair_relation"]) == {"same_index", "temporal_overlap_8m"}
    selected_pairs = selected.drop_duplicates(
        [
            "match_scope",
            "camera1",
            "camera2",
            "pair_relation",
            "frame1",
            "frame2",
            "timestamp1_ns",
            "timestamp2_ns",
            "frame_offset",
            "baseline_target_m",
        ]
    )
    assert len(selected_pairs) == 2


def test_build_reprojection_qc_matches_combines_before_and_after_fields():
    before = pd.DataFrame(
        {
            "match_scope": ["intra_camera"],
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "pair_relation": ["baseline_2m"],
            "baseline_target_m": [2.0],
            "frame1": [1],
            "frame2": [2],
            "timestamp1_ns": [100],
            "timestamp2_ns": [200],
            "frame_offset": [1],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "reprojected1_x": [11.0],
            "reprojected1_y": [21.0],
            "reprojected2_x": [31.0],
            "reprojected2_y": [41.0],
            "reprojection_rms_px": [2.5],
        }
    )
    after = before.copy()
    after[[
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "reprojection_rms_px",
    ]] = [[10.5, 20.5, 30.5, 40.5, 1.0]]

    compact = _build_reprojection_qc_matches(before, after, sample_ratio=1.0)

    assert len(compact) == 1
    assert compact.loc[0, "before_reprojection_rms_px"] == pytest.approx(2.5)
    assert compact.loc[0, "after_reprojection_rms_px"] == pytest.approx(1.0)
    assert compact.loc[0, "before_reprojected1_x"] == pytest.approx(11.0)
    assert compact.loc[0, "after_reprojected1_x"] == pytest.approx(10.5)


def test_absorb_camera_common_frame_disturbance_moves_common_mode_to_body():
    sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, 0.0, 0.1], [0.0, 0.1, 0.0]),
        "camera_b": _transform_from_rt([-0.2, 0.1, 0.0], [0.05, 0.0, -0.02]),
        "camera_c": _transform_from_rt([0.0, -0.2, 0.2], [-0.03, 0.02, 0.04]),
    }
    common_body = _transform_from_rt([0.03, -0.02, 0.04], [0.02, -0.03, 0.01])
    frame_rotvecs = {}
    for camera, transform in sensor_to_car.items():
        camera_delta = np.linalg.inv(transform) @ common_body @ transform
        frame_rotvecs[(camera, 7)] = _disturbance_from_transform(camera_delta, dof=6)
    body_frame_disturbances = {7: np.zeros(6, dtype=np.float64)}

    before_poses = {
        camera: _rotation_delta_transform(body_frame_disturbances[7])
        @ sensor_to_car[camera]
        @ _rotation_delta_transform(frame_rotvecs[(camera, 7)])
        for camera in sensor_to_car
    }

    new_frames, new_body, report = _absorb_camera_common_frame_disturbance(
        frame_rotvecs=frame_rotvecs,
        body_frame_disturbances=body_frame_disturbances,
        sensor_to_car_by_camera=sensor_to_car,
        min_cameras=3,
        min_median_cos=0.5,
        min_positive_projection_fraction=0.8,
        body_disturbance_dof=6,
    )

    assert report["frames_absorbed_rotation"] == 1
    assert report["frames_absorbed_translation"] == 1
    np.testing.assert_allclose(
        _rotation_delta_transform(new_body[7]),
        common_body,
        atol=1e-10,
    )
    residual_body_vectors = []
    for camera, transform in sensor_to_car.items():
        after_pose = (
            _rotation_delta_transform(new_body[7])
            @ transform
            @ _rotation_delta_transform(new_frames[(camera, 7)])
        )
        np.testing.assert_allclose(after_pose, before_poses[camera], atol=1e-10)
        residual_body = (
            transform
            @ _rotation_delta_transform(new_frames[(camera, 7)])
            @ np.linalg.inv(transform)
        )
        residual_body_vectors.append(_disturbance_from_transform(residual_body, dof=6))
    np.testing.assert_allclose(
        np.mean(np.vstack(residual_body_vectors), axis=0),
        np.zeros(6, dtype=np.float64),
        atol=1e-10,
    )


def test_camera_frame_common_mode_loss_penalizes_same_frame_shared_direction():
    coherent = {
        ("camera_a", 3): np.array([0.02, 0.0, 0.0, 0.01, 0.0, 0.0]),
        ("camera_b", 3): np.array([0.02, 0.0, 0.0, 0.01, 0.0, 0.0]),
        ("camera_c", 3): np.array([0.02, 0.0, 0.0, 0.01, 0.0, 0.0]),
    }
    cancelling = {
        ("camera_a", 3): np.array([0.02, 0.0, 0.0, 0.01, 0.0, 0.0]),
        ("camera_b", 3): np.array([-0.02, 0.0, 0.0, -0.01, 0.0, 0.0]),
        ("camera_c", 3): np.zeros(6, dtype=np.float64),
    }

    coherent_loss = _camera_frame_common_mode_loss_numpy(
        coherent,
        dof=6,
        rotation_weight=500.0,
        translation_weight=1000.0,
        min_cameras=2,
    )
    cancelling_loss = _camera_frame_common_mode_loss_numpy(
        cancelling,
        dof=6,
        rotation_weight=500.0,
        translation_weight=1000.0,
        min_cameras=2,
    )

    assert coherent_loss > 0.0
    assert cancelling_loss == pytest.approx(0.0)
    assert coherent_loss > cancelling_loss


def test_absorb_camera_common_frame_disturbance_skips_incoherent_frames():
    sensor_to_car = {
        "camera_a": np.eye(4, dtype=np.float64),
        "camera_b": np.eye(4, dtype=np.float64),
        "camera_c": np.eye(4, dtype=np.float64),
    }
    frame_rotvecs = {
        ("camera_a", 3): np.array([0.03, 0.0, 0.0, 0.02, 0.0, 0.0]),
        ("camera_b", 3): np.array([-0.03, 0.0, 0.0, -0.02, 0.0, 0.0]),
        ("camera_c", 3): np.array([0.0, 0.03, 0.0, 0.0, 0.02, 0.0]),
    }
    body_frame_disturbances = {3: np.zeros(6, dtype=np.float64)}

    new_frames, new_body, report = _absorb_camera_common_frame_disturbance(
        frame_rotvecs=frame_rotvecs,
        body_frame_disturbances=body_frame_disturbances,
        sensor_to_car_by_camera=sensor_to_car,
        min_cameras=3,
        min_median_cos=0.9,
        min_positive_projection_fraction=1.0,
        body_disturbance_dof=6,
    )

    assert report["frames_absorbed_rotation"] == 0
    assert report["frames_absorbed_translation"] == 0
    for key, value in frame_rotvecs.items():
        np.testing.assert_allclose(new_frames[key], value)
    np.testing.assert_allclose(new_body[3], body_frame_disturbances[3])


def test_absorb_camera_long_term_frame_disturbance_moves_stable_bias_to_fixed_extrinsic():
    sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, 0.0, 0.1], [0.0, 0.1, 0.0]),
        "camera_b": _transform_from_rt([-0.2, 0.1, 0.0], [0.05, 0.0, -0.02]),
    }
    long_term_delta = _transform_from_rt([0.02, -0.01, 0.03], [0.01, -0.02, 0.005])
    frame_rotvecs = {}
    before_products = {}
    for frame in range(5):
        small_residual = _transform_from_rt(
            [0.001 * frame, 0.0, 0.0],
            [0.0005 * frame, 0.0, 0.0],
        )
        camera_delta = long_term_delta @ small_residual
        frame_rotvecs[("camera_a", frame)] = _disturbance_from_transform(
            camera_delta,
            dof=6,
        )
        before_products[("camera_a", frame)] = sensor_to_car["camera_a"] @ camera_delta
        frame_rotvecs[("camera_b", frame)] = np.array(
            [0.02, 0.0, 0.0, 0.01, 0.0, 0.0]
            if frame % 2 == 0
            else [-0.02, 0.0, 0.0, -0.01, 0.0, 0.0],
            dtype=np.float64,
        )

    new_sensor_to_car, new_frames, report = _absorb_camera_long_term_frame_disturbance(
        frame_rotvecs=frame_rotvecs,
        sensor_to_car_by_camera=sensor_to_car,
        min_frames=5,
        min_median_cos=0.5,
        min_positive_projection_fraction=0.8,
        min_mean_to_rms=0.35,
    )

    assert report["cameras_absorbed_rotation"] == 1
    assert report["cameras_absorbed_translation"] == 1
    assert report["per_camera"][0]["camera"] == "camera_a"
    np.testing.assert_allclose(
        new_sensor_to_car["camera_a"],
        sensor_to_car["camera_a"] @ long_term_delta,
        atol=5e-3,
    )
    for frame in range(5):
        after_product = (
            new_sensor_to_car["camera_a"]
            @ _rotation_delta_transform(new_frames[("camera_a", frame)])
        )
        np.testing.assert_allclose(after_product, before_products[("camera_a", frame)], atol=1e-10)
    residual_vectors = np.vstack(
        [new_frames[("camera_a", frame)] for frame in range(5)]
    )
    assert np.linalg.norm(residual_vectors[:, :3].mean(axis=0)) < 0.002
    assert np.linalg.norm(residual_vectors[:, 3:6].mean(axis=0)) < 0.002


def test_absorb_camera_long_term_frame_disturbance_skips_incoherent_camera():
    sensor_to_car = {"camera_a": np.eye(4, dtype=np.float64)}
    frame_rotvecs = {
        ("camera_a", frame): np.array(
            [0.02, 0.0, 0.0, 0.01, 0.0, 0.0]
            if frame % 2 == 0
            else [-0.02, 0.0, 0.0, -0.01, 0.0, 0.0],
            dtype=np.float64,
        )
        for frame in range(6)
    }

    new_sensor_to_car, new_frames, report = _absorb_camera_long_term_frame_disturbance(
        frame_rotvecs=frame_rotvecs,
        sensor_to_car_by_camera=sensor_to_car,
        min_frames=5,
        min_median_cos=0.9,
        min_positive_projection_fraction=1.0,
        min_mean_to_rms=0.35,
    )

    assert report["cameras_absorbed_rotation"] == 0
    assert report["cameras_absorbed_translation"] == 0
    np.testing.assert_allclose(new_sensor_to_car["camera_a"], sensor_to_car["camera_a"])
    for key, value in frame_rotvecs.items():
        np.testing.assert_allclose(new_frames[key], value)


def test_project_time_varying_modes_moves_common_and_long_term_modes():
    frame_keys = [
        ("camera_a", 1),
        ("camera_b", 1),
        ("camera_a", 2),
        ("camera_b", 2),
    ]
    camera_to_index = {"camera_a": 0, "camera_b": 1}
    body_frame_to_index = {1: 0, 2: 1}
    fixed_parameters = np.zeros((2, 6), dtype=np.float64)
    frame_parameters = np.array(
        [
            [0.11, 0.02, 0.0, 0.21, 0.02, 0.0],
            [0.09, 0.04, 0.0, 0.19, 0.04, 0.0],
            [0.13, 0.02, 0.0, 0.23, 0.02, 0.0],
            [0.07, 0.04, 0.0, 0.17, 0.04, 0.0],
        ],
        dtype=np.float64,
    )
    body_parameters = np.zeros((2, 6), dtype=np.float64)
    before_total = (
        fixed_parameters[[camera_to_index[c] for c, _ in frame_keys]]
        + frame_parameters
        + body_parameters[[body_frame_to_index[f] for _, f in frame_keys]]
    )

    report = _project_time_varying_modes_numpy(
        fixed_parameters=fixed_parameters,
        frame_parameters=frame_parameters,
        body_parameters=body_parameters,
        frame_keys=frame_keys,
        camera_to_index=camera_to_index,
        body_frame_to_index=body_frame_to_index,
        frame_mean_groups=[[0, 2], [1, 3]],
        frame_common_groups=[[0, 1], [2, 3]],
        frame_dof=6,
        body_dof=6,
        absorb_frame_common=True,
        absorb_camera_long_term=True,
    )

    after_total = (
        fixed_parameters[[camera_to_index[c] for c, _ in frame_keys]]
        + frame_parameters
        + body_parameters[[body_frame_to_index[f] for _, f in frame_keys]]
    )
    np.testing.assert_allclose(after_total, before_total, atol=1e-12)
    np.testing.assert_allclose(frame_parameters[[0, 2]].mean(axis=0), np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(frame_parameters[[1, 3]].mean(axis=0), np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(frame_parameters[[0, 1]].mean(axis=0), np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(frame_parameters[[2, 3]].mean(axis=0), np.zeros(6), atol=1e-12)
    assert report["frame_common_projection_count"] == 2
    assert report["camera_long_term_projection_count"] == 2


def test_project_time_varying_modes_preserves_camera_to_world_products():
    sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, 0.02, -0.01]),
        "camera_b": _transform_from_rt([-0.3, 0.05, 0.2], [-0.02, 0.01, 0.03]),
    }
    frame_keys = [("camera_a", 4), ("camera_b", 4)]
    camera_to_index = {"camera_a": 0, "camera_b": 1}
    body_frame_to_index = {4: 0}
    body_common = _transform_from_rt([0.02, -0.01, 0.03], [0.01, -0.015, 0.005])
    long_term = _transform_from_rt([0.01, 0.02, -0.005], [0.004, -0.003, 0.002])
    fixed_parameters = np.zeros((2, 6), dtype=np.float64)
    frame_parameters = np.zeros((2, 6), dtype=np.float64)
    before_products = {}
    body_parameters = np.zeros((1, 6), dtype=np.float64)
    for index, (camera, _frame) in enumerate(frame_keys):
        transform = sensor_to_car[camera]
        camera_common = np.linalg.inv(transform) @ body_common @ transform
        frame_delta = camera_common @ long_term
        frame_parameters[index] = _disturbance_from_transform(frame_delta, dof=6)
        before_products[camera] = (
            _rotation_delta_transform(body_parameters[0])
            @ transform
            @ _rotation_delta_transform(frame_parameters[index])
        )

    _project_time_varying_modes_numpy(
        fixed_parameters=fixed_parameters,
        frame_parameters=frame_parameters,
        body_parameters=body_parameters,
        frame_keys=frame_keys,
        camera_to_index=camera_to_index,
        body_frame_to_index=body_frame_to_index,
        frame_mean_groups=[[0], [1]],
        frame_common_groups=[[0, 1]],
        frame_dof=6,
        body_dof=6,
        absorb_frame_common=True,
        absorb_camera_long_term=True,
        sensor_to_car_by_camera=sensor_to_car,
    )

    for index, (camera, _frame) in enumerate(frame_keys):
        after_sensor_to_car = (
            sensor_to_car[camera]
            @ _rotation_delta_transform(fixed_parameters[camera_to_index[camera]])
        )
        after_product = (
            _rotation_delta_transform(body_parameters[0])
            @ after_sensor_to_car
            @ _rotation_delta_transform(frame_parameters[index])
        )
        np.testing.assert_allclose(after_product, before_products[camera], atol=1e-10)


def test_sample_matches_per_image_pair_limits_each_pair_with_seed():
    rows = []
    for frame1, count in [(10, 8), (20, 3)]:
        for index in range(count):
            rows.append(
                {
                    "match_scope": "intra_camera",
                    "camera1": "camera_a",
                    "camera2": "camera_a",
                    "pair_relation": np.nan,
                    "baseline_target_m": 2.0,
                    "frame1": frame1,
                    "frame2": frame1 + 4,
                    "timestamp1_ns": frame1 * 1000,
                    "timestamp2_ns": (frame1 + 4) * 1000,
                    "frame_offset": 4,
                    "match_id": f"{frame1}_{index}",
                }
            )
    matches = pd.DataFrame(rows)

    sampled_a = _sample_matches_per_image_pair(matches, max_matches_per_pair=5, seed=123)
    sampled_b = _sample_matches_per_image_pair(matches, max_matches_per_pair=5, seed=123)
    sampled_c = _sample_matches_per_image_pair(matches, max_matches_per_pair=5, seed=456)

    counts = sampled_a.groupby(["frame1", "frame2"]).size().to_dict()
    assert counts[(10, 14)] == 5
    assert counts[(20, 24)] == 3
    assert sampled_a["match_id"].tolist() == sampled_b["match_id"].tolist()
    assert sampled_a.loc[sampled_a["frame1"].eq(10), "match_id"].tolist() != sampled_c.loc[
        sampled_c["frame1"].eq(10), "match_id"
    ].tolist()


def test_sample_matches_per_image_pair_spatial_grid_random_uses_both_views():
    rows = []
    for index in range(60):
        if index < 40:
            x1, y1 = 10.0 + index * 0.1, 10.0 + index * 0.1
        else:
            grid_index = index - 40
            x1 = 20.0 + (grid_index % 4) * 120.0
            y1 = 20.0 + (grid_index // 4) * 120.0
        if index < 40:
            x2 = 20.0 + (index % 5) * 110.0
            y2 = 20.0 + (index // 5) * 50.0
        else:
            x2, y2 = 15.0 + index * 0.1, 15.0 + index * 0.1
        rows.append(
            {
                "match_scope": "intra_camera",
                "camera1": "camera_a",
                "camera2": "camera_a",
                "pair_relation": np.nan,
                "baseline_target_m": 2.0,
                "frame1": 10,
                "frame2": 14,
                "timestamp1_ns": 10_000,
                "timestamp2_ns": 14_000,
                "frame_offset": 4,
                "match_id": f"m{index}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "backend_score": float(index),
                "reprojection_rms_px": float(1000 - index),
            }
        )
    matches = pd.DataFrame(rows)

    sampled_a = _sample_matches_per_image_pair(
        matches,
        max_matches_per_pair=18,
        seed=20260704,
        strategy="spatial_grid_random",
        grid_cols=3,
        grid_rows=3,
    )
    sampled_b = _sample_matches_per_image_pair(
        matches,
        max_matches_per_pair=18,
        seed=20260704,
        strategy="spatial_grid_random",
        grid_cols=3,
        grid_rows=3,
    )
    sampled_c = _sample_matches_per_image_pair(
        matches,
        max_matches_per_pair=18,
        seed=20260705,
        strategy="spatial_grid_random",
        grid_cols=3,
        grid_rows=3,
    )

    assert len(sampled_a) == 18
    assert sampled_a["match_id"].tolist() == sampled_b["match_id"].tolist()
    assert sampled_a["match_id"].tolist() != sampled_c["match_id"].tolist()

    def occupied_cells(frame: pd.DataFrame, x_name: str, y_name: str) -> set[tuple[int, int]]:
        x = frame[x_name].to_numpy(dtype=np.float64)
        y = frame[y_name].to_numpy(dtype=np.float64)
        x_bins = np.linspace(float(x.min()), float(x.max()), 4)
        y_bins = np.linspace(float(y.min()), float(y.max()), 4)
        cols = np.clip(np.searchsorted(x_bins, x, side="right") - 1, 0, 2)
        rows_ = np.clip(np.searchsorted(y_bins, y, side="right") - 1, 0, 2)
        return set(zip(cols.tolist(), rows_.tolist()))

    assert len(occupied_cells(sampled_a, "x1", "y1")) >= 3
    assert len(occupied_cells(sampled_a, "x2", "y2")) >= 3


def test_sample_multicamera_matches_uses_explicit_seed():
    rows = []
    for scope, count in [("intra_camera", 12), ("inter_camera", 12)]:
        for index in range(count):
            rows.append(
                {
                    "match_scope": scope,
                    "camera1": "camera_a",
                    "camera2": "camera_b" if scope == "inter_camera" else "camera_a",
                    "frame_offset": 1 if scope == "intra_camera" else 0,
                    "match_id": f"{scope}_{index}",
                }
            )
    matches = pd.DataFrame(rows)

    sampled_a = extrinsic_optimization._sample_multicamera_matches(
        matches, max_observations=8, seed=20260704
    )
    sampled_b = extrinsic_optimization._sample_multicamera_matches(
        matches, max_observations=8, seed=20260704
    )
    sampled_c = extrinsic_optimization._sample_multicamera_matches(
        matches, max_observations=8, seed=20260705
    )

    assert sampled_a["match_id"].tolist() == sampled_b["match_id"].tolist()
    assert sampled_a["match_id"].tolist() != sampled_c["match_id"].tolist()
    assert sampled_a.groupby("match_scope").size().to_dict() == {
        "inter_camera": 4,
        "intra_camera": 4,
    }


def test_select_optimization_matches_prefers_precision_filter_valid():
    matches = pd.DataFrame(
        {
            "pose_quality_valid": [False, True, True],
            "reprojection_threshold_inlier": [False, True, False],
            "precision_filter_valid": [True, False, True],
        }
    )

    selected = extrinsic_optimization._select_optimization_matches(matches)

    assert selected.index.tolist() == [0, 2]


def test_multi_camera_optimization_records_pair_sampling_settings(tmp_path: Path):
    navigation = _make_navigation()
    sensor_to_car_a = _transform_from_rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_to_car_b = _transform_from_rt([0.2, 0.0, 0.0], [0.0, 0.0, 0.02])
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", sensor_to_car_a),
        "camera_b": _make_named_calibration("camera_b", sensor_to_car_b),
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        {"camera_a": sensor_to_car_a, "camera_b": sensor_to_car_b},
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        pair_sample_max_matches=3,
        pair_sample_seed=42,
        pair_sample_strategy="spatial_grid_random",
        pair_sample_grid_cols=4,
        pair_sample_grid_rows=2,
        evaluation_max_observations=20,
        optimizer_method="scipy_least_squares",
        max_nfev=1,
    )

    assert result.solver_status["pair_sample_max_matches"] == 3
    assert result.solver_status["pair_sample_seed"] == 42
    assert result.solver_status["pair_sample_strategy"] == "spatial_grid_random"
    assert result.solver_status["pair_sample_grid_cols"] == 4
    assert result.solver_status["pair_sample_grid_rows"] == 2
    assert result.solver_status["pair_sampled_observations"] <= result.solver_status[
        "retained_observations"
    ]


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


def _make_named_calibration(camera: str, sensor_to_car: np.ndarray) -> CameraCalibration:
    base = _make_calibration(sensor_to_car)
    return replace(base, camera=camera)


def _make_navigation() -> NavigationTrajectory:
    timestamps_ns = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.2, 0.0]],
        dtype=np.float64,
    )
    quaternions_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    return NavigationTrajectory(timestamps_ns, positions, quaternions_xyzw)


def _make_synthetic_matches(
    calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    true_sensor_to_car: np.ndarray,
) -> pd.DataFrame:
    world_points = np.array(
        [
            [1.0, -0.8, 8.0],
            [1.5, -0.2, 10.0],
            [2.0, 0.5, 9.0],
            [2.5, 0.8, 11.0],
            [3.0, -0.5, 12.0],
            [3.5, 0.2, 13.0],
            [4.0, 0.7, 14.0],
            [4.5, -0.6, 15.0],
        ],
        dtype=np.float64,
    )
    records: list[dict[str, float | int | str | bool]] = []
    pair_specs = [(0, 1, 1), (1, 2, 1), (0, 2, 2)]
    for first_idx, second_idx, offset in pair_specs:
        first_body = navigation.pose_at(int(navigation.timestamps_ns[first_idx]))
        second_body = navigation.pose_at(int(navigation.timestamps_ns[second_idx]))
        first_world_to_camera = np.linalg.inv(first_body @ true_sensor_to_car)
        second_world_to_camera = np.linalg.inv(second_body @ true_sensor_to_car)
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
                    "frame_offset": offset,
                    "frame1": first_idx,
                    "frame2": second_idx,
                    "timestamp1_ns": int(navigation.timestamps_ns[first_idx]),
                    "timestamp2_ns": int(navigation.timestamps_ns[second_idx]),
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


def _project_world_points(
    world_points: np.ndarray,
    calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    timestamp_ns: int,
    sensor_to_car: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_to_world = navigation.pose_at(timestamp_ns) @ sensor_to_car
    world_to_camera = np.linalg.inv(camera_to_world)
    world_h = np.column_stack([world_points, np.ones(len(world_points))])
    points_camera = (world_to_camera @ world_h.T).T[:, :3]
    pixels = project_points(points_camera, calibration)
    normalized = points_camera[:, :2] / points_camera[:, 2, None]
    return pixels, normalized


def _make_multi_camera_matches(
    calibrations: dict[str, CameraCalibration],
    navigation: NavigationTrajectory,
    true_sensor_to_car: dict[str, np.ndarray],
) -> pd.DataFrame:
    world_points = np.array(
        [
            [2.0, -0.8, 8.0],
            [2.5, -0.2, 10.0],
            [3.0, 0.5, 9.0],
            [3.5, 0.8, 11.0],
            [4.0, -0.5, 12.0],
            [4.5, 0.2, 13.0],
            [5.0, 0.7, 14.0],
            [5.5, -0.6, 15.0],
            [6.0, 0.1, 16.0],
            [6.5, 0.4, 17.0],
        ],
        dtype=np.float64,
    )
    records: list[dict[str, float | int | str | bool]] = []

    for camera in ("camera_a", "camera_b"):
        for first_idx, second_idx, offset in [(0, 1, 1), (1, 2, 1), (0, 2, 2)]:
            pixels1, norm1 = _project_world_points(
                world_points,
                calibrations[camera],
                navigation,
                int(navigation.timestamps_ns[first_idx]),
                true_sensor_to_car[camera],
            )
            pixels2, norm2 = _project_world_points(
                world_points,
                calibrations[camera],
                navigation,
                int(navigation.timestamps_ns[second_idx]),
                true_sensor_to_car[camera],
            )
            for point_idx in range(len(world_points)):
                records.append(
                    {
                        "match_scope": "intra_camera",
                        "camera1": camera,
                        "camera2": camera,
                        "camera": camera,
                        "frame_offset": offset,
                        "frame1": first_idx,
                        "frame2": second_idx,
                        "timestamp1_ns": int(navigation.timestamps_ns[first_idx]),
                        "timestamp2_ns": int(navigation.timestamps_ns[second_idx]),
                        "x1": float(pixels1[point_idx, 0]),
                        "y1": float(pixels1[point_idx, 1]),
                        "x2": float(pixels2[point_idx, 0]),
                        "y2": float(pixels2[point_idx, 1]),
                        "nx1": float(norm1[point_idx, 0]),
                        "ny1": float(norm1[point_idx, 1]),
                        "nx2": float(norm2[point_idx, 0]),
                        "ny2": float(norm2[point_idx, 1]),
                        "pose_quality_valid": True,
                        "reprojection_threshold_inlier": True,
                    }
                )

    for frame_idx in (0, 1, 2):
        pixels1, norm1 = _project_world_points(
            world_points,
            calibrations["camera_a"],
            navigation,
            int(navigation.timestamps_ns[frame_idx]),
            true_sensor_to_car["camera_a"],
        )
        pixels2, norm2 = _project_world_points(
            world_points,
            calibrations["camera_b"],
            navigation,
            int(navigation.timestamps_ns[frame_idx]),
            true_sensor_to_car["camera_b"],
        )
        for point_idx in range(len(world_points)):
            records.append(
                {
                    "match_scope": "inter_camera",
                    "camera1": "camera_a",
                    "camera2": "camera_b",
                    "frame_offset": 0,
                    "frame1": frame_idx,
                    "frame2": frame_idx,
                    "timestamp1_ns": int(navigation.timestamps_ns[frame_idx]),
                    "timestamp2_ns": int(navigation.timestamps_ns[frame_idx]),
                    "x1": float(pixels1[point_idx, 0]),
                    "y1": float(pixels1[point_idx, 1]),
                    "x2": float(pixels2[point_idx, 0]),
                    "y2": float(pixels2[point_idx, 1]),
                    "nx1": float(norm1[point_idx, 0]),
                    "ny1": float(norm1[point_idx, 1]),
                    "nx2": float(norm2[point_idx, 0]),
                    "ny2": float(norm2[point_idx, 1]),
                    "pose_quality_valid": True,
                    "reprojection_threshold_inlier": True,
                }
            )
    return pd.DataFrame.from_records(records)


def test_optimize_multi_camera_extrinsics_uses_intra_and_inter_edges(tmp_path: Path):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    synthetic_projection_calibrations = {
        camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
        for camera in calibrations
    }
    matches = _make_multi_camera_matches(
        synthetic_projection_calibrations,
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=25,
        evaluation_max_observations=30,
        torch_device="cpu",
        torch_learning_rate=0.01,
    )

    assert set(result.optimized_sensor_to_car) == {"camera_a", "camera_b"}
    assert len(result.before_matches) == 30
    assert len(result.after_matches) == 30
    assert result.before_summary["reprojection_median_px"] > result.after_summary[
        "reprojection_median_px"
    ]
    assert result.before_summary["sampson_mean"] > result.after_summary["sampson_mean"]
    assert set(result.edge_summary["match_scope"]) == {"intra_camera", "inter_camera"}
    assert (tmp_path / "optimized_multi_camera_extrinsics.yaml").exists()
    assert (tmp_path / "metrics" / "camera_delta_summary.csv").exists()
    assert (tmp_path / "observations" / "qc_matches.parquet").exists()
    assert not (tmp_path / "observations" / "before_matches.parquet").exists()
    assert not (tmp_path / "observations" / "after_matches.parquet").exists()
    assert not (tmp_path / "observations" / "after_matches_reprojection_inliers.parquet").exists()
    assert result.solver_status["max_nfev"] == 25
    assert result.solver_status["solver_method"] == "torch_sparse_graph_full_batch"


def test_optimize_multi_camera_extrinsics_can_optimize_time_varying_disturbance(
    tmp_path: Path,
):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    synthetic_projection_calibrations = {
        camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
        for camera in calibrations
    }
    matches = _make_multi_camera_matches(
        synthetic_projection_calibrations,
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        max_nfev=5,
        evaluation_max_observations=30,
        torch_device="cpu",
        torch_learning_rate=0.005,
        optimize_time_varying_disturbance=True,
        time_varying_disturbance_dof=6,
        optimize_body_time_varying_disturbance=True,
        body_time_varying_disturbance_dof=6,
        time_varying_rotation_bound_deg=1.0,
        time_varying_translation_bound_m=0.05,
        fixed_translation_l2_weight=0.2,
        fixed_rotation_l2_weight=0.3,
        time_varying_translation_l2_weight=0.4,
        time_varying_rotation_l2_weight=0.5,
        time_varying_translation_smooth_l2_weight=0.6,
        time_varying_rotation_smooth_l2_weight=0.7,
        time_varying_translation_mean_l2_weight=0.8,
        time_varying_rotation_mean_l2_weight=0.9,
        mode_projection_enabled=True,
        mode_projection_interval_steps=2,
    )

    disturbance = pd.read_csv(tmp_path / "metrics" / "frame_disturbance_summary.csv")
    assert result.solver_status["time_varying_frame_variable_count"] == len(disturbance)
    assert result.solver_status["time_varying_disturbance_dof"] == 6
    assert result.solver_status["body_time_varying_frame_variable_count"] > 0
    assert result.solver_status["body_time_varying_disturbance_dof"] == 6
    assert result.solver_status["unique_endpoint_count"] == 6
    assert result.solver_status["observation_endpoint_count"] == 2 * len(matches)
    assert result.solver_status["navigation_pose_evaluation_count"] == 6
    assert result.solver_status["fixed_translation_l2_weight"] == pytest.approx(0.2)
    assert result.solver_status["fixed_rotation_l2_weight"] == pytest.approx(0.3)
    assert result.solver_status["time_varying_translation_l2_weight"] == pytest.approx(0.4)
    assert result.solver_status["time_varying_rotation_l2_weight"] == pytest.approx(0.5)
    assert result.solver_status["time_varying_translation_smooth_l2_weight"] == pytest.approx(0.6)
    assert result.solver_status["time_varying_rotation_smooth_l2_weight"] == pytest.approx(0.7)
    assert result.solver_status["time_varying_translation_mean_l2_weight"] == pytest.approx(0.8)
    assert result.solver_status["time_varying_rotation_mean_l2_weight"] == pytest.approx(0.9)
    assert result.solver_status["mode_projection_enabled"] is True
    assert result.solver_status["mode_projection_interval_steps"] == 2
    assert result.solver_status["mode_projection_count"] == 2
    assert result.solver_status["mode_projection_frame_common_projection_count"] > 0
    assert result.solver_status["mode_projection_camera_long_term_projection_count"] > 0
    assert result.solver_status["absorb_camera_common_mode"] is True
    assert "camera_common_mode_absorption" in result.solver_status
    assert result.solver_status["camera_common_mode_absorption"]["enabled"] is True
    assert result.solver_status["absorb_camera_long_term_mode"] is True
    assert "camera_long_term_mode_absorption" in result.solver_status
    assert result.solver_status["camera_long_term_mode_absorption"]["enabled"] is True
    assert "translation_prior_sigma_m" not in result.solver_status
    assert "rotation_prior_sigma_deg" not in result.solver_status
    assert "time_varying_prior_sigma_deg" not in result.solver_status
    assert "time_varying_translation_prior_sigma_m" not in result.solver_status
    assert "time_varying_smooth_sigma_deg" not in result.solver_status
    assert "time_varying_translation_smooth_sigma_m" not in result.solver_status
    assert {
        "camera",
        "frame",
        "delta_norm_deg",
        "delta_tx_m",
        "delta_ty_m",
        "delta_tz_m",
        "delta_translation_norm_m",
    } <= set(disturbance.columns)
    assert "after_frame_delta_norm_deg" in result.after_matches.columns
    assert "after_body_delta_norm_deg" in result.after_matches.columns
    assert np.isfinite(result.after_matches["after_frame_delta_norm_deg"]).all()
    assert np.isfinite(result.after_matches["after_body_delta_norm_deg"]).all()
    body_disturbance = pd.read_csv(
        tmp_path / "metrics" / "body_pose_disturbance_summary.csv"
    )
    assert {"frame", "delta_tx_m", "delta_translation_norm_m"} <= set(
        body_disturbance.columns
    )


def test_multi_camera_torch_optimizer_supports_step_learning_rate_schedule(
    tmp_path: Path,
):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        {
            camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
            for camera in calibrations
        },
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=7,
        evaluation_max_observations=30,
        torch_device="cpu",
        torch_learning_rate=0.003,
        torch_lr_schedule="step",
        torch_lr_step_size=3,
        torch_lr_decay=0.5,
        torch_lr_min_factor=0.25,
    )

    assert result.solver_status["lr_schedule"] == "step"
    assert result.solver_status["learning_rate_first_steps"][:3] == [0.003, 0.003, 0.003]
    assert result.solver_status["learning_rate_last_steps"][-1] == pytest.approx(0.00075)


def test_multi_camera_torch_optimizer_reports_torch_dtype(tmp_path: Path):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        {
            camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
            for camera in calibrations
        },
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=3,
        evaluation_max_observations=30,
        torch_device="cpu",
        torch_dtype="float32",
    )

    assert result.solver_status["torch_dtype"] == "float32"


def test_multi_camera_legacy_evaluation_caches_navigation_poses_per_call():
    base_navigation = _make_navigation()

    class CountingNavigation(NavigationTrajectory):
        def __init__(self, base: NavigationTrajectory):
            super().__init__(
                base.timestamps_ns,
                base.positions,
                base.quaternions_xyzw,
            )
            self.pose_at_call_count = 0

        def pose_at(self, timestamp_ns: int) -> np.ndarray:
            self.pose_at_call_count += 1
            return super().pose_at(timestamp_ns)

    navigation = CountingNavigation(base_navigation)
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, true_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        true_sensor_to_car,
    )
    navigation.pose_at_call_count = 0

    evaluated = _evaluate_multicamera_matches_legacy(
        matches,
        navigation,
        calibrations,
        true_sensor_to_car,
    )

    assert len(evaluated) == len(matches)
    expected_pose_count = len(
        set(matches["timestamp1_ns"].to_numpy()) | set(matches["timestamp2_ns"].to_numpy())
    )
    assert navigation.pose_at_call_count == expected_pose_count


def test_multicamera_evaluation_torch_backend_matches_legacy_for_undistorted_data():
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, true_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        true_sensor_to_car,
    )

    legacy = _evaluate_multicamera_matches_legacy(
        matches,
        navigation,
        calibrations,
        true_sensor_to_car,
    )
    accelerated = _evaluate_multicamera_matches(
        matches,
        navigation,
        calibrations,
        true_sensor_to_car,
        evaluation_backend="torch",
        evaluation_device="cpu",
        evaluation_batch_size=16,
    )

    assert list(accelerated.columns) == list(legacy.columns)
    for column in (
        "nav_sampson_px2",
        "symmetric_epipolar_px",
        "triangulation_angle_deg",
        "ray_closest_distance_m",
        "reprojection1_px",
        "reprojection2_px",
        "reprojection_rms_px",
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
    ):
        np.testing.assert_allclose(
            accelerated[column].to_numpy(dtype=np.float64),
            legacy[column].to_numpy(dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
        )
    assert accelerated["triangulation_valid_depth"].tolist() == legacy[
        "triangulation_valid_depth"
    ].tolist()


def test_auto_evaluation_backend_prefers_cuda_torch_when_supported(monkeypatch):
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", np.eye(4, dtype=np.float64)),
    }
    monkeypatch.setattr(
        extrinsic_optimization,
        "_torch_device_available",
        lambda device: str(device).startswith("cuda"),
    )

    assert (
        _resolve_multicamera_evaluation_backend(
            "auto",
            evaluation_device="cuda:0",
            calibrations=calibrations,
        )
        == "torch"
    )


def test_auto_evaluation_backend_falls_back_to_legacy_without_cuda(monkeypatch):
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", np.eye(4, dtype=np.float64)),
    }
    monkeypatch.setattr(
        extrinsic_optimization,
        "_torch_device_available",
        lambda _device: False,
    )

    assert (
        _resolve_multicamera_evaluation_backend(
            "auto",
            evaluation_device="cuda:0",
            calibrations=calibrations,
        )
        == "legacy"
    )


def test_auto_torch_evaluation_refines_large_reprojection_candidates_with_legacy(
    monkeypatch,
):
    matches = pd.DataFrame(
        {
            "camera1": ["camera_a", "camera_a"],
            "camera2": ["camera_a", "camera_a"],
            "frame1": [0, 0],
            "frame2": [1, 1],
            "timestamp1_ns": [0, 0],
            "timestamp2_ns": [1, 1],
            "reprojection_rms_px": [999.0, 999.0],
            "nav_sampson_px2": [999.0, 999.0],
        }
    )
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", np.eye(4, dtype=np.float64)),
    }
    torch_result = matches.copy()
    torch_result["reprojection_rms_px"] = [2.0, 15.0]
    torch_result["nav_sampson_px2"] = [1.0, 225.0]
    legacy_calls = []

    def fake_legacy(candidate_matches, *_args, **_kwargs):
        legacy_calls.append(candidate_matches.copy())
        result = candidate_matches.reset_index(drop=True).copy()
        result["reprojection_rms_px"] = 8.0
        result["nav_sampson_px2"] = 4.0
        return result

    monkeypatch.setattr(
        extrinsic_optimization,
        "_torch_device_available",
        lambda _device: True,
    )
    monkeypatch.setattr(
        extrinsic_optimization,
        "_evaluate_multicamera_matches_torch",
        lambda *_args, **_kwargs: torch_result.copy(),
    )
    monkeypatch.setattr(
        extrinsic_optimization,
        "_evaluate_multicamera_matches_legacy",
        fake_legacy,
    )

    evaluated = _evaluate_multicamera_matches(
        matches,
        _make_navigation(),
        calibrations,
        {"camera_a": np.eye(4, dtype=np.float64)},
        evaluation_backend="auto",
        evaluation_device="cuda:0",
    )

    assert len(legacy_calls) == 1
    assert len(legacy_calls[0]) == 1
    assert evaluated["reprojection_rms_px"].tolist() == [2.0, 8.0]
    assert evaluated.attrs["legacy_refinement_count"] == 1


def test_vectorized_multicamera_edge_summary_matches_groupwise_reference():
    before = pd.DataFrame(
        {
            "match_scope": ["intra_camera"] * 4 + ["inter_camera"] * 3,
            "camera1": ["a"] * 4 + ["a"] * 3,
            "camera2": ["a"] * 4 + ["b"] * 3,
            "frame_offset": [2] * 4 + [0] * 3,
            "nav_sampson_px2": [1.0, 4.0, 9.0, 16.0, 1.0, 9.0, 25.0],
            "reprojection_rms_px": [2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 9.0],
        }
    )
    after = before.copy()
    after["nav_sampson_px2"] = before["nav_sampson_px2"] * 0.25
    after["reprojection_rms_px"] = before["reprojection_rms_px"] * 0.5

    actual = _summarize_multicamera_edges(before, after).sort_values(
        ["match_scope", "camera1", "camera2", "frame_offset"]
    ).reset_index(drop=True)

    records = []
    group_keys = ["match_scope", "camera1", "camera2", "frame_offset"]
    for key, before_group in before.groupby(group_keys, sort=True):
        after_group = after.loc[before_group.index]
        before_summary = extrinsic_optimization._summarize_match_quality(before_group)
        after_summary = extrinsic_optimization._summarize_match_quality(after_group)
        records.append(
            {
                "match_scope": key[0],
                "camera1": key[1],
                "camera2": key[2],
                "frame_offset": key[3],
                "count": before_summary["count"],
                "before_sampson_mean": before_summary["sampson_mean"],
                "after_sampson_mean": after_summary["sampson_mean"],
                "before_reprojection_median_px": before_summary[
                    "reprojection_median_px"
                ],
                "after_reprojection_median_px": after_summary[
                    "reprojection_median_px"
                ],
                "before_reprojection_p95_px": before_summary[
                    "reprojection_p95_px"
                ],
                "after_reprojection_p95_px": after_summary[
                    "reprojection_p95_px"
                ],
            }
        )
    expected = pd.DataFrame.from_records(records).sort_values(group_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_multi_camera_extrinsics_reports_phase_timings(tmp_path: Path):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        {
            camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
            for camera in calibrations
        },
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=3,
        evaluation_max_observations=30,
        torch_device="cpu",
    )

    timings = result.solver_status["phase_timings_seconds"]
    assert {
        "prepare_inputs",
        "before_evaluation",
        "adaptive_weights",
        "optimization",
        "after_evaluation",
        "reporting_and_export",
        "total",
    } <= set(timings)
    assert timings["total"] >= timings["optimization"] >= 0.0
    persisted = json.loads((tmp_path / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert persisted["solver"]["phase_timings_seconds"]["total"] == pytest.approx(
        timings["total"]
    )


def test_multi_camera_torch_optimizer_can_stop_early(tmp_path: Path):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, true_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=50,
        evaluation_max_observations=30,
        torch_device="cpu",
        torch_early_stopping_patience=2,
        torch_early_stopping_min_delta=1.0,
        torch_early_stopping_min_steps=0,
    )

    assert result.solver_status["stopped_early"] is True
    assert result.solver_status["nfev"] < 50


def test_multi_camera_extrinsics_exports_static_adaptive_group_weights(tmp_path: Path):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    initial_sensor_to_car = {
        "camera_a": _transform_from_rt([0.35, -0.05, 0.18], [0.04, -0.05, 0.02]),
        "camera_b": _transform_from_rt([-0.10, 0.05, 0.2], [-0.04, 0.04, -0.03]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, initial_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        {
            camera: replace(calibrations[camera], sensor_to_car=true_sensor_to_car[camera])
            for camera in calibrations
        },
        navigation,
        true_sensor_to_car,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        max_nfev=3,
        evaluation_max_observations=30,
        torch_device="cpu",
        adaptive_weight_mode="group_static",
        adaptive_weight_min_group_count=1,
    )

    weights = pd.read_csv(tmp_path / "metrics" / "adaptive_group_weights.csv")
    assert result.solver_status["adaptive_weight_mode"] == "group_static"
    assert {"group_key", "weight", "median_reprojection_px"} <= set(weights.columns)
    assert len(weights) > 0


def test_multi_camera_adaptive_weights_reuse_before_evaluation_when_unsampled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    navigation = _make_navigation()
    true_sensor_to_car = {
        "camera_a": _transform_from_rt([0.2, -0.1, 0.1], [0.01, -0.02, 0.005]),
        "camera_b": _transform_from_rt([-0.25, 0.15, 0.12], [-0.015, 0.01, -0.008]),
    }
    calibrations = {
        camera: _make_named_calibration(camera, true_sensor_to_car[camera])
        for camera in ("camera_a", "camera_b")
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        true_sensor_to_car,
    )

    original_evaluate = extrinsic_optimization._evaluate_multicamera_matches
    evaluate_call_count = 0

    def counting_evaluate(*args, **kwargs):
        nonlocal evaluate_call_count
        evaluate_call_count += 1
        return original_evaluate(*args, **kwargs)

    def fake_torch_optimizer(*args, **kwargs):
        cameras = kwargs["cameras"]
        return (
            np.zeros(len(cameras) * 6, dtype=np.float64),
            {},
            {},
            {
                "success": True,
                "status": 0,
                "message": "fake optimizer",
                "nfev": 0,
                "cost": 0.0,
                "solver_method": "torch_sparse_graph",
            },
        )

    monkeypatch.setattr(
        extrinsic_optimization,
        "_evaluate_multicamera_matches",
        counting_evaluate,
    )
    monkeypatch.setattr(
        extrinsic_optimization,
        "_torch_optimize_multi_camera_extrinsics",
        fake_torch_optimizer,
    )

    result = optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        max_observations=None,
        evaluation_max_observations=None,
        translation_bound_m=1.0,
        rotation_bound_deg=5.0,
        fixed_translation_l2_weight=1e-3,
        fixed_rotation_l2_weight=1e-3,
        adaptive_weight_mode="group_static",
        adaptive_weight_min_group_count=1,
        torch_device="cpu",
    )

    assert evaluate_call_count == 2
    assert result.solver_status["adaptive_weights_reused_before_evaluation"] is True


def test_static_adaptive_group_weights_boost_bad_groups_but_guard_long_tails():
    evaluated = pd.DataFrame(
        {
            "match_scope": ["intra_camera"] * 6 + ["inter_camera"] * 6 + ["intra_camera"] * 6,
            "camera1": ["camera_a"] * 6 + ["camera_a"] * 6 + ["camera_c"] * 6,
            "camera2": ["camera_a"] * 6 + ["camera_b"] * 6 + ["camera_c"] * 6,
            "frame_offset": [5] * 6 + [0] * 6 + [5] * 6,
            "reprojection_rms_px": [
                1.0,
                1.0,
                1.0,
                1.2,
                1.2,
                1.2,
                4.0,
                4.0,
                4.0,
                4.2,
                4.2,
                4.2,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                30.0,
            ],
        }
    )

    weights, diagnostics = _compute_static_adaptive_group_weights(
        evaluated,
        alpha=0.5,
        min_weight=0.5,
        max_weight=3.0,
        long_tail_ratio_guard=4.0,
        min_group_count=6,
    )

    good = diagnostics.loc[diagnostics["group_key"] == "intra_camera|camera_a|camera_a|5"].iloc[0]
    bad = diagnostics.loc[diagnostics["group_key"] == "inter_camera|camera_a|camera_b|0"].iloc[0]
    long_tail = diagnostics.loc[
        diagnostics["group_key"] == "intra_camera|camera_c|camera_c|5"
    ].iloc[0]
    assert bad["weight"] > good["weight"]
    assert bad["weight"] <= 3.0
    assert long_tail["weight"] == pytest.approx(1.0)
    assert weights[evaluated["camera2"].eq("camera_b").to_numpy()].mean() == pytest.approx(
        bad["weight"]
    )


def test_alternating_optimization_orders_blocks_by_current_intra_error_and_carries_forward(
    monkeypatch, tmp_path: Path
):
    navigation = _make_navigation()
    calibrations = {
        camera: _make_named_calibration(camera, np.eye(4))
        for camera in ("camera_a", "camera_b")
    }
    matches = pd.DataFrame(
        {
            "match_scope": ["intra_camera", "intra_camera", "inter_camera"],
            "camera1": ["camera_a", "camera_b", "camera_a"],
            "camera2": ["camera_a", "camera_b", "camera_b"],
            "frame_offset": [1, 1, 0],
            "frame1": [0, 0, 0],
            "frame2": [1, 1, 0],
            "timestamp1_ns": [0, 0, 0],
            "timestamp2_ns": [1_000_000_000, 1_000_000_000, 0],
            "x1": [1.0, 1.0, 1.0],
            "y1": [1.0, 1.0, 1.0],
            "x2": [2.0, 2.0, 2.0],
            "y2": [2.0, 2.0, 2.0],
            "nx1": [0.1, 0.1, 0.1],
            "ny1": [0.1, 0.1, 0.1],
            "nx2": [0.2, 0.2, 0.2],
            "ny2": [0.2, 0.2, 0.2],
            "pose_quality_valid": [True, True, True],
            "reprojection_threshold_inlier": [True, True, True],
        }
    )
    calls: list[tuple[str, dict[tuple[str, int], np.ndarray]]] = []

    def fake_evaluate(matches_arg, *_args, frame_rotvecs=None, **_kwargs):
        result = matches_arg.copy()
        result["reprojection_rms_px"] = np.where(
            result["camera1"].astype(str) == "camera_b",
            10.0,
            2.0,
        )
        if frame_rotvecs:
            result["reprojection_rms_px"] = result["reprojection_rms_px"] - 1.0
        result["nav_sampson_px2"] = result["reprojection_rms_px"]
        return result

    def fake_block(*, target_camera, initial_frame_rotvecs, **_kwargs):
        calls.append(
            (
                target_camera,
                {key: value.copy() for key, value in initial_frame_rotvecs.items()},
            )
        )
        return {
            **initial_frame_rotvecs,
            (target_camera, 0): np.array([0.01, 0.0, 0.0], dtype=np.float64),
        }, {"target_camera": target_camera}

    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization._evaluate_multicamera_matches",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization._optimize_single_camera_time_varying_block",
        fake_block,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        lambda **kwargs: None,
    )

    result = optimize_multi_camera_alternating_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path,
        block_rounds=1,
        block_max_nfev=2,
        run_global_refine=False,
        torch_device="cpu",
    )

    assert result.solver_status["block_order"] == ["camera_b", "camera_a"]
    assert [camera for camera, _seen in calls] == ["camera_b", "camera_a"]
    assert calls[0][1] == {}
    assert ("camera_b", 0) in calls[1][1]
    assert result.solver_status["global_refine_output"] is None


def test_optimize_multi_camera_extrinsics_from_match_cache_loads_intra_and_inter(
    monkeypatch, tmp_path: Path
):
    cache_root = tmp_path / "data_cache" / "matches" / "cache"
    intra = cache_root / "intra_camera" / "camera_a" / "offsets_5"
    inter = cache_root / "inter_camera" / "camera_a__camera_b" / "same_index"
    intra.mkdir(parents=True)
    inter.mkdir(parents=True)
    base = pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "camera": ["camera_a"],
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [5],
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
    base.to_parquet(intra / "retained_matches.parquet", index=False)
    base.assign(camera2="camera_b", frame_offset=0).to_parquet(
        inter / "retained_matches.parquet",
        index=False,
    )
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, calibrations, **kwargs):
        seen["matches"] = matches.copy()
        seen["calibrations"] = calibrations
        seen["kwargs"] = kwargs
        return None

    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[5],
        inter_pairs=["camera_a__camera_b"],
        max_observations=10,
        max_nfev=7,
        evaluation_max_observations=8,
        torch_device="cpu",
        torch_dtype="float32",
        torch_early_stopping_patience=9,
        torch_early_stopping_min_delta=1e-4,
        torch_early_stopping_min_steps=25,
        evaluation_backend="torch",
        evaluation_device="cuda:0",
        evaluation_batch_size=1234,
        optimize_time_varying_disturbance=True,
    )

    assert set(seen["calibrations"]) == {"camera_a", "camera_b"}
    assert set(seen["matches"]["match_scope"]) == {"intra_camera", "inter_camera"}
    assert len(seen["matches"]) == 2
    assert seen["kwargs"]["max_nfev"] == 7
    assert seen["kwargs"]["evaluation_max_observations"] == 8
    assert seen["kwargs"]["torch_device"] == "cpu"
    assert seen["kwargs"]["torch_dtype"] == "float32"
    assert seen["kwargs"]["torch_early_stopping_patience"] == 9
    assert seen["kwargs"]["torch_early_stopping_min_delta"] == pytest.approx(1e-4)
    assert seen["kwargs"]["torch_early_stopping_min_steps"] == 25
    assert seen["kwargs"]["evaluation_backend"] == "torch"
    assert seen["kwargs"]["evaluation_device"] == "cuda:0"
    assert seen["kwargs"]["evaluation_batch_size"] == 1234
    assert seen["kwargs"]["optimize_time_varying_disturbance"] is True


def test_optimize_match_cache_uses_pair_specific_inter_relations(
    monkeypatch, tmp_path: Path
):
    cache_root = tmp_path / "data_cache" / "matches" / "cache"
    same_index = cache_root / "inter_camera" / "camera_a__camera_b" / "same_index"
    temporal = (
        cache_root
        / "inter_camera"
        / "camera_a__camera_b"
        / "temporal_overlap_8m"
    )
    same_index.mkdir(parents=True)
    temporal.mkdir(parents=True)
    base = pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_b"],
            "frame_offset": [0],
            "frame1": [0],
            "frame2": [0],
            "timestamp1_ns": [100],
            "timestamp2_ns": [100],
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
    base.to_parquet(same_index / "retained_matches.parquet", index=False)
    base.assign(frame2=8, timestamp2_ns=108).to_parquet(
        temporal / "retained_matches.parquet", index=False
    )
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, **kwargs):
        seen["matches"] = matches.copy()
        return None

    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.load_config",
        lambda _path: fake_config,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.CameraCalibration",
        StubCalibration,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.NavigationTrajectory",
        StubNavigation,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[],
        inter_pairs=["camera_a__camera_b"],
        inter_relations_by_pair={"camera_a__camera_b": ["same_index"]},
    )

    assert len(seen["matches"]) == 1
    assert seen["matches"]["frame2"].tolist() == [0]


def test_match_cache_loader_preserves_inter_camera_order_recorded_in_parquet(
    monkeypatch, tmp_path: Path
):
    cache_root = tmp_path / "data_cache" / "matches" / "cache"
    inter = cache_root / "inter_camera" / "camera_a__camera_b" / "same_index"
    inter.mkdir(parents=True)
    pd.DataFrame(
        {
            "camera1": ["camera_b"],
            "camera2": ["camera_a"],
            "frame_offset": [0],
            "frame1": [12],
            "frame2": [12],
            "timestamp1_ns": [100],
            "timestamp2_ns": [100],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    ).to_parquet(inter / "retained_matches.parquet", index=False)
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, **kwargs):
        seen["matches"] = matches.copy()
        return None

    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[],
        inter_pairs=["camera_a__camera_b"],
    )

    inter_matches = seen["matches"].loc[seen["matches"]["match_scope"] == "inter_camera"]
    assert inter_matches["camera1"].tolist() == ["camera_b"]
    assert inter_matches["camera2"].tolist() == ["camera_a"]


def test_match_cache_loader_includes_all_inter_relation_dirs_by_default(
    monkeypatch, tmp_path: Path
):
    cache_root = tmp_path / "data_cache" / "matches" / "cache"
    same_index = cache_root / "inter_camera" / "camera_a__camera_b" / "same_index"
    temporal = cache_root / "inter_camera" / "camera_a__camera_b" / "temporal_overlap_12m"
    same_index.mkdir(parents=True)
    temporal.mkdir(parents=True)
    pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_b"],
            "frame_offset": [0],
            "frame1": [10],
            "frame2": [10],
            "timestamp1_ns": [100],
            "timestamp2_ns": [100],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    ).to_parquet(same_index / "retained_matches.parquet", index=False)
    pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_b"],
            "frame_offset": [25],
            "frame1": [10],
            "frame2": [35],
            "timestamp1_ns": [100],
            "timestamp2_ns": [350],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    ).to_parquet(temporal / "retained_matches.parquet", index=False)
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, **kwargs):
        seen["matches"] = matches.copy()
        return None

    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[],
        inter_pairs=["camera_a__camera_b"],
    )

    inter_matches = seen["matches"].loc[seen["matches"]["match_scope"] == "inter_camera"]
    assert sorted(inter_matches["frame_offset"].tolist()) == [0, 25]


def test_match_cache_loader_reads_temporal_overlap_when_explicitly_requested(
    monkeypatch, tmp_path: Path
):
    cache_root = tmp_path / "data_cache" / "matches" / "cache"
    inter = cache_root / "inter_camera" / "camera_a__camera_b" / "temporal_overlap_12m"
    inter.mkdir(parents=True)
    pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_b"],
            "frame_offset": [25],
            "frame1": [10],
            "frame2": [35],
            "timestamp1_ns": [100],
            "timestamp2_ns": [350],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "pose_quality_valid": [True],
            "reprojection_threshold_inlier": [True],
        }
    ).to_parquet(inter / "retained_matches.parquet", index=False)
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, **kwargs):
        seen["matches"] = matches.copy()
        return None

    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[],
        inter_pairs=["camera_a__camera_b"],
        inter_relations=["temporal_overlap"],
    )

    inter_matches = seen["matches"].loc[seen["matches"]["match_scope"] == "inter_camera"]
    assert len(inter_matches) == 1
    assert inter_matches["frame1"].tolist() == [10]
    assert inter_matches["frame2"].tolist() == [35]


def test_optimize_multi_camera_extrinsics_from_match_cache_accepts_multiple_roots(
    monkeypatch, tmp_path: Path
):
    primary_root = tmp_path / "matches" / "primary"
    secondary_root = tmp_path / "matches" / "secondary"
    intra = primary_root / "intra_camera" / "camera_a" / "offsets_5"
    inter = secondary_root / "inter_camera" / "camera_a__camera_b" / "same_index"
    intra.mkdir(parents=True)
    inter.mkdir(parents=True)
    base = pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "camera": ["camera_a"],
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [5],
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
    base.to_parquet(intra / "retained_matches.parquet", index=False)
    base.assign(camera2="camera_b", frame_offset=0).to_parquet(
        inter / "retained_matches.parquet",
        index=False,
    )
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    seen = {}

    class StubCalibration:
        sensor_to_car = np.eye(4)

        @classmethod
        def from_dataset(cls, _root, camera):
            item = cls()
            item.camera = camera
            item.intrinsic = np.eye(3)
            return item

    class StubNavigation:
        @classmethod
        def from_csv(cls, _path):
            return cls()

    def fake_optimize(*, matches, **kwargs):
        seen["matches"] = matches.copy()
        return None

    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.CameraCalibration", StubCalibration)
    monkeypatch.setattr("dongfeng_registration.extrinsic_optimization.NavigationTrajectory", StubNavigation)
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=tmp_path / "config.yaml",
        match_cache_root=[primary_root, secondary_root],
        output=tmp_path / "out",
        cameras=["camera_a", "camera_b"],
        intra_offsets=[5],
        inter_pairs=["camera_a__camera_b"],
    )

    assert set(seen["matches"]["match_scope"]) == {"intra_camera", "inter_camera"}
    assert len(seen["matches"]) == 2


def test_pkl_match_cache_export_metadata_uses_image_root_not_compat_dataset(
    monkeypatch,
    tmp_path: Path,
):
    cache_root = tmp_path / "matches"
    intra = cache_root / "intra_camera" / "camera_a" / "offsets_1"
    intra.mkdir(parents=True)
    pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "camera": ["camera_a"],
            "frame_offset": [1],
            "frame1": [0],
            "frame2": [1],
            "timestamp1_ns": [1_000_000],
            "timestamp2_ns": [2_000_000],
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
    ).to_parquet(intra / "retained_matches.parquet", index=False)

    image_root = tmp_path / "camera_distorted"
    image_camera = image_root / "camera_a"
    image_camera.mkdir(parents=True)
    (image_camera / "1000.jpg").write_bytes(b"fake")
    (image_camera / "2000.jpg").write_bytes(b"fake")
    compat_camera = tmp_path / "compat_dataset" / "camera" / "camera_a"
    compat_camera.mkdir(parents=True)
    (compat_camera / "stale.jpg").write_bytes(b"stale")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_root: {(tmp_path / 'compat_dataset').as_posix()}",
                f"output_root: {(tmp_path / 'outputs').as_posix()}",
                "cameras: [camera_a]",
                "assumptions:",
                "  pose_backend: result_interpolation_pkl",
                "  result_interpolation_pkl_path: result_interpolation.pkl",
                f"  result_interpolation_image_root: {image_root.as_posix()}",
                "  result_interpolation_ego_field: ego2global_transformation_matrix_camera_reference_offset",
                "  result_interpolation_optimized_ego_output_field: ego2global_transformation_matrix_camera_reference_offset_optimized",
            ]
        ),
        encoding="utf-8",
    )
    seen = {}

    class StubNavigation:
        timestamps_ns = np.array([0, 3_000_000], dtype=np.int64)

        def pose_at(self, _timestamp_ns):
            return np.eye(4)

    class StubCalibration:
        sensor_to_car = np.eye(4)
        intrinsic = np.eye(3)

    def fake_optimize(*, frame_metadata, **kwargs):
        seen["frame_metadata"] = frame_metadata
        seen["kwargs"] = kwargs
        return None

    def fake_navigation(_path, **kwargs):
        seen["navigation_kwargs"] = kwargs
        return StubNavigation()

    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.navigation_from_result_interpolation",
        fake_navigation,
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.calibrations_from_result_interpolation",
        lambda _path, cameras, **_kwargs: {camera: StubCalibration() for camera in cameras},
    )
    monkeypatch.setattr(
        "dongfeng_registration.extrinsic_optimization.optimize_multi_camera_extrinsics_from_observations",
        fake_optimize,
    )

    optimize_multi_camera_extrinsics_from_match_cache(
        config_path=config_path,
        match_cache_root=cache_root,
        output=tmp_path / "out",
        cameras=["camera_a"],
        intra_offsets=[1],
    )

    metadata = seen["frame_metadata"]["camera_a"]
    assert metadata[0]["image_path"] == str(image_camera / "1000.jpg")
    assert metadata[1]["image_path"] == str(image_camera / "2000.jpg")
    assert seen["kwargs"]["result_interpolation_pkl_path"] == (
        tmp_path / "compat_dataset" / "result_interpolation.pkl"
    )
    assert seen["navigation_kwargs"] == {
        "ego_field": "ego2global_transformation_matrix_camera_reference_offset",
        "allow_end_extrapolation": False,
    }
    assert seen["kwargs"]["result_interpolation_source_ego_field"] == (
        "ego2global_transformation_matrix_camera_reference_offset"
    )
    assert seen["kwargs"]["result_interpolation_optimized_ego_output_field"] == (
        "ego2global_transformation_matrix_camera_reference_offset_optimized"
    )


def test_multi_camera_optimization_exports_image_camera_to_world_csvs(tmp_path: Path):
    navigation = _make_navigation()
    sensor_to_car_a = _transform_from_rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_to_car_b = _transform_from_rt([0.2, 0.0, 0.0], [0.0, 0.0, 0.02])
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", sensor_to_car_a),
        "camera_b": _make_named_calibration("camera_b", sensor_to_car_b),
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        {"camera_a": sensor_to_car_a, "camera_b": sensor_to_car_b},
    )

    optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path / "out",
        max_observations=20,
        evaluation_max_observations=20,
        optimizer_method="scipy_least_squares",
        max_nfev=1,
    )

    deliverable_root = tmp_path / "out" / "deliverables" / "image_extrinsics"
    before_csv = deliverable_root / "image_camera_to_world_before.csv"
    after_csv = deliverable_root / "image_camera_to_world_after.csv"
    assert before_csv.exists()
    assert after_csv.exists()
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)
    assert set(before["camera"]) == {"camera_a", "camera_b"}
    assert set(after["camera"]) == {"camera_a", "camera_b"}
    assert before["frame"].tolist() == after["frame"].tolist()
    assert {"T_cw_00", "T_cw_03", "T_cw_33"}.issubset(before.columns)
    assert (deliverable_root / "before" / "camera_a_image_camera_to_world_before.csv").exists()
    assert (deliverable_root / "after" / "camera_b_image_camera_to_world_after.csv").exists()


def test_multi_camera_optimization_exports_optimized_result_interpolation_pkl(
    monkeypatch,
    tmp_path: Path,
):
    navigation = _make_navigation()
    sensor_to_car_a = _transform_from_rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_to_car_b = _transform_from_rt([0.2, 0.0, 0.0], [0.0, 0.0, 0.02])
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", sensor_to_car_a),
        "camera_b": _make_named_calibration("camera_b", sensor_to_car_b),
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        {"camera_a": sensor_to_car_a, "camera_b": sensor_to_car_b},
    )
    input_pkl = tmp_path / "dataset" / "result_interpolation.pkl"
    input_pkl.parent.mkdir()
    input_pkl.write_bytes(b"original")
    calls: dict[str, Path] = {}

    def fake_generate(
        *,
        input_pkl,
        optimized_yaml,
        image_extrinsics_csv,
        output_pkl,
        manifest_json,
        validation_json,
        source_ego_field,
        optimized_ego_output_field,
    ):
        calls["input_pkl"] = Path(input_pkl)
        calls["optimized_yaml"] = Path(optimized_yaml)
        calls["image_extrinsics_csv"] = Path(image_extrinsics_csv)
        calls["output_pkl"] = Path(output_pkl)
        calls["manifest_json"] = Path(manifest_json)
        calls["validation_json"] = Path(validation_json)
        calls["source_ego_field"] = source_ego_field
        calls["optimized_ego_output_field"] = optimized_ego_output_field
        Path(output_pkl).write_bytes(b"optimized")
        Path(manifest_json).write_text("{}", encoding="utf-8")
        Path(validation_json).write_text("{}", encoding="utf-8")
        return {"infos_updated": 1}

    monkeypatch.setattr(
        extrinsic_optimization,
        "generate_optimized_result_interpolation_pkl",
        fake_generate,
        raising=False,
    )

    optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path / "out",
        max_observations=20,
        evaluation_max_observations=20,
        optimizer_method="scipy_least_squares",
        max_nfev=1,
        result_interpolation_pkl_path=input_pkl,
        result_interpolation_source_ego_field=(
            "ego2global_transformation_matrix_camera_reference_offset"
        ),
        result_interpolation_optimized_ego_output_field=(
            "ego2global_transformation_matrix_camera_reference_offset_optimized"
        ),
    )

    deliverables = tmp_path / "out" / "deliverables"
    assert calls == {
        "input_pkl": input_pkl,
        "optimized_yaml": tmp_path / "out" / "optimized_multi_camera_extrinsics.yaml",
        "image_extrinsics_csv": deliverables
        / "image_extrinsics"
        / "image_camera_to_world_after.csv",
        "output_pkl": deliverables / "result_interpolation_optimized.pkl",
        "manifest_json": deliverables / "result_interpolation_optimized_manifest.json",
        "validation_json": deliverables / "result_interpolation_optimized_validation.json",
        "source_ego_field": "ego2global_transformation_matrix_camera_reference_offset",
        "optimized_ego_output_field": (
            "ego2global_transformation_matrix_camera_reference_offset_optimized"
        ),
    }


def test_multi_camera_image_camera_to_world_export_skips_metadata_outside_navigation(
    tmp_path: Path,
):
    navigation = _make_navigation()
    sensor_to_car_a = _transform_from_rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_to_car_b = _transform_from_rt([0.2, 0.0, 0.0], [0.0, 0.0, 0.02])
    calibrations = {
        "camera_a": _make_named_calibration("camera_a", sensor_to_car_a),
        "camera_b": _make_named_calibration("camera_b", sensor_to_car_b),
    }
    matches = _make_multi_camera_matches(
        calibrations,
        navigation,
        {"camera_a": sensor_to_car_a, "camera_b": sensor_to_car_b},
    )
    valid_frame = int(matches["frame1"].iloc[0])
    valid_timestamp = int(matches["timestamp1_ns"].iloc[0])
    frame_metadata = {
        "camera_a": {
            valid_frame: {
                "timestamp_ns": valid_timestamp,
                "image_filename": "valid.jpg",
                "image_path": "/data/valid.jpg",
            },
            999: {
                "timestamp_ns": int(navigation.timestamps_ns[-1]) + 1_000_000_000,
                "image_filename": "outside.jpg",
                "image_path": "/data/outside.jpg",
            },
        }
    }

    optimize_multi_camera_extrinsics_from_observations(
        matches=matches,
        navigation=navigation,
        calibrations=calibrations,
        output=tmp_path / "out",
        max_observations=20,
        evaluation_max_observations=20,
        optimizer_method="scipy_least_squares",
        max_nfev=1,
        frame_metadata=frame_metadata,
    )

    before = pd.read_csv(
        tmp_path
        / "out"
        / "deliverables"
        / "image_extrinsics"
        / "image_camera_to_world_before.csv"
    )
    assert 999 not in set(before["frame"])
    assert "outside.jpg" not in set(before["image_filename"])


