from __future__ import annotations

import numpy as np
import importlib.util
from pathlib import Path

from dongfeng_registration.lidar_scan_registration import chain_pair_transforms
from dongfeng_registration.lidar_scan_registration import direction_adaptive_lidar_point_filter_mask
from dongfeng_registration.lidar_scan_registration import interpolate_ego_world_pose
from dongfeng_registration.lidar_scan_registration import lidar_point_filter_mask
from dongfeng_registration.lidar_scan_registration import normalize_registration_method
from dongfeng_registration.lidar_scan_registration import optimize_world_pose_graph
from dongfeng_registration.lidar_scan_registration import resolve_point_coordinate_frame
from dongfeng_registration.lidar_scan_registration import summarize_pair_metrics
from dongfeng_registration.lidar_scan_registration import transform_point_records
from dongfeng_registration.lidar_scan_registration import voxel_downsample_indices
from dongfeng_registration.lidar_scan_registration import write_point_records_ply_ascii


def _translation_transform(x_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = float(x_m)
    return transform


def test_optimize_world_pose_graph_recovers_global_poses_from_pairwise_edges():
    navigation_world_poses = [
        _translation_transform(0.0),
        _translation_transform(1.20),
        _translation_transform(2.40),
    ]
    edges = [
        {"target_index": 0, "source_index": 1, "target_from_source": _translation_transform(1.0), "weight": 10.0},
        {"target_index": 1, "source_index": 2, "target_from_source": _translation_transform(1.0), "weight": 10.0},
        {"target_index": 0, "source_index": 2, "target_from_source": _translation_transform(2.0), "weight": 10.0},
    ]

    optimized, summary = optimize_world_pose_graph(
        navigation_world_poses,
        edges,
        navigation_translation_weight=0.05,
        navigation_rotation_weight=0.05,
        rotation_residual_scale_m_per_rad=5.0,
        max_nfev=100,
    )

    assert summary["edge_count"] == 3
    assert summary["success"] is True
    np.testing.assert_allclose(
        [pose[0, 3] for pose in optimized],
        [0.0, 1.0, 2.0],
        atol=2e-2,
    )


def test_interpolate_ego_world_pose_uses_requested_optimized_field():
    raw0 = _translation_transform(0.0)
    raw1 = _translation_transform(10.0)
    optimized0 = _translation_transform(100.0)
    optimized1 = _translation_transform(120.0)
    infos = [
        {"timestamp": 1_000_000_000_000_000_000, "ego2global_transformation_matrix": raw0, "ego_optimized": optimized0},
        {"timestamp": 1_000_000_002_000_000_000, "ego2global_transformation_matrix": raw1, "ego_optimized": optimized1},
    ]

    interpolated, source = interpolate_ego_world_pose(
        infos,
        time_s=1_000_000_001.0,
        ego_field="ego_optimized",
    )

    assert source == "interpolated"
    np.testing.assert_allclose(interpolated[:3, 3], [110.0, 0.0, 0.0])


def test_chain_pair_transforms_accumulates_source_to_previous_edges():
    edge_01 = np.eye(4, dtype=np.float64)
    edge_01[0, 3] = 1.0
    edge_12 = np.eye(4, dtype=np.float64)
    edge_12[1, 3] = 2.0

    chained = chain_pair_transforms([edge_01, edge_12])

    assert len(chained) == 3
    np.testing.assert_allclose(chained[0], np.eye(4))
    np.testing.assert_allclose(chained[1], edge_01)
    expected_02 = edge_01 @ edge_12
    np.testing.assert_allclose(chained[2], expected_02)


def test_summarize_pair_metrics_reports_max_corrections_and_average_runtime():
    summary = summarize_pair_metrics(
        [
            {
                "source_frame": 1,
                "target_frame": 0,
                "delta_from_init_translation_m": 0.02,
                "delta_from_init_rotation_deg": 0.3,
                "icp_total_time_s": 1.0,
            },
            {
                "source_frame": 2,
                "target_frame": 1,
                "delta_from_init_translation_m": 0.05,
                "delta_from_init_rotation_deg": 0.1,
                "icp_total_time_s": 3.0,
            },
        ]
    )

    assert summary["pair_count"] == 2
    assert summary["mean_icp_time_s"] == 2.0
    assert summary["max_translation_source_frame"] == 2
    assert summary["max_translation_target_frame"] == 1
    assert summary["max_translation_m"] == 0.05
    assert summary["max_rotation_source_frame"] == 1
    assert summary["max_rotation_target_frame"] == 0
    assert summary["max_rotation_deg"] == 0.3


def test_normalize_registration_method_accepts_translation_only_icp_and_gicp():
    assert normalize_registration_method("translation-only") == "translation_only"
    assert normalize_registration_method("translation") == "translation_only"
    assert normalize_registration_method("ICP") == "icp"
    assert normalize_registration_method("gicp") == "gicp"


def test_resolve_point_coordinate_frame_treats_perception_as_ego_in_auto_mode():
    assert resolve_point_coordinate_frame("auto", lidar_name="perception") == "ego"
    assert resolve_point_coordinate_frame("auto", lidar_name="front_lidar") == "lidar"
    assert resolve_point_coordinate_frame("ego", lidar_name="front_lidar") == "ego"


def test_resolve_point_coordinate_frame_treats_compensation_lidar_root_as_ego():
    assert (
        resolve_point_coordinate_frame(
            "auto",
            lidar_name="top_center_lidar",
            lidar_root=Path("L29_new/L29_new/lidar/top_center_lidar_compensation"),
        )
        == "ego"
    )
    assert (
        resolve_point_coordinate_frame(
            "auto",
            lidar_name="top_center_lidar",
            lidar_root=Path("L29_new/L29_new/lidar/top_center_lidar"),
        )
        == "lidar"
    )


def test_scan_to_scan_script_defaults_do_not_filter_height():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_lidar_scan_to_scan_fusion.py"
    spec = importlib.util.spec_from_file_location("compare_lidar_scan_to_scan_fusion", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.DEFAULT_MIN_Z_M == -float("inf")
    assert module.DEFAULT_MAX_Z_M == float("inf")


def test_l29_projection_lidar_points_for_info_uses_requested_lidar_name(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_l29_lidar_projection_pkl.py"
    spec = importlib.util.spec_from_file_location("visualize_l29_lidar_projection_pkl", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    expected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    lidar_root = tmp_path / "left_lidar_compensation"
    lidar_root.mkdir()
    expected_path = lidar_root / "left_frame.pcd"
    expected_path.write_text("stub", encoding="utf-8")

    def fake_read_point_cloud_xyz(pcd_path):
        assert pcd_path == expected_path
        return expected

    monkeypatch.setattr(module, "read_point_cloud_xyz", fake_read_point_cloud_xyz)
    info = {
        "sensors": {
            "lidar": {
                "perception": {"data_path": "/data/perception_frame.pcd"},
                "left_lidar": {"data_path": "/data/left_frame.pcd"},
            }
        }
    }

    pcd_path, points = module.lidar_points_for_info(
        info,
        lidar_root,
        lidar_name="left_lidar",
    )

    assert pcd_path == expected_path
    np.testing.assert_array_equal(points, expected)


def test_l29_projection_lidar_points_for_info_applies_ego_shift(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_l29_lidar_projection_pkl.py"
    spec = importlib.util.spec_from_file_location("visualize_l29_lidar_projection_pkl", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    lidar_root = tmp_path / "top_center_lidar_compensation"
    lidar_root.mkdir()
    expected_path = lidar_root / "top_frame.pcd"
    expected_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "read_point_cloud_xyz",
        lambda _path: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    info = {
        "sensors": {
            "lidar": {
                "top_center_lidar": {"data_path": "/data/top_frame.pcd"},
            }
        }
    }

    _pcd_path, points = module.lidar_points_for_info(
        info,
        lidar_root,
        lidar_name="top_center_lidar",
        ego_shift_m=np.array([0.0, 0.8, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(points, np.array([[1.0, 2.8, 3.0]], dtype=np.float32))


def test_compare_lidar_scan_to_scan_fusion_t3_defaults_point_to_standard_inputs():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_lidar_scan_to_scan_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("compare_lidar_scan_to_scan_fusion", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    defaults = module.build_t3_defaults()

    assert defaults["pkl"] == Path(r"D:\DongFeng\data\T3_canonical\result_interpolation_standard.pkl")
    assert defaults["lidar_root"] == Path(
        r"D:\DongFeng\data\T3_canonical\lidar\front_lidar_compensation"
    )
    assert defaults["output_dir"] == Path(r"D:\DongFeng\outputs\T3_front_lidar_multiframe_fusion")


def test_direction_adaptive_range_filter_keeps_lateral_points_farther_than_forward_points():
    points = np.asarray(
        [
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [0.0, 70.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 101.0, 0.0],
        ],
        dtype=np.float32,
    )

    keep, per_point_max_range = direction_adaptive_lidar_point_filter_mask(
        points,
        motion_direction_ego=np.array([1.0, 0.0, 0.0]),
        min_range_m=1.5,
        longitudinal_max_range_m=60.0,
        lateral_max_range_m=100.0,
        gamma=1.5,
    )

    np.testing.assert_array_equal(keep, np.array([True, False, True, True, False]))
    np.testing.assert_allclose(per_point_max_range[[0, 1]], [60.0, 60.0])
    np.testing.assert_allclose(per_point_max_range[[2, 3, 4]], [100.0, 100.0, 100.0])


def test_compare_lidar_scan_to_scan_fusion_translation_fallback_aligns_simple_shift():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_lidar_scan_to_scan_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("compare_lidar_scan_to_scan_fusion", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    target = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    source = target + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    transform, details = module.run_multiscale_translation_only_icp(
        source_points=source,
        target_points=target,
        init_transform=np.eye(4, dtype=np.float64),
        voxel_sizes=[1.0],
        max_correspondence_distances=[2.0],
        max_iterations=[5],
    )

    assert len(details) == 1
    np.testing.assert_allclose(transform[:3, 3], np.array([-1.0, 0.0, 0.0]), atol=1e-6)


def test_compare_lidar_scan_to_scan_fusion_both_runs_icp_and_gicp():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_lidar_scan_to_scan_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("compare_lidar_scan_to_scan_fusion", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.registration_methods_to_run("both") == ["icp", "gicp"]
    assert module.registration_methods_to_run("all") == ["translation_only", "icp", "gicp"]
    assert module.registration_methods_to_run("translation-only") == ["translation_only"]
    assert module.registration_methods_to_run("icp") == ["icp"]
    assert module.registration_methods_to_run("gicp") == ["gicp"]


def test_compare_lidar_scan_to_scan_fusion_builds_unique_multiframe_edges():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_lidar_scan_to_scan_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("compare_lidar_scan_to_scan_fusion", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    pairs = module.build_pose_graph_edge_pairs(12, [1, 5, 10])

    assert pairs[0] == (0, 1)
    assert pairs[-1] == (1, 11)
    assert len(pairs) == (11 + 7 + 2)
    assert len(pairs) == len(set(pairs))
    assert all(target < source for target, source in pairs)


def test_optimized_pose_fusion_uses_pcd_filename_as_scan_reference_time():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_lidar_optimized_pose.py"
    spec = importlib.util.spec_from_file_location("fuse_lidar_optimized_pose", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    time_s = module.pcd_filename_reference_time_s(Path("1772699910207617000.pcd"))

    assert time_s == 1772699910.207617


def test_optimized_pose_fusion_builds_named_output_in_selected_directory():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_lidar_optimized_pose.py"
    spec = importlib.util.spec_from_file_location("fuse_lidar_optimized_pose", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output = module.output_ply_path(
        Path("outputs/fused"),
        "source_ego_interpolated_world_fused.ply",
    )

    assert output == Path("outputs/fused/source_ego_interpolated_world_fused.ply")


def test_optimized_pose_fusion_builds_named_metadata_outputs_in_selected_directory():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_lidar_optimized_pose.py"
    spec = importlib.util.spec_from_file_location("fuse_lidar_optimized_pose", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output_dir = Path("outputs/fused")

    assert module.output_metadata_path(
        output_dir,
        "source_ego_frame_pose_interpolation.csv",
        suffix=".csv",
    ) == Path("outputs/fused/source_ego_frame_pose_interpolation.csv")
    assert module.output_metadata_path(
        output_dir,
        "source_ego_interpolation_summary.json",
        suffix=".json",
    ) == Path("outputs/fused/source_ego_interpolation_summary.json")


def test_fused_projection_uses_selected_raw_ego_field_not_precomputed_camera_pose():
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_fused_lidar_projection.py"
    spec = importlib.util.spec_from_file_location("visualize_fused_lidar_projection", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    raw_ego = _translation_transform(10.0)
    misleading_optimized_camera_pose = _translation_transform(100.0)
    info = {
        "ego2global_transformation_matrix": raw_ego,
        "sensors": {
            "cams": {
                "center_camera_fov30": {
                    "extrinsic": np.eye(4, dtype=np.float64),
                    "camera_to_world_optimized": misleading_optimized_camera_pose,
                }
            }
        },
    }

    world_to_camera = module.world_to_camera_from_info(
        info,
        camera="center_camera_fov30",
        ego_field="ego2global_transformation_matrix",
    )
    projected_origin = module.transform_points_world_to_camera(
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        world_to_camera,
    )

    np.testing.assert_allclose(projected_origin, np.zeros((1, 3)), atol=1e-7)


def test_write_point_records_ply_ascii_preserves_lidar_scalar_attributes(tmp_path):
    records = np.zeros(
        3,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("ring", "<u2"),
            ("timestamp", "<f8"),
        ],
    )
    records["x"] = [0.0, 0.04, 1.0]
    records["y"] = [0.0, 0.04, 2.0]
    records["z"] = [0.0, 0.04, 3.0]
    records["intensity"] = [10.0, 20.0, 30.0]
    records["ring"] = [1, 2, 3]
    records["timestamp"] = [100.0, 101.0, 102.0]

    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 5.0
    transformed = transform_point_records(records, transform)
    keep = lidar_point_filter_mask(
        np.column_stack([transformed["x"], transformed["y"], transformed["z"]]),
        min_range_m=0.0,
        max_range_m=10.0,
        min_z_m=-1.0,
        max_z_m=10.0,
    )
    selected = transformed[keep]
    down_indices = voxel_downsample_indices(
        np.column_stack([selected["x"], selected["y"], selected["z"]]),
        0.5,
    )

    out = tmp_path / "cloud.ply"
    write_point_records_ply_ascii(out, selected[down_indices])

    text = out.read_text(encoding="ascii")
    assert "property float intensity" in text
    assert "property ushort ring" in text
    assert "property double timestamp" in text
    assert "5 0 0 10 1 100" in text
