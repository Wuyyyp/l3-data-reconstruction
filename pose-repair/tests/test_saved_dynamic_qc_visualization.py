from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_saved_qc_script():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    path = scripts_dir / "visualize_saved_dynamic_compensation_qc.py"
    spec = importlib.util.spec_from_file_location("visualize_saved_dynamic_compensation_qc", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.path.insert(0, str(scripts_dir))
    spec.loader.exec_module(module)
    return module


def test_compensated_qc_uses_before_box_point_indices_for_after_projection():
    module = load_saved_qc_script()
    original = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    compensated = original + np.asarray([10.0, 0.0, 0.0])
    before_corners = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    before_points, after_points, mask = module.select_points_compensated_from_before_box(
        original,
        compensated,
        before_corners,
        margin_m=0.0,
    )

    assert mask.tolist() == [True, True, False]
    np.testing.assert_allclose(before_points, original[:2])
    np.testing.assert_allclose(after_points, compensated[:2])


def test_compensated_qc_rejects_mismatched_before_after_point_order():
    module = load_saved_qc_script()
    before_corners = np.zeros((8, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="same point count/order"):
        module.select_points_compensated_from_before_box(
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            before_corners,
            margin_m=0.0,
        )


def test_source_ego_points_are_transformed_to_camera_time_ego_for_qc_projection():
    module = load_saved_qc_script()
    source_ego_to_world = np.eye(4, dtype=np.float64)
    target_ego_to_world = np.eye(4, dtype=np.float64)
    target_ego_to_world[0, 3] = 2.0
    source_points = np.asarray([[5.0, 0.0, 0.0]], dtype=np.float64)

    target_points = module.transform_source_ego_to_target_ego(
        source_points,
        source_ego_to_world=source_ego_to_world,
        target_ego_to_world=target_ego_to_world,
    )

    np.testing.assert_allclose(target_points, [[3.0, 0.0, 0.0]])


def test_qc_image_and_intrinsic_are_resized_to_configured_panel_width():
    module = load_saved_qc_script()
    image = np.zeros((1080, 3840, 3), dtype=np.uint8)
    intrinsic = np.asarray(
        [[2000.0, 0.0, 1920.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    resized, scaled_intrinsic = module.resize_image_and_intrinsic(
        image, intrinsic, max_width=1280
    )

    assert resized.shape == (360, 1280, 3)
    np.testing.assert_allclose(
        scaled_intrinsic,
        [[2000.0 / 3.0, 0.0, 640.0], [0.0, 2000.0 / 3.0, 180.0], [0.0, 0.0, 1.0]],
    )


def test_qc_cloud_cache_reads_each_path_once(tmp_path: Path):
    module = load_saved_qc_script()
    path = tmp_path / "cloud.pcd"
    calls = []

    def loader(value):
        calls.append(value)
        return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)

    cache = {}
    first = module.load_xyz_cached(path, cache, loader)
    second = module.load_xyz_cached(path, cache, loader)

    assert calls == [path]
    assert first is second
