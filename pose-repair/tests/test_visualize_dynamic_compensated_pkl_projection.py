from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from visualize_dynamic_compensated_pkl_projection import (  # noqa: E402
    offset_ego_to_world_for_info,
    points_in_offset_ego,
    resolve_pkl_relative_sensor_file,
)


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def test_points_in_offset_ego_uses_original_source_ego_and_offset_target_ego():
    source_points = np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32)

    shifted = points_in_offset_ego(
        source_points,
        source_ego_to_world=_translation(0.0),
        offset_ego_to_world=_translation(2.0),
    )

    np.testing.assert_allclose(shifted, [[3.0, 0.0, 0.0]])


def test_offset_ego_to_world_for_info_reads_new_deliverable_field():
    info = {"ego2global_transformation_matrix_camera_reference_offset": _translation(3.0).tolist()}

    np.testing.assert_allclose(offset_ego_to_world_for_info(info), _translation(3.0))


def test_resolve_pkl_relative_sensor_file_prefers_pkl_parent_for_relative_path(tmp_path: Path):
    pkl_path = tmp_path / "result.pkl"
    pkl_path.write_bytes(b"")
    expected = tmp_path / "compensated_perception" / "1.pcd"
    expected.parent.mkdir()
    expected.write_text("pcd", encoding="utf-8")

    assert (
        resolve_pkl_relative_sensor_file(
            pkl_path=pkl_path,
            data_path="compensated_perception/1.pcd",
            fallback_root=tmp_path / "dataset",
            fallback_folder="perception",
        )
        == expected
    )


def test_resolve_pkl_relative_sensor_file_uses_explicit_point_cloud_root(
    tmp_path: Path,
):
    pkl_path = tmp_path / "deliverables" / "result_interpolation_optimized.pkl"
    pkl_path.parent.mkdir()
    pkl_path.write_bytes(b"")
    point_cloud_root = tmp_path / "dynamic" / "compensated_perception"
    expected = point_cloud_root / "1.pcd"
    point_cloud_root.mkdir(parents=True)
    expected.write_text("pcd", encoding="utf-8")

    assert (
        resolve_pkl_relative_sensor_file(
            pkl_path=pkl_path,
            data_path="compensated_perception/1.pcd",
            fallback_root=tmp_path / "dataset",
            fallback_folder="perception",
            point_cloud_root=point_cloud_root,
        )
        == expected
    )
