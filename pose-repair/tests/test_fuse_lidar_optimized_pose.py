from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_lidar_optimized_pose.py"
    spec = importlib.util.spec_from_file_location("fuse_lidar_optimized_pose", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t3_fusion_defaults_use_lidar_scan_time_ego_pose(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["fuse_lidar_optimized_pose.py"])

    args = module.parse_args()

    assert args.ego_field == "ego2global_transformation_matrix"
    assert args.point_coordinate_frame == "ego"
    assert args.filter_dynamic_semantic is True
