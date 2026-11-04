from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from dongfeng_registration.lidar_image_edge_cache import (
    build_lidar_image_edge_cache_paths,
    image_edge_cache_key,
    lidar_edge_cache_key,
    write_json,
    write_rows_csv,
)


def load_prepare_cache_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_lidar_image_edge_cache.py"
    spec = importlib.util.spec_from_file_location("prepare_lidar_image_edge_cache", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_cache_default_diagnostic_preview_scale_is_half_resolution():
    module = load_prepare_cache_script()

    assert module.DEFAULT_DIAGNOSTIC_PREVIEW_SCALE == 0.50


def test_build_lidar_image_edge_cache_paths_uses_stable_layout(tmp_path):
    paths = build_lidar_image_edge_cache_paths(
        cache_root=tmp_path,
        sequence="M18",
        camera="center_camera_fov120",
        lidar_name="front_lidar",
        image_key=image_edge_cache_key(canny_low=40, canny_high=100),
        lidar_key=lidar_edge_cache_key(
            min_percentile=10.0,
            max_percentile=90.0,
            edge_side="near",
            require_support=True,
            min_component_pixels=5,
        ),
    )

    assert paths.root == tmp_path / "M18" / "lidar_image_edges"
    assert (
        paths.image_dir
        == paths.root
        / "image_edges"
        / "canny_l40_h100"
        / "center_camera_fov120"
    )
    assert (
        paths.lidar_dir
        == paths.root
        / "lidar_edges"
        / "range_native_p10_p90_near_support5"
        / "front_lidar"
    )
    assert (
        paths.pair_dir
        == paths.root / "pairs" / "center_camera_fov120__front_lidar"
    )
    assert paths.diagnostics_dir == paths.pair_dir / "diagnostics"


def test_write_cache_metadata_files(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "sequence": "M18",
            "version": 1,
            "camera": "center_camera_fov120",
        },
    )

    rows_path = tmp_path / "pair_index.csv"
    write_rows_csv(
        rows_path,
        [
            {
                "frame_index": 0,
                "camera_name": "center_camera_fov120",
                "lidar_name": "front_lidar",
                "camera_timestamp_us": 1766546037101098,
                "lidar_timestamp_ns": 1766546037100006000,
                "time_diff_ms": 1.092,
                "image_edge_npz": "image_edges/x.npz",
                "lidar_edge_npz": "lidar_edges/y.npz",
                "valid_pair": 1,
                "notes": "",
            }
        ],
    )

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sequence"] == "M18"
    with rows_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["time_diff_ms"] == "1.092"
    assert rows[0]["valid_pair"] == "1"
