from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


def load_projection_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_l29_lidar_projection_pkl.py"
    spec = importlib.util.spec_from_file_location("visualize_l29_lidar_projection_pkl", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multicamera_shift_compare_resizes_all_tiles_to_same_width():
    module = load_projection_script()
    before = {
        "cam_a": np.full((10, 20, 3), 10, dtype=np.uint8),
        "cam_b": np.full((30, 15, 3), 20, dtype=np.uint8),
    }
    after = {
        "cam_a": np.full((10, 20, 3), 30, dtype=np.uint8),
        "cam_b": np.full((30, 15, 3), 40, dtype=np.uint8),
    }

    canvas = module.build_multicamera_shift_compare_canvas(
        before_by_camera=before,
        after_by_camera=after,
        camera_order=["cam_a", "cam_b"],
        tile_width=40,
        title="dx=-0.10 dz=-0.10",
    )

    # cam_a: 20->40 width doubles height 10->20.
    # cam_b: 15->40 width scales height 30->80.
    # Each row has before+after tiles, so final width is 80.
    assert canvas.shape == (100, 80, 3)


def test_image_for_camera_accepts_canonical_camera_directory(tmp_path):
    module = load_projection_script()
    image_dir = tmp_path / "center_camera_fov30"
    image_dir.mkdir()
    expected_path = image_dir / "frame.jpg"
    assert cv2.imwrite(str(expected_path), np.zeros((8, 12, 3), dtype=np.uint8))

    resolved = module.image_for_camera(
        {"data_path": "images/center_camera_fov30/frame.jpg"},
        image_root=tmp_path,
        camera="center_camera_fov30",
    )

    assert resolved is not None
    path, image = resolved
    assert path == expected_path
    assert image.shape == (8, 12, 3)
