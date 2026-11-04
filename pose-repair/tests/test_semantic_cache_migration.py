from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dongfeng_registration.semantic_cache_migration import migrate_semantic_cache_layout


def test_migrate_semantic_cache_layout_moves_legacy_model_cache(tmp_path: Path):
    cache_root = tmp_path / "semantic_masks"
    legacy_base = cache_root / "legacy-model-key" / "left_front_camera"
    legacy_base.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    (legacy_base / "previews").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(legacy_base / "100.png"), np.zeros((10, 10), dtype=np.uint8))
    cv2.imwrite(str(legacy_base / "previews" / "100.jpg"), np.zeros((10, 20, 3), dtype=np.uint8))
    (legacy_base / "100.json").write_text(
        json.dumps(
            {
                "backend": "mask2former",
                "timestamp_ns": 100,
            }
        ),
        encoding="utf-8",
    )

    summary = migrate_semantic_cache_layout(cache_root)

    assert summary["moved_masks"] == 1
    assert (
        cache_root
        / "backends"
        / "mask2former"
        / "legacy-model-key"
        / "left_front_camera"
        / "masks"
        / "100.png"
    ).exists()
    assert (
        cache_root
        / "backends"
        / "mask2former"
        / "legacy-model-key"
        / "left_front_camera"
        / "metadata"
        / "100.json"
    ).exists()


def test_migrate_semantic_cache_layout_moves_legacy_reports_and_removes_legacy_dirs(tmp_path: Path):
    cache_root = tmp_path / "semantic_masks"
    reports_root = cache_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "left_front_camera.csv").write_text("camera\nleft_front_camera\n", encoding="utf-8")
    (reports_root / "left_front_camera.config.json").write_text("{}", encoding="utf-8")
    legacy_stage = cache_root / "left_front_camera" / "semantic"
    legacy_stage.mkdir(parents=True, exist_ok=True)
    (legacy_stage / "segment_report.csv").write_text("camera\nleft_front_camera\n", encoding="utf-8")

    summary = migrate_semantic_cache_layout(cache_root, remove_legacy=True)

    assert summary["moved_reports"] >= 1
    assert (cache_root / "reports" / "left_front_camera" / "segment_report.csv").exists()
    assert (cache_root / "reports" / "left_front_camera" / "config.json").exists()
    assert not (cache_root / "left_front_camera").exists()
