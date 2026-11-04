from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _camera_pair_key(first_camera: str, second_camera: str) -> str:
    return f"{first_camera}__{second_camera}"


def match_cache_key(
    *,
    backend: str,
    image_size: int,
    max_keypoints: int,
    score_threshold: float,
    semantic_backend: str,
    semantic_model_name: str,
    semantic_image_size: int,
    semantic_dilation_pixels: int,
) -> str:
    return f"{backend}__{semantic_backend}"


def intra_camera_cache_dir(
    root: Path,
    *,
    cache_key: str,
    camera: str,
    frame_offset: int,
) -> Path:
    return (
        Path(root)
        / cache_key
        / "intra_camera"
        / str(camera)
        / f"offsets_{int(frame_offset)}"
    )


def intra_camera_baseline_cache_dir(
    root: Path,
    *,
    cache_key: str,
    camera: str,
    target_baseline_m: float,
) -> Path:
    value = float(target_baseline_m)
    label = f"{value:g}m".replace(".", "p")
    return (
        Path(root)
        / cache_key
        / "intra_camera"
        / str(camera)
        / f"baselines_{label}"
    )


def inter_camera_cache_dir(
    root: Path,
    *,
    cache_key: str,
    first_camera: str,
    second_camera: str,
    relation: str = "same_index",
) -> Path:
    return (
        Path(root)
        / cache_key
        / "inter_camera"
        / _camera_pair_key(first_camera, second_camera)
        / str(relation)
    )


def write_match_cache_artifacts(
    cache_dir: Path,
    *,
    matches: pd.DataFrame,
    pair_metrics: pd.DataFrame | None,
    manifest: dict[str, Any],
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pair_metrics, pd.DataFrame):
        pair_metrics.to_csv(cache_dir / "pair_metrics.csv", index=False)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cache_dir
