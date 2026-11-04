from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LidarImageEdgeCachePaths:
    root: Path
    image_dir: Path
    lidar_dir: Path
    pair_dir: Path
    diagnostics_dir: Path
    image_index_csv: Path
    lidar_index_csv: Path
    pair_index_csv: Path
    pair_config_json: Path
    manifest_json: Path


def _format_percentile(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}".replace(".", "p")


def image_edge_cache_key(*, canny_low: int, canny_high: int) -> str:
    return f"canny_l{int(canny_low)}_h{int(canny_high)}"


def lidar_edge_cache_key(
    *,
    min_percentile: float,
    max_percentile: float,
    edge_side: str,
    require_support: bool,
    min_component_pixels: int,
) -> str:
    support = f"support{int(min_component_pixels)}" if require_support else "nosupport"
    return (
        "range_native_"
        f"p{_format_percentile(min_percentile)}_"
        f"p{_format_percentile(max_percentile)}_"
        f"{str(edge_side)}_"
        f"{support}"
    )


def build_lidar_image_edge_cache_paths(
    *,
    cache_root: Path,
    sequence: str,
    camera: str,
    lidar_name: str,
    image_key: str,
    lidar_key: str,
) -> LidarImageEdgeCachePaths:
    root = Path(cache_root) / str(sequence) / "lidar_image_edges"
    image_dir = root / "image_edges" / str(image_key) / str(camera)
    lidar_dir = root / "lidar_edges" / str(lidar_key) / str(lidar_name)
    pair_dir = root / "pairs" / f"{camera}__{lidar_name}"
    diagnostics_dir = pair_dir / "diagnostics"
    return LidarImageEdgeCachePaths(
        root=root,
        image_dir=image_dir,
        lidar_dir=lidar_dir,
        pair_dir=pair_dir,
        diagnostics_dir=diagnostics_dir,
        image_index_csv=image_dir / "index.csv",
        lidar_index_csv=lidar_dir / "index.csv",
        pair_index_csv=pair_dir / "pair_index.csv",
        pair_config_json=pair_dir / "config.json",
        manifest_json=root / "manifest.json",
    )


def make_frame_stem(frame_index: int, timestamp: int | str) -> str:
    return f"{int(frame_index):06d}_{int(timestamp)}"


def relative_to(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
