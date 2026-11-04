#!/usr/bin/env python3
"""Derive the OpenMVS depth search range of every view from COLMAP sparse points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_points(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: malformed point record")
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
    if not points:
        raise ValueError(f"no sparse points found in {path}")
    return points


def read_images(path: Path) -> list[tuple[str, np.ndarray, np.ndarray, list[int]]]:
    raw_lines = path.read_text().splitlines()
    records: list[tuple[str, np.ndarray, np.ndarray, list[int]]] = []
    index = 0
    while index < len(raw_lines):
        header = raw_lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        fields = header.split()
        if len(fields) < 10:
            raise ValueError(f"{path}:{index}: malformed image record")
        while index < len(raw_lines) and raw_lines[index].lstrip().startswith("#"):
            index += 1
        observations = raw_lines[index].split() if index < len(raw_lines) else []
        index += 1
        point_ids = [int(value) for value in observations[2::3] if int(value) >= 0]
        records.append(
            (
                Path(fields[9]).stem,
                np.asarray(fields[1:5], dtype=np.float64),
                np.asarray(fields[5:8], dtype=np.float64),
                point_ids,
            )
        )
    if not records or len({record[0] for record in records}) != len(records):
        raise ValueError(f"could not read unique image records from {path}")
    return records


def frame_position(label: str, order: int) -> tuple[str, float]:
    frame, camera = label.rsplit("_", 1)
    try:
        position = float(frame)
    except ValueError:
        position = float(order)
    return camera, position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="COLMAP text model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-scale", type=float, default=0.9)
    parser.add_argument("--far-scale", type=float, default=1.1)
    args = parser.parse_args()
    if not 0 < args.near_scale < 1:
        parser.error("expected 0 < --near-scale < 1")
    if args.far_scale <= 1:
        parser.error("expected --far-scale > 1")

    points = read_points(args.model / "points3D.txt")
    images = read_images(args.model / "images.txt")
    direct: dict[str, dict[str, object]] = {}
    missing: list[tuple[int, str]] = []
    positions = {label: frame_position(label, order) for order, (label, *_rest) in enumerate(images)}

    for order, (label, quaternion, translation, point_ids) in enumerate(images):
        rotation = quaternion_to_rotation(quaternion)
        depths = np.asarray(
            [(rotation @ points[point_id] + translation)[2] for point_id in set(point_ids) if point_id in points],
            dtype=np.float64,
        )
        depths = depths[np.isfinite(depths) & (depths > 0)]
        if depths.size:
            direct[label] = {
                "depth_min_m": float(args.near_scale * depths.min()),
                "depth_max_m": float(args.far_scale * depths.max()),
                "sparse_points": int(depths.size),
                "source": (
                    f"visible_sparse_points_minmax_times_{args.near_scale:g}_{args.far_scale:g}"
                ),
            }
        else:
            missing.append((order, label))

    ranges = dict(direct)
    for order, label in missing:
        camera, position = positions[label]
        candidates = [
            (abs(positions[candidate][1] - position), candidate, values)
            for candidate, values in direct.items()
            if positions[candidate][0] == camera
        ]
        if not candidates:
            raise RuntimeError(f"no sparse depth range or same-camera fallback for {label}")
        _distance, neighbor, values = min(candidates)
        ranges[label] = {
            **values,
            "sparse_points": 0,
            "source": f"nearest_same_camera_sparse_range:{neighbor}",
        }

    if set(ranges) != {record[0] for record in images}:
        raise RuntimeError("sparse range/image mismatch")
    values = list(ranges.values())
    minimums = [float(value["depth_min_m"]) for value in values]
    maximums = [float(value["depth_max_m"]) for value in values]
    report = {
        "method": "OpenMVS sparse-point range: per-image visible positive camera-Z min*0.9 and max*1.1",
        "near_scale": args.near_scale,
        "far_scale": args.far_scale,
        "fixed_0_100_fallback": False,
        "image_count": len(ranges),
        "direct_count": len(direct),
        "nearest_same_camera_fallback_count": len(missing),
        "depth_min_m_summary": {
            "min": min(minimums),
            "median": float(np.median(minimums)),
            "max": max(minimums),
        },
        "depth_max_m_summary": {
            "min": min(maximums),
            "median": float(np.median(maximums)),
            "max": max(maximums),
        },
        "ranges": dict(sorted(ranges.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "ranges"}, indent=2))


if __name__ == "__main__":
    main()
