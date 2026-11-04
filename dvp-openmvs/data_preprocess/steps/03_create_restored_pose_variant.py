#!/usr/bin/env python3
"""Create a converted-data view that replaces only camera-to-world poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CAMERAS = {
    "00": "left_front_camera",
    "01": "right_front_camera",
    "02": "rear_camera",
    "03": "left_rear_camera",
    "04": "right_rear_camera",
    "09": "front_camera_fov30",
    "10": "front_camera_fov120",
}
TIMESTAMP_ALIASES = {
    "front_camera_fov30": "center_camera_fov30",
    "front_camera_fov120": "center_camera_fov120",
}


def load_pose_rows(path: Path, start: int, end: int):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            camera = row["camera"]
            if start <= frame <= end and camera in CAMERAS.values():
                matrix = np.array([
                    [float(row[f"T_cw_{i}{j}"]) for j in range(4)]
                    for i in range(4)
                ])
                rows[(frame, camera)] = matrix, int(row["timestamp_ns"])
    return rows


def link_shared_inputs(source: Path, output: Path) -> None:
    for entry in source.iterdir():
        if entry.name in {"ego_pose", "input_manifest.json"}:
            continue
        target = output / entry.name
        if target.is_symlink():
            if target.resolve() != entry.resolve():
                raise RuntimeError(f"wrong existing symlink: {target}")
        elif target.exists():
            raise RuntimeError(f"refusing to replace existing path: {target}")
        else:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path,
                        help="Existing normalized converted directory")
    parser.add_argument("--pose-csv", required=True, type=Path,
                        help="image_camera_to_world_after.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output converted-data view")
    parser.add_argument("--original-frame-start", required=True, type=int)
    parser.add_argument("--frame-count", type=int, default=101)
    args = parser.parse_args()

    if args.frame_count <= 0:
        parser.error("--frame-count must be positive")
    source = args.source.resolve()
    pose_csv = args.pose_csv.resolve()
    if not (source / "timestamps_specific.json").is_file():
        raise RuntimeError(f"missing timestamps_specific.json in {source}")
    if not pose_csv.is_file():
        raise RuntimeError(f"missing pose CSV: {pose_csv}")

    end = args.original_frame_start + args.frame_count - 1
    pose_rows = load_pose_rows(pose_csv, args.original_frame_start, end)
    expected = args.frame_count * len(CAMERAS)
    if len(pose_rows) != expected:
        raise RuntimeError(f"expected {expected} pose rows, found {len(pose_rows)}")

    timestamps = json.loads((source / "timestamps_specific.json").read_text())
    if len(timestamps) != args.frame_count:
        raise RuntimeError(
            f"expected {args.frame_count} timestamp records, found {len(timestamps)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    link_shared_inputs(source, args.output)
    pose_output = args.output / "ego_pose"
    pose_output.mkdir(exist_ok=True)

    translation_changes = []
    rotation_changes = []
    max_timestamp_difference_ns = 0
    for local_frame in range(args.frame_count):
        original_frame = args.original_frame_start + local_frame
        camera_timestamps = {
            camera: int(value["timestamp"]) * 1000
            for camera, value in timestamps[local_frame]["cameras"].items()
        }
        for code, csv_camera in CAMERAS.items():
            matrix, csv_timestamp = pose_rows[(original_frame, csv_camera)]
            timestamp_camera = TIMESTAMP_ALIASES.get(csv_camera, csv_camera)
            difference = abs(csv_timestamp - camera_timestamps[timestamp_camera])
            max_timestamp_difference_ns = max(max_timestamp_difference_ns, difference)

            label = f"{local_frame:06d}_{code}"
            original = np.loadtxt(source / "ego_pose" / f"{label}.txt")
            translation_changes.append(float(np.linalg.norm(
                original[:3, 3] - matrix[:3, 3]
            )))
            rotation_changes.append(float(np.linalg.norm(
                original[:3, :3] - matrix[:3, :3]
            )))
            np.savetxt(pose_output / f"{label}.txt", matrix, fmt="%.18e")

    output_poses = list(pose_output.glob("??????_??.txt"))
    if len(output_poses) != expected:
        raise RuntimeError(f"expected {expected} output poses, found {len(output_poses)}")
    if max_timestamp_difference_ns != 0:
        raise RuntimeError(
            f"pose/image timestamp mismatch: max {max_timestamp_difference_ns} ns"
        )

    report = {
        "status": "PASS",
        "source": str(source),
        "pose_csv": str(pose_csv),
        "original_frame_range_inclusive": [args.original_frame_start, end],
        "normalized_frame_range_inclusive": [0, args.frame_count - 1],
        "images_and_poses": expected,
        "max_timestamp_difference_ns": max_timestamp_difference_ns,
        "translation_change_m": {
            "median": float(np.median(translation_changes)),
            "max": float(np.max(translation_changes)),
        },
        "rotation_change_frobenius": {
            "median": float(np.median(rotation_changes)),
            "max": float(np.max(rotation_changes)),
        },
        "shared_inputs": "all converted inputs except ego_pose",
    }
    audit = args.output / "restored_pose_variant_audit.json"
    audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
