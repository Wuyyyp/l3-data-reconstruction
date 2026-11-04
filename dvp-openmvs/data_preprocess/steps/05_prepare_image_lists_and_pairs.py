#!/usr/bin/env python3
"""Validate the converted image inventory and build COLMAP image/pair lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


CAMERAS = ("00", "01", "02", "03", "04", "09", "10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pair-radius", type=int, default=20)
    args = parser.parse_args()

    manifest_path = args.source / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_count = int(manifest["frame_count"])
    calibrations = {item["camera_code"]: item for item in manifest["calibrations"].values()}
    if set(calibrations) != set(CAMERAS):
        raise RuntimeError(f"camera set mismatch: {sorted(calibrations)}")

    expected = [f"{frame:06d}_{camera}.jpg" for frame in range(frame_count) for camera in CAMERAS]
    actual = sorted(path.name for path in (args.source / "images").glob("*.jpg"))
    if actual != sorted(expected):
        raise RuntimeError(f"image inventory mismatch: expected={len(expected)} actual={len(actual)}")

    for name in expected:
        camera = Path(name).stem.rsplit("_", 1)[1]
        calibration = calibrations[camera]
        image = cv2.imread(str(args.source / "images" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot decode image: {name}")
        expected_size = (int(calibration["width"]), int(calibration["height"]))
        actual_size = (image.shape[1], image.shape[0])
        if actual_size != expected_size:
            raise RuntimeError(f"{name}: expected {expected_size}, got {actual_size}")

    args.output.mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        names = [f"{frame:06d}_{camera}.jpg" for frame in range(frame_count)]
        (args.output / f"images_camera_{camera}.txt").write_text("\n".join(names) + "\n")

    pairs: list[str] = []
    for frame_a in range(frame_count):
        for frame_b in range(frame_a, min(frame_count, frame_a + args.pair_radius + 1)):
            for camera_a in CAMERAS:
                for camera_b in CAMERAS:
                    if frame_a == frame_b and camera_a >= camera_b:
                        continue
                    pairs.append(
                        f"{frame_a:06d}_{camera_a}.jpg {frame_b:06d}_{camera_b}.jpg"
                    )
    pair_file = args.output / "colmap_match_pairs.txt"
    pair_file.write_text("\n".join(pairs) + "\n")
    report = {
        "status": "PASS",
        "frames": frame_count,
        "cameras": list(CAMERAS),
        "images": len(actual),
        "pair_radius": args.pair_radius,
        "pairs": len(pairs),
    }
    (args.output / "image_pair_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
