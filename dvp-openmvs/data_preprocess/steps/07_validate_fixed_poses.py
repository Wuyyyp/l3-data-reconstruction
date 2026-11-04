#!/usr/bin/env python3
"""Check that COLMAP triangulation preserved every supplied fixed pose."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_text(path: Path) -> dict[str, list[float]]:
    poses = {}
    for line in path.open():
        fields = line.split()
        if len(fields) >= 10 and fields[9].lower().endswith((".jpg", ".png", ".jpeg")):
            poses[fields[9]] = list(map(float, fields[1:8]))
    return poses


def read_binary(path: Path) -> dict[str, tuple[float, ...]]:
    poses = {}
    with path.open("rb") as handle:
        for _ in range(struct.unpack("<Q", handle.read(8))[0]):
            row = struct.unpack("<i7di", handle.read(64))
            name = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise EOFError("truncated COLMAP image name")
                if char == b"\0":
                    break
                name.extend(char)
            decoded = name.decode("utf-8")
            poses[decoded] = row[1:8]
            point_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(24 * point_count, 1)
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-images", required=True, type=Path)
    parser.add_argument("--triangulated-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    expected = read_text(args.known_images)
    actual = read_binary(args.triangulated_images)
    same_names = bool(expected) and set(expected) == set(actual)
    finite = all(math.isfinite(value) for data in (expected, actual) for pose in data.values() for value in pose)
    quaternion_error = translation_error = None
    if same_names and finite:
        quaternion_error = max(
            min(
                max(abs(expected[name][i] - actual[name][i]) for i in range(4)),
                max(abs(expected[name][i] + actual[name][i]) for i in range(4)),
            )
            for name in expected
        )
        translation_error = max(
            max(abs(expected[name][i] - actual[name][i]) for i in range(4, 7))
            for name in expected
        )
    passed = (
        same_names
        and finite
        and max(quaternion_error or 0.0, translation_error or 0.0) <= args.tolerance
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "images": len(actual),
        "same_image_names": same_names,
        "finite": finite,
        "max_quaternion_abs_error_sign_invariant": quaternion_error,
        "max_translation_abs_error": translation_error,
        "tolerance": args.tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
