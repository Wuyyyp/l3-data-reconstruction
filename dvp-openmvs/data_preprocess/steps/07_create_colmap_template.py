#!/usr/bin/env python3
"""Create the camera/image text skeleton used for fixed-pose triangulation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    calibrations = {item["camera_code"]: item for item in manifest["calibrations"].values()}
    with sqlite3.connect(args.database) as connection:
        rows = connection.execute(
            "select image_id, name, camera_id from images order by image_id"
        ).fetchall()
    by_code: dict[str, set[int]] = {}
    for _, name, camera_id in rows:
        code = Path(name).stem.rsplit("_", 1)[1]
        by_code.setdefault(code, set()).add(int(camera_id))
    if any(len(camera_ids) != 1 for camera_ids in by_code.values()):
        raise RuntimeError(f"each camera code must map to one COLMAP camera: {by_code}")

    args.output.mkdir(parents=True, exist_ok=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    for code, camera_ids in sorted(by_code.items()):
        item = calibrations[code]
        matrix = item["K"]
        cameras.append(
            f"{next(iter(camera_ids))} PINHOLE {item['width']} {item['height']} "
            f"{matrix[0][0]} {matrix[1][1]} {matrix[0][2]} {matrix[1][2]}"
        )
    (args.output / "cameras.txt").write_text("\n".join(cameras) + "\n")

    images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for image_id, name, camera_id in rows:
        images.extend([f"{image_id} 1 0 0 0 0 0 0 {camera_id} {name}", ""])
    (args.output / "images.txt").write_text("\n".join(images) + "\n")
    (args.output / "points3D.txt").write_text("# empty\n")
    print(json.dumps({"status": "PASS", "images": len(rows), "cameras": len(by_code)}))


if __name__ == "__main__":
    main()
