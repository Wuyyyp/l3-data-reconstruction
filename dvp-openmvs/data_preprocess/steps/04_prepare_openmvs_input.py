#!/usr/bin/env python3
"""Create a lightweight manifest view over an existing converted sample."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


CAMERAS = {
    "00": "left_front_camera", "01": "right_front_camera", "02": "rear_camera",
    "03": "left_rear_camera", "04": "right_rear_camera",
    "09": "center_camera_fov30", "10": "center_camera_fov120",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "intrinsics", "extrinsics", "ego_pose", "dynamic_mask",
                 "timestamps.json", "timestamps_specific.json", "pointcloud.npz",
                 "dynamic_mask_policy.json"):
        target = args.output / name
        if not target.exists():
            target.symlink_to((args.source / name).resolve(), target_is_directory=(args.source / name).is_dir())

    records = []
    calibrations = {}
    for code, camera in CAMERAS.items():
        image = cv2.imread(str(args.source / "images" / f"000000_{code}.jpg"))
        if image is None:
            raise FileNotFoundError(code)
        height, width = image.shape[:2]
        calibrations[camera] = {
            "camera_code": code, "width": width, "height": height,
            "K": np.loadtxt(args.source / "intrinsics" / f"000000_{code}.txt").tolist(),
            "distortion_assumption": "converted image; zero distortion",
        }
    for frame in range(101):
        for code, camera in CAMERAS.items():
            label = f"{frame:06d}_{code}.jpg"
            records.append({
                "frame_index": frame, "camera": camera, "camera_code": code,
                "label": label, "image": str(args.output / "images" / label),
                "camera_to_world": np.loadtxt(args.source / "ego_pose" / f"{frame:06d}_{code}.txt").tolist(),
            })
    match = re.search(r"_f(\d+)_(\d+)_", args.source.parent.name)
    start, end = (map(int, match.groups()) if match else (0, 100))
    manifest = {
        "source_raw_clip": "", "frame_range_inclusive": [start, end], "frame_count": 101,
        "camera_names": list(CAMERAS.values()), "image_count": len(records),
        "records": records, "calibrations": calibrations,
        "pose_formula": "camera_to_world = converted/ego_pose/{frame}_{camera}.txt",
        "fixed_pose_requested": True, "bundle_adjustment": False,
    }
    (args.output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "source": str(args.source), "output": str(args.output),
                      "images": len(records)}))


if __name__ == "__main__":
    main()
