#!/usr/bin/env python3
"""Create zero-valued OpenMVS masks when semantic sky masking is disabled."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("00", "01", "02", "03", "04", "09", "10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=101)
    args = parser.parse_args()
    masks = args.output / "openmvs_masks"
    masks.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in range(args.frame_count):
        for camera in CAMERAS:
            stem = f"{frame:06d}_{camera}"
            image = cv2.imread(str(args.images / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(stem)
            if not cv2.imwrite(str(masks / f"{stem}.mask.png"), np.zeros_like(image)):
                raise RuntimeError(f"cannot write mask for {stem}")
            count += 1
    report = {"status": "PASS", "mode": "empty_masks", "image_count": count}
    (args.output / "mask_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
