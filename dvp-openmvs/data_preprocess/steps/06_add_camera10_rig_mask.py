#!/usr/bin/env python3
"""Merge the fixed camera-10 ego-rig polygon into OpenMVS ignore masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TOPLINE = (
    "0.000,0.815;0.080,0.805;0.180,0.775;0.320,0.750;0.430,0.725;"
    "0.570,0.722;0.680,0.745;0.820,0.780;0.920,0.808;1.000,0.815"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=101)
    parser.add_argument("--topline", default=DEFAULT_TOPLINE)
    args = parser.parse_args()

    top_normalized = [tuple(map(float, point.split(","))) for point in args.topline.split(";")]
    preview = args.mask_root / "camera10_rig_previews"
    preview.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(args.frame_count):
        stem = f"{frame:06d}_10"
        rgb = cv2.imread(str(args.images / f"{stem}.jpg"))
        mask_path = args.mask_root / "openmvs_masks" / f"{stem}.mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise FileNotFoundError(stem)
        height, width = mask.shape
        top = np.asarray(
            [[round(x * (width - 1)), round(y * (height - 1))] for x, y in top_normalized],
            dtype=np.int32,
        )
        polygon = np.vstack([top, [[width - 1, height - 1], [0, height - 1]]]).astype(np.int32)
        rig = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(rig, [polygon], 255)
        merged = cv2.bitwise_or(mask, rig)
        if not cv2.imwrite(str(mask_path), merged):
            raise RuntimeError(f"cannot write {mask_path}")
        if frame in {0, 25, 50, 75, 100}:
            overlay = rgb.copy()
            overlay[rig > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)
            cv2.imwrite(str(preview / f"{stem}.jpg"), overlay)
        rows.append(
            {
                "stem": stem,
                "rig_pixels": int(np.count_nonzero(rig)),
                "final_ignored_pixels": int(np.count_nonzero(merged)),
            }
        )
    report = {"status": "PASS", "topline_normalized": top_normalized, "rows": rows}
    (args.mask_root / "camera10_rig_mask_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"status": "PASS", "frames": len(rows)}))


if __name__ == "__main__":
    main()
