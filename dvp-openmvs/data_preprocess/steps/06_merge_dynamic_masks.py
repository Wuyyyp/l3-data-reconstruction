#!/usr/bin/env python3
"""Merge converted dynamic-object masks into OpenMVS ignore masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from M18proc.dynamic_policy import VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--dynamic-masks", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--dilate-px", type=int, default=0)
    args = parser.parse_args()
    if args.dilate_px < 0:
        parser.error("--dilate-px must be non-negative")

    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"no input images in {args.images}")
    policy_path = args.dynamic_masks.resolve().parent / 'dynamic_mask_policy.json'
    policy = json.loads(policy_path.read_text())
    if policy.get('version') != VERSION:
        raise RuntimeError('Regenerate dynamic masks with the current map XY >3 m window policy')
    frame_count = len({path.stem.split('_')[0] for path in image_paths})
    if len(policy.get('frame_ids', [])) != frame_count:
        raise RuntimeError('Dynamic policy window does not match the reconstruction frame count')

    openmvs_masks = args.mask_root / "openmvs_masks"
    if not openmvs_masks.is_dir():
        raise RuntimeError(f"missing OpenMVS mask directory: {openmvs_masks}")

    kernel = None
    if args.dilate_px:
        size = 2 * args.dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    totals = defaultdict(int)
    per_camera = defaultdict(lambda: defaultdict(int))
    per_image = {}
    masks_with_dynamic_objects = 0

    for image_path in image_paths:
        stem = image_path.stem
        dynamic_path = args.dynamic_masks / f"{stem}.png"
        openmvs_path = openmvs_masks / f"{stem}.mask.png"
        dynamic_raw = cv2.imread(str(dynamic_path), cv2.IMREAD_UNCHANGED)
        base = cv2.imread(str(openmvs_path), cv2.IMREAD_GRAYSCALE)
        if dynamic_raw is None:
            raise RuntimeError(f"missing dynamic mask: {dynamic_path}")
        if base is None:
            raise RuntimeError(f"missing OpenMVS mask: {openmvs_path}")
        if dynamic_raw.shape[:2] != base.shape:
            raise RuntimeError(
                f"mask size mismatch for {stem}: dynamic={dynamic_raw.shape[:2]}, base={base.shape}"
            )

        if dynamic_raw.ndim == 3:
            dynamic = np.any(dynamic_raw != 0, axis=2).astype(np.uint8)
        else:
            dynamic = (dynamic_raw != 0).astype(np.uint8)
        raw_dynamic_pixels = int(np.count_nonzero(dynamic))
        if kernel is not None and raw_dynamic_pixels:
            dynamic = cv2.dilate(dynamic, kernel, iterations=1)
        dynamic_pixels = int(np.count_nonzero(dynamic))
        masks_with_dynamic_objects += int(dynamic_pixels > 0)

        base_ignored = base >= 128
        merged = np.where(base_ignored | (dynamic > 0), 255, 0).astype(np.uint8)
        temp = openmvs_path.with_name(openmvs_path.stem + ".tmp.png")
        if not cv2.imwrite(str(temp), merged):
            raise RuntimeError(f"cannot write {temp}")
        os.replace(temp, openmvs_path)

        camera = stem.rsplit("_", 1)[-1]
        values = {
            "pixels": int(merged.size),
            "base_ignored_pixels": int(np.count_nonzero(base_ignored)),
            "raw_dynamic_pixels": raw_dynamic_pixels,
            "dynamic_pixels": dynamic_pixels,
            "merged_ignored_pixels": int(np.count_nonzero(merged)),
        }
        per_image[stem] = values
        for key, value in values.items():
            totals[key] += value
            per_camera[camera][key] += value

    expected = {path.stem for path in image_paths}
    actual = {path.stem for path in args.dynamic_masks.glob("*.png")}
    extras = sorted(actual - expected)
    if extras:
        raise RuntimeError(f"unexpected dynamic masks (first 10): {extras[:10]}")

    report = {
        "status": "PASS",
        "image_count": len(image_paths),
        "dynamic_mask_count": len(actual),
        "masks_with_dynamic_objects": masks_with_dynamic_objects,
        "dynamic_dilation_px": args.dilate_px,
        "dynamic_policy": VERSION,
        "dynamic_policy_report": str(policy_path),
        "mask_semantics": "nonzero dynamic input is converted to OpenMVS ignore label 255",
        "totals": dict(totals),
        "dynamic_fraction": totals["dynamic_pixels"] / max(1, totals["pixels"]),
        "per_camera": {camera: dict(data) for camera, data in sorted(per_camera.items())},
        "per_image": per_image,
    }
    output = args.mask_root / "dynamic_mask_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "image_count", "dynamic_mask_count",
        "masks_with_dynamic_objects", "dynamic_fraction",
    )}, indent=2))


if __name__ == "__main__":
    main()
