#!/usr/bin/env python3
"""Recompute one MoGe-3 metric prior without an upper depth cutoff."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import torch

from export_metric_depth_ply import make_points, read_dmb, read_camera, resize_inputs, write_ply


def write_dmb(path: Path, depth: np.ndarray) -> None:
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    path.write_bytes(struct.pack("<4i", 1, depth.shape[0], depth.shape[1], 5) + depth.tobytes())


def colorize_unbounded(depth: np.ndarray, minimum: float, display_maximum: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= minimum)
    normalized = np.clip((depth - minimum) / (display_maximum - minimum), 0.0, 1.0)
    bgr = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def depth_stats(depth: np.ndarray) -> dict[str, float | int]:
    values = depth[np.isfinite(depth) & (depth > 0)]
    result: dict[str, float | int] = {
        "valid_pixels": int(values.size),
        "valid_fraction": float(values.size / depth.size),
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
    }
    for percentile in (50, 90, 95, 99, 99.9):
        result[f"p{percentile:g}_m"] = float(np.percentile(values, percentile))
    for threshold in (40, 60, 75, 80, 100, 150, 200):
        result[f"pixels_gt_{threshold}m"] = int(np.count_nonzero(values > threshold))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--openmvs-mask", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--resize-factor", type=int, default=4)
    parser.add_argument("--display-maximum", type=float, default=200.0)
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text())
    records = {item["source_label"]: item for item in manifest["images"]}
    if args.label not in records:
        parser.error(f"Unknown label: {args.label}")
    record = records[args.label]
    image_id = int(record["id"])
    stem = f"{image_id:08d}"
    source = Path(manifest["source"])
    args.output.mkdir(parents=True, exist_ok=True)

    source_bgr = cv2.imread(str(source / "images" / f"{args.label}.jpg"), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise RuntimeError("Could not read source image")
    target_width, target_height = manifest["output_size"]
    source_height, source_width = source_bgr.shape[:2]
    scale_xy = min(target_width / source_width, target_height / source_height)
    content_width = int(round(source_width * scale_xy))
    content_height = int(round(source_height * scale_xy))
    pad_x = (target_width - content_width) // 2
    pad_y = (target_height - content_height) // 2
    content_bgr = cv2.resize(source_bgr, (content_width, content_height), interpolation=cv2.INTER_AREA)
    content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)

    from moge.model import import_model_class_by_version

    model = import_model_class_by_version("v3").from_pretrained(str(args.weights)).to(args.device).eval()
    tensor = torch.from_numpy(content_rgb).to(args.device).permute(2, 0, 1).float().div_(255.0)
    output = model.infer(
        tensor,
        resolution_level=args.resolution_level,
        refine_steps=args.refine_steps,
        apply_mask=False,
        use_fp16=True,
    )
    predicted = output["depth"].detach().cpu().numpy().astype(np.float32)
    model_mask = output["mask"].detach().cpu().numpy().astype(bool)
    del tensor, output, model

    if predicted.shape != (content_height, content_width):
        predicted = cv2.resize(predicted, (content_width, content_height), interpolation=cv2.INTER_LINEAR)
        model_mask = cv2.resize(
            model_mask.astype(np.uint8), (content_width, content_height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    openmvs_mask = cv2.imread(str(args.openmvs_mask), cv2.IMREAD_GRAYSCALE)
    if openmvs_mask is None or openmvs_mask.shape != (source_height, source_width):
        raise RuntimeError(f"Invalid OpenMVS mask: {args.openmvs_mask}")
    ignored = cv2.resize(openmvs_mask, (content_width, content_height), interpolation=cv2.INTER_NEAREST) >= 128
    geometry_valid = model_mask & ~ignored & np.isfinite(predicted) & (predicted > 0)

    lidar = np.load(source / "lidar_depth" / f"{args.label}.npy", allow_pickle=True).item()
    source_y, source_x = np.nonzero(lidar["mask"])
    lidar_values = np.asarray(lidar["value"], dtype=np.float32)
    resized_x = np.clip(np.rint((source_x + 0.5) * scale_xy - 0.5).astype(int), 0, content_width - 1)
    resized_y = np.clip(np.rint((source_y + 0.5) * scale_xy - 0.5).astype(int), 0, content_height - 1)
    sampled = predicted[resized_y, resized_x]
    calibration_base = (
        np.isfinite(sampled)
        & (sampled > 0)
        & geometry_valid[resized_y, resized_x]
        & np.isfinite(lidar_values)
        & (lidar_values > 2.0)
    )
    calibration_under_80m = calibration_base & (lidar_values < 80.0)
    ratios_all = lidar_values[calibration_base] / sampled[calibration_base]
    ratios_under_80m = lidar_values[calibration_under_80m] / sampled[calibration_under_80m]
    if ratios_all.size < 100:
        raise RuntimeError(f"Only {ratios_all.size} unbounded LiDAR calibration pixels")
    scale_unbounded = float(np.median(ratios_all))
    scale_under_80m = float(np.median(ratios_under_80m))

    unmasked_content_depth = predicted * scale_unbounded
    valid_content_depth = np.where(geometry_valid, unmasked_content_depth, 0.0).astype(np.float32)
    unmasked_depth = np.zeros((target_height, target_width), np.float32)
    valid_depth = np.zeros_like(unmasked_depth)
    bgr = np.zeros((target_height, target_width, 3), np.uint8)
    unmasked_depth[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = unmasked_content_depth
    valid_depth[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = valid_content_depth
    bgr[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = content_bgr

    old_depth = read_dmb(args.run / "scene" / "metric_prior" / f"{stem}.dmb")
    world_to_camera, intrinsic = read_camera(args.run / "scene" / "cams" / f"{stem}_cam.txt")

    cv2.imwrite(str(args.output / f"{args.label}_rgb.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(args.output / f"{args.label}_moge3_raw_unmasked_m.npy", unmasked_depth)
    np.save(args.output / f"{args.label}_moge3_unbounded_valid_m.npy", valid_depth)
    write_dmb(args.output / f"{stem}_moge3_unbounded.dmb", valid_depth)
    valid_mask_image = (valid_depth > 0).astype(np.uint8) * 255
    cv2.imwrite(str(args.output / f"{args.label}_valid_mask.png"), valid_mask_image)
    far_mask = np.isfinite(valid_depth) & (valid_depth > 80.0)
    cv2.imwrite(str(args.output / f"{args.label}_far_gt80m_mask.png"), far_mask.astype(np.uint8) * 255)
    far_overlay = bgr.copy()
    far_overlay[far_mask] = np.rint(
        0.35 * far_overlay[far_mask].astype(np.float32) + 0.65 * np.array([0, 0, 255], np.float32)
    ).astype(np.uint8)
    cv2.imwrite(str(args.output / f"{args.label}_far_gt80m_overlay.jpg"), far_overlay)

    old_vis = colorize_unbounded(old_depth, 0.5, 80.0)
    new_vis_80 = colorize_unbounded(valid_depth, 0.5, 80.0)
    new_vis_long = colorize_unbounded(valid_depth, 0.5, args.display_maximum)
    cv2.imwrite(str(args.output / f"{args.label}_old_clipped_0_80m.png"), old_vis)
    cv2.imwrite(str(args.output / f"{args.label}_unbounded_vis_0_80m.png"), new_vis_80)
    cv2.imwrite(
        str(args.output / f"{args.label}_unbounded_vis_0_{args.display_maximum:g}m.png"), new_vis_long
    )

    points = make_points(valid_depth, bgr, world_to_camera, intrinsic, 0.5, float("inf"), False)
    far_points = make_points(valid_depth, bgr, world_to_camera, intrinsic, 80.0, float("inf"), False)
    write_ply(args.output / f"{args.label}_moge3_unbounded_world.ply", points)
    write_ply(args.output / f"{args.label}_moge3_far_gt80m_world.ply", far_points)

    resized_depth, resized_bgr, resized_intrinsic = resize_inputs(
        valid_depth, bgr, intrinsic, args.resize_factor
    )
    resized_vis = colorize_unbounded(resized_depth, 0.5, args.display_maximum)
    suffix = f"_r{args.resize_factor}"
    cv2.imwrite(str(args.output / f"{args.label}_rgb{suffix}.jpg"), resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(args.output / f"{args.label}_moge3_unbounded_m{suffix}.npy", resized_depth)
    cv2.imwrite(
        str(args.output / f"{args.label}_unbounded_vis_0_{args.display_maximum:g}m{suffix}.png"), resized_vis
    )
    resized_points = make_points(
        resized_depth, resized_bgr, world_to_camera, resized_intrinsic, 0.5, float("inf"), False
    )
    write_ply(args.output / f"{args.label}_moge3_unbounded_world{suffix}.ply", resized_points)

    comparison = np.concatenate((bgr, old_vis, new_vis_80, new_vis_long), axis=1)
    labels = ("RGB", "old prior 0-80m", "new unbounded (display 0-80m)", f"new unbounded (display 0-{args.display_maximum:g}m)")
    for index, label in enumerate(labels):
        cv2.putText(comparison, label, (index * target_width + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output / f"{args.label}_comparison.png"), comparison)

    report = {
        "label": args.label,
        "image_id": image_id,
        "model": "MoGe-3 ViT-L",
        "weights": str(args.weights),
        "resolution_level": args.resolution_level,
        "refine_steps": args.refine_steps,
        "upper_depth_cutoff_m": None,
        "model_apply_mask": False,
        "point_cloud_mask": "MoGe geometry-valid mask AND inverse OpenMVS ignore mask; no upper depth cutoff",
        "output_resolution": [target_width, target_height],
        "resize_factor": args.resize_factor,
        "resized_output_resolution": [resized_depth.shape[1], resized_depth.shape[0]],
        "calibration": {
            "unbounded_lidar_pixels": int(ratios_all.size),
            "unbounded_scale": scale_unbounded,
            "under_80m_lidar_pixels": int(ratios_under_80m.size),
            "under_80m_scale_for_comparison": scale_under_80m,
            "previous_manifest_scale": float(record["moge_scale_multiplier"]),
            "lidar_depth_maximum_m": float(lidar_values[calibration_base].max()),
        },
        "old_clipped_prior": depth_stats(old_depth),
        "new_unbounded_valid_depth": depth_stats(valid_depth),
        "new_unmasked_all_pixel_depth": depth_stats(unmasked_depth),
        "full_ply_points": int(len(points)),
        "far_gt80m_ply_points": int(len(far_points)),
        "resize4_ply_points": int(len(resized_points)),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
