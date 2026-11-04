#!/usr/bin/env python3
"""Export saved metric-prior DMB depth maps as world-coordinate PLY files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


def read_dmb(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        channels, height, width, type_code = struct.unpack("<4i", stream.read(16))
        if channels != 1 or type_code != 5:
            raise ValueError(f"Unsupported DMB header in {path}: {(channels, height, width, type_code)}")
        depth = np.fromfile(stream, dtype="<f4")
    if depth.size != height * width:
        raise ValueError(f"Truncated DMB file: {path}")
    return depth.reshape(height, width)


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    extrinsic_at = lines.index("extrinsic")
    intrinsic_at = lines.index("intrinsic")
    world_to_camera = np.array(
        [[float(value) for value in lines[extrinsic_at + 1 + row].split()] for row in range(4)],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [[float(value) for value in lines[intrinsic_at + 1 + row].split()] for row in range(3)],
        dtype=np.float64,
    )
    return world_to_camera, intrinsic


def colorize_depth(depth: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Map metric Z depth to Turbo colors, leaving invalid pixels black."""
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    normalized = np.clip((depth - minimum) / max(maximum - minimum, 1e-6), 0, 1)
    gray = np.rint(normalized * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def resize_inputs(
    depth: np.ndarray,
    bgr: np.ndarray,
    intrinsic: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor == 1:
        return depth, bgr, intrinsic
    height, width = depth.shape
    if width % factor or height % factor:
        raise ValueError(f"Input size {width}x{height} is not divisible by resize factor {factor}")
    output_size = (width // factor, height // factor)
    nearest = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    resized_depth = cv2.resize(depth, output_size, interpolation=nearest)
    resized_bgr = cv2.resize(bgr, output_size, interpolation=cv2.INTER_AREA)
    resized_intrinsic = intrinsic.copy()
    resized_intrinsic[0, 0] /= factor
    resized_intrinsic[1, 1] /= factor
    resized_intrinsic[0, 2] = (intrinsic[0, 2] + 0.5) / factor - 0.5
    resized_intrinsic[1, 2] = (intrinsic[1, 2] + 0.5) / factor - 0.5
    return resized_depth, resized_bgr, resized_intrinsic


def make_points(
    depth: np.ndarray,
    bgr: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    minimum: float,
    maximum: float,
    depth_colors: bool,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    y, x = np.nonzero(valid)
    z = depth[y, x].astype(np.float64)
    xyz_camera = np.column_stack(
        ((x - intrinsic[0, 2]) * z / intrinsic[0, 0],
         (y - intrinsic[1, 2]) * z / intrinsic[1, 1],
         z)
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    xyz_world = xyz_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    ranges = np.linalg.norm(xyz_camera, axis=1)

    if depth_colors:
        normalized = np.clip((z - minimum) / max(maximum - minimum, 1e-6), 0, 1)
        color_bgr = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = color_bgr[:, 0, ::-1]
    else:
        rgb = bgr[y, x, ::-1]

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("depth_z", "<f4"), ("range", "<f4"),
    ])
    points = np.empty(len(z), dtype=dtype)
    points["x"], points["y"], points["z"] = xyz_world.T.astype(np.float32)
    points["red"], points["green"], points["blue"] = rgb.T
    points["depth_z"] = z.astype(np.float32)
    points["range"] = ranges.astype(np.float32)
    return points


def write_ply(path: Path, points: np.ndarray) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property float depth_z", "property float range", "end_header", "",
    ]).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)


def write_overview(
    path: Path,
    panels: list[tuple[str, np.ndarray, int]],
    minimum: float,
    maximum: float,
    columns: int = 4,
) -> None:
    source_height, source_width = panels[0][1].shape[:2]
    overview_scale = min(1.0, 480 / source_width, 320 / source_height)
    tile_width = int(round(source_width * overview_scale))
    tile_height = int(round(source_height * overview_scale))
    rows = (len(panels) + columns - 1) // columns
    canvas = np.zeros((rows * tile_height + 80, columns * tile_width, 3), np.uint8)
    for index, (label, image, count) in enumerate(panels):
        row, column = divmod(index, columns)
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        text = f"{label}  valid={count:,}"
        cv2.putText(tile, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
        y, x = row * tile_height, column * tile_width
        canvas[y:y + tile_height, x:x + tile_width] = tile

    bar_y = rows * tile_height + 22
    bar_x0, bar_x1 = 80, columns * tile_width - 80
    ramp = np.linspace(0, 255, bar_x1 - bar_x0, dtype=np.uint8)[None, :]
    colorbar = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    canvas[bar_y:bar_y + 24, bar_x0:bar_x1] = colorbar
    for value in np.linspace(minimum, maximum, 5):
        x = int(round(bar_x0 + (value - minimum) / (maximum - minimum) * (bar_x1 - bar_x0 - 1)))
        cv2.line(canvas, (x, bar_y + 24), (x, bar_y + 31), (255, 255, 255), 1)
        cv2.putText(canvas, f"{value:g}m", (x - 18, bar_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-min", type=float, default=0.5)
    parser.add_argument("--depth-max", type=float, default=80.0)
    parser.add_argument("--far-min", type=float, default=40.0)
    parser.add_argument("--write-depth-rgb", action="store_true")
    parser.add_argument("--write-rgb", action="store_true")
    parser.add_argument("--write-depth-npy", action="store_true")
    parser.add_argument("--full-depth-colors", action="store_true")
    parser.add_argument("--skip-far", action="store_true")
    parser.add_argument("--resize-factor", type=int, default=1)
    parser.add_argument("--overview-columns", type=int, default=4)
    args = parser.parse_args()
    if args.resize_factor < 1:
        parser.error("--resize-factor must be at least 1")

    manifest = json.loads((args.run / "manifest.json").read_text())
    by_label = {item["source_label"]: item for item in manifest["images"]}
    args.output.mkdir(parents=True, exist_ok=True)
    combined = []
    panels = []
    report = {
        "depth_convention": "camera optical-axis Z in meters",
        "visualization_range_m": [args.depth_min, args.depth_max],
        "visualization_colormap": "OpenCV Turbo; invalid pixels are black",
        "resize_factor": args.resize_factor,
        "items": [],
    }

    for label in args.labels:
        item = by_label[label]
        image_id = int(item["id"])
        stem = f"{image_id:08d}"
        depth = read_dmb(args.run / "scene" / "metric_prior" / f"{stem}.dmb")
        bgr = cv2.imread(str(args.run / "scene" / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != depth.shape:
            raise ValueError(f"Missing or mismatched image for {label}")
        world_to_camera, intrinsic = read_camera(args.run / "scene" / "cams" / f"{stem}_cam.txt")
        depth, bgr, intrinsic = resize_inputs(depth, bgr, intrinsic, args.resize_factor)
        suffix = f"_r{args.resize_factor}" if args.resize_factor > 1 else ""

        if args.write_rgb:
            rgb_path = args.output / f"{label}_rgb{suffix}.jpg"
            if not cv2.imwrite(str(rgb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {rgb_path}")
        if args.write_depth_npy:
            np.save(args.output / f"{label}_depth_m{suffix}.npy", np.ascontiguousarray(depth, dtype=np.float32))

        full = make_points(
            depth, bgr, world_to_camera, intrinsic, args.depth_min, args.depth_max,
            args.full_depth_colors,
        )
        valid_depths = depth[np.isfinite(depth) & (depth >= args.depth_min) & (depth <= args.depth_max)]
        if args.write_depth_rgb:
            depth_bgr = colorize_depth(depth, args.depth_min, args.depth_max)
            depth_path = args.output / f"{label}_depth_rgb{suffix}.png"
            if not cv2.imwrite(str(depth_path), depth_bgr):
                raise RuntimeError(f"Could not write {depth_path}")
            panels.append((label, depth_bgr, len(full)))
        write_ply(args.output / f"{label}_depth{suffix}.ply", full)
        item_report = {
            "label": label, "image_id": image_id, "full_points": len(full),
            "depth_z_min": float(full["depth_z"].min()), "depth_z_max": float(full["depth_z"].max()),
            "depth_z_p50": float(np.quantile(valid_depths, 0.50)),
            "depth_z_p90": float(np.quantile(valid_depths, 0.90)),
            "depth_z_p99": float(np.quantile(valid_depths, 0.99)),
            "points_ge_40m": int(np.count_nonzero(valid_depths >= 40.0)),
            "points_ge_60m": int(np.count_nonzero(valid_depths >= 60.0)),
            "points_ge_75m": int(np.count_nonzero(valid_depths >= 75.0)),
            "euclidean_range_max": float(full["range"].max()),
        }
        if not args.skip_far:
            far = make_points(depth, bgr, world_to_camera, intrinsic, args.far_min, args.depth_max, True)
            write_ply(args.output / f"{label}_moge_metric_far_{args.far_min:g}_{args.depth_max:g}m.ply", far)
            combined.append(far)
            item_report.update({
                "far_points": len(far),
                "far_fraction": len(far) / max(len(full), 1),
            })
        report["items"].append(item_report)

    if combined:
        merged = np.concatenate(combined)
        write_ply(args.output / f"combined_moge_metric_far_{args.far_min:g}_{args.depth_max:g}m.ply", merged)
        report["combined_far_points"] = len(merged)
    if panels:
        write_overview(
            args.output / f"depth_rgb_overview_{len(panels)}{suffix}.png",
            panels, args.depth_min, args.depth_max, args.overview_columns,
        )
    report["aggregate"] = {
        "selected_images": len(report["items"]),
        "valid_points": sum(item["full_points"] for item in report["items"]),
        "points_ge_40m": sum(item["points_ge_40m"] for item in report["items"]),
        "points_ge_60m": sum(item["points_ge_60m"] for item in report["items"]),
        "points_ge_75m": sum(item["points_ge_75m"] for item in report["items"]),
    }
    if report["items"]:
        report["output_resolution"] = [int(depth.shape[1]), int(depth.shape[0])]
    (args.output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
