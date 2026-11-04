#!/usr/bin/env python3
"""Convert DVP-MVS depths directly to OpenMVS DMAP files.

The OpenMVS image IDs and view graph come from the exact COLMAP/OpenMVS scene;
depth, masks and resize/padding metadata come from the matching DVP run.
"""

from __future__ import annotations

from mask_policy import block_path

import argparse
import json
import re
import struct
from pathlib import Path

import cv2
import numpy as np


HEADER = struct.Struct("<HBBIIIIff")
MAGIC = int.from_bytes(b"DR", "little")
HAS_DEPTH = 1
HAS_NORMAL = 2
IMAGE_PATTERN = re.compile(rb"([0-9]{6}_(?:00|01|02|03|04|09|10)\.jpg)")


def read_colmap_image_ids(path: Path) -> dict[str, int]:
    lines = [line.strip() for line in path.open() if not line.lstrip().startswith("#")]
    if len(lines) % 2:
        raise ValueError(f"malformed COLMAP images file: {path}")
    result: dict[str, int] = {}
    for header in lines[0::2]:
        fields = header.split()
        if len(fields) < 10:
            raise ValueError(f"malformed COLMAP image record: {header!r}")
        image_id = int(fields[0])
        label = Path(fields[9]).stem
        if label in result:
            raise ValueError(f"duplicate COLMAP image label: {label}")
        result[label] = image_id
    return result


def read_scene_names(path: Path) -> list[str]:
    names = [value.decode() for value in IMAGE_PATTERN.findall(path.read_bytes())]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"could not recover unique image names from {path}: {len(names)}")
    return [Path(name).stem for name in names]


def read_neighbors(path: Path, scene_names: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        indexes = [int(value) for value in line.split()]
        if len(indexes) < 2:
            raise ValueError(f"{path}:{line_number}: expected reference and source IDs")
        if any(index < 0 or index >= len(scene_names) for index in indexes):
            raise ValueError(f"{path}:{line_number}: scene image ID out of range")
        reference = scene_names[indexes[0]]
        if reference in result:
            raise ValueError(f"{path}:{line_number}: duplicate reference {reference}")
        result[reference] = [scene_names[index] for index in indexes[1:]]
    return result


def read_dmb(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        version, height, width, type_code = struct.unpack("<4i", handle.read(16))
        if version != 1 or type_code != 5:
            raise ValueError(f"unsupported depth DMB header in {path}")
        data = np.fromfile(handle, dtype="<f4")
    if data.size != height * width:
        raise ValueError(f"truncated depth DMB: {path}")
    return data.reshape(height, width)


def camera_normals(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    x = (xx.astype(np.float32) - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (yy.astype(np.float32) - intrinsic[1, 2]) / intrinsic[1, 1]
    xyz = np.stack((x * depth, y * depth, depth), axis=-1)
    dx = np.empty_like(xyz)
    dy = np.empty_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dx[:, 0] = xyz[:, 1] - xyz[:, 0]
    dx[:, -1] = xyz[:, -1] - xyz[:, -2]
    dy[1:-1] = xyz[2:] - xyz[:-2]
    dy[0] = xyz[1] - xyz[0]
    dy[-1] = xyz[-1] - xyz[-2]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=2)
    valid = depth > 0
    good = valid & np.isfinite(length) & (length > 1e-8)
    normal[good] /= length[good, None]
    ray = np.stack((x, y, np.ones_like(x)), axis=-1)
    ray /= np.linalg.norm(ray, axis=2, keepdims=True)
    normal[valid & ~good] = -ray[valid & ~good]
    flip = valid & (np.sum(normal * ray, axis=2) > 0)
    normal[flip] *= -1
    normal[~valid] = 0
    return np.ascontiguousarray(normal, dtype="<f4")


def write_dmap(
    path: Path,
    image_name: bytes,
    image_ids: list[int],
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
    depth_min: float,
    depth_max: float,
    depth: np.ndarray,
    normal: np.ndarray,
) -> None:
    header = (
        MAGIC,
        HAS_DEPTH | HAS_NORMAL,
        0,
        width,
        height,
        width,
        height,
        depth_min,
        depth_max,
    )
    if depth.shape != (height, width) or normal.shape != (height, width, 3):
        raise ValueError(f"DMAP payload shape mismatch for {path}")
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        handle.write(HEADER.pack(*header))
        handle.write(struct.pack("<H", len(image_name)))
        handle.write(image_name)
        handle.write(struct.pack("<I", len(image_ids)))
        handle.write(np.asarray(image_ids, dtype="<u4").tobytes())
        handle.write(np.asarray(intrinsic, dtype="<f8").tobytes())
        handle.write(np.asarray(rotation, dtype="<f8").tobytes())
        handle.write(np.asarray(center, dtype="<f8").tobytes())
        handle.write(np.ascontiguousarray(depth, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(normal, dtype="<f4").tobytes())
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="DVP run containing manifest.json")
    parser.add_argument("--source", type=Path, required=True, help="normalized converted directory")
    parser.add_argument("--scene", type=Path, required=True, help="OpenMVS scene.mvs")
    parser.add_argument("--images-txt", type=Path, required=True, help="COLMAP text model images.txt")
    parser.add_argument("--neighbors", type=Path, required=True, help="OpenMVS numeric neighbor list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-ranges-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text())
    range_report = json.loads(args.depth_ranges_json.read_text())
    depth_ranges = range_report["ranges"]
    records = {item["source_label"]: item for item in manifest["images"]}
    colmap_ids = read_colmap_image_ids(args.images_txt)
    scene_names = read_scene_names(args.scene)
    neighbors = read_neighbors(args.neighbors, scene_names)
    expected = set(records)
    for name, values in (
        ("COLMAP", set(colmap_ids)),
        ("OpenMVS scene", set(scene_names)),
        ("OpenMVS neighbors", set(neighbors)),
        ("sparse depth ranges", set(depth_ranges)),
    ):
        if values != expected:
            missing = sorted(expected - values)[:5]
            extra = sorted(values - expected)[:5]
            raise RuntimeError(f"{name}/DVP image mismatch; missing={missing}, extra={extra}")

    args.output.mkdir(parents=True, exist_ok=True)
    expected_files: set[str] = set()
    valid_counts: list[int] = []
    for index, label in enumerate(sorted(records), 1):
        record = records[label]
        depth_range = depth_ranges[label]
        depth_min = float(depth_range["depth_min_m"])
        depth_max = float(depth_range["depth_max_m"])
        if not 0 < depth_min < depth_max:
            raise RuntimeError(f"invalid sparse depth range for {label}: {depth_min}-{depth_max}")
        source_width, source_height = record["source_size"]
        content_width, content_height = record["content_size"]
        image_path = args.source / "images" / f"{label}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != (source_width, source_height):
            raise RuntimeError(f"missing or mismatched source image: {image_path}")

        intrinsic = np.loadtxt(args.source / "intrinsics" / f"{label}.txt", dtype=np.float64)
        intrinsic[0, :] *= content_width / source_width
        intrinsic[1, :] *= content_height / source_height
        # OpenMVS shifts the resized principal point by half a pixel.
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5
        camera_to_world = np.loadtxt(args.source / "ego_pose" / f"{label}.txt", dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)

        dvp_id = int(record["id"])
        depth = read_dmb(args.run / "scene" / "APD" / f"{dvp_id:08d}" / "depths.dmb")
        mask = cv2.imread(
            str(block_path(args.run / "scene", dvp_id)),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None or mask.shape != depth.shape:
            raise RuntimeError(f"missing or mismatched DVP mask for {label}")
        pad_x, pad_y = record["padding_xy"]
        depth = depth[pad_y : pad_y + content_height, pad_x : pad_x + content_width].copy()
        mask = mask[pad_y : pad_y + content_height, pad_x : pad_x + content_width]
        valid = (
            np.isfinite(depth)
            & (depth >= depth_min)
            & (depth <= depth_max)
            & (mask >= 128)
        )
        depth[~valid] = 0
        depth = np.ascontiguousarray(depth, dtype="<f4")
        normal = camera_normals(depth, intrinsic)
        valid_counts.append(int(valid.sum()))

        reference_id = colmap_ids[label]
        image_ids = [reference_id] + [colmap_ids[source] for source in neighbors[label]]
        filename = f"depth{reference_id:04d}.dmap"
        expected_files.add(filename)
        write_dmap(
            args.output / filename,
            str(image_path.resolve()).encode(),
            image_ids,
            intrinsic,
            world_to_camera[:3, :3],
            camera_to_world[:3, 3],
            content_width,
            content_height,
            depth_min,
            depth_max,
            depth,
            normal,
        )
        if index % 50 == 0 or index == len(records):
            print(
                f"converted {index}/{len(records)} DMAPs; last={label}; valid={valid_counts[-1]}",
                flush=True,
            )

    stale = [path for path in args.output.glob("depth*.dmap") if path.name not in expected_files]
    if stale:
        raise RuntimeError(f"stale DMAP files in {args.output}: {[path.name for path in stale[:5]]}")
    stats = {
        "files": len(valid_counts),
        "valid_depth_pixels": int(sum(valid_counts)),
        "valid_per_image_min": int(min(valid_counts)),
        "valid_per_image_median": float(np.median(valid_counts)),
        "valid_per_image_max": int(max(valid_counts)),
        "depth_range_mode": "per-view sparse camera-Z min/max with margins",
        "fixed_depth_range": False,
        "depth_min_m_summary": range_report["depth_min_m_summary"],
        "depth_max_m_summary": range_report["depth_max_m_summary"],
        "dmap_channels": ["depth", "camera_normal"],
    }
    (args.output.parent / "conversion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"PASS: wrote {len(expected_files)} OpenMVS DMAP files in {args.output}")


if __name__ == "__main__":
    main()
