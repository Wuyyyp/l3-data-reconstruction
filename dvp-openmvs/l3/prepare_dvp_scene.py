#!/usr/bin/env python3
"""Infer aspect-preserving MoGe-3 priors and prepare masked DVP inputs."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'mvs_process'))
from moge_overlap import raw_prediction
from mask_policy import resize_ignore
import hashlib

import argparse
import json
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(os.environ.get("DVP_RUN_ROOT", Path(__file__).resolve().parent))
SOURCE = Path(os.environ.get(
    "DVP_SOURCE",
    ROOT/"converted",
))
OPENMVS_MASKS = Path(os.environ.get(
    "DVP_OPENMVS_MASKS",
    ROOT/"openmvs_masks",
))
SCENE = ROOT / "scene"
WEIGHTS = Path(os.environ.get(
    "DVP_MOGE_WEIGHTS",
    ROOT/"weights/moge-3-vitl/model.pt",
))
WIDTH, HEIGHT = 960, 640


def atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def write_dmb(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    cv_type = 5 if array.ndim == 2 else 21
    payload = struct.pack("4i", 1, array.shape[0], array.shape[1], cv_type)
    atomic_bytes(path, payload + array.tobytes())


def valid_dmb(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == 16 + WIDTH * HEIGHT * 4


def metadata_matches_request(
    path: Path,
    depth_range: dict[str, object],
    unbounded_lidar_calibration: bool,
    mask_sha256: str,
) -> bool:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    expected_upper_bound = None if unbounded_lidar_calibration else 80.0
    return (
        metadata.get("mask_sha256") == mask_sha256
        and metadata.get("dvp_depth_range_m")
        == [depth_range["depth_min_m"], depth_range["depth_max_m"]]
        and metadata.get("depth_range_source") == depth_range["source"]
        and metadata.get("lidar_calibration_upper_bound_m") == expected_upper_bound
    )


def write_camera(
    path: Path,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> None:
    text = "extrinsic\n"
    text += "\n".join(" ".join(f"{v:.12g}" for v in row) for row in extrinsic)
    text += "\n\nintrinsic\n"
    text += "\n".join(" ".join(f"{v:.12g}" for v in row) for row in intrinsic)
    # APD expands the stored endpoints by 0.6 and 1.2 when creating its search range.
    stored_min = depth_min / 0.6
    stored_max = depth_max / 1.2
    interval = 0.25
    depth_num = max(2, int(round((stored_max - stored_min) / interval)) + 1)
    text += f"\n\n{stored_min:.12g} {interval:.12g} {depth_num} {stored_max:.12g}\n"
    atomic_bytes(path, text.encode())


def geometry(source_width: int, source_height: int):
    scale = min(WIDTH / source_width, HEIGHT / source_height)
    content_width = int(round(source_width * scale))
    content_height = int(round(source_height * scale))
    pad_x = (WIDTH - content_width) // 2
    pad_y = (HEIGHT - content_height) // 2
    return scale, content_width, content_height, pad_x, pad_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument(
        "--depth-ranges-json",
        type=Path,
        required=True,
        help="Per-view sparse-point ranges from derive_sparse_depth_ranges.py.",
    )
    parser.add_argument(
        "--unbounded-lidar-calibration",
        action="store_true",
        help="Use every finite LiDAR depth above 2 m when calibrating the MoGe scale.",
    )
    args = parser.parse_args()
    range_report = json.loads(args.depth_ranges_json.read_text())
    depth_ranges = range_report["ranges"]

    for name in ("images", "cams", "metric_prior", "metadata", "blocks"):
        (SCENE / name).mkdir(parents=True, exist_ok=True)

    image_paths = sorted((SOURCE / "images").glob("*.jpg"))
    image_labels = {path.stem for path in image_paths}
    if set(depth_ranges) != image_labels:
        missing = sorted(image_labels - set(depth_ranges))[:5]
        extra = sorted(set(depth_ranges) - image_labels)[:5]
        raise RuntimeError(f"sparse range/image mismatch; missing={missing}, extra={extra}")
    selected = [(i, p) for i, p in enumerate(image_paths) if i % args.num_shards == args.shard]
    pending = []
    for image_id, image_path in selected:
        stem = f"{image_id:08d}"
        required = [
            SCENE / "images" / f"{stem}.jpg",
            SCENE / "cams" / f"{stem}_cam.txt",
            SCENE / "blocks" / f"mask_{image_id}.png",
        ]
        metadata_path = SCENE / "metadata" / f"{stem}.json"
        depth_range = depth_ranges[image_path.stem]
        if (
            valid_dmb(SCENE / "metric_prior" / f"{stem}.dmb")
            and all(p.is_file() for p in required)
            and metadata_matches_request(
                metadata_path,
                depth_range,
                args.unbounded_lidar_calibration,
                hashlib.sha256((OPENMVS_MASKS / f"{image_path.stem}.mask.png").read_bytes()).hexdigest(),
            )
        ):
            print(f"[{args.shard}] skip complete {stem} {image_path.stem}", flush=True)
        else:
            pending.append((image_id, image_path))

    if not pending:
        print(f"[{args.shard}] all {len(selected)} images already complete", flush=True)
        return

    model = None
    cache = Path(os.environ['WYS_CASE_ROOT'])/'.moge_overlap_cache'
    reused = 0
    started = time.time()
    for done, (image_id, image_path) in enumerate(pending, 1):
        label = image_path.stem
        stem = f"{image_id:08d}"
        depth_range = depth_ranges[label]
        depth_min = float(depth_range["depth_min_m"])
        depth_max = float(depth_range["depth_max_m"])
        if not 0 < depth_min < depth_max:
            raise RuntimeError(f"invalid sparse depth range for {label}: {depth_min}-{depth_max}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read {image_path}")
        source_height, source_width = bgr.shape[:2]
        scale_xy, content_width, content_height, pad_x, pad_y = geometry(source_width, source_height)
        content = cv2.resize(bgr, (content_width, content_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
        model, predicted, model_mask, hit = raw_prediction(
            model,rgb,WEIGHTS,args.resolution_level,args.refine_steps,cache)
        reused += int(hit)
        if hit and (reused==1 or reused%50==0):
            print('REUSED pre-BA MoGe views',reused,flush=True)
        if predicted.shape != (content_height, content_width):
            predicted = cv2.resize(predicted, (content_width, content_height), interpolation=cv2.INTER_LINEAR)
            model_mask = cv2.resize(model_mask.astype(np.uint8), (content_width, content_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        openmvs_mask = cv2.imread(str(OPENMVS_MASKS / f"{label}.mask.png"), cv2.IMREAD_GRAYSCALE)
        if openmvs_mask is None or openmvs_mask.shape != (source_height, source_width):
            raise RuntimeError(f"Invalid OpenMVS mask for {label}")
        ignored_content = resize_ignore(openmvs_mask, (content_width, content_height))
        keep_content = ~ignored_content

        lidar = np.load(SOURCE / "lidar_depth" / f"{label}.npy", allow_pickle=True).item()
        source_y, source_x = np.nonzero(lidar["mask"])
        lidar_values = np.asarray(lidar["value"], dtype=np.float32)
        resized_x = np.clip(np.rint((source_x + 0.5) * scale_xy - 0.5).astype(int), 0, content_width - 1)
        resized_y = np.clip(np.rint((source_y + 0.5) * scale_xy - 0.5).astype(int), 0, content_height - 1)
        sampled = predicted[resized_y, resized_x]
        calibration = (
            np.isfinite(sampled)
            & (sampled > 0)
            & model_mask[resized_y, resized_x]
            & keep_content[resized_y, resized_x]
            & np.isfinite(lidar_values)
            & (lidar_values > 2.0)
        )
        if not args.unbounded_lidar_calibration:
            calibration &= lidar_values < 80.0
        ratios = lidar_values[calibration] / sampled[calibration]
        if ratios.size < 100:
            print(
                f"Deferred scale fit: {label} has {ratios.size} LiDAR calibration pixels",
                flush=True,
            )
        # Finalization replaces this neutral placeholder with the robust clip
        # median of this camera, or the all-camera median when a camera has no
        # valid LiDAR fits anywhere in the segment.
        raw_scale = float(np.median(ratios)) if ratios.size >= 100 else 1.0

        content_depth = np.where(
            model_mask & keep_content & np.isfinite(predicted) & (predicted > 0),
            predicted * raw_scale,
            0.0,
        ).astype(np.float32)
        depth = np.zeros((HEIGHT, WIDTH), np.float32)
        depth[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = content_depth

        # APD has no standard MVSNet mask input. Black out ignored pixels in its
        # private image copy; per-iteration depth output is masked again in C++.
        dvp_content = content.copy()
        dvp_content[ignored_content] = 0
        padded = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        padded[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = dvp_content
        block = np.zeros((HEIGHT, WIDTH), np.uint8)
        block[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = keep_content.astype(np.uint8) * 255

        intrinsic = np.loadtxt(SOURCE / "intrinsics" / f"{label}.txt", dtype=np.float64)
        camera_to_world = np.loadtxt(SOURCE / "ego_pose" / f"{label}.txt", dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        intrinsic[0, :] *= scale_xy
        intrinsic[1, :] *= scale_xy
        intrinsic[0, 2] += pad_x
        intrinsic[1, 2] += pad_y

        write_dmb(SCENE / "metric_prior" / f"{stem}.dmb", depth)
        image_temp = SCENE / "images" / f"{stem}.tmp.jpg"
        block_temp = SCENE / "blocks" / f"mask_{image_id}.tmp.png"
        if not cv2.imwrite(str(image_temp), padded, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Cannot write {image_temp}")
        if not cv2.imwrite(str(block_temp), block, [cv2.IMWRITE_JPEG_QUALITY, 100]):
            raise RuntimeError(f"Cannot write {block_temp}")
        os.replace(image_temp, SCENE / "images" / f"{stem}.jpg")
        os.replace(block_temp, SCENE / "blocks" / f"mask_{image_id}.png")
        write_camera(
            SCENE / "cams" / f"{stem}_cam.txt",
            world_to_camera,
            intrinsic,
            depth_min,
            depth_max,
        )

        metadata = {
            "mask_sha256": hashlib.sha256((OPENMVS_MASKS / f"{label}.mask.png").read_bytes()).hexdigest(),
            "id": image_id,
            "source_label": label,
            "source_size": [source_width, source_height],
            "content_size": [content_width, content_height],
            "padding_xy": [pad_x, pad_y],
            "resize_scale": scale_xy,
            "raw_moge_scale_multiplier": raw_scale,
            "lidar_calibration_pixels": int(ratios.size),
            "raw_valid_prior_fraction": float(np.mean(depth > 0)),
            "block_keep_fraction": float(np.mean(block >= 128)),
            "lidar_calibration_upper_bound_m": None if args.unbounded_lidar_calibration else 80.0,
            "dvp_depth_range_m": [depth_min, depth_max],
            "depth_range_source": depth_range["source"],
            "sparse_range_points": int(depth_range["sparse_points"]),
        }
        atomic_bytes(SCENE / "metadata" / f"{stem}.json", (json.dumps(metadata, indent=2) + "\n").encode())
        elapsed = time.time() - started
        print(
            f"[{args.shard}] {done}/{len(pending)} id={stem} label={label} "
            f"shape={content_width}x{content_height}+{pad_x}+{pad_y} "
            f"scale={raw_scale:.5f} valid={metadata['raw_valid_prior_fraction']:.3f} elapsed={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
