#!/usr/bin/env python3
"""Robustly calibrate priors, filter them, and build the verified 20-view graph."""

from __future__ import annotations

from mask_policy import block_path

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(os.environ.get("DVP_RUN_ROOT", Path(__file__).resolve().parent))
SCENE = ROOT / "scene"
SOURCE = Path(os.environ.get(
    "DVP_SOURCE",
    ROOT/"converted",
))
NEIGHBORS_TSV = Path(os.environ.get(
    "DVP_NEIGHBORS_TSV",
    ROOT/"neighbors.tsv",
))
WIDTH, HEIGHT = 960, 640
MIN_PRIOR_COMPONENT = 20


def atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def read_dmb(path: Path, channels: int) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(16)
        version, rows, cols, cv_type = struct.unpack("4i", header)
        expected_type = 5 if channels == 1 else 21
        if (version, rows, cols, cv_type) != (1, HEIGHT, WIDTH, expected_type):
            raise RuntimeError(f"Invalid DMB header: {path}")
        data = np.frombuffer(f.read(), dtype=np.float32)
    expected = HEIGHT * WIDTH * channels
    if data.size != expected:
        raise RuntimeError(f"Invalid DMB payload: {path}")
    shape = (HEIGHT, WIDTH) if channels == 1 else (HEIGHT, WIDTH, channels)
    return data.reshape(shape).copy()


def write_dmb(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    cv_type = 5 if array.ndim == 2 else 21
    atomic_bytes(path, struct.pack("4i", 1, array.shape[0], array.shape[1], cv_type) + array.tobytes())


def write_camera(
    path: Path,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> None:
    text = "extrinsic\n"
    text += "\n".join(" ".join(f"{value:.12g}" for value in row) for row in extrinsic)
    text += "\n\nintrinsic\n"
    text += "\n".join(" ".join(f"{value:.12g}" for value in row) for row in intrinsic)
    # APD expands these stored endpoints by 0.6 and 1.2 internally.
    stored_min = depth_min / 0.6
    stored_max = depth_max / 1.2
    interval = 0.25
    depth_num = max(2, int(round((stored_max - stored_min) / interval)) + 1)
    text += f"\n\n{stored_min:.12g} {interval:.12g} {depth_num} {stored_max:.12g}\n"
    atomic_bytes(path, text.encode())


def remove_small_components(depth: np.ndarray) -> int:
    valid = (depth > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    small = np.zeros(count, dtype=bool)
    if count > 1:
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < MIN_PRIOR_COMPONENT
    remove = small[labels]
    removed = int(remove.sum())
    depth[remove] = 0
    return removed


def make_normals(depth: np.ndarray, intrinsic: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    xyz = np.stack(
        (
            (xx - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (yy - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        axis=-1,
    )
    normal_camera = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    normal_camera /= np.maximum(np.linalg.norm(normal_camera, axis=-1, keepdims=True), 1e-8)
    normal_camera[np.sum(normal_camera * xyz, axis=-1) > 0] *= -1
    neighbors_valid = depth > 0
    neighbors_valid[1:-1, 1:-1] &= (
        (depth[:-2, 1:-1] > 0)
        & (depth[2:, 1:-1] > 0)
        & (depth[1:-1, :-2] > 0)
        & (depth[1:-1, 2:] > 0)
    )
    neighbors_valid[[0, -1], :] = False
    neighbors_valid[:, [0, -1]] = False
    normal_world = normal_camera @ world_to_camera[:3, :3]
    normal_world[~neighbors_valid] = 0
    return normal_world.astype(np.float32)


def calibrate_scales(by_camera: dict[str, list[dict]]) -> dict[str, dict]:
    """Robustly calibrate valid LiDAR fits and fill views without enough LiDAR."""
    is_valid = lambda record: int(record.get("lidar_calibration_pixels", 0)) >= 100
    all_logs = [
        math.log(record["raw_moge_scale_multiplier"])
        for records in by_camera.values()
        for record in records
        if is_valid(record)
    ]
    if not all_logs:
        raise RuntimeError("No view in this segment has a valid LiDAR scale fit")

    report = {}
    for camera, records in sorted(by_camera.items()):
        camera_logs = np.asarray(
            [math.log(record["raw_moge_scale_multiplier"]) for record in records if is_valid(record)],
            dtype=np.float64,
        )
        fallback_source = "same_camera_segment_median"
        if not camera_logs.size:
            camera_logs = np.asarray(all_logs, dtype=np.float64)
            fallback_source = "all_cameras_segment_median"
        center = float(np.median(camera_logs))
        mad = float(np.median(np.abs(camera_logs - center)))
        robust_sigma = 1.4826 * mad
        limit = min(math.log(1.15), max(math.log(1.10), 3.0 * robust_sigma))
        for record in records:
            raw_log = math.log(record["raw_moge_scale_multiplier"])
            used_log = float(np.clip(raw_log, center - limit, center + limit)) if is_valid(record) else center
            record["moge_scale_multiplier"] = math.exp(used_log)
            record["scale_was_clipped"] = bool(is_valid(record) and abs(used_log - raw_log) > 1e-8)
            record["scale_fallback_source"] = "none" if is_valid(record) else fallback_source
        report[camera] = {
            "count": len(records),
            "median_scale": math.exp(center),
            "log_mad": mad,
            "allowed_ratio": math.exp(limit),
            "clipped_images": sum(int(record["scale_was_clipped"]) for record in records),
            "valid_lidar_fits": sum(int(is_valid(record)) for record in records),
            "fallback_images": sum(int(not is_valid(record)) for record in records),
            "fallback_source": fallback_source,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-ranges-json", type=Path, required=True)
    parser.add_argument(
        "--save-unbounded-prior",
        action="store_true",
        help="Save calibrated geometry-valid depth and normals before DVP range filtering.",
    )
    args = parser.parse_args()
    range_bytes = args.depth_ranges_json.read_bytes()
    range_report = json.loads(range_bytes)
    depth_ranges = range_report["ranges"]
    range_digest = hashlib.sha256(range_bytes).hexdigest()[:12]
    finalization_version = f"clean7-v3-sparse-range-{range_digest}"
    unbounded_dir = SCENE / "metric_prior_unbounded"
    if args.save_unbounded_prior:
        unbounded_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted((SOURCE / "images").glob("*.jpg"))
    if not source_images:
        raise RuntimeError(f"No source images found in {SOURCE / 'images'}")
    source_labels = {path.stem for path in source_images}
    if set(depth_ranges) != source_labels:
        missing = sorted(source_labels - set(depth_ranges))[:5]
        extra = sorted(set(depth_ranges) - source_labels)[:5]
        raise RuntimeError(f"sparse range/image mismatch; missing={missing}, extra={extra}")
    records = []
    by_camera = defaultdict(list)
    for image_id, source_image in enumerate(source_images):
        stem = f"{image_id:08d}"
        metadata_path = SCENE / "metadata" / f"{stem}.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing {metadata_path}")
        record = json.loads(metadata_path.read_text())
        if record["source_label"] != source_image.stem:
            raise RuntimeError(f"ID mapping mismatch for {stem}")
        records.append(record)
        camera = source_image.stem.rsplit("_", 1)[1]
        by_camera[camera].append(record)

    calibration_report = calibrate_scales(by_camera)

    total_prior_speckles = 0
    for record in records:
        image_id = record["id"]
        stem = f"{image_id:08d}"
        depth_range = depth_ranges[record["source_label"]]
        depth_min = float(depth_range["depth_min_m"])
        depth_max = float(depth_range["depth_max_m"])
        if not 0 < depth_min < depth_max:
            raise RuntimeError(
                f"invalid sparse depth range for {record['source_label']}: {depth_min}-{depth_max}"
            )
        raw_scale = record["raw_moge_scale_multiplier"]
        used_scale = record["moge_scale_multiplier"]
        depth_path = SCENE / "metric_prior" / f"{stem}.dmb"
        unbounded_depth_path = unbounded_dir / f"{stem}.dmb"
        unbounded_normal_path = unbounded_dir / f"{stem}_normal.dmb"
        unbounded_output_complete = (
            not args.save_unbounded_prior
            or (unbounded_depth_path.is_file() and unbounded_normal_path.is_file())
        )
        if record.get("finalization_version") == finalization_version and unbounded_output_complete:
            total_prior_speckles += int(record.get("prior_small_component_pixels_removed", 0))
            continue
        # The preserved prior already contains the robust per-camera scale. Reuse it
        # without applying that scale a second time when changing the DVP range.
        reuse_unbounded = (
            unbounded_depth_path.is_file()
            and record.get("unbounded_valid_prior_fraction") is not None
        )
        if reuse_unbounded:
            depth = read_dmb(unbounded_depth_path, 1)
        else:
            depth = read_dmb(depth_path, 1)
            depth *= used_scale / raw_scale
        block = cv2.imread(str(block_path(SCENE, image_id)), cv2.IMREAD_GRAYSCALE)
        if block is None or block.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"Invalid DVP block mask for {stem}")
        depth[(block < 128) | ~np.isfinite(depth) | (depth <= 0)] = 0

        label = record["source_label"]
        source_width, source_height = record["source_size"]
        resize_scale = record["resize_scale"]
        pad_x, pad_y = record["padding_xy"]
        intrinsic = np.loadtxt(SOURCE / "intrinsics" / f"{label}.txt", dtype=np.float64)
        intrinsic[0, :] *= resize_scale
        intrinsic[1, :] *= resize_scale
        intrinsic[0, 2] += pad_x
        intrinsic[1, 2] += pad_y
        camera_to_world = np.loadtxt(SOURCE / "ego_pose" / f"{label}.txt", dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        write_camera(
            SCENE / "cams" / f"{stem}_cam.txt",
            world_to_camera,
            intrinsic,
            depth_min,
            depth_max,
        )

        if args.save_unbounded_prior:
            unbounded_normals = make_normals(depth, intrinsic, world_to_camera)
            write_dmb(unbounded_depth_path, depth)
            write_dmb(unbounded_normal_path, unbounded_normals)
            record["unbounded_valid_prior_fraction"] = float(np.mean(depth > 0))
            unbounded_values = depth[depth > 0]
            record["unbounded_depth_max_m"] = (
                float(unbounded_values.max()) if unbounded_values.size else None
            )

        depth[(depth < depth_min) | (depth > depth_max)] = 0
        removed = remove_small_components(depth)
        total_prior_speckles += removed

        normals = make_normals(depth, intrinsic, world_to_camera)
        write_dmb(depth_path, depth)
        write_dmb(SCENE / "metric_prior" / f"{stem}_normal.dmb", normals)
        record["valid_prior_fraction"] = float(np.mean(depth > 0))
        record["prior_small_component_pixels_removed"] = removed
        record["dvp_depth_range_m"] = [depth_min, depth_max]
        record["depth_range_source"] = depth_range["source"]
        record["sparse_range_points"] = int(depth_range["sparse_points"])
        record["finalization_version"] = finalization_version
        atomic_bytes(SCENE / "metadata" / f"{stem}.json", (json.dumps(record, indent=2) + "\n").encode())

    label_to_id = {record["source_label"]: record["id"] for record in records}
    selected = defaultdict(list)
    with NEIGHBORS_TSV.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ref = row["reference"]
            src = row["source"]
            selected[ref].append((int(row["rank"]), src, int(row["shared_sparse_points"])))

    pair_lines = [str(len(records))]
    cross_counts = []
    for record in records:
        label = record["source_label"]
        neighbors = sorted(selected[label])
        if len(neighbors) != 20:
            raise RuntimeError(f"Expected 20 verified neighbors for {label}, got {len(neighbors)}")
        sources = [(label_to_id[src], evidence) for _, src, evidence in neighbors]
        pair_lines.append(str(record["id"]))
        pair_lines.append("20 " + " ".join(f"{source_id} {max(evidence, 1):.6f}" for source_id, evidence in sources))
        record["source_views"] = [source_id for source_id, _ in sources]
        ref_camera = label.rsplit("_", 1)[1]
        cross = sum(records[source_id]["source_label"].rsplit("_", 1)[1] != ref_camera for source_id, _ in sources)
        cross_counts.append(cross)

    atomic_bytes(SCENE / "pair.txt", ("\n".join(pair_lines) + "\n").encode())
    manifest = {
        "source": str(SOURCE),
        "output_size": [WIDTH, HEIGHT],
        "image_count": len(records),
        "camera_count": len(by_camera),
        "cameras": sorted(by_camera),
        "changes": {
            "masks": "combined OpenMVS ignore masks (sky, converted dynamic objects and camera10 ego rig); applied to priors/DVP depth estimation and passed to OpenMVS fusion",
            "pairing": "20 COLMAP-SIFT-verified OpenMVS views per reference, including cross-camera and +/-20-frame diversity",
            "dvp_native_fusion": "disabled by the mvs_process batch with DVP_SKIP_FUSION=1",
            "resize": "aspect-preserving fit into 960x640; camera09/10 are 960x540 with 50-pixel top/bottom padding",
            "scale_filter": "per-camera log-scale median/MAD clipping, constrained to 10--15 percent; missing LiDAR fits use the same-camera segment median or all-camera segment median",
            "prior_component_filter_pixels": MIN_PRIOR_COMPONENT,
            "depth_range": "per-view sparse camera-Z min/max with 0.9/1.1 margins",
            "fixed_depth_range": False,
            "sparse_depth_range_report": str(args.depth_ranges_json),
            "depth_min_m_summary": range_report["depth_min_m_summary"],
            "depth_max_m_summary": range_report["depth_max_m_summary"],
            "unbounded_metric_prior_saved": args.save_unbounded_prior,
            "unbounded_metric_prior_directory": "scene/metric_prior_unbounded" if args.save_unbounded_prior else None,
        },
        "calibration": calibration_report,
        "prior_small_component_pixels_removed": total_prior_speckles,
        "source_views_min_max": [20, 20],
        "cross_camera_sources_min_max": [min(cross_counts), max(cross_counts)],
        "images": records,
    }
    atomic_bytes(ROOT / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({key: value for key, value in manifest.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
