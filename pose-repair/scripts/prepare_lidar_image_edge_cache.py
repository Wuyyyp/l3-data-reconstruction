from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dongfeng_registration.lidar_edges import (  # noqa: E402
    distance_transform_from_edges,
    extract_image_edges,
    infer_lidar_range_axes,
    select_source_indices_by_range_percentile_band,
)
from dongfeng_registration.lidar_image_edge_cache import (  # noqa: E402
    build_lidar_image_edge_cache_paths,
    image_edge_cache_key,
    lidar_edge_cache_key,
    make_frame_stem,
    relative_to,
    write_json,
    write_rows_csv,
)
from extract_lidar_image_edges import (  # noqa: E402
    DEFAULT_CANNY_HIGH,
    DEFAULT_CANNY_LOW,
    DEFAULT_EDGE_MAX_RANGE_PERCENTILE,
    DEFAULT_EDGE_MIN_RANGE_PERCENTILE,
    DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS,
    DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION,
    lidar_points_for_info,
    lidar_range_edge_source_indices,
    make_edge_panel,
    source_indices_to_range_edge_mask,
)
from visualize_l29_lidar_projection_pkl import (  # noqa: E402
    CAMERA_TO_IMAGE_FOLDER,
    image_for_camera,
    load_infos,
)

DEFAULT_DIAGNOSTIC_PREVIEW_SCALE = 0.50


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else np.asarray(image)
    gray_f = gray.astype(np.float32) / 255.0
    dx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    confidence = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
    return dx.astype(np.float32, copy=False), dy.astype(np.float32, copy=False), confidence


def save_image_edge_npz(
    path: Path,
    *,
    image: np.ndarray,
    edge_mask: np.ndarray,
    distance_transform_px: np.ndarray,
    frame_index: int,
    timestamp_us: int,
    camera: str,
    image_path: Path,
    canny_low: int,
    canny_high: int,
    write_gradients: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "image_edge_mask": np.asarray(edge_mask, dtype=np.bool_),
        "distance_transform_px": np.asarray(distance_transform_px, dtype=np.float32),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "timestamp_us": np.asarray(timestamp_us, dtype=np.int64),
        "camera_name": np.asarray(str(camera)),
        "image_path": np.asarray(str(image_path)),
        "canny_low": np.asarray(canny_low, dtype=np.int32),
        "canny_high": np.asarray(canny_high, dtype=np.int32),
    }
    if write_gradients:
        gradient_dx, gradient_dy, edge_confidence = image_gradients(image)
        payload.update(
            {
                "gradient_dx": gradient_dx,
                "gradient_dy": gradient_dy,
                "edge_confidence": edge_confidence,
            }
        )
    np.savez_compressed(path, **payload)


def save_lidar_edge_npz(
    path: Path,
    *,
    raw_points_lidar: np.ndarray,
    edge_source_indices: np.ndarray,
    range_edge_mask: np.ndarray,
    range_valid_mask: np.ndarray,
    range_image: np.ndarray,
    frame_index: int,
    timestamp_ns: int,
    lidar_name: str,
    pcd_path: Path,
    range_low_m: float,
    range_high_m: float,
    metadata: dict[str, float | int | str],
) -> None:
    edge_indices = np.asarray(edge_source_indices, dtype=np.int64)
    valid_edge_indices = edge_indices[(edge_indices >= 0) & (edge_indices < len(raw_points_lidar))]
    edge_points = np.asarray(raw_points_lidar, dtype=np.float32)[valid_edge_indices]
    edge_ranges = np.linalg.norm(edge_points.astype(np.float64), axis=1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edge_source_indices=valid_edge_indices,
        edge_points_lidar=edge_points,
        edge_ranges_m=edge_ranges,
        range_edge_mask=np.asarray(range_edge_mask, dtype=np.bool_),
        range_valid_mask=np.asarray(range_valid_mask, dtype=np.bool_),
        range_image=np.asarray(range_image, dtype=np.float32),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        timestamp_ns=np.asarray(timestamp_ns, dtype=np.int64),
        lidar_name=np.asarray(str(lidar_name)),
        pcd_path=np.asarray(str(pcd_path)),
        all_points_count=np.asarray(len(raw_points_lidar), dtype=np.int64),
        edge_points_count=np.asarray(len(valid_edge_indices), dtype=np.int64),
        range_low_m=np.asarray(range_low_m, dtype=np.float32),
        range_high_m=np.asarray(range_high_m, dtype=np.float32),
        range_image_rows=np.asarray(metadata.get("range_image_rows", 0), dtype=np.int32),
        range_image_cols=np.asarray(metadata.get("range_image_cols", 0), dtype=np.int32),
        range_forward_axis=np.asarray(metadata.get("range_forward_axis", 0), dtype=np.int32),
        range_lateral_axis=np.asarray(metadata.get("range_lateral_axis", 1), dtype=np.int32),
        range_vertical_axis=np.asarray(metadata.get("range_vertical_axis", 2), dtype=np.int32),
    )


def time_diff_ms(camera_timestamp_us: int, lidar_timestamp_ns: int) -> float:
    return (float(camera_timestamp_us) - float(lidar_timestamp_ns) / 1000.0) / 1000.0


def prepare_cache(args: argparse.Namespace) -> tuple[int, Path]:
    dataset_root = Path(args.dataset_root)
    image_root = Path(args.image_root) if args.image_root is not None else dataset_root
    lidar_root = Path(args.lidar_root) if args.lidar_root is not None else dataset_root / args.lidar_name
    image_key = image_edge_cache_key(canny_low=args.canny_low, canny_high=args.canny_high)
    lidar_key = lidar_edge_cache_key(
        min_percentile=args.edge_min_range_percentile,
        max_percentile=args.edge_max_range_percentile,
        edge_side=args.range_edge_side,
        require_support=not args.no_range_edge_support,
        min_component_pixels=args.range_edge_min_component_pixels,
    )
    paths = build_lidar_image_edge_cache_paths(
        cache_root=args.cache_root,
        sequence=args.sequence,
        camera=args.camera,
        lidar_name=args.lidar_name,
        image_key=image_key,
        lidar_key=lidar_key,
    )
    for directory in (paths.image_dir, paths.lidar_dir, paths.pair_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.no_diagnostics:
        paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(args.before_pkl)
    frame_end = len(infos) if args.end is None else min(int(args.end), len(infos))
    image_rows: list[dict[str, Any]] = []
    lidar_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for frame_index in range(int(args.start), frame_end, max(1, int(args.stride))):
        info = infos[frame_index]
        camera_data = info.get("sensors", {}).get("cams", {}).get(args.camera)
        lidar_data = info.get("sensors", {}).get("lidar", {}).get(args.lidar_name)
        if not isinstance(camera_data, dict) or not isinstance(lidar_data, dict):
            continue

        image_payload = image_for_camera(camera_data, image_root=image_root, camera=args.camera)
        lidar_payload = lidar_points_for_info(
            info,
            lidar_root=lidar_root,
            lidar_name=args.lidar_name,
        )
        if image_payload is None or lidar_payload is None:
            pair_rows.append(
                {
                    "frame_index": frame_index,
                    "camera_name": args.camera,
                    "lidar_name": args.lidar_name,
                    "camera_timestamp_us": camera_data.get("timestamp", ""),
                    "lidar_timestamp_ns": lidar_data.get("timestamp", ""),
                    "time_diff_ms": "",
                    "image_edge_npz": "",
                    "lidar_edge_npz": "",
                    "diagnostic_image": "",
                    "valid_pair": 0,
                    "notes": "missing_image_or_lidar_file",
                }
            )
            continue

        image_path, image = image_payload
        pcd_path, raw_points_lidar, lidar_extrinsic = lidar_payload
        camera_timestamp_us = int(camera_data["timestamp"])
        lidar_timestamp_ns = int(lidar_data["timestamp"])

        image_edges = extract_image_edges(
            image,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
        image_distance = distance_transform_from_edges(image_edges)
        image_npz = paths.image_dir / f"{make_frame_stem(frame_index, camera_timestamp_us)}.npz"
        save_image_edge_npz(
            image_npz,
            image=image,
            edge_mask=image_edges,
            distance_transform_px=image_distance,
            frame_index=frame_index,
            timestamp_us=camera_timestamp_us,
            camera=args.camera,
            image_path=image_path,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            write_gradients=args.write_image_gradients,
        )

        range_axes = infer_lidar_range_axes(
            raw_points_lidar,
            lidar_extrinsic=lidar_extrinsic,
        )
        range_edge_indices, range_projection, _full_range_edge_mask, range_metadata = (
            lidar_range_edge_source_indices(
                raw_points_lidar,
                num_rows=args.lidar_rows,
                azimuth_resolution_deg=args.lidar_azimuth_resolution_deg,
                vertical_fov_deg=(
                    None
                    if args.lidar_vertical_min_deg is None
                    or args.lidar_vertical_max_deg is None
                    else (args.lidar_vertical_min_deg, args.lidar_vertical_max_deg)
                ),
                range_edge_abs_m=args.range_edge_abs_m,
                range_edge_rel=args.range_edge_rel,
                include_silhouette=not args.no_range_silhouette,
                edge_side=args.range_edge_side,
                min_component_pixels=args.range_edge_min_component_pixels,
                silhouette_ignore_lower_fraction=args.range_silhouette_ignore_lower_fraction,
                range_axes=range_axes,
                require_depth_jump_support=not args.no_range_edge_support,
            )
        )
        filtered_range_edge_indices, range_low_m, range_high_m = (
            select_source_indices_by_range_percentile_band(
                range_edge_indices,
                raw_points_lidar,
                min_percentile=args.edge_min_range_percentile,
                max_percentile=args.edge_max_range_percentile,
            )
        )
        filtered_range_edge_mask = source_indices_to_range_edge_mask(
            range_projection.source_index_image,
            filtered_range_edge_indices,
        )
        lidar_npz = paths.lidar_dir / f"{make_frame_stem(frame_index, lidar_timestamp_ns)}.npz"
        save_lidar_edge_npz(
            lidar_npz,
            raw_points_lidar=raw_points_lidar,
            edge_source_indices=filtered_range_edge_indices,
            range_edge_mask=filtered_range_edge_mask,
            range_valid_mask=range_projection.valid_mask,
            range_image=range_projection.range_image,
            frame_index=frame_index,
            timestamp_ns=lidar_timestamp_ns,
            lidar_name=args.lidar_name,
            pcd_path=pcd_path,
            range_low_m=range_low_m,
            range_high_m=range_high_m,
            metadata=range_metadata,
        )

        image_rel = relative_to(image_npz, paths.root)
        lidar_rel = relative_to(lidar_npz, paths.root)
        diagnostic_rel = ""
        if not args.no_diagnostics:
            panel, _stats = make_edge_panel(
                image=image,
                display_points_lidar=raw_points_lidar,
                edge_points_lidar=raw_points_lidar,
                edge_source_indices=filtered_range_edge_indices,
                range_image=range_projection.range_image,
                range_valid_mask=range_projection.valid_mask,
                range_edge_mask=filtered_range_edge_mask,
                middle_panel_mode="native_range",
                lidar_extrinsic=lidar_extrinsic,
                ego_to_camera=np.asarray(camera_data["extrinsic"], dtype=np.float64),
                intrinsic=np.asarray(camera_data["cam_intrinsic"], dtype=np.float64),
                lidar_extrinsic_direction="lidar_to_ego",
                max_depth_m=args.max_depth_m,
                depth_jump_abs_m=args.range_edge_abs_m,
                depth_jump_rel=args.range_edge_rel,
                depth_edge_mode="combined",
                depth_edge_radius_px=5,
                min_valid_neighbors=2,
                occupancy_fill_radius_px=12,
                occupancy_boundary_width_px=1,
                occupancy_return_radius_px=1,
                occupancy_min_area_px=1000,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                max_edge_distance_px=20.0,
                preview_scale=args.preview_scale,
                lidar_edge_visual_radius=args.lidar_edge_visual_radius,
                projected_point_radius=args.projected_point_radius,
            )
            diagnostic_path = (
                paths.diagnostics_dir
                / f"edge_before_{frame_index:06d}_{image_path.name}"
            )
            cv2.imwrite(str(diagnostic_path), panel)
            diagnostic_rel = relative_to(diagnostic_path, paths.root)
        image_rows.append(
            {
                "frame_index": frame_index,
                "camera_name": args.camera,
                "timestamp_us": camera_timestamp_us,
                "image_path": str(image_path),
                "image_edge_npz": image_rel,
                "image_edge_pixels": int(np.count_nonzero(image_edges)),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
        )
        lidar_rows.append(
            {
                "frame_index": frame_index,
                "lidar_name": args.lidar_name,
                "timestamp_ns": lidar_timestamp_ns,
                "pcd_path": str(pcd_path),
                "lidar_edge_npz": lidar_rel,
                "all_points_count": int(len(raw_points_lidar)),
                "edge_points_count": int(len(filtered_range_edge_indices)),
                "range_low_m": float(range_low_m),
                "range_high_m": float(range_high_m),
            }
        )
        pair_rows.append(
            {
                "frame_index": frame_index,
                "camera_name": args.camera,
                "lidar_name": args.lidar_name,
                "camera_timestamp_us": camera_timestamp_us,
                "lidar_timestamp_ns": lidar_timestamp_ns,
                "time_diff_ms": round(time_diff_ms(camera_timestamp_us, lidar_timestamp_ns), 6),
                "image_edge_npz": image_rel,
                "lidar_edge_npz": lidar_rel,
                "diagnostic_image": diagnostic_rel,
                "valid_pair": 1,
                "notes": "",
            }
        )
        print(
            "[cache] "
            f"frame={frame_index:06d} image_edges={int(np.count_nonzero(image_edges))} "
            f"lidar_edges={len(filtered_range_edge_indices)} "
            f"dt_ms={time_diff_ms(camera_timestamp_us, lidar_timestamp_ns):.3f}"
        )

    config = {
        "sequence": args.sequence,
        "camera": args.camera,
        "lidar_name": args.lidar_name,
        "before_pkl": str(args.before_pkl),
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "lidar_root": str(lidar_root),
        "image_key": image_key,
        "lidar_key": lidar_key,
        "canny_low": int(args.canny_low),
        "canny_high": int(args.canny_high),
        "write_image_gradients": int(args.write_image_gradients),
        "edge_source": "lidar_range",
        "edge_min_range_percentile": float(args.edge_min_range_percentile),
        "edge_max_range_percentile": float(args.edge_max_range_percentile),
        "lidar_rows": int(args.lidar_rows),
        "lidar_azimuth_resolution_deg": float(args.lidar_azimuth_resolution_deg),
        "range_edge_abs_m": float(args.range_edge_abs_m),
        "range_edge_rel": float(args.range_edge_rel),
        "range_edge_side": str(args.range_edge_side),
        "range_edge_min_component_pixels": int(args.range_edge_min_component_pixels),
        "range_edge_require_support": int(not args.no_range_edge_support),
        "range_silhouette_ignore_lower_fraction": float(
            args.range_silhouette_ignore_lower_fraction
        ),
        "range_include_silhouette": int(not args.no_range_silhouette),
        "diagnostics": int(not args.no_diagnostics),
        "preview_scale": float(args.preview_scale),
        "lidar_edge_visual_radius": int(args.lidar_edge_visual_radius),
        "projected_point_radius": int(args.projected_point_radius),
        "max_depth_m": float(args.max_depth_m),
        "start": int(args.start),
        "end": None if args.end is None else int(args.end),
        "stride": int(args.stride),
        "valid_pairs": int(sum(1 for row in pair_rows if row.get("valid_pair") == 1)),
        "total_pair_rows": int(len(pair_rows)),
    }
    write_rows_csv(paths.image_index_csv, image_rows)
    write_rows_csv(paths.lidar_index_csv, lidar_rows)
    write_rows_csv(paths.pair_index_csv, pair_rows)
    write_json(paths.pair_config_json, config)
    write_json(
        paths.manifest_json,
        {
            "version": 1,
            "description": "Image edges and native LiDAR range-image edge points for LiDAR-camera edge constraints.",
            "last_sequence": args.sequence,
            "last_camera": args.camera,
            "last_lidar_name": args.lidar_name,
            "last_pair_dir": relative_to(paths.pair_dir, paths.root),
        },
    )
    return len(pair_rows), paths.root


def parse_args() -> argparse.Namespace:
    default_root = Path("data/M18_canonical")
    parser = argparse.ArgumentParser(
        description=(
            "Prepare reusable image-edge and native LiDAR range-edge caches for "
            "LiDAR-camera edge distance-transform constraints."
        )
    )
    parser.add_argument("--sequence", default="M18")
    parser.add_argument("--cache-root", type=Path, default=Path("data_cache"))
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument(
        "--before-pkl",
        type=Path,
        default=default_root / "result_interpolation_standard.pkl",
    )
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--lidar-root", type=Path)
    parser.add_argument("--camera", default="center_camera_fov120", choices=sorted(CAMERA_TO_IMAGE_FOLDER))
    parser.add_argument("--lidar-name", default="front_lidar")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--canny-low", type=int, default=DEFAULT_CANNY_LOW)
    parser.add_argument("--canny-high", type=int, default=DEFAULT_CANNY_HIGH)
    parser.add_argument(
        "--write-image-gradients",
        action="store_true",
        help=(
            "Also cache full-resolution image gradient fields. Disabled by default "
            "because 2160x3840 float32 gradients make each frame much larger."
        ),
    )
    parser.add_argument(
        "--edge-min-range-percentile",
        type=float,
        default=DEFAULT_EDGE_MIN_RANGE_PERCENTILE,
    )
    parser.add_argument(
        "--edge-max-range-percentile",
        type=float,
        default=DEFAULT_EDGE_MAX_RANGE_PERCENTILE,
    )
    parser.add_argument("--lidar-rows", type=int, default=128)
    parser.add_argument("--lidar-azimuth-resolution-deg", type=float, default=0.2)
    parser.add_argument("--lidar-vertical-min-deg", type=float)
    parser.add_argument("--lidar-vertical-max-deg", type=float)
    parser.add_argument("--range-edge-abs-m", type=float, default=0.8)
    parser.add_argument("--range-edge-rel", type=float, default=0.08)
    parser.add_argument("--range-edge-side", default="near", choices=("near", "both", "far"))
    parser.add_argument(
        "--range-edge-min-component-pixels",
        type=int,
        default=DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS,
    )
    parser.add_argument("--no-range-edge-support", action="store_true")
    parser.add_argument(
        "--range-silhouette-ignore-lower-fraction",
        type=float,
        default=DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION,
    )
    parser.add_argument("--no-range-silhouette", action="store_true")
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not write per-frame 2x3 diagnostic PNGs.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=DEFAULT_DIAGNOSTIC_PREVIEW_SCALE,
        help="Scale for saved diagnostic PNGs only.",
    )
    parser.add_argument("--lidar-edge-visual-radius", type=int, default=2)
    parser.add_argument("--projected-point-radius", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    rows, root = prepare_cache(parse_args())
    print(f"[done] wrote {rows} pair rows -> {root}")


if __name__ == "__main__":
    main()
