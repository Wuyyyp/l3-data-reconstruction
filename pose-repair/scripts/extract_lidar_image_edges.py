from __future__ import annotations

import argparse
import csv
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
    binary_mask_to_visual,
    crop_range_display_to_valid_region,
    distance_transform_from_edges,
    draw_binary_mask_overlay,
    draw_depth_projection_overlay,
    draw_edge_mask_visual,
    extract_depth_jump_edges,
    extract_image_edges,
    extract_lidar_range_edge_mask,
    extract_lidar_range_edge_source_indices,
    extract_occupancy_boundary_edges,
    extract_projected_lidar_depth_edge_source_indices,
    infer_lidar_range_axes,
    extract_sparse_depth_jump_edges,
    filter_points_by_range,
    make_lidar_diagnostic_grid,
    project_points_to_lidar_range_image,
    project_points_to_depth_map,
    range_image_to_visual,
    rotate_range_display_180,
    select_source_indices_by_range_percentile_band,
    roll_range_display_180,
    summarize_edge_alignment,
    transform_points,
)
from visualize_l29_lidar_projection_pkl import (  # noqa: E402
    CAMERA_TO_IMAGE_FOLDER,
    image_for_camera,
    load_infos,
    read_point_cloud_xyz,
)

DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 120
DEFAULT_EDGE_MIN_RANGE_PERCENTILE = 10.0
DEFAULT_EDGE_MAX_RANGE_PERCENTILE = 90.0
DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS = 5
DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION = 0.3


def lidar_points_for_info(
    info: dict[str, Any],
    *,
    lidar_root: Path,
    lidar_name: str,
) -> tuple[Path, np.ndarray, np.ndarray | None] | None:
    lidar_data = info.get("sensors", {}).get("lidar", {}).get(lidar_name)
    if not isinstance(lidar_data, dict):
        return None
    data_path = lidar_data.get("data_path")
    if not data_path:
        return None
    pcd_path = lidar_root / Path(str(data_path)).name
    if not pcd_path.exists():
        return None
    extrinsic = lidar_data.get("extrinsic")
    matrix = None if extrinsic is None else np.asarray(extrinsic, dtype=np.float64)
    return pcd_path, read_point_cloud_xyz(pcd_path), matrix


def lidar_to_camera_points(
    *,
    points_lidar: np.ndarray,
    lidar_extrinsic: np.ndarray | None,
    ego_to_camera: np.ndarray,
    lidar_extrinsic_direction: str,
) -> np.ndarray:
    if lidar_extrinsic is None:
        points_ego = points_lidar
    elif str(lidar_extrinsic_direction) == "lidar_to_ego":
        points_ego = transform_points(points_lidar, lidar_extrinsic)
    elif str(lidar_extrinsic_direction) == "ego_to_lidar":
        points_ego = transform_points(points_lidar, np.linalg.inv(lidar_extrinsic))
    else:
        raise ValueError(
            "lidar_extrinsic_direction must be lidar_to_ego or ego_to_lidar"
        )
    return transform_points(points_ego, ego_to_camera)


def prepare_range_display(
    *,
    range_image: np.ndarray,
    valid_mask: np.ndarray,
    edge_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rolled_range, rolled_valid, rolled_edge = roll_range_display_180(
        range_image,
        valid_mask,
        edge_mask,
    )
    cropped_range, cropped_valid, cropped_edge = crop_range_display_to_valid_region(
        rolled_range,
        rolled_valid,
        rolled_edge,
    )
    if cropped_range.size == 0:
        return cropped_range, cropped_valid, cropped_edge
    return rotate_range_display_180(cropped_range, cropped_valid, cropped_edge)



def make_edge_panel(
    *,
    image: np.ndarray,
    display_points_lidar: np.ndarray,
    edge_points_lidar: np.ndarray,
    edge_source_indices: np.ndarray | None,
    range_image: np.ndarray | None,
    range_valid_mask: np.ndarray | None,
    range_edge_mask: np.ndarray | None,
    middle_panel_mode: str,
    lidar_extrinsic: np.ndarray | None,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    lidar_extrinsic_direction: str,
    max_depth_m: float,
    depth_jump_abs_m: float,
    depth_jump_rel: float,
    depth_edge_mode: str,
    depth_edge_radius_px: int,
    min_valid_neighbors: int,
    occupancy_fill_radius_px: int,
    occupancy_boundary_width_px: int,
    occupancy_return_radius_px: int,
    occupancy_min_area_px: int,
    canny_low: int,
    canny_high: int,
    max_edge_distance_px: float,
    preview_scale: float,
    lidar_edge_visual_radius: int,
    projected_point_radius: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    image_edges = extract_image_edges(image, canny_low=canny_low, canny_high=canny_high)
    image_distance = distance_transform_from_edges(image_edges)
    points_camera = lidar_to_camera_points(
        points_lidar=display_points_lidar,
        lidar_extrinsic=lidar_extrinsic,
        ego_to_camera=ego_to_camera,
        lidar_extrinsic_direction=lidar_extrinsic_direction,
    )
    projection = project_points_to_depth_map(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=image.shape[:2],
        max_depth_m=max_depth_m,
    )
    if edge_source_indices is None and str(depth_edge_mode) == "adjacent":
        lidar_edges = extract_depth_jump_edges(
            projection.depth_map,
            projection.valid_mask,
            absolute_threshold_m=depth_jump_abs_m,
            relative_threshold=depth_jump_rel,
        )
    elif edge_source_indices is None and str(depth_edge_mode) in {"sparse", "combined"}:
        lidar_edges = extract_sparse_depth_jump_edges(
            projection.depth_map,
            projection.valid_mask,
            radius_px=depth_edge_radius_px,
            min_valid_neighbors=min_valid_neighbors,
            absolute_threshold_m=depth_jump_abs_m,
            relative_threshold=depth_jump_rel,
        )
        if str(depth_edge_mode) == "combined":
            lidar_edges |= extract_occupancy_boundary_edges(
                projection.valid_mask,
                fill_radius_px=occupancy_fill_radius_px,
                boundary_width_px=occupancy_boundary_width_px,
                return_radius_px=occupancy_return_radius_px,
                min_component_area_px=occupancy_min_area_px,
            )
    elif edge_source_indices is None and str(depth_edge_mode) == "occupancy":
        lidar_edges = extract_occupancy_boundary_edges(
            projection.valid_mask,
            fill_radius_px=occupancy_fill_radius_px,
            boundary_width_px=occupancy_boundary_width_px,
            return_radius_px=occupancy_return_radius_px,
            min_component_area_px=occupancy_min_area_px,
        )
    else:
        edge_points_camera = lidar_to_camera_points(
            points_lidar=edge_points_lidar,
            lidar_extrinsic=lidar_extrinsic,
            ego_to_camera=ego_to_camera,
            lidar_extrinsic_direction=lidar_extrinsic_direction,
        )
        edge_projection = project_points_to_depth_map(
            points_camera=edge_points_camera,
            intrinsic=intrinsic,
            image_shape=image.shape[:2],
            max_depth_m=max_depth_m,
        )
        lidar_edges = np.zeros(edge_projection.depth_map.shape, dtype=bool)
        edge_sources = set(np.asarray(edge_source_indices, dtype=np.int64).tolist())
        if edge_sources:
            keep = np.array(
                [int(index) in edge_sources for index in edge_projection.source_indices],
                dtype=bool,
            )
            lidar_edges[edge_projection.ys[keep], edge_projection.xs[keep]] = True
    lidar_edges_visual = lidar_edges
    radius = max(0, int(lidar_edge_visual_radius))
    if radius > 0 and np.any(lidar_edges):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        lidar_edges_visual = cv2.dilate(lidar_edges.astype(np.uint8), kernel) > 0
    edge_mask_visual = draw_edge_mask_visual(
        image_edges=image_edges,
        lidar_edges=lidar_edges_visual,
    )
    projected_lidar_image = draw_depth_projection_overlay(
        image,
        projection,
        point_radius=projected_point_radius,
    )
    image_edge_overlay = draw_binary_mask_overlay(
        image,
        image_edges,
        color=(0, 255, 0),
        alpha=0.85,
    )
    if str(middle_panel_mode) == "camera_projection":
        native_range_image = draw_depth_projection_overlay(
            np.zeros_like(image),
            projection,
            point_radius=projected_point_radius,
        )
        native_range_edges = draw_binary_mask_overlay(
            native_range_image,
            lidar_edges_visual,
            color=(0, 0, 255),
            alpha=1.0,
        )
    elif range_image is None:
        native_range_image = np.zeros_like(image)
        native_range_edges = np.zeros_like(image)
    else:
        display_range, display_valid, display_edges = prepare_range_display(
            range_image=range_image,
            valid_mask=(
                np.isfinite(range_image)
                if range_valid_mask is None
                else range_valid_mask
            ),
            edge_mask=range_edge_mask,
        )
        if display_range.size == 0:
            native_range_image = np.zeros_like(image)
            native_range_edges = np.zeros_like(image)
        else:
            native_range_image = range_image_to_visual(display_range, display_valid)
            edge_mask_for_display = (
                np.zeros(display_range.shape, dtype=bool)
                if display_edges is None
                else display_edges
            )
            native_range_edges = draw_binary_mask_overlay(
                native_range_image,
                edge_mask_for_display,
                color=(0, 0, 255),
                alpha=1.0,
            )
            if display_edges is not None:
                native_range_edges = cv2.addWeighted(
                    native_range_edges,
                    0.85,
                    binary_mask_to_visual(display_edges, color=(0, 0, 255)),
                    0.15,
                    0.0,
                )
    panel = make_lidar_diagnostic_grid(
        original_image=image,
        native_range_image=native_range_image,
        projected_lidar_image=projected_lidar_image,
        image_edge_overlay=image_edge_overlay,
        native_range_edges=native_range_edges,
        projected_edge_overlay=edge_mask_visual,
        preview_scale=preview_scale,
    )
    stats = summarize_edge_alignment(
        lidar_edges=lidar_edges,
        image_edge_distance=image_distance,
        max_distance_px=max_edge_distance_px,
    )
    stats.update(
        {
            "projected_lidar_pixels": int(np.count_nonzero(projection.valid_mask)),
            "image_edge_pixels": int(np.count_nonzero(image_edges)),
        }
    )
    return panel, stats


def lidar_range_edge_source_indices(
    points_lidar: np.ndarray,
    *,
    num_rows: int,
    azimuth_resolution_deg: float,
    vertical_fov_deg: tuple[float, float] | None,
    range_edge_abs_m: float,
    range_edge_rel: float,
    include_silhouette: bool,
    edge_side: str,
    min_component_pixels: int,
    silhouette_ignore_lower_fraction: float,
    range_axes: tuple[int, int, int],
    require_depth_jump_support: bool,
) -> tuple[np.ndarray, Any, np.ndarray, dict[str, float | int | str]]:
    forward_axis, lateral_axis, vertical_axis = range_axes
    projection = project_points_to_lidar_range_image(
        points_lidar,
        num_rows=num_rows,
        azimuth_resolution_deg=azimuth_resolution_deg,
        vertical_fov_deg=vertical_fov_deg,
        forward_axis=forward_axis,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
    )
    edge_mask = extract_lidar_range_edge_mask(
        projection,
        absolute_threshold_m=range_edge_abs_m,
        relative_threshold=range_edge_rel,
        include_silhouette=include_silhouette,
        edge_side=edge_side,
        min_component_pixels=min_component_pixels,
        silhouette_ignore_lower_fraction=silhouette_ignore_lower_fraction,
        require_depth_jump_support=require_depth_jump_support,
    )
    source_indices = extract_lidar_range_edge_source_indices(
        projection,
        absolute_threshold_m=range_edge_abs_m,
        relative_threshold=range_edge_rel,
        include_silhouette=include_silhouette,
        edge_side=edge_side,
        min_component_pixels=min_component_pixels,
        silhouette_ignore_lower_fraction=silhouette_ignore_lower_fraction,
        require_depth_jump_support=require_depth_jump_support,
    )
    metadata: dict[str, float | int | str] = {
        "range_image_rows": int(projection.num_rows),
        "range_image_cols": int(projection.num_cols),
        "range_image_azimuth_resolution_deg": float(projection.azimuth_resolution_deg),
        "range_image_vertical_min_deg": float(projection.vertical_fov_deg[0]),
        "range_image_vertical_max_deg": float(projection.vertical_fov_deg[1]),
        "range_image_valid_pixels": int(np.count_nonzero(projection.valid_mask)),
        "range_image_edge_points": int(len(source_indices)),
        "range_image_edge_pixels": int(np.count_nonzero(edge_mask)),
        "range_edge_side": str(edge_side),
        "range_edge_min_component_pixels": int(min_component_pixels),
        "range_edge_require_support": int(bool(require_depth_jump_support)),
        "range_silhouette_ignore_lower_fraction": float(
            silhouette_ignore_lower_fraction
        ),
        "range_forward_axis": int(forward_axis),
        "range_lateral_axis": int(lateral_axis),
        "range_vertical_axis": int(vertical_axis),
    }
    return source_indices, projection, edge_mask, metadata


def source_indices_to_range_edge_mask(
    source_index_image: np.ndarray,
    source_indices: np.ndarray | None,
) -> np.ndarray:
    if source_indices is None:
        return np.zeros(source_index_image.shape, dtype=bool)
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.zeros(source_index_image.shape, dtype=bool)
    source_image = np.asarray(source_index_image, dtype=np.int64)
    return (source_image >= 0) & np.isin(source_image, indices)


def extract_reference_edge_source_indices(
    *,
    image_shape: tuple[int, int],
    points_lidar: np.ndarray,
    lidar_extrinsic: np.ndarray | None,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    lidar_extrinsic_direction: str,
    max_depth_m: float,
    depth_jump_abs_m: float,
    depth_jump_rel: float,
    depth_edge_mode: str,
    depth_edge_radius_px: int,
    min_valid_neighbors: int,
    occupancy_fill_radius_px: int,
    occupancy_boundary_width_px: int,
    occupancy_return_radius_px: int,
    occupancy_min_area_px: int,
) -> np.ndarray:
    points_camera = lidar_to_camera_points(
        points_lidar=points_lidar,
        lidar_extrinsic=lidar_extrinsic,
        ego_to_camera=ego_to_camera,
        lidar_extrinsic_direction=lidar_extrinsic_direction,
    )
    projection = project_points_to_depth_map(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=image_shape,
        max_depth_m=max_depth_m,
    )
    if str(depth_edge_mode) == "adjacent":
        edge_mask = extract_depth_jump_edges(
            projection.depth_map,
            projection.valid_mask,
            absolute_threshold_m=depth_jump_abs_m,
            relative_threshold=depth_jump_rel,
        )
    elif str(depth_edge_mode) in {"sparse", "combined"}:
        edge_mask = extract_sparse_depth_jump_edges(
            projection.depth_map,
            projection.valid_mask,
            radius_px=depth_edge_radius_px,
            min_valid_neighbors=min_valid_neighbors,
            absolute_threshold_m=depth_jump_abs_m,
            relative_threshold=depth_jump_rel,
        )
        if str(depth_edge_mode) == "combined":
            edge_mask |= extract_occupancy_boundary_edges(
                projection.valid_mask,
                fill_radius_px=occupancy_fill_radius_px,
                boundary_width_px=occupancy_boundary_width_px,
                return_radius_px=occupancy_return_radius_px,
                min_component_area_px=occupancy_min_area_px,
            )
    elif str(depth_edge_mode) == "occupancy":
        edge_mask = extract_occupancy_boundary_edges(
            projection.valid_mask,
            fill_radius_px=occupancy_fill_radius_px,
            boundary_width_px=occupancy_boundary_width_px,
            return_radius_px=occupancy_return_radius_px,
            min_component_area_px=occupancy_min_area_px,
        )
    else:
        raise ValueError("depth_edge_mode must be combined, sparse, adjacent, or occupancy")
    keep = edge_mask[projection.ys, projection.xs]
    return np.unique(projection.source_indices[keep]).astype(np.int64)


def process_frames(
    *,
    before_pkl: Path,
    after_pkl: Path | None,
    compare_after: bool,
    lidar_root: Path,
    image_root: Path,
    output_dir: Path,
    camera: str,
    lidar_name: str,
    lidar_extrinsic_direction: str,
    start: int,
    end: int | None,
    stride: int,
    max_lidar_range_m: float | None,
    edge_min_range_percentile: float,
    edge_max_range_percentile: float,
    max_depth_m: float,
    depth_jump_abs_m: float,
    depth_jump_rel: float,
    depth_edge_mode: str,
    depth_edge_radius_px: int,
    min_valid_neighbors: int,
    occupancy_fill_radius_px: int,
    occupancy_boundary_width_px: int,
    occupancy_return_radius_px: int,
    occupancy_min_area_px: int,
    canny_low: int,
    canny_high: int,
    max_edge_distance_px: float,
    preview_scale: float,
    lidar_edge_visual_radius: int,
    projected_point_radius: int,
    edge_reference: str,
    edge_source: str,
    lidar_rows: int,
    lidar_azimuth_resolution_deg: float,
    lidar_vertical_fov_deg: tuple[float, float] | None,
    range_edge_abs_m: float,
    range_edge_rel: float,
    range_edge_side: str,
    range_edge_min_component_pixels: int,
    range_edge_require_support: bool,
    range_silhouette_ignore_lower_fraction: float,
    range_include_silhouette: bool,
    projected_edge_search_radius_px: int,
    projected_edge_require_support: bool,
) -> list[dict[str, float | int | str]]:
    before_infos = load_infos(before_pkl)
    after_infos = load_infos(after_pkl) if compare_after and after_pkl is not None else None
    if compare_after and after_infos is None:
        raise ValueError("--compare-after requires --after-pkl")
    frame_end = len(before_infos) if end is None else min(int(end), len(before_infos))
    if after_infos is not None:
        frame_end = min(frame_end, len(after_infos))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for frame_index in range(int(start), frame_end, max(1, int(stride))):
        before_info = before_infos[frame_index]
        before_cam = before_info.get("sensors", {}).get("cams", {}).get(camera)
        if not isinstance(before_cam, dict):
            continue
        after_cam = None
        if after_infos is not None:
            after_info = after_infos[frame_index]
            candidate_after_cam = after_info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(candidate_after_cam, dict):
                continue
            after_cam = candidate_after_cam
        lidar_payload = lidar_points_for_info(
            before_info,
            lidar_root=lidar_root,
            lidar_name=lidar_name,
        )
        image_payload = image_for_camera(before_cam, image_root=image_root, camera=camera)
        if lidar_payload is None or image_payload is None:
            continue
        pcd_path, raw_points_lidar, lidar_extrinsic = lidar_payload
        display_points_lidar = raw_points_lidar
        if max_lidar_range_m is None:
            edge_points_lidar = raw_points_lidar
            lidar_range_keep = np.ones(len(raw_points_lidar), dtype=bool)
            resolved_min_lidar_range_m = float("nan")
            resolved_max_lidar_range_m = float("nan")
            range_filter_mode = "post_edge_percentile_band"
        else:
            edge_points_lidar, lidar_range_keep = filter_points_by_range(
                raw_points_lidar,
                max_range_m=max_lidar_range_m,
            )
            resolved_min_lidar_range_m = 0.0
            resolved_max_lidar_range_m = float(max_lidar_range_m)
            range_filter_mode = "explicit"
        image_path, image = image_payload
        intrinsic = np.asarray(before_cam["cam_intrinsic"], dtype=np.float64)
        range_metadata: dict[str, float | int | str] = {}
        range_axes = infer_lidar_range_axes(
            display_points_lidar,
            lidar_extrinsic=lidar_extrinsic,
        )
        range_forward_axis, range_lateral_axis, range_vertical_axis = range_axes
        display_range_projection = project_points_to_lidar_range_image(
            display_points_lidar,
            num_rows=lidar_rows,
            azimuth_resolution_deg=lidar_azimuth_resolution_deg,
            vertical_fov_deg=lidar_vertical_fov_deg,
            forward_axis=range_forward_axis,
            lateral_axis=range_lateral_axis,
            vertical_axis=range_vertical_axis,
        )
        edge_range_vertical_fov_deg = (
            lidar_vertical_fov_deg
            if lidar_vertical_fov_deg is not None
            else display_range_projection.vertical_fov_deg
        )
        range_projection = display_range_projection
        range_edge_mask = None
        if str(edge_source) == "lidar_range":
            (
                full_edge_source_indices,
                edge_range_projection,
                range_edge_mask,
                range_metadata,
            ) = lidar_range_edge_source_indices(
                display_points_lidar,
                num_rows=lidar_rows,
                azimuth_resolution_deg=lidar_azimuth_resolution_deg,
                vertical_fov_deg=edge_range_vertical_fov_deg,
                range_edge_abs_m=range_edge_abs_m,
                range_edge_rel=range_edge_rel,
                edge_side=range_edge_side,
                min_component_pixels=range_edge_min_component_pixels,
                silhouette_ignore_lower_fraction=range_silhouette_ignore_lower_fraction,
                include_silhouette=range_include_silhouette,
                range_axes=range_axes,
                require_depth_jump_support=range_edge_require_support,
            )
            if max_lidar_range_m is None:
                (
                    edge_source_indices,
                    resolved_min_lidar_range_m,
                    resolved_max_lidar_range_m,
                ) = select_source_indices_by_range_percentile_band(
                    full_edge_source_indices,
                    display_points_lidar,
                    min_percentile=edge_min_range_percentile,
                    max_percentile=edge_max_range_percentile,
                )
                ranges = np.linalg.norm(display_points_lidar, axis=1)
                lidar_range_keep = (
                    np.isfinite(display_points_lidar).all(axis=1)
                    & np.isfinite(ranges)
                    & (ranges >= resolved_min_lidar_range_m)
                    & (ranges <= resolved_max_lidar_range_m)
                )
            else:
                edge_keep_indices = np.flatnonzero(lidar_range_keep).astype(np.int64)
                edge_keep_set = set(edge_keep_indices.tolist())
                edge_source_indices = np.asarray(
                    [
                        int(index)
                        for index in full_edge_source_indices
                        if int(index) in edge_keep_set
                    ],
                    dtype=np.int64,
                )
            range_metadata["edge_range_image_valid_pixels"] = range_metadata.pop(
                "range_image_valid_pixels"
            )
            full_range_edge_pixels = range_metadata.pop("range_image_edge_pixels")
            range_edge_mask = source_indices_to_range_edge_mask(
                display_range_projection.source_index_image,
                edge_source_indices,
            )
            range_metadata["edge_range_image_edge_pixels"] = int(
                np.count_nonzero(range_edge_mask)
            )
            range_metadata["full_edge_range_image_edge_pixels"] = int(
                full_range_edge_pixels
            )
            range_metadata["full_edge_range_image_edge_points"] = range_metadata.pop(
                "range_image_edge_points"
            )
            range_metadata["edge_range_image_edge_points"] = int(len(edge_source_indices))
            range_metadata["range_image_valid_pixels"] = int(
                np.count_nonzero(display_range_projection.valid_mask)
            )
            range_metadata["range_image_display_points"] = int(len(display_points_lidar))
            range_metadata["edge_range_image_points"] = int(len(display_points_lidar))
        else:
            edge_source_indices = None
        if str(edge_source) == "camera_projected_lidar":
            reference_cam = before_cam if str(edge_reference) == "before" else after_cam
            if reference_cam is None:
                reference_cam = before_cam
            candidate_points_lidar = (
                display_points_lidar
                if max_lidar_range_m is None
                else edge_points_lidar
            )
            candidate_points_camera = lidar_to_camera_points(
                points_lidar=candidate_points_lidar,
                lidar_extrinsic=lidar_extrinsic,
                ego_to_camera=np.asarray(reference_cam["extrinsic"], dtype=np.float64),
                lidar_extrinsic_direction=lidar_extrinsic_direction,
            )
            full_edge_source_indices = extract_projected_lidar_depth_edge_source_indices(
                points_camera=candidate_points_camera,
                intrinsic=np.asarray(reference_cam["cam_intrinsic"], dtype=np.float64),
                image_shape=image.shape[:2],
                max_depth_m=max_depth_m,
                search_radius_px=projected_edge_search_radius_px,
                require_same_depth_support=projected_edge_require_support,
                absolute_threshold_m=depth_jump_abs_m,
                relative_threshold=depth_jump_rel,
                edge_side=range_edge_side,
            )
            if max_lidar_range_m is None:
                (
                    edge_source_indices,
                    resolved_min_lidar_range_m,
                    resolved_max_lidar_range_m,
                ) = select_source_indices_by_range_percentile_band(
                    full_edge_source_indices,
                    display_points_lidar,
                    min_percentile=edge_min_range_percentile,
                    max_percentile=edge_max_range_percentile,
                )
                ranges = np.linalg.norm(display_points_lidar, axis=1)
                lidar_range_keep = (
                    np.isfinite(display_points_lidar).all(axis=1)
                    & np.isfinite(ranges)
                    & (ranges >= resolved_min_lidar_range_m)
                    & (ranges <= resolved_max_lidar_range_m)
                )
            else:
                edge_source_indices = full_edge_source_indices
            range_metadata["camera_view_full_edge_points"] = int(
                len(full_edge_source_indices)
            )
            range_metadata["camera_view_edge_points"] = int(len(edge_source_indices))
            range_metadata["camera_view_search_radius_px"] = int(
                projected_edge_search_radius_px
            )
            range_metadata["camera_view_require_support"] = int(
                bool(projected_edge_require_support)
            )
        if after_cam is not None and str(edge_source) == "camera_projection":
            reference_cam = before_cam if str(edge_reference) == "before" else after_cam
            edge_source_indices = extract_reference_edge_source_indices(
                image_shape=image.shape[:2],
                points_lidar=edge_points_lidar,
                lidar_extrinsic=lidar_extrinsic,
                ego_to_camera=np.asarray(reference_cam["extrinsic"], dtype=np.float64),
                intrinsic=np.asarray(reference_cam["cam_intrinsic"], dtype=np.float64),
                lidar_extrinsic_direction=lidar_extrinsic_direction,
                max_depth_m=max_depth_m,
                depth_jump_abs_m=depth_jump_abs_m,
                depth_jump_rel=depth_jump_rel,
                depth_edge_mode=depth_edge_mode,
                depth_edge_radius_px=depth_edge_radius_px,
                min_valid_neighbors=min_valid_neighbors,
                occupancy_fill_radius_px=occupancy_fill_radius_px,
                occupancy_boundary_width_px=occupancy_boundary_width_px,
                occupancy_return_radius_px=occupancy_return_radius_px,
                occupancy_min_area_px=occupancy_min_area_px,
            )
        if range_edge_mask is None and edge_source_indices is not None:
            range_edge_mask = source_indices_to_range_edge_mask(
                display_range_projection.source_index_image,
                edge_source_indices,
            )
        before_panel, before_stats = make_edge_panel(
            image=image,
            display_points_lidar=display_points_lidar,
            edge_points_lidar=(
                display_points_lidar
                if str(edge_source) == "lidar_range"
                else edge_points_lidar
            ),
            edge_source_indices=edge_source_indices,
            range_image=None if range_projection is None else range_projection.range_image,
            range_valid_mask=None if range_projection is None else range_projection.valid_mask,
            range_edge_mask=range_edge_mask,
            middle_panel_mode="native_range",
            lidar_extrinsic=lidar_extrinsic,
            ego_to_camera=np.asarray(before_cam["extrinsic"], dtype=np.float64),
            intrinsic=intrinsic,
            lidar_extrinsic_direction=lidar_extrinsic_direction,
            max_depth_m=max_depth_m,
            depth_jump_abs_m=depth_jump_abs_m,
            depth_jump_rel=depth_jump_rel,
            depth_edge_mode=depth_edge_mode,
            depth_edge_radius_px=depth_edge_radius_px,
            min_valid_neighbors=min_valid_neighbors,
            occupancy_fill_radius_px=occupancy_fill_radius_px,
            occupancy_boundary_width_px=occupancy_boundary_width_px,
            occupancy_return_radius_px=occupancy_return_radius_px,
            occupancy_min_area_px=occupancy_min_area_px,
            canny_low=canny_low,
            canny_high=canny_high,
            max_edge_distance_px=max_edge_distance_px,
            preview_scale=preview_scale,
            lidar_edge_visual_radius=lidar_edge_visual_radius,
            projected_point_radius=projected_point_radius,
        )
        output_image = before_panel
        after_stats = None
        output_prefix = "edge_before"
        if after_cam is not None:
            after_panel, after_stats = make_edge_panel(
                image=image,
                display_points_lidar=display_points_lidar,
                edge_points_lidar=(
                    display_points_lidar
                    if str(edge_source) == "lidar_range"
                    else edge_points_lidar
                ),
                edge_source_indices=edge_source_indices,
                range_image=None if range_projection is None else range_projection.range_image,
                range_valid_mask=None if range_projection is None else range_projection.valid_mask,
                range_edge_mask=range_edge_mask,
                middle_panel_mode="native_range",
                lidar_extrinsic=lidar_extrinsic,
                ego_to_camera=np.asarray(after_cam["extrinsic"], dtype=np.float64),
                intrinsic=np.asarray(after_cam["cam_intrinsic"], dtype=np.float64),
                lidar_extrinsic_direction=lidar_extrinsic_direction,
                max_depth_m=max_depth_m,
                depth_jump_abs_m=depth_jump_abs_m,
                depth_jump_rel=depth_jump_rel,
                depth_edge_mode=depth_edge_mode,
                depth_edge_radius_px=depth_edge_radius_px,
                min_valid_neighbors=min_valid_neighbors,
                occupancy_fill_radius_px=occupancy_fill_radius_px,
                occupancy_boundary_width_px=occupancy_boundary_width_px,
                occupancy_return_radius_px=occupancy_return_radius_px,
                occupancy_min_area_px=occupancy_min_area_px,
                canny_low=canny_low,
                canny_high=canny_high,
                max_edge_distance_px=max_edge_distance_px,
                preview_scale=preview_scale,
                lidar_edge_visual_radius=lidar_edge_visual_radius,
                projected_point_radius=projected_point_radius,
            )
            output_image = np.concatenate([before_panel, after_panel], axis=0)
            output_prefix = "edge_compare"
        output_path = output_dir / f"{output_prefix}_{frame_index:06d}_{image_path.name}"
        cv2.imwrite(str(output_path), output_image)
        range_depth_output = ""
        range_edges_output = ""
        range_display_rows = 0
        range_display_cols = 0
        if range_projection is not None:
            range_dir = output_dir / "range_native"
            range_dir.mkdir(parents=True, exist_ok=True)
            display_range, display_valid, display_edges = prepare_range_display(
                range_image=range_projection.range_image,
                valid_mask=range_projection.valid_mask,
                edge_mask=range_edge_mask,
            )
            if display_range.size > 0:
                range_display_rows = int(display_range.shape[0])
                range_display_cols = int(display_range.shape[1])
                range_depth = range_image_to_visual(display_range, display_valid)
                range_edges = draw_binary_mask_overlay(
                    range_depth,
                    np.zeros(display_range.shape, dtype=bool)
                    if display_edges is None
                    else display_edges,
                    color=(0, 0, 255),
                    alpha=1.0,
                )
                range_depth_path = range_dir / f"range_depth_{frame_index:06d}_{pcd_path.stem}.png"
                range_edges_path = range_dir / f"range_edges_{frame_index:06d}_{pcd_path.stem}.png"
                cv2.imwrite(str(range_depth_path), range_depth)
                cv2.imwrite(str(range_edges_path), range_edges)
                range_depth_output = str(range_depth_path)
                range_edges_output = str(range_edges_path)
        row: dict[str, float | int | str] = {
            "frame": int(frame_index),
            "image": image_path.name,
            "pcd": pcd_path.name,
            "lidar": lidar_name,
            "camera": camera,
            "raw_lidar_points": int(len(raw_points_lidar)),
            "display_lidar_points": int(len(display_points_lidar)),
            "edge_range_filtered_lidar_points": int(np.count_nonzero(lidar_range_keep)),
            "range_filtered_fraction": float(
                1.0 - (np.count_nonzero(lidar_range_keep) / max(len(raw_points_lidar), 1))
            ),
            "range_filter_mode": range_filter_mode,
            "min_lidar_range_m": float(resolved_min_lidar_range_m),
            "max_lidar_range_m": float(resolved_max_lidar_range_m),
            "edge_min_range_percentile": float(edge_min_range_percentile),
            "edge_max_range_percentile": float(edge_max_range_percentile),
            "edge_source": edge_source,
            "lidar_extrinsic_direction": lidar_extrinsic_direction,
            "depth_edge_mode": depth_edge_mode,
            "depth_edge_radius_px": int(depth_edge_radius_px),
            "min_valid_neighbors": int(min_valid_neighbors),
            "occupancy_fill_radius_px": int(occupancy_fill_radius_px),
            "occupancy_boundary_width_px": int(occupancy_boundary_width_px),
            "occupancy_return_radius_px": int(occupancy_return_radius_px),
            "occupancy_min_area_px": int(occupancy_min_area_px),
            "projected_point_radius": int(projected_point_radius),
            "output": str(output_path),
            "range_depth_output": range_depth_output,
            "range_edges_output": range_edges_output,
            "range_display_rows": int(range_display_rows),
            "range_display_cols": int(range_display_cols),
        }
        if edge_source_indices is not None:
            row["edge_reference"] = edge_reference
            row["reference_lidar_edge_points"] = int(len(edge_source_indices))
        row.update(range_metadata)
        for key, value in before_stats.items():
            row[f"before_{key}"] = value
        if after_stats is not None:
            for key, value in after_stats.items():
                row[f"after_{key}"] = value
        rows.append(row)
        message = f"[{frame_index:06d}] before median={row['before_median_distance_px']:.2f}px"
        if after_stats is not None:
            message += f" after median={row['after_median_distance_px']:.2f}px"
        print(f"{message} -> {output_path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    default_root = Path("data/M18_canonical")
    parser = argparse.ArgumentParser(
        description=(
            "Extract before-PKL LiDAR-image edge diagnostics. "
            "Each output image is a 2x3 grid: image, native LiDAR range, "
            "LiDAR projection, image edges, native range edges, projected edge overlay."
        )
    )
    parser.add_argument(
        "--before-pkl",
        type=Path,
        default=default_root / "result_interpolation_standard.pkl",
    )
    parser.add_argument(
        "--after-pkl",
        type=Path,
        help="Optional optimized PKL, used only with --compare-after.",
    )
    parser.add_argument(
        "--compare-after",
        action="store_true",
        help="Optional diagnostic mode: stack before over after using one shared LiDAR edge set.",
    )
    parser.add_argument("--image-root", type=Path, default=default_root)
    parser.add_argument("--lidar-root", type=Path, default=default_root / "front_lidar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m18_lidar_image_edges_front_lidar_center_fov30"),
    )
    parser.add_argument("--camera", default="center_camera_fov30", choices=sorted(CAMERA_TO_IMAGE_FOLDER))
    parser.add_argument("--lidar-name", default="front_lidar")
    parser.add_argument(
        "--lidar-extrinsic-direction",
        default="lidar_to_ego",
        choices=("lidar_to_ego", "ego_to_lidar"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-lidar-range-m",
        type=float,
        default=None,
        help=(
            "Compatibility option for edge candidates only; default uses "
            "--edge-min-range-percentile/--edge-max-range-percentile."
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
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--depth-jump-abs-m", type=float, default=0.8)
    parser.add_argument("--depth-jump-rel", type=float, default=0.08)
    parser.add_argument(
        "--depth-edge-mode",
        default="combined",
        choices=("combined", "sparse", "adjacent", "occupancy"),
        help=(
            "combined = sparse depth jumps plus LiDAR occupancy silhouettes; "
            "occupancy catches valid-to-empty object/sky boundaries."
        ),
    )
    parser.add_argument("--depth-edge-radius-px", type=int, default=5)
    parser.add_argument("--min-valid-neighbors", type=int, default=2)
    parser.add_argument("--occupancy-fill-radius-px", type=int, default=12)
    parser.add_argument("--occupancy-boundary-width-px", type=int, default=1)
    parser.add_argument("--occupancy-return-radius-px", type=int, default=1)
    parser.add_argument("--occupancy-min-area-px", type=int, default=1000)
    parser.add_argument(
        "--edge-source",
        default="lidar_range",
        choices=("lidar_range", "camera_projection", "camera_projected_lidar"),
        help=(
            "lidar_range extracts native LiDAR range-image edges; "
            "camera_projected_lidar extracts LiDAR depth edges in the target camera view."
        ),
    )
    parser.add_argument("--lidar-rows", type=int, default=128)
    parser.add_argument("--lidar-azimuth-resolution-deg", type=float, default=0.2)
    parser.add_argument("--lidar-vertical-min-deg", type=float)
    parser.add_argument("--lidar-vertical-max-deg", type=float)
    parser.add_argument("--range-edge-abs-m", type=float, default=0.8)
    parser.add_argument("--range-edge-rel", type=float, default=0.08)
    parser.add_argument(
        "--range-edge-side",
        default="near",
        choices=("near", "both", "far"),
        help="Depth-jump side to keep in LiDAR range image; near is recommended for constraints.",
    )
    parser.add_argument(
        "--range-edge-min-component-pixels",
        type=int,
        default=DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS,
        help="Remove LiDAR range edge connected components smaller than this pixel count.",
    )
    parser.add_argument(
        "--no-range-edge-support",
        action="store_true",
        help=(
            "Disable same-side depth support for native LiDAR range depth-jump edges."
        ),
    )
    parser.add_argument(
        "--range-silhouette-ignore-lower-fraction",
        type=float,
        default=DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION,
        help=(
            "Ignore silhouette edges in the lowest-elevation fraction of the "
            "LiDAR range image; 0.3 suppresses the displayed bottom 30 percent."
        ),
    )
    parser.add_argument(
        "--no-range-silhouette",
        action="store_true",
        help="Disable valid-to-empty silhouette edges in LiDAR range image.",
    )
    parser.add_argument(
        "--projected-edge-search-radius-px",
        type=int,
        default=2,
        help=(
            "8-direction sparse depth search radius for camera_projected_lidar; "
            "2 means check the immediate 8-neighborhood, then one expanded ring."
        ),
    )
    parser.add_argument(
        "--no-projected-edge-support",
        action="store_true",
        help=(
            "Disable same-depth support filtering for camera_projected_lidar; "
            "useful for sparse side LiDARs."
        ),
    )
    parser.add_argument(
        "--edge-reference",
        default="before",
        choices=("before", "after"),
        help="Pose used only to select one shared LiDAR edge point set for before/after projection.",
    )
    parser.add_argument("--canny-low", type=int, default=DEFAULT_CANNY_LOW)
    parser.add_argument("--canny-high", type=int, default=DEFAULT_CANNY_HIGH)
    parser.add_argument("--max-edge-distance-px", type=float, default=20.0)
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Scale saved diagnostic images only; CSV metrics stay at full resolution.",
    )
    parser.add_argument(
        "--lidar-edge-visual-radius",
        type=int,
        default=2,
        help="Dilate LiDAR edge pixels for visualization only; CSV metrics use raw edges.",
    )
    parser.add_argument(
        "--projected-point-radius",
        type=int,
        default=2,
        help="Point radius for the top-right LiDAR projection overlay only.",
    )
    parser.add_argument(
        "--depth-visual-radius",
        type=int,
        default=0,
        help="Deprecated compatibility option; native LiDAR range images are never radius-dilated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = process_frames(
        before_pkl=args.before_pkl,
        after_pkl=args.after_pkl,
        compare_after=args.compare_after,
        lidar_root=args.lidar_root,
        image_root=args.image_root,
        output_dir=args.output_dir,
        camera=args.camera,
        lidar_name=args.lidar_name,
        lidar_extrinsic_direction=args.lidar_extrinsic_direction,
        start=args.start,
        end=args.end,
        stride=args.stride,
        max_lidar_range_m=args.max_lidar_range_m,
        edge_min_range_percentile=args.edge_min_range_percentile,
        edge_max_range_percentile=args.edge_max_range_percentile,
        max_depth_m=args.max_depth_m,
        depth_jump_abs_m=args.depth_jump_abs_m,
        depth_jump_rel=args.depth_jump_rel,
        depth_edge_mode=args.depth_edge_mode,
        depth_edge_radius_px=args.depth_edge_radius_px,
        min_valid_neighbors=args.min_valid_neighbors,
        occupancy_fill_radius_px=args.occupancy_fill_radius_px,
        occupancy_boundary_width_px=args.occupancy_boundary_width_px,
        occupancy_return_radius_px=args.occupancy_return_radius_px,
        occupancy_min_area_px=args.occupancy_min_area_px,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        max_edge_distance_px=args.max_edge_distance_px,
        preview_scale=args.preview_scale,
        lidar_edge_visual_radius=args.lidar_edge_visual_radius,
        projected_point_radius=args.projected_point_radius,
        edge_reference=args.edge_reference,
        edge_source=args.edge_source,
        lidar_rows=args.lidar_rows,
        lidar_azimuth_resolution_deg=args.lidar_azimuth_resolution_deg,
        lidar_vertical_fov_deg=(
            None
            if args.lidar_vertical_min_deg is None or args.lidar_vertical_max_deg is None
            else (args.lidar_vertical_min_deg, args.lidar_vertical_max_deg)
        ),
        range_edge_abs_m=args.range_edge_abs_m,
        range_edge_rel=args.range_edge_rel,
        range_edge_side=args.range_edge_side,
        range_edge_min_component_pixels=args.range_edge_min_component_pixels,
        range_edge_require_support=not args.no_range_edge_support,
        range_silhouette_ignore_lower_fraction=args.range_silhouette_ignore_lower_fraction,
        range_include_silhouette=not args.no_range_silhouette,
        projected_edge_search_radius_px=args.projected_edge_search_radius_px,
        projected_edge_require_support=not args.no_projected_edge_support,
    )
    csv_path = args.output_dir / "edge_alignment_summary.csv"
    write_csv(csv_path, rows)
    print(f"[done] {len(rows)} frames -> {args.output_dir}")
    if rows:
        print(f"[done] summary -> {csv_path}")


if __name__ == "__main__":
    main()
