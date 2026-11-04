from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthProjection:
    depth_map: np.ndarray
    valid_mask: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    depths: np.ndarray
    points_camera: np.ndarray
    source_indices: np.ndarray


@dataclass(frozen=True)
class LidarRangeProjection:
    range_image: np.ndarray
    valid_mask: np.ndarray
    source_index_image: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    ranges: np.ndarray
    source_indices: np.ndarray
    num_rows: int
    num_cols: int
    azimuth_resolution_deg: float
    vertical_fov_deg: tuple[float, float]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.copy()
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_h = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    return (matrix @ points_h.T).T[:, :3]


def filter_points_by_range(
    points: np.ndarray,
    *,
    max_range_m: float | None,
    min_range_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.copy(), np.zeros((0,), dtype=bool)
    ranges = np.linalg.norm(values, axis=1)
    keep = np.isfinite(values).all(axis=1) & np.isfinite(ranges) & (ranges >= float(min_range_m))
    if max_range_m is not None:
        keep &= ranges <= float(max_range_m)
    return values[keep], keep


def filter_points_by_range_percentile(
    points: np.ndarray,
    *,
    percentile: float = 95.0,
    min_range_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.copy(), np.zeros((0,), dtype=bool), float("nan")
    ranges = np.linalg.norm(values, axis=1)
    finite = np.isfinite(values).all(axis=1) & np.isfinite(ranges) & (ranges >= float(min_range_m))
    if not np.any(finite):
        return values[:0].copy(), np.zeros(len(values), dtype=bool), float("nan")
    threshold = float(np.percentile(ranges[finite], float(percentile)))
    keep = finite & (ranges <= threshold)
    return values[keep], keep, threshold


def filter_points_by_range_percentile_band(
    points: np.ndarray,
    *,
    min_percentile: float = 5.0,
    max_percentile: float = 80.0,
    min_range_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.copy(), np.zeros((0,), dtype=bool), float("nan"), float("nan")
    ranges = np.linalg.norm(values, axis=1)
    finite = np.isfinite(values).all(axis=1) & np.isfinite(ranges) & (ranges >= float(min_range_m))
    if not np.any(finite):
        return values[:0].copy(), np.zeros(len(values), dtype=bool), float("nan"), float("nan")
    low = float(np.percentile(ranges[finite], float(min_percentile)))
    high = float(np.percentile(ranges[finite], float(max_percentile)))
    if high < low:
        low, high = high, low
    keep = finite & (ranges >= low) & (ranges <= high)
    return values[keep], keep, low, high


def select_source_indices_by_range_percentile_band(
    source_indices: np.ndarray,
    points: np.ndarray,
    *,
    min_percentile: float = 5.0,
    max_percentile: float = 80.0,
    min_range_m: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if values.size == 0 or indices.size == 0:
        return indices[:0].copy(), float("nan"), float("nan")

    ranges = np.linalg.norm(values, axis=1)
    finite = np.isfinite(values).all(axis=1) & np.isfinite(ranges) & (ranges >= float(min_range_m))
    if not np.any(finite):
        return indices[:0].copy(), float("nan"), float("nan")

    low = float(np.percentile(ranges[finite], float(min_percentile)))
    high = float(np.percentile(ranges[finite], float(max_percentile)))
    if high < low:
        low, high = high, low

    valid_indices = indices[(indices >= 0) & (indices < len(values))]
    selected_ranges = ranges[valid_indices]
    keep = (
        np.isfinite(selected_ranges)
        & (selected_ranges >= low)
        & (selected_ranges <= high)
    )
    return valid_indices[keep].astype(np.int64), low, high


def project_points_to_lidar_range_image(
    points_lidar: np.ndarray,
    *,
    num_rows: int = 128,
    azimuth_resolution_deg: float = 0.1,
    vertical_fov_deg: tuple[float, float] | None = None,
    min_range_m: float = 0.01,
    max_range_m: float | None = None,
    forward_axis: int = 0,
    lateral_axis: int = 1,
    vertical_axis: int = 2,
) -> LidarRangeProjection:
    values = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    forward_axis, lateral_axis, vertical_axis = _validate_range_axes(
        forward_axis,
        lateral_axis,
        vertical_axis,
    )
    num_rows = max(1, int(num_rows))
    azimuth_resolution_deg = float(azimuth_resolution_deg)
    num_cols = max(1, int(round(360.0 / azimuth_resolution_deg)))
    range_image = np.full((num_rows, num_cols), np.inf, dtype=np.float32)
    source_index_image = np.full((num_rows, num_cols), -1, dtype=np.int64)
    if values.size == 0:
        return LidarRangeProjection(
            range_image=range_image,
            valid_mask=np.zeros_like(range_image, dtype=bool),
            source_index_image=source_index_image,
            rows=np.empty((0,), dtype=np.int32),
            cols=np.empty((0,), dtype=np.int32),
            ranges=np.empty((0,), dtype=np.float32),
            source_indices=np.empty((0,), dtype=np.int64),
            num_rows=num_rows,
            num_cols=num_cols,
            azimuth_resolution_deg=azimuth_resolution_deg,
            vertical_fov_deg=(float("nan"), float("nan")),
        )

    source_indices = np.arange(len(values), dtype=np.int64)
    ranges = np.linalg.norm(values, axis=1)
    finite = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(min_range_m))
    )
    if max_range_m is not None:
        finite &= ranges <= float(max_range_m)
    values = values[finite]
    ranges = ranges[finite]
    source_indices = source_indices[finite]
    if values.size == 0:
        return LidarRangeProjection(
            range_image=range_image,
            valid_mask=np.zeros_like(range_image, dtype=bool),
            source_index_image=source_index_image,
            rows=np.empty((0,), dtype=np.int32),
            cols=np.empty((0,), dtype=np.int32),
            ranges=np.empty((0,), dtype=np.float32),
            source_indices=np.empty((0,), dtype=np.int64),
            num_rows=num_rows,
            num_cols=num_cols,
            azimuth_resolution_deg=azimuth_resolution_deg,
            vertical_fov_deg=(float("nan"), float("nan")),
        )

    forward_values = values[:, forward_axis]
    lateral_values = values[:, lateral_axis]
    vertical_values = values[:, vertical_axis]
    azimuth_deg = (
        np.degrees(np.arctan2(lateral_values, forward_values)) + 360.0
    ) % 360.0
    elevation_deg = np.degrees(
        np.arcsin(
            np.clip(vertical_values / np.maximum(ranges, 1e-12), -1.0, 1.0)
        )
    )
    if vertical_fov_deg is None:
        v_min = float(np.percentile(elevation_deg, 0.5))
        v_max = float(np.percentile(elevation_deg, 99.5))
        if v_max <= v_min:
            v_min -= 0.5
            v_max += 0.5
    else:
        v_min, v_max = float(vertical_fov_deg[0]), float(vertical_fov_deg[1])
    vertical_span = max(v_max - v_min, 1e-6)
    rows = np.floor((elevation_deg - v_min) / vertical_span * num_rows).astype(np.int32)
    cols = np.floor(azimuth_deg / azimuth_resolution_deg).astype(np.int32)
    valid = (rows >= 0) & (rows < num_rows) & (cols >= 0) & (cols < num_cols)
    rows = rows[valid]
    cols = cols[valid]
    ranges_valid = ranges[valid].astype(np.float32)
    source_valid = source_indices[valid]
    if len(rows) == 0:
        return LidarRangeProjection(
            range_image=range_image,
            valid_mask=np.zeros_like(range_image, dtype=bool),
            source_index_image=source_index_image,
            rows=np.empty((0,), dtype=np.int32),
            cols=np.empty((0,), dtype=np.int32),
            ranges=np.empty((0,), dtype=np.float32),
            source_indices=np.empty((0,), dtype=np.int64),
            num_rows=num_rows,
            num_cols=num_cols,
            azimuth_resolution_deg=azimuth_resolution_deg,
            vertical_fov_deg=(v_min, v_max),
        )

    order = np.argsort(ranges_valid)
    rows = rows[order]
    cols = cols[order]
    ranges_valid = ranges_valid[order]
    source_valid = source_valid[order]
    flat = rows * num_cols + cols
    _, first_indices = np.unique(flat, return_index=True)
    first_indices.sort()
    rows_nearest = rows[first_indices]
    cols_nearest = cols[first_indices]
    ranges_nearest = ranges_valid[first_indices]
    source_nearest = source_valid[first_indices]
    range_image[rows_nearest, cols_nearest] = ranges_nearest
    source_index_image[rows_nearest, cols_nearest] = source_nearest
    return LidarRangeProjection(
        range_image=range_image,
        valid_mask=np.isfinite(range_image),
        source_index_image=source_index_image,
        rows=rows_nearest.astype(np.int32, copy=False),
        cols=cols_nearest.astype(np.int32, copy=False),
        ranges=ranges_nearest.astype(np.float32, copy=False),
        source_indices=source_nearest.astype(np.int64, copy=False),
        num_rows=num_rows,
        num_cols=num_cols,
        azimuth_resolution_deg=azimuth_resolution_deg,
        vertical_fov_deg=(v_min, v_max),
    )


def infer_lidar_range_axes(
    points_lidar: np.ndarray,
    *,
    lidar_extrinsic: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Infer forward/lateral/vertical axes for LiDAR native range images."""

    vertical_axis = 2
    if lidar_extrinsic is not None:
        matrix = np.asarray(lidar_extrinsic, dtype=np.float64).reshape(4, 4)
        # Columns are LiDAR local axes expressed in ego coordinates.
        vertical_axis = int(np.argmax(np.abs(matrix[:3, :3][2, :])))

    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
    values = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        forward_axis = horizontal_axes[0]
    else:
        finite = values[np.isfinite(values).all(axis=1)]
        if finite.size == 0:
            forward_axis = horizontal_axes[0]
        else:
            positive_extent = [
                float(np.percentile(finite[:, axis], 99.0)) for axis in horizontal_axes
            ]
            forward_axis = horizontal_axes[int(np.argmax(positive_extent))]
    lateral_axis = next(axis for axis in horizontal_axes if axis != forward_axis)
    return _validate_range_axes(forward_axis, lateral_axis, vertical_axis)


def _validate_range_axes(
    forward_axis: int,
    lateral_axis: int,
    vertical_axis: int,
) -> tuple[int, int, int]:
    axes = (int(forward_axis), int(lateral_axis), int(vertical_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError(
            "forward_axis, lateral_axis, and vertical_axis must be a permutation of 0, 1, 2"
        )
    return axes


def extract_lidar_range_edge_source_indices(
    projection: LidarRangeProjection,
    *,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
    include_silhouette: bool = True,
    edge_side: str = "near",
    min_component_pixels: int = 1,
    silhouette_ignore_lower_fraction: float = 0.0,
    require_depth_jump_support: bool = False,
) -> np.ndarray:
    edge_mask = extract_lidar_range_edge_mask(
        projection,
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
        include_silhouette=include_silhouette,
        edge_side=edge_side,
        min_component_pixels=min_component_pixels,
        silhouette_ignore_lower_fraction=silhouette_ignore_lower_fraction,
        require_depth_jump_support=require_depth_jump_support,
    )
    source_indices = projection.source_index_image[edge_mask]
    source_indices = source_indices[source_indices >= 0]
    return np.unique(source_indices).astype(np.int64)


def extract_lidar_range_edge_mask(
    projection: LidarRangeProjection,
    *,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
    include_silhouette: bool = True,
    edge_side: str = "near",
    min_component_pixels: int = 1,
    silhouette_ignore_lower_fraction: float = 0.0,
    require_depth_jump_support: bool = False,
) -> np.ndarray:
    edge_mask = extract_range_depth_jump_edges(
        projection.range_image,
        projection.valid_mask,
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
        edge_side=edge_side,
        require_same_depth_support=require_depth_jump_support,
    )
    if include_silhouette:
        silhouette_mask = extract_occupancy_boundary_edges(
            projection.valid_mask,
            fill_radius_px=1,
            boundary_width_px=1,
            return_radius_px=0,
            min_component_area_px=2,
        )
        silhouette_mask = suppress_lower_range_rows(
            silhouette_mask,
            lower_fraction=silhouette_ignore_lower_fraction,
        )
        edge_mask |= silhouette_mask
    return remove_small_edge_components(
        edge_mask,
        min_component_pixels=min_component_pixels,
    )


def suppress_lower_range_rows(
    mask: np.ndarray,
    *,
    lower_fraction: float,
) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    fraction = float(lower_fraction)
    if fraction <= 0.0 or output.size == 0:
        return output
    rows = output.shape[0]
    cutoff = int(np.ceil(rows * min(fraction, 1.0)))
    if cutoff > 0:
        output[:cutoff, :] = False
    return output


def remove_small_edge_components(
    edge_mask: np.ndarray,
    *,
    min_component_pixels: int = 1,
) -> np.ndarray:
    mask = np.asarray(edge_mask, dtype=bool)
    min_pixels = max(1, int(min_component_pixels))
    if min_pixels <= 1 or not np.any(mask):
        return mask.copy()
    return _remove_small_binary_components(mask, min_area_px=min_pixels)


def extract_range_depth_jump_edges(
    depth_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
    edge_side: str = "near",
    require_same_depth_support: bool = False,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    edges = np.zeros(depth.shape, dtype=bool)
    side = str(edge_side)
    if side not in {"near", "both", "far"}:
        raise ValueError("edge_side must be near, both, or far")

    _mark_range_depth_jump_pairs(
        edges,
        depth[:, :-1],
        depth[:, 1:],
        valid[:, :-1] & valid[:, 1:],
        first_support=_range_horizontal_first_support(
            depth,
            valid,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
        if require_same_depth_support
        else None,
        second_support=_range_horizontal_second_support(
            depth,
            valid,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
        if require_same_depth_support
        else None,
        left_slice=(slice(None), slice(None, -1)),
        right_slice=(slice(None), slice(1, None)),
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
        edge_side=side,
    )
    _mark_range_depth_jump_pairs(
        edges,
        depth[:-1, :],
        depth[1:, :],
        valid[:-1, :] & valid[1:, :],
        first_support=_range_vertical_first_support(
            depth,
            valid,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
        if require_same_depth_support
        else None,
        second_support=_range_vertical_second_support(
            depth,
            valid,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
        if require_same_depth_support
        else None,
        left_slice=(slice(None, -1), slice(None)),
        right_slice=(slice(1, None), slice(None)),
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
        edge_side=side,
    )
    return edges


def project_points_to_depth_map(
    *,
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    min_depth_m: float = 0.01,
    max_depth_m: float = 200.0,
) -> DepthProjection:
    height, width = int(image_shape[0]), int(image_shape[1])
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    values = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    source_indices = np.arange(len(values), dtype=np.int64)
    if values.size == 0:
        return DepthProjection(
            depth_map=depth_map,
            valid_mask=np.zeros_like(depth_map, dtype=bool),
            xs=np.empty((0,), dtype=np.int32),
            ys=np.empty((0,), dtype=np.int32),
            depths=np.empty((0,), dtype=np.float32),
            points_camera=values,
            source_indices=np.empty((0,), dtype=np.int64),
        )

    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    depth = values[:, 2]
    finite_depth = (
        np.isfinite(values).all(axis=1)
        & (depth > float(min_depth_m))
        & (depth < float(max_depth_m))
    )
    source_indices = source_indices[finite_depth]
    values = values[finite_depth]
    if values.size == 0:
        return DepthProjection(
            depth_map=depth_map,
            valid_mask=np.zeros_like(depth_map, dtype=bool),
            xs=np.empty((0,), dtype=np.int32),
            ys=np.empty((0,), dtype=np.int32),
            depths=np.empty((0,), dtype=np.float32),
            points_camera=np.empty((0, 3), dtype=np.float64),
            source_indices=np.empty((0,), dtype=np.int64),
        )

    depth = values[:, 2]
    projected = (intrinsic @ values.T).T
    uv = projected[:, :2] / np.clip(projected[:, 2:3], 1e-8, None)
    xs = np.rint(uv[:, 0]).astype(np.int32)
    ys = np.rint(uv[:, 1]).astype(np.int32)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (xs >= 0)
        & (xs < width)
        & (ys >= 0)
        & (ys < height)
    )
    if not np.any(valid):
        return DepthProjection(
            depth_map=depth_map,
            valid_mask=np.zeros_like(depth_map, dtype=bool),
            xs=np.empty((0,), dtype=np.int32),
            ys=np.empty((0,), dtype=np.int32),
            depths=np.empty((0,), dtype=np.float32),
            points_camera=np.empty((0, 3), dtype=np.float64),
            source_indices=np.empty((0,), dtype=np.int64),
        )

    xs_valid = xs[valid]
    ys_valid = ys[valid]
    depths_valid = depth[valid].astype(np.float32)
    points_valid = values[valid]
    source_valid = source_indices[valid]
    order = np.argsort(depths_valid)
    xs_valid = xs_valid[order]
    ys_valid = ys_valid[order]
    depths_valid = depths_valid[order]
    points_valid = points_valid[order]
    source_valid = source_valid[order]

    flat = ys_valid * width + xs_valid
    _, first_indices = np.unique(flat, return_index=True)
    first_indices.sort()
    xs_nearest = xs_valid[first_indices]
    ys_nearest = ys_valid[first_indices]
    depths_nearest = depths_valid[first_indices]
    points_nearest = points_valid[first_indices]
    source_nearest = source_valid[first_indices]
    depth_map[ys_nearest, xs_nearest] = depths_nearest
    return DepthProjection(
        depth_map=depth_map,
        valid_mask=np.isfinite(depth_map),
        xs=xs_nearest.astype(np.int32, copy=False),
        ys=ys_nearest.astype(np.int32, copy=False),
        depths=depths_nearest.astype(np.float32, copy=False),
        points_camera=points_nearest,
        source_indices=source_nearest.astype(np.int64, copy=False),
    )


def extract_depth_jump_edges(
    depth_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    edges = np.zeros(depth.shape, dtype=bool)

    _mark_depth_jump_pairs(
        edges,
        depth[:, :-1],
        depth[:, 1:],
        valid[:, :-1] & valid[:, 1:],
        left_slice=(slice(None), slice(None, -1)),
        right_slice=(slice(None), slice(1, None)),
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
    )
    _mark_depth_jump_pairs(
        edges,
        depth[:-1, :],
        depth[1:, :],
        valid[:-1, :] & valid[1:, :],
        left_slice=(slice(None, -1), slice(None)),
        right_slice=(slice(1, None), slice(None)),
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
    )
    return edges


def extract_sparse_depth_jump_edges(
    depth_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    radius_px: int = 5,
    min_valid_neighbors: int = 2,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
) -> np.ndarray:
    """Find sparse LiDAR depth discontinuities without requiring adjacent pixels."""

    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)

    radius = max(1, int(radius_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * radius + 1, 2 * radius + 1),
    )
    finite_depth = np.where(valid, depth, 0.0).astype(np.float32)
    local_max = cv2.dilate(finite_depth, kernel)
    local_min_source = np.where(valid, depth, np.finfo(np.float32).max).astype(np.float32)
    local_min = -cv2.dilate(-local_min_source, kernel)
    support = cv2.filter2D(
        valid.astype(np.uint8),
        ddepth=cv2.CV_16U,
        kernel=np.ones(kernel.shape, dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )

    local_range = local_max - local_min
    threshold = np.maximum(
        float(absolute_threshold_m),
        np.maximum(local_min, 1e-6) * float(relative_threshold),
    )
    return valid & (support >= int(min_valid_neighbors)) & (local_range > threshold)


def extract_projected_lidar_depth_edge_source_indices(
    *,
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    min_depth_m: float = 0.01,
    max_depth_m: float = 200.0,
    search_radius_px: int = 2,
    require_same_depth_support: bool = True,
    absolute_threshold_m: float = 0.8,
    relative_threshold: float = 0.08,
    edge_side: str = "near",
) -> np.ndarray:
    """Extract LiDAR depth boundaries in the target camera image view.

    This operates on projected, z-buffered LiDAR points, so it does not depend
    on LiDAR ring count, azimuth resolution, or native range-image layout.
    """

    projection = project_points_to_depth_map(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=image_shape,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if len(projection.xs) < 2:
        return np.empty((0,), dtype=np.int64)

    side = str(edge_side)
    if side not in {"near", "both", "far"}:
        raise ValueError("edge_side must be near, both, or far")

    edge_visible = np.zeros(len(projection.xs), dtype=bool)
    depths = projection.depths.astype(np.float32, copy=False)
    visible_index_image = np.full(projection.depth_map.shape, -1, dtype=np.int64)
    visible_index_image[projection.ys, projection.xs] = np.arange(
        len(projection.xs),
        dtype=np.int64,
    )
    visible_xs = projection.xs.astype(np.int32, copy=False)
    visible_ys = projection.ys.astype(np.int32, copy=False)
    directions = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    max_radius = max(1, int(search_radius_px))
    height, width = projection.depth_map.shape
    for i, (x, y) in enumerate(zip(projection.xs, projection.ys, strict=False)):
        for dx, dy in directions:
            for radius in range(1, max_radius + 1):
                nx = int(x) + dx * radius
                ny = int(y) + dy * radius
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    break
                j = int(visible_index_image[ny, nx])
                if j < 0:
                    continue
                if j != i:
                    _mark_projected_depth_edge_pair(
                        edge_visible,
                        depths,
                        visible_index_image,
                        visible_xs,
                        visible_ys,
                        i,
                        j,
                        direction=(dx, dy),
                        search_radius_px=max_radius,
                        require_same_depth_support=require_same_depth_support,
                        absolute_threshold_m=absolute_threshold_m,
                        relative_threshold=relative_threshold,
                        edge_side=side,
                    )
                break

    return np.unique(projection.source_indices[edge_visible]).astype(np.int64)


def _mark_projected_depth_edge_pair(
    edge_visible: np.ndarray,
    depths: np.ndarray,
    visible_index_image: np.ndarray,
    visible_xs: np.ndarray,
    visible_ys: np.ndarray,
    first_index: int,
    second_index: int,
    *,
    direction: tuple[int, int],
    search_radius_px: int,
    require_same_depth_support: bool,
    absolute_threshold_m: float,
    relative_threshold: float,
    edge_side: str,
) -> None:
    near_depth = min(float(depths[first_index]), float(depths[second_index]))
    threshold = max(
        float(absolute_threshold_m),
        max(near_depth, 1e-6) * float(relative_threshold),
    )
    if abs(float(depths[first_index]) - float(depths[second_index])) <= threshold:
        return
    if not bool(require_same_depth_support):
        if edge_side == "both":
            edge_visible[first_index] = True
            edge_visible[second_index] = True
        elif (depths[first_index] <= depths[second_index]) == (edge_side == "near"):
            edge_visible[first_index] = True
        else:
            edge_visible[second_index] = True
        return
    if edge_side == "both":
        if _has_similar_depth_support(
            visible_index_image,
            visible_xs,
            visible_ys,
            depths,
            first_index,
            support_direction=(-int(direction[0]), -int(direction[1])),
            search_radius_px=search_radius_px,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        ):
            edge_visible[first_index] = True
        if _has_similar_depth_support(
            visible_index_image,
            visible_xs,
            visible_ys,
            depths,
            second_index,
            support_direction=(int(direction[0]), int(direction[1])),
            search_radius_px=search_radius_px,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        ):
            edge_visible[second_index] = True
    elif (depths[first_index] <= depths[second_index]) == (edge_side == "near"):
        if _has_similar_depth_support(
            visible_index_image,
            visible_xs,
            visible_ys,
            depths,
            first_index,
            support_direction=(-int(direction[0]), -int(direction[1])),
            search_radius_px=search_radius_px,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        ):
            edge_visible[first_index] = True
    else:
        if _has_similar_depth_support(
            visible_index_image,
            visible_xs,
            visible_ys,
            depths,
            second_index,
            support_direction=(int(direction[0]), int(direction[1])),
            search_radius_px=search_radius_px,
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        ):
            edge_visible[second_index] = True


def _has_similar_depth_support(
    visible_index_image: np.ndarray,
    visible_xs: np.ndarray,
    visible_ys: np.ndarray,
    depths: np.ndarray,
    index: int,
    *,
    support_direction: tuple[int, int],
    search_radius_px: int,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> bool:
    x = int(visible_xs[index])
    y = int(visible_ys[index])
    dx, dy = int(support_direction[0]), int(support_direction[1])
    if dx == 0 and dy == 0:
        return False
    height, width = visible_index_image.shape
    depth = float(depths[index])
    threshold = max(
        float(absolute_threshold_m),
        max(depth, 1e-6) * float(relative_threshold),
    )
    for radius in range(1, max(1, int(search_radius_px)) + 1):
        nx = x + dx * radius
        ny = y + dy * radius
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            return False
        support_index = int(visible_index_image[ny, nx])
        if support_index < 0:
            continue
        return abs(float(depths[support_index]) - depth) <= threshold
    return False


def extract_occupancy_boundary_edges(
    valid_mask: np.ndarray,
    *,
    fill_radius_px: int = 3,
    boundary_width_px: int = 1,
    return_radius_px: int = 1,
    min_component_area_px: int = 10,
) -> np.ndarray:
    """Find silhouettes of sparse projected LiDAR occupancy regions."""

    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return np.zeros(valid.shape, dtype=bool)

    fill_radius = max(1, int(fill_radius_px))
    fill_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * fill_radius + 1, 2 * fill_radius + 1),
    )
    occupancy = cv2.dilate(valid.astype(np.uint8), fill_kernel) > 0
    occupancy = _remove_small_binary_components(
        occupancy,
        min_area_px=max(1, int(min_component_area_px)),
    )
    if not np.any(occupancy):
        return np.zeros(valid.shape, dtype=bool)

    boundary_width = max(1, int(boundary_width_px))
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * boundary_width + 1, 2 * boundary_width + 1),
    )
    eroded = cv2.erode(occupancy.astype(np.uint8), boundary_kernel) > 0
    boundary_band = occupancy & ~eroded

    return_radius = max(0, int(return_radius_px)) + fill_radius
    if return_radius > 0:
        return_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * return_radius + 1, 2 * return_radius + 1),
        )
        boundary_band = cv2.dilate(boundary_band.astype(np.uint8), return_kernel) > 0
    return valid & boundary_band


def _remove_small_binary_components(mask: np.ndarray, *, min_area_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area_px)
    return keep[labels]


def _mark_depth_jump_pairs(
    edges: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    valid_pair: np.ndarray,
    *,
    left_slice: tuple[slice, slice],
    right_slice: tuple[slice, slice],
    absolute_threshold_m: float,
    relative_threshold: float,
) -> None:
    local_near = np.minimum(first_depth, second_depth)
    threshold = np.maximum(
        float(absolute_threshold_m),
        np.maximum(local_near, 1e-6) * float(relative_threshold),
    )
    diff = np.zeros_like(first_depth, dtype=np.float32)
    np.subtract(first_depth, second_depth, out=diff, where=valid_pair)
    jump = valid_pair & (np.abs(diff) > threshold)
    edges[left_slice] |= jump
    edges[right_slice] |= jump


def _mark_range_depth_jump_pairs(
    edges: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    valid_pair: np.ndarray,
    *,
    first_support: np.ndarray | None = None,
    second_support: np.ndarray | None = None,
    left_slice: tuple[slice, slice],
    right_slice: tuple[slice, slice],
    absolute_threshold_m: float,
    relative_threshold: float,
    edge_side: str,
) -> None:
    local_near = np.minimum(first_depth, second_depth)
    threshold = np.maximum(
        float(absolute_threshold_m),
        np.maximum(local_near, 1e-6) * float(relative_threshold),
    )
    diff = np.zeros_like(first_depth, dtype=np.float32)
    np.subtract(first_depth, second_depth, out=diff, where=valid_pair)
    jump = valid_pair & (np.abs(diff) > threshold)
    first_ok = np.ones_like(jump, dtype=bool) if first_support is None else first_support
    second_ok = np.ones_like(jump, dtype=bool) if second_support is None else second_support
    if edge_side == "both":
        edges[left_slice] |= jump & first_ok
        edges[right_slice] |= jump & second_ok
        return
    first_is_near = first_depth <= second_depth
    if edge_side == "near":
        edges[left_slice] |= jump & first_is_near & first_ok
        edges[right_slice] |= jump & ~first_is_near & second_ok
        return
    edges[left_slice] |= jump & ~first_is_near & first_ok
    edges[right_slice] |= jump & first_is_near & second_ok


def _range_horizontal_first_support(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> np.ndarray:
    support = np.zeros((depth.shape[0], max(depth.shape[1] - 1, 0)), dtype=bool)
    if depth.shape[1] > 2:
        support[:, 1:] = _similar_depth_values(
            depth[:, 1:-1],
            depth[:, :-2],
            valid[:, 1:-1] & valid[:, :-2],
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
    return support


def _range_horizontal_second_support(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> np.ndarray:
    support = np.zeros((depth.shape[0], max(depth.shape[1] - 1, 0)), dtype=bool)
    if depth.shape[1] > 2:
        support[:, :-1] = _similar_depth_values(
            depth[:, 1:-1],
            depth[:, 2:],
            valid[:, 1:-1] & valid[:, 2:],
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
    return support


def _range_vertical_first_support(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> np.ndarray:
    support = np.zeros((max(depth.shape[0] - 1, 0), depth.shape[1]), dtype=bool)
    if depth.shape[0] > 2:
        support[1:, :] = _similar_depth_values(
            depth[1:-1, :],
            depth[:-2, :],
            valid[1:-1, :] & valid[:-2, :],
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
    return support


def _range_vertical_second_support(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> np.ndarray:
    support = np.zeros((max(depth.shape[0] - 1, 0), depth.shape[1]), dtype=bool)
    if depth.shape[0] > 2:
        support[:-1, :] = _similar_depth_values(
            depth[1:-1, :],
            depth[2:, :],
            valid[1:-1, :] & valid[2:, :],
            absolute_threshold_m=absolute_threshold_m,
            relative_threshold=relative_threshold,
        )
    return support


def _similar_depth_values(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    valid_pair: np.ndarray,
    *,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> np.ndarray:
    local_near = np.minimum(first_depth, second_depth)
    threshold = np.maximum(
        float(absolute_threshold_m),
        np.maximum(local_near, 1e-6) * float(relative_threshold),
    )
    diff = np.zeros_like(first_depth, dtype=np.float32)
    np.subtract(first_depth, second_depth, out=diff, where=valid_pair)
    return valid_pair & (np.abs(diff) <= threshold)


def extract_image_edges(
    image: np.ndarray,
    *,
    canny_low: int = 80,
    canny_high: int = 160,
    blur_kernel: int = 5,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else np.asarray(image)
    kernel = max(1, int(blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.Canny(gray, int(canny_low), int(canny_high)) > 0


def distance_transform_from_edges(edges: np.ndarray) -> np.ndarray:
    edge_mask = np.asarray(edges, dtype=bool)
    if not np.any(edge_mask):
        return np.full(edge_mask.shape, np.inf, dtype=np.float32)
    inverse = np.where(edge_mask, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def summarize_edge_alignment(
    *,
    lidar_edges: np.ndarray,
    image_edge_distance: np.ndarray,
    valid_mask: np.ndarray | None = None,
    max_distance_px: float = 20.0,
) -> dict[str, float | int]:
    lidar_mask = np.asarray(lidar_edges, dtype=bool)
    if valid_mask is not None:
        lidar_mask &= np.asarray(valid_mask, dtype=bool)
    distances = np.asarray(image_edge_distance, dtype=np.float32)[lidar_mask]
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return {
            "lidar_edge_pixels": 0,
            "mean_distance_px": float("nan"),
            "median_distance_px": float("nan"),
            "p75_distance_px": float("nan"),
            "inlier_fraction": float("nan"),
        }
    return {
        "lidar_edge_pixels": int(distances.size),
        "mean_distance_px": float(np.mean(distances)),
        "median_distance_px": float(np.median(distances)),
        "p75_distance_px": float(np.percentile(distances, 75)),
        "inlier_fraction": float(np.mean(distances <= float(max_distance_px))),
    }


def draw_edge_overlay(
    image: np.ndarray,
    *,
    image_edges: np.ndarray,
    lidar_edges: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    output = image.copy()
    image_mask = np.asarray(image_edges, dtype=bool)
    lidar_mask = np.asarray(lidar_edges, dtype=bool)
    overlay = output.copy()
    overlay[image_mask] = (0, 255, 0)
    overlay[lidar_mask] = (0, 0, 255)
    both = image_mask & lidar_mask
    overlay[both] = (0, 255, 255)
    return cv2.addWeighted(overlay, float(alpha), output, 1.0 - float(alpha), 0.0)


def draw_edge_mask_visual(
    *,
    image_edges: np.ndarray,
    lidar_edges: np.ndarray,
) -> np.ndarray:
    image_mask = np.asarray(image_edges, dtype=bool)
    lidar_mask = np.asarray(lidar_edges, dtype=bool)
    output = np.zeros((*image_mask.shape, 3), dtype=np.uint8)
    output[image_mask] = (0, 255, 0)
    output[lidar_mask] = (0, 0, 255)
    output[image_mask & lidar_mask] = (0, 255, 255)
    return output


def depth_map_to_visual(
    depth_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    visual_radius: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output
    values = depth[valid]
    near = float(np.percentile(values, 1))
    far = float(np.percentile(values, 99))
    if far <= near:
        far = near + 1.0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip((depth[valid] - near) / (far - near) * 255.0, 0, 255).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
    output[valid] = colored[valid]
    radius = max(0, int(visual_radius))
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        output = cv2.dilate(output, kernel)
    return output


def range_image_to_visual(
    range_image: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Colorize a native LiDAR range image without changing its pixel/bin layout."""

    return depth_map_to_visual(range_image, valid_mask, visual_radius=0)


def roll_range_display_180(
    range_image: np.ndarray,
    valid_mask: np.ndarray,
    edge_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Circularly shift range-image columns by 180 degrees for display."""

    cols = int(np.asarray(range_image).shape[1])
    shift = cols // 2
    rolled_range = np.roll(np.asarray(range_image), shift=shift, axis=1)
    rolled_valid = np.roll(np.asarray(valid_mask, dtype=bool), shift=shift, axis=1)
    rolled_edge = (
        None
        if edge_mask is None
        else np.roll(np.asarray(edge_mask, dtype=bool), shift=shift, axis=1)
    )
    return rolled_range, rolled_valid, rolled_edge


def crop_range_display_to_valid_region(
    range_image: np.ndarray,
    valid_mask: np.ndarray,
    edge_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop a range-image display to the bounding box containing valid bins."""

    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        empty_range = np.asarray(range_image)[:0, :0].copy()
        empty_valid = valid[:0, :0].copy()
        empty_edge = None if edge_mask is None else np.asarray(edge_mask, dtype=bool)[:0, :0].copy()
        return empty_range, empty_valid, empty_edge

    rows = np.flatnonzero(np.any(valid, axis=1))
    cols = np.flatnonzero(np.any(valid, axis=0))
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    col_slice = slice(int(cols[0]), int(cols[-1]) + 1)
    cropped_range = np.asarray(range_image)[row_slice, col_slice].copy()
    cropped_valid = valid[row_slice, col_slice].copy()
    cropped_edge = (
        None
        if edge_mask is None
        else np.asarray(edge_mask, dtype=bool)[row_slice, col_slice].copy()
    )
    return cropped_range, cropped_valid, cropped_edge


def rotate_range_display_180(
    range_image: np.ndarray,
    valid_mask: np.ndarray,
    edge_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Rotate a range-image display by 180 degrees, keeping masks aligned."""

    rotated_range = np.rot90(np.asarray(range_image), 2).copy()
    rotated_valid = np.rot90(np.asarray(valid_mask, dtype=bool), 2).copy()
    rotated_edge = (
        None
        if edge_mask is None
        else np.rot90(np.asarray(edge_mask, dtype=bool), 2).copy()
    )
    return rotated_range, rotated_valid, rotated_edge


def binary_mask_to_visual(
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    output = np.zeros((*binary.shape, 3), dtype=np.uint8)
    output[binary] = np.asarray(color, dtype=np.uint8)
    return output


def draw_binary_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.75,
) -> np.ndarray:
    output = np.asarray(image).copy()
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return output
    overlay = output.copy()
    overlay[binary] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, float(alpha), output, 1.0 - float(alpha), 0.0)


def draw_depth_projection_overlay(
    image: np.ndarray,
    projection: DepthProjection,
    *,
    point_radius: int = 1,
) -> np.ndarray:
    output = np.asarray(image).copy()
    if len(projection.xs) == 0:
        return output

    depths = projection.depths.astype(np.float32, copy=False)
    near = float(np.percentile(depths, 1))
    far = float(np.percentile(depths, 99))
    if far <= near:
        far = near + 1.0
    normalized = np.clip((depths - near) / (far - near), 0.0, 1.0)
    colors = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_PLASMA,
    ).reshape(-1, 3)

    radius = max(1, int(point_radius))
    for x, y, color in zip(projection.xs, projection.ys, colors, strict=False):
        cv2.circle(output, (int(x), int(y)), radius, color.tolist(), -1)
    return output


def make_lidar_diagnostic_grid(
    *,
    original_image: np.ndarray,
    native_range_image: np.ndarray,
    projected_lidar_image: np.ndarray,
    image_edge_overlay: np.ndarray,
    native_range_edges: np.ndarray,
    projected_edge_overlay: np.ndarray,
    preview_scale: float = 1.0,
) -> np.ndarray:
    """Build the requested 2x3 LiDAR-camera diagnostic layout."""

    cell_shape = np.asarray(original_image).shape[:2]
    top = np.concatenate(
        [
            _resize_panel_to_cell(original_image, cell_shape),
            _resize_panel_to_cell(native_range_image, cell_shape, preserve_aspect=True),
            _resize_panel_to_cell(projected_lidar_image, cell_shape),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            _resize_panel_to_cell(image_edge_overlay, cell_shape),
            _resize_panel_to_cell(native_range_edges, cell_shape, preserve_aspect=True),
            _resize_panel_to_cell(projected_edge_overlay, cell_shape),
        ],
        axis=1,
    )
    grid = np.concatenate([top, bottom], axis=0)
    if float(preview_scale) != 1.0:
        scale = max(0.05, float(preview_scale))
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grid


def _resize_panel_to_cell(
    panel: np.ndarray,
    cell_shape: tuple[int, int],
    *,
    preserve_aspect: bool = False,
) -> np.ndarray:
    height, width = int(cell_shape[0]), int(cell_shape[1])
    image = np.asarray(panel)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.shape[:2] == (height, width):
        return image.copy()
    if preserve_aspect:
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        scale = min(width / src_w, height / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_NEAREST,
        )
        output = np.zeros((height, width, 3), dtype=resized.dtype)
        y0 = (height - resized_h) // 2
        x0 = (width - resized_w) // 2
        output[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
        return output
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
