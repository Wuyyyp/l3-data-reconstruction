import numpy as np
import warnings

from dongfeng_registration.lidar_edges import (
    binary_mask_to_visual,
    crop_range_display_to_valid_region,
    depth_map_to_visual,
    distance_transform_from_edges,
    draw_binary_mask_overlay,
    draw_edge_mask_visual,
    extract_depth_jump_edges,
    extract_lidar_range_edge_mask,
    extract_lidar_range_edge_source_indices,
    extract_occupancy_boundary_edges,
    extract_projected_lidar_depth_edge_source_indices,
    extract_range_depth_jump_edges,
    extract_sparse_depth_jump_edges,
    filter_points_by_range_percentile_band,
    filter_points_by_range_percentile,
    filter_points_by_range,
    infer_lidar_range_axes,
    make_lidar_diagnostic_grid,
    project_points_to_lidar_range_image,
    project_points_to_depth_map,
    range_image_to_visual,
    rotate_range_display_180,
    roll_range_display_180,
    select_source_indices_by_range_percentile_band,
    remove_small_edge_components,
    suppress_lower_range_rows,
)


def test_project_points_to_depth_map_keeps_nearest_depth_per_pixel():
    points = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 4.0],
            [0.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [10.0, 0.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = project_points_to_depth_map(
        points_camera=points,
        intrinsic=intrinsic,
        image_shape=(12, 12),
    )

    assert result.depth_map[5, 5] == 3.0
    assert result.depth_map[5, 8] == 4.0
    assert result.source_indices[result.depths == 3.0][0] == 1
    assert result.valid_mask[5, 5]
    assert result.valid_mask[5, 8]


def test_filter_points_by_range_excludes_far_lidar_returns():
    points = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, 30.0],
            [np.nan, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    filtered, keep = filter_points_by_range(points, max_range_m=20.0)

    assert keep.tolist() == [True, True, False, False]
    assert filtered.shape == (2, 3)
    assert np.allclose(filtered[:, 2], [0.0, 12.0])


def test_filter_points_by_range_percentile_uses_p95_threshold():
    points = np.array([[float(i), 0.0, 0.0] for i in range(1, 101)], dtype=np.float64)

    filtered, keep, threshold = filter_points_by_range_percentile(points, percentile=95.0)

    assert threshold == 95.05
    assert keep.sum() == 95
    assert filtered[-1, 0] == 95.0


def test_filter_points_by_range_percentile_band_excludes_near_and_far_returns():
    points = np.array([[float(i), 0.0, 0.0] for i in range(1, 101)], dtype=np.float64)

    filtered, keep, low, high = filter_points_by_range_percentile_band(
        points,
        min_percentile=5.0,
        max_percentile=80.0,
    )

    assert low == 5.95
    assert high == 80.2
    assert keep.sum() == 75
    assert filtered[0, 0] == 6.0
    assert filtered[-1, 0] == 80.0


def test_select_source_indices_by_range_percentile_band_filters_after_edge_extraction():
    points = np.array([[float(i), 0.0, 0.0] for i in range(1, 101)], dtype=np.float64)
    source_indices = np.array([0, 4, 5, 49, 79, 80, 99], dtype=np.int64)

    selected, low, high = select_source_indices_by_range_percentile_band(
        source_indices,
        points,
        min_percentile=5.0,
        max_percentile=80.0,
    )

    assert low == 5.95
    assert high == 80.2
    assert selected.tolist() == [5, 49, 79]


def test_extract_depth_jump_edges_flags_valid_neighbor_discontinuity():
    depth_map = np.full((5, 6), np.inf, dtype=np.float32)
    depth_map[:, :3] = 5.0
    depth_map[:, 3:] = 9.0
    valid = np.isfinite(depth_map)

    edges = extract_depth_jump_edges(
        depth_map,
        valid,
        absolute_threshold_m=1.0,
        relative_threshold=0.1,
    )

    assert edges[:, 2].all()
    assert edges[:, 3].all()
    assert not edges[:, 0].any()
    assert not edges[:, -1].any()


def test_extract_sparse_depth_jump_edges_detects_discontinuity_across_projection_gaps():
    depth_map = np.full((9, 11), np.inf, dtype=np.float32)
    depth_map[4, 2] = 6.0
    depth_map[4, 8] = 13.0
    valid = np.isfinite(depth_map)

    strict_edges = extract_depth_jump_edges(
        depth_map,
        valid,
        absolute_threshold_m=1.0,
        relative_threshold=0.1,
    )
    sparse_edges = extract_sparse_depth_jump_edges(
        depth_map,
        valid,
        radius_px=6,
        absolute_threshold_m=1.0,
        relative_threshold=0.1,
    )

    assert not strict_edges.any()
    assert sparse_edges[4, 2]
    assert sparse_edges[4, 8]


def test_extract_sparse_depth_jump_edges_respects_local_support_count():
    depth_map = np.full((7, 7), np.inf, dtype=np.float32)
    depth_map[3, 3] = 8.0
    valid = np.isfinite(depth_map)

    sparse_edges = extract_sparse_depth_jump_edges(
        depth_map,
        valid,
        radius_px=3,
        min_valid_neighbors=2,
        absolute_threshold_m=1.0,
        relative_threshold=0.1,
    )

    assert not sparse_edges.any()


def test_extract_projected_lidar_depth_edge_source_indices_marks_near_side_only():
    points_camera = np.array(
        [
            [-1.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 12.0],
            [2.0, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [10.0, 0.0, 10.0],
            [0.0, 10.0, 10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    indices = extract_projected_lidar_depth_edge_source_indices(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=(24, 24),
        search_radius_px=2,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
    )

    assert indices.tolist() == [1]


def test_extract_projected_lidar_depth_edge_source_indices_uses_image_view_neighbors():
    points_camera = np.array(
        [
            [-0.1, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [20.0, 0.0, 5.0],
            [0.2, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [100.0, 0.0, 20.0],
            [0.0, 100.0, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    indices = extract_projected_lidar_depth_edge_source_indices(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=(40, 500),
        search_radius_px=2,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
    )

    assert indices.tolist() == [1]


def test_extract_projected_lidar_depth_edge_source_indices_does_not_knn_across_non_directional_gap():
    points_camera = np.array(
        [
            [50.0, 50.0, 5.0],
            [144.0, 132.0, 12.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.eye(3, dtype=np.float64)

    indices = extract_projected_lidar_depth_edge_source_indices(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=(30, 30),
        search_radius_px=2,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
    )

    assert indices.tolist() == []


def test_extract_projected_lidar_depth_edge_source_indices_suppresses_smooth_depth_ramp():
    points_camera = np.array(
        [
            [-0.1, 0.0, 5.0],
            [0.0, 0.0, 8.0],
            [0.1, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [40.0, 0.0, 10.0],
            [0.0, 40.0, 10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    indices = extract_projected_lidar_depth_edge_source_indices(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=(24, 24),
        search_radius_px=2,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
    )

    assert indices.tolist() == []


def test_extract_projected_lidar_depth_edge_source_indices_can_disable_same_depth_support():
    points_camera = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.2, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [100.0, 0.0, 20.0],
            [0.0, 100.0, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    indices = extract_projected_lidar_depth_edge_source_indices(
        points_camera=points_camera,
        intrinsic=intrinsic,
        image_shape=(40, 80),
        search_radius_px=2,
        require_same_depth_support=False,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
    )

    assert indices.tolist() == [0]


def test_extract_occupancy_boundary_edges_marks_sparse_object_silhouette_not_interior():
    valid = np.zeros((25, 27), dtype=bool)
    valid[5:20:2, 5:22:2] = True

    edges = extract_occupancy_boundary_edges(
        valid,
        fill_radius_px=2,
        boundary_width_px=1,
        return_radius_px=1,
    )

    assert edges[5, 5]
    assert edges[5, 21]
    assert edges[19, 5]
    assert edges[19, 21]
    assert not edges[11, 13]
    assert not edges[13, 15]


def test_extract_occupancy_boundary_edges_does_not_mark_isolated_single_returns():
    valid = np.zeros((9, 9), dtype=bool)
    valid[4, 4] = True

    edges = extract_occupancy_boundary_edges(
        valid,
        fill_radius_px=2,
        boundary_width_px=1,
        return_radius_px=1,
        min_component_area_px=20,
    )

    assert not edges.any()


def test_project_points_to_lidar_range_image_keeps_nearest_point_per_native_bin():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ],
        dtype=np.float64,
    )

    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-10.0, 10.0),
    )

    assert projection.range_image[2, 0] == 5.0
    assert projection.source_index_image[2, 0] == 1
    assert projection.range_image[2, 1] == 8.0
    assert projection.source_index_image[2, 1] == 2


def test_project_points_to_lidar_range_image_supports_non_z_vertical_axis():
    points = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 8.0, 0.0],
            [2.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )

    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-30.0, 30.0),
        forward_axis=2,
        lateral_axis=1,
        vertical_axis=0,
    )

    assert projection.range_image[2, 0] == 10.0
    assert projection.source_index_image[2, 0] == 0
    assert projection.range_image[2, 1] == 8.0
    assert projection.source_index_image[2, 1] == 1


def test_infer_lidar_range_axes_detects_side_lidar_layout():
    left_lidar_rotation = np.array(
        [
            [0.001, -0.007, -1.0],
            [0.030, 1.000, -0.007],
            [1.000, -0.030, 0.001],
        ],
        dtype=np.float64,
    )
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = left_lidar_rotation
    points = np.array(
        [
            [-0.5, -2.0, 8.0],
            [-0.8, 1.0, 15.0],
            [0.2, 3.0, 20.0],
        ],
        dtype=np.float64,
    )

    axes = infer_lidar_range_axes(points, lidar_extrinsic=extrinsic)

    assert axes == (2, 1, 0)


def test_extract_lidar_range_edge_source_indices_detects_native_depth_jump():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-10.0, 10.0),
    )

    source_indices = extract_lidar_range_edge_source_indices(
        projection,
        absolute_threshold_m=5.0,
        relative_threshold=0.1,
        include_silhouette=False,
        edge_side="both",
    )

    assert source_indices.tolist() == [0, 1]


def test_extract_lidar_range_edge_mask_marks_native_depth_jump_bins():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-10.0, 10.0),
    )

    edge_mask = extract_lidar_range_edge_mask(
        projection,
        absolute_threshold_m=5.0,
        relative_threshold=0.1,
        include_silhouette=False,
        edge_side="both",
    )

    assert edge_mask[2, 0]
    assert edge_mask[2, 1]
    assert not edge_mask[2, 2]


def test_extract_lidar_range_edge_mask_near_side_marks_only_nearer_jump_bin():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-10.0, 10.0),
    )

    edge_mask = extract_lidar_range_edge_mask(
        projection,
        absolute_threshold_m=5.0,
        relative_threshold=0.1,
        include_silhouette=False,
        edge_side="near",
    )

    assert edge_mask[2, 0]
    assert not edge_mask[2, 1]
    assert not edge_mask[2, 2]


def test_extract_range_depth_jump_edges_with_support_keeps_supported_foreground_side():
    depth_map = np.array([[5.0, 5.0, 12.0]], dtype=np.float32)
    valid = np.isfinite(depth_map)

    edges = extract_range_depth_jump_edges(
        depth_map,
        valid,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
        require_same_depth_support=True,
    )

    assert edges.tolist() == [[False, True, False]]


def test_extract_range_depth_jump_edges_with_support_suppresses_smooth_ramp():
    depth_map = np.array([[5.0, 8.0, 12.0]], dtype=np.float32)
    valid = np.isfinite(depth_map)

    edges = extract_range_depth_jump_edges(
        depth_map,
        valid,
        absolute_threshold_m=2.0,
        relative_threshold=0.1,
        edge_side="near",
        require_same_depth_support=True,
    )

    assert not edges.any()


def test_extract_lidar_range_edge_mask_far_side_marks_only_farther_jump_bin():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    projection = project_points_to_lidar_range_image(
        points,
        num_rows=4,
        azimuth_resolution_deg=90.0,
        vertical_fov_deg=(-10.0, 10.0),
    )

    edge_mask = extract_lidar_range_edge_mask(
        projection,
        absolute_threshold_m=5.0,
        relative_threshold=0.1,
        include_silhouette=False,
        edge_side="far",
    )

    assert not edge_mask[2, 0]
    assert edge_mask[2, 1]
    assert not edge_mask[2, 2]


def test_distance_transform_from_edges_is_zero_on_edges_and_grows_away():
    edges = np.zeros((7, 7), dtype=bool)
    edges[3, 3] = True

    distance = distance_transform_from_edges(edges)

    assert distance[3, 3] == 0.0
    assert distance[3, 4] < distance[0, 0]
    assert distance[3, 4] > 0.0


def test_depth_map_to_visual_can_dilate_sparse_depth_pixels_for_display_only():
    depth_map = np.full((7, 7), np.inf, dtype=np.float32)
    depth_map[3, 3] = 12.0

    point_visual = depth_map_to_visual(depth_map, visual_radius=0)
    dilated_visual = depth_map_to_visual(depth_map, visual_radius=1)

    assert point_visual[3, 3].sum() > 0
    assert point_visual[3, 4].sum() == 0
    assert dilated_visual[3, 4].sum() > 0


def test_range_image_to_visual_keeps_one_native_bin_as_one_pixel():
    range_image = np.full((4, 5), np.inf, dtype=np.float32)
    range_image[1, 2] = 8.0
    range_image[2, 4] = 16.0

    visual = range_image_to_visual(range_image)

    non_black = np.any(visual > 0, axis=2)
    assert non_black.sum() == 2
    assert non_black[1, 2]
    assert non_black[2, 4]
    assert not non_black[1, 3]


def test_roll_range_display_180_moves_azimuth_seam_to_image_center():
    range_image = np.full((2, 8), np.inf, dtype=np.float32)
    valid = np.zeros((2, 8), dtype=bool)
    edge = np.zeros((2, 8), dtype=bool)
    range_image[0, 0] = 10.0
    range_image[0, 7] = 11.0
    valid[0, 0] = True
    valid[0, 7] = True
    edge[0, 7] = True

    rolled_range, rolled_valid, rolled_edge = roll_range_display_180(
        range_image,
        valid,
        edge,
    )

    assert rolled_valid[0, 3]
    assert rolled_valid[0, 4]
    assert rolled_range[0, 4] == 10.0
    assert rolled_range[0, 3] == 11.0
    assert rolled_edge[0, 3]


def test_crop_range_display_to_valid_region_keeps_only_valid_bbox():
    range_image = np.full((4, 8), np.inf, dtype=np.float32)
    valid = np.zeros((4, 8), dtype=bool)
    edge = np.zeros((4, 8), dtype=bool)
    range_image[1, 3] = 5.0
    range_image[2, 5] = 6.0
    valid[1, 3] = True
    valid[2, 5] = True
    edge[2, 5] = True

    cropped_range, cropped_valid, cropped_edge = crop_range_display_to_valid_region(
        range_image,
        valid,
        edge,
    )

    assert cropped_range.shape == (2, 3)
    assert cropped_valid.tolist() == [
        [True, False, False],
        [False, False, True],
    ]
    assert cropped_edge.tolist() == [
        [False, False, False],
        [False, False, True],
    ]


def test_rotate_range_display_180_rotates_range_valid_and_edges_together():
    range_image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [True, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    edge = np.array(
        [
            [False, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )

    rotated_range, rotated_valid, rotated_edge = rotate_range_display_180(
        range_image,
        valid,
        edge,
    )

    assert rotated_range.tolist() == [
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0],
    ]
    assert rotated_valid.tolist() == [
        [True, False, False],
        [False, False, True],
    ]
    assert rotated_edge.tolist() == [
        [True, False, False],
        [False, True, False],
    ]


def test_draw_binary_mask_overlay_marks_only_requested_mask_pixels():
    base = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = True

    overlay = draw_binary_mask_overlay(base, mask, color=(0, 0, 255), alpha=1.0)

    assert overlay[1, 2].tolist() == [0, 0, 255]
    assert overlay[1, 3].tolist() == [0, 0, 0]


def test_draw_edge_mask_visual_uses_black_background_without_original_image():
    image_edges = np.zeros((3, 4), dtype=bool)
    lidar_edges = np.zeros((3, 4), dtype=bool)
    image_edges[0, 1] = True
    lidar_edges[1, 2] = True
    image_edges[2, 3] = True
    lidar_edges[2, 3] = True

    visual = draw_edge_mask_visual(image_edges=image_edges, lidar_edges=lidar_edges)

    assert visual[0, 1].tolist() == [0, 255, 0]
    assert visual[1, 2].tolist() == [0, 0, 255]
    assert visual[2, 3].tolist() == [0, 255, 255]
    assert visual[0, 0].tolist() == [0, 0, 0]


def test_remove_small_edge_components_drops_isolated_edge_pixel():
    edge_mask = np.zeros((5, 6), dtype=bool)
    edge_mask[1, 1] = True
    edge_mask[3, 3] = True
    edge_mask[3, 4] = True

    filtered = remove_small_edge_components(edge_mask, min_component_pixels=2)

    assert not filtered[1, 1]
    assert filtered[3, 3]
    assert filtered[3, 4]


def test_suppress_lower_range_rows_removes_lower_fraction_only():
    mask = np.ones((10, 4), dtype=bool)

    filtered = suppress_lower_range_rows(mask, lower_fraction=0.3)

    assert not filtered[:3].any()
    assert filtered[3:].all()


def test_make_lidar_diagnostic_grid_places_panels_in_requested_2x3_layout():
    panels = [
        np.full((2, 3, 3), color, dtype=np.uint8)
        for color in (
            (10, 0, 0),
            (20, 0, 0),
            (30, 0, 0),
            (40, 0, 0),
            (50, 0, 0),
            (60, 0, 0),
        )
    ]

    grid = make_lidar_diagnostic_grid(
        original_image=panels[0],
        native_range_image=panels[1],
        projected_lidar_image=panels[2],
        image_edge_overlay=panels[3],
        native_range_edges=panels[4],
        projected_edge_overlay=panels[5],
    )

    assert grid.shape == (4, 9, 3)
    assert grid[0, 0].tolist() == [10, 0, 0]
    assert grid[0, 3].tolist() == [20, 0, 0]
    assert grid[0, 6].tolist() == [30, 0, 0]
    assert grid[2, 0].tolist() == [40, 0, 0]
    assert grid[2, 3].tolist() == [50, 0, 0]
    assert grid[2, 6].tolist() == [60, 0, 0]


def test_make_lidar_diagnostic_grid_preserves_range_panel_aspect_ratio():
    original = np.full((6, 6, 3), (10, 0, 0), dtype=np.uint8)
    wide_range = np.full((2, 6, 3), (50, 0, 0), dtype=np.uint8)
    projected = np.full((6, 6, 3), (30, 0, 0), dtype=np.uint8)

    grid = make_lidar_diagnostic_grid(
        original_image=original,
        native_range_image=wide_range,
        projected_lidar_image=projected,
        image_edge_overlay=original,
        native_range_edges=wide_range,
        projected_edge_overlay=projected,
    )

    top_middle = grid[:6, 6:12]
    bottom_middle = grid[6:12, 6:12]
    assert np.any(top_middle[2:4] == np.array([50, 0, 0], dtype=np.uint8))
    assert not np.any(top_middle[:2])
    assert not np.any(top_middle[4:])
    assert np.any(bottom_middle[2:4] == np.array([50, 0, 0], dtype=np.uint8))
    assert not np.any(bottom_middle[:2])
    assert not np.any(bottom_middle[4:])


def test_project_points_to_depth_map_ignores_invalid_points_without_warnings():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 5.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.eye(3, dtype=np.float64)
    intrinsic[0, 0] = 10.0
    intrinsic[1, 1] = 10.0
    intrinsic[0, 2] = 4.0
    intrinsic[1, 2] = 4.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = project_points_to_depth_map(
            points_camera=points,
            intrinsic=intrinsic,
            image_shape=(8, 8),
        )

    assert result.valid_mask[4, 4]
    assert result.depth_map[4, 4] == 4.0


def test_extract_depth_jump_edges_ignores_invalid_neighbors_without_warnings():
    depth_map = np.full((3, 4), np.inf, dtype=np.float32)
    depth_map[:, :2] = 5.0
    valid = np.isfinite(depth_map)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        edges = extract_depth_jump_edges(depth_map, valid)

    assert not edges.any()
