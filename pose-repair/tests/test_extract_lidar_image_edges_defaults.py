from scripts.extract_lidar_image_edges import (
    DEFAULT_CANNY_HIGH,
    DEFAULT_CANNY_LOW,
    DEFAULT_EDGE_MAX_RANGE_PERCENTILE,
    DEFAULT_EDGE_MIN_RANGE_PERCENTILE,
    DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS,
    DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION,
    make_edge_panel,
)

import numpy as np


def test_default_image_edge_thresholds_are_relaxed():
    assert DEFAULT_CANNY_LOW == 50
    assert DEFAULT_CANNY_HIGH == 120


def test_default_lidar_edge_distance_band_is_p10_to_p90():
    assert DEFAULT_EDGE_MIN_RANGE_PERCENTILE == 10.0
    assert DEFAULT_EDGE_MAX_RANGE_PERCENTILE == 90.0


def test_default_lidar_edge_filter_removes_single_pixel_components():
    assert DEFAULT_RANGE_EDGE_MIN_COMPONENT_PIXELS == 5


def test_default_silhouette_ignores_lower_30_percent_rows():
    assert DEFAULT_RANGE_SILHOUETTE_IGNORE_LOWER_FRACTION == 0.3


def test_camera_projection_middle_panel_shows_projected_lidar_edges():
    image = np.full((20, 20, 3), (10, 20, 30), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    intrinsic = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    panel, _stats = make_edge_panel(
        image=image,
        display_points_lidar=points,
        edge_points_lidar=points,
        edge_source_indices=np.array([0], dtype=np.int64),
        range_image=None,
        range_valid_mask=None,
        range_edge_mask=None,
        middle_panel_mode="camera_projection",
        lidar_extrinsic=None,
        ego_to_camera=np.eye(4, dtype=np.float64),
        intrinsic=intrinsic,
        lidar_extrinsic_direction="lidar_to_ego",
        max_depth_m=120.0,
        depth_jump_abs_m=0.8,
        depth_jump_rel=0.08,
        depth_edge_mode="combined",
        depth_edge_radius_px=5,
        min_valid_neighbors=2,
        occupancy_fill_radius_px=12,
        occupancy_boundary_width_px=1,
        occupancy_return_radius_px=1,
        occupancy_min_area_px=1000,
        canny_low=50,
        canny_high=120,
        max_edge_distance_px=20.0,
        preview_scale=1.0,
        lidar_edge_visual_radius=0,
        projected_point_radius=1,
    )

    bottom_middle = panel[20:40, 20:40]
    top_middle = panel[:20, 20:40]
    assert np.any(top_middle[10, 11] > 0)
    assert bottom_middle[10, 11].tolist() == top_middle[10, 11].tolist()
    assert bottom_middle[10, 10].tolist() == [0, 0, 255]
