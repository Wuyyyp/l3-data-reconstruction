from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from dongfeng_registration.camera_models import CameraCalibration, project_points
from dongfeng_registration.geometry import triangulate_and_reproject
from dongfeng_registration.visualization import (
    _save,
    draw_ransac_retained_reprojection_qc,
    draw_epipolar_overlay,
    draw_match_overlay,
    draw_reprojection_qc_overview,
    draw_reprojection_overlay,
    draw_tiled_before_after_reprojection_overlay,
    draw_tiled_match_overlay,
    draw_tiled_reprojection_overlay,
    resize_with_padding,
)
from tests.conftest import real_t3_data_available


DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)

requires_real_t3_data = pytest.mark.skipif(
    not real_t3_data_available(DATA_ROOT),
    reason=f"optional real T3 dataset not available: {DATA_ROOT}",
)


def test_save_creates_missing_nested_windows_directories(tmp_path):
    paths = [tmp_path / "one" / "two" / f"image_{index}.jpg" for index in range(16)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                _save,
                path,
                np.zeros((8, 8, 3), dtype=np.uint8),
                jpeg_quality=90,
            )
            for path in paths
        ]
        for future in futures:
            future.result()

    assert all(path.exists() for path in paths)


@requires_real_t3_data
def test_pinhole_and_fisheye_projection_are_finite():
    points = np.array([[0.2, 0.1, 3.0], [-0.5, 0.2, 5.0]])
    for camera in ("center_camera_fov30", "front_camera_fov195"):
        calibration = CameraCalibration.from_dataset(DATA_ROOT, camera)
        pixels = project_points(points, calibration)
        assert pixels.shape == (2, 2)
        assert np.isfinite(pixels).all()


def test_triangulation_reprojection_recovers_synthetic_correspondences():
    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[0, 3] = -1.0
    world_points = np.array([[0.2, 0.1, 4.0], [-0.4, 0.3, 6.0]])
    first_points = world_points[:, :2] / world_points[:, 2, None]
    second_camera_points = world_points + np.array([-1.0, 0.0, 0.0])
    second_points = second_camera_points[:, :2] / second_camera_points[:, 2, None]
    result = triangulate_and_reproject(
        first_points, second_points, first_pose, second_pose
    )
    assert result["valid_depth"].all()
    assert np.max(result["reprojection_error_normalized"]) < 1e-9
    assert result["first_camera_points"].shape == (2, 3)
    assert result["second_camera_points"].shape == (2, 3)


def test_visual_overlays_write_nonempty_images(tmp_path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    first = np.array([[60.0, 80.0], [100.0, 120.0]])
    second = np.array([[65.0, 80.0], [108.0, 120.0]])
    fundamental = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    paths = [
        tmp_path / "matches.jpg",
        tmp_path / "epipolar.jpg",
        tmp_path / "reprojection.jpg",
    ]
    draw_match_overlay(image, image, first, second, paths[0])
    draw_epipolar_overlay(image, image, first, second, fundamental, paths[1])
    draw_reprojection_overlay(
        image, second, second + np.array([[2.0, 0.0], [0.0, 3.0]]), paths[2]
    )
    for path in paths:
        rendered = cv2.imread(str(path))
        assert path.exists()
        assert rendered is not None
        assert rendered.mean() > 0


def test_visual_overlay_writes_long_paths(tmp_path):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    first = np.array([[10.0, 12.0], [20.0, 18.0]])
    second = np.array([[12.0, 12.0], [23.0, 18.0]])
    long_dir = tmp_path / (
        "front_camera_fov30__right_front_camera"
        "_temporal_overlap_10m_long_windows_path_segment"
    )
    long_name = (
        "front_camera_fov30__right_front_camera_0005_0025_offset_+020"
        "_qc_with_long_name.jpg"
    )
    path = long_dir / long_name

    draw_match_overlay(image, image, first, second, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_reprojection_qc_skips_extreme_projected_points(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    points = np.array([[20.0, 30.0], [40.0, 50.0]])
    extreme = np.array([[1.0e18, 1.0e18], [42.0, 51.0]])
    path = tmp_path / "qc_extreme.jpg"

    draw_reprojection_qc_overview(
        image,
        image,
        points,
        points,
        extreme,
        points + 1.0,
        extreme,
        points + 1.0,
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_tiled_match_and_reprojection_overlays_expand_to_multiple_rows(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    count = 305
    x = np.linspace(10, 140, count)
    y = np.linspace(20, 100, count)
    first = np.stack([x, y], axis=1)
    second = np.stack([x + 5, y], axis=1)
    reprojected = second + np.stack([np.full(count, 2.0), np.zeros(count)], axis=1)

    match_path = tmp_path / "matches_tiled.jpg"
    reproj_path = tmp_path / "reprojection_tiled.jpg"
    match_layout = draw_tiled_match_overlay(
        image, image, first, second, match_path, max_per_row=120
    )
    reproj_layout = draw_tiled_reprojection_overlay(
        image, second, reprojected, reproj_path, max_per_row=120
    )

    match_image = cv2.imread(str(match_path))
    reproj_image = cv2.imread(str(reproj_path))
    assert match_image is not None
    assert reproj_image is not None
    assert len(match_layout) == 3
    assert len(reproj_layout) == 3
    assert match_layout[-1]["end_index"] == count
    assert reproj_layout[-1]["end_index"] == count
    assert match_image.shape[0] > image.shape[0]
    assert reproj_image.shape[0] > image.shape[0]


def test_tiled_match_overlay_can_draw_fixed_red_lines(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    first = np.array([[30.0, 40.0], [80.0, 70.0]])
    second = np.array([[40.0, 42.0], [92.0, 74.0]])
    path = tmp_path / "matches_red.jpg"

    draw_tiled_match_overlay(
        image,
        image,
        first,
        second,
        path,
        max_per_row=10,
        line_color=(0, 0, 255),
        title_prefix="red=test",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None


def test_ransac_retained_reprojection_qc_draws_red_and_green_lines(tmp_path):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    first = np.array([[60.0, 200.0], [240.0, 300.0], [440.0, 380.0]])
    second = np.array([[80.0, 205.0], [270.0, 310.0], [480.0, 390.0]])
    retained = np.array([True, False, True])
    ransac = np.array([True, True, True])
    path = tmp_path / "qc.jpg"

    draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=ransac,
        retained_mask=retained,
        reprojected_first=first + np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        reprojected_second=second + np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        metadata={
            "pose_degraded": True,
            "reprojection_median_px": 100.0,
            "reprojection_p75_px": 120.0,
            "ray_gap_median_m": 0.7,
            "ray_gap_p75_m": 1.3,
        },
        path=path,
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert rendered.shape[0] > image.shape[0]
    green_pixels = (rendered[:, :, 1] > 120) & (rendered[:, :, 2] < 80)
    match_panel = rendered[: image.shape[0]]
    red_pixels = (match_panel[:, :, 2] > 120) & (match_panel[:, :, 1] < 80)
    assert int(green_pixels.sum()) > 0
    assert int(red_pixels.sum()) > 0


def test_ransac_retained_reprojection_qc_can_draw_retained_only(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    first = np.array([[30.0, 40.0], [80.0, 70.0], [120.0, 90.0]])
    second = np.array([[40.0, 42.0], [92.0, 74.0], [130.0, 92.0]])
    retained = np.array([True, False, True])
    ransac = np.array([True, True, True])
    path = tmp_path / "retained_only.jpg"

    draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=ransac,
        retained_mask=retained,
        reprojected_first=first,
        reprojected_second=second,
        metadata={"pose_degraded": False},
        path=path,
        include_rejected_ransac=False,
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    green_pixels = (rendered[:, :, 1] > 120) & (rendered[:, :, 2] < 80)
    match_panel = rendered[: image.shape[0]]
    red_pixels = (match_panel[:, :, 2] > 120) & (match_panel[:, :, 1] < 80)
    assert int(green_pixels.sum()) > 0
    assert int(red_pixels.sum()) == 0


def test_ransac_retained_reprojection_qc_can_keep_retained_view_in_one_row(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    count = 12
    first = np.column_stack([np.linspace(20.0, 140.0, count), np.linspace(20.0, 100.0, count)])
    second = first + np.array([8.0, 2.0])
    path = tmp_path / "retained_one_row.jpg"

    layout = draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=np.ones(count, dtype=bool),
        retained_mask=np.ones(count, dtype=bool),
        reprojected_first=first,
        reprojected_second=second,
        metadata={"pose_degraded": False},
        path=path,
        include_rejected_ransac=False,
        max_per_row=count,
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert len(layout) == 1
    assert rendered.shape[0] == image.shape[0] * 3


def test_ransac_retained_reprojection_qc_can_place_reprojections_side_by_side(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    count = 6
    first = np.column_stack([np.linspace(20.0, 140.0, count), np.linspace(20.0, 100.0, count)])
    second = first + np.array([8.0, 2.0])
    path = tmp_path / "retained_side_by_side.jpg"

    layout = draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=np.ones(count, dtype=bool),
        retained_mask=np.ones(count, dtype=bool),
        reprojected_first=first,
        reprojected_second=second,
        metadata={"pose_degraded": False},
        path=path,
        include_rejected_ransac=False,
        max_per_row=count,
        reprojection_layout="side_by_side",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert len(layout) == 1
    assert rendered.shape[0] == image.shape[0] * 2


def test_ransac_retained_reprojection_qc_keeps_a_left_b_right_in_reprojection_row(
    tmp_path,
):
    first_image = np.zeros((40, 60, 3), dtype=np.uint8)
    first_image[:, :] = (255, 0, 0)
    second_image = np.zeros((40, 60, 3), dtype=np.uint8)
    second_image[:, :] = (0, 255, 0)
    first = np.array([[20.0, 20.0]], dtype=np.float32)
    second = np.array([[30.0, 20.0]], dtype=np.float32)
    path = tmp_path / "retained_ab_order.jpg"

    draw_ransac_retained_reprojection_qc(
        first_image,
        second_image,
        first,
        second,
        ransac_inlier=np.ones(1, dtype=bool),
        retained_mask=np.ones(1, dtype=bool),
        reprojected_first=first,
        reprojected_second=second,
        metadata={"pose_degraded": False},
        path=path,
        include_rejected_ransac=False,
        max_per_row=1,
        reprojection_layout="side_by_side",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    reprojection_row = rendered[first_image.shape[0] :, :]
    left_half = reprojection_row[:, : first_image.shape[1]]
    right_half = reprojection_row[:, first_image.shape[1] :]
    assert left_half[:, :, 0].mean() > 180
    assert left_half[:, :, 1].mean() < 80
    assert right_half[:, :, 1].mean() > 180
    assert right_half[:, :, 0].mean() < 80


def test_ransac_retained_reprojection_qc_can_draw_reprojection_for_all_ransac_inliers(
    tmp_path,
):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    first = np.array([[30.0, 40.0], [80.0, 70.0], [120.0, 90.0]])
    second = np.array([[40.0, 42.0], [92.0, 74.0], [130.0, 92.0]])
    retained = np.array([True, False, False])
    ransac = np.array([True, True, False])
    path = tmp_path / "qc_all_ransac_reprojection.jpg"

    draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=ransac,
        retained_mask=retained,
        reprojected_first=first + np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 0.0]]),
        reprojected_second=second + np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 0.0]]),
        metadata={"pose_degraded": True},
        path=path,
        include_rejected_ransac=True,
        reprojection_layout="side_by_side",
        reprojection_points="ransac_inliers",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert rendered.shape[0] == image.shape[0] * 2
    # The second RANSAC inlier is not retained, but has visible red reprojection overlay.
    red_pixels = (rendered[:, :, 2] > 120) & (rendered[:, :, 1] < 100)
    green_pixels = (rendered[:, :, 1] > 120) & (rendered[:, :, 2] < 100)
    assert int(red_pixels.sum()) > 0
    assert int(green_pixels.sum()) > 0


def test_ransac_retained_reprojection_qc_skips_extreme_reprojected_points(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    first = np.array([[30.0, 40.0], [80.0, 70.0]])
    second = np.array([[40.0, 42.0], [92.0, 74.0]])
    path = tmp_path / "qc_extreme_reprojection.jpg"

    draw_ransac_retained_reprojection_qc(
        image,
        image,
        first,
        second,
        ransac_inlier=np.ones(2, dtype=bool),
        retained_mask=np.ones(2, dtype=bool),
        reprojected_first=np.array([[1.0e18, 1.0e18], [81.0, 71.0]]),
        reprojected_second=np.array([[1.0e18, 1.0e18], [93.0, 75.0]]),
        metadata={"pose_degraded": False},
        path=path,
        include_rejected_ransac=False,
        reprojection_layout="side_by_side",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert path.stat().st_size > 0


def test_before_after_reprojection_overlay_draws_red_and_green(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    observed = np.array([[50.0, 60.0], [80.0, 70.0]])
    before = observed + np.array([[10.0, 0.0], [0.0, 10.0]])
    after = observed + np.array([[2.0, 0.0], [0.0, 2.0]])

    path = tmp_path / "before_after.jpg"
    layout = draw_tiled_before_after_reprojection_overlay(
        image, observed, before, after, path, max_per_row=1
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert len(layout) == 2
    red_pixels = (rendered[:, :, 2] > 180) & (rendered[:, :, 1] < 120)
    green_pixels = (rendered[:, :, 1] > 180) & (rendered[:, :, 2] < 120)
    assert red_pixels.any()
    assert green_pixels.any()


def test_resize_with_padding_preserves_aspect_and_transforms_points():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    points = np.array([[0.0, 0.0], [200.0, 100.0], [100.0, 50.0]])

    resized, transformed, meta = resize_with_padding(
        image, points, target_size=(300, 300)
    )

    assert resized.shape == (300, 300, 3)
    assert meta["scale"] == 1.5
    assert meta["pad_x"] == 0
    assert meta["pad_y"] == 75
    np.testing.assert_allclose(
        transformed,
        np.array([[0.0, 75.0], [300.0, 225.0], [150.0, 150.0]]),
    )


def test_reprojection_qc_overview_uses_fixed_size_and_draws_all_layers(tmp_path):
    first_image = np.zeros((120, 240, 3), dtype=np.uint8)
    second_image = np.zeros((200, 100, 3), dtype=np.uint8)
    first_points = np.array([[40.0, 30.0], [160.0, 90.0]])
    second_points = np.array([[30.0, 60.0], [70.0, 140.0]])
    before_first = first_points + np.array([[12.0, 0.0], [0.0, 12.0]])
    after_first = first_points + np.array([[3.0, 0.0], [0.0, 3.0]])
    before_second = second_points + np.array([[8.0, 0.0], [0.0, 8.0]])
    after_second = second_points + np.array([[2.0, 0.0], [0.0, 2.0]])

    path = tmp_path / "qc.jpg"
    layout = draw_reprojection_qc_overview(
        first_image,
        second_image,
        first_points,
        second_points,
        before_first,
        after_first,
        before_second,
        after_second,
        path,
        tile_size=(320, 240),
        title="qc test",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    assert rendered.shape == (480, 640, 3)
    assert layout["tile_width"] == 320
    assert layout["tile_height"] == 240
    red_pixels = (rendered[:, :, 2] > 180) & (rendered[:, :, 1] < 120)
    green_pixels = (rendered[:, :, 1] > 180) & (rendered[:, :, 2] < 120)
    white_pixels = (
        (rendered[:, :, 0] > 180)
        & (rendered[:, :, 1] > 180)
        & (rendered[:, :, 2] > 180)
    )
    assert red_pixels.any()
    assert green_pixels.any()
    assert white_pixels.any()


def test_reprojection_qc_overview_adapts_to_largest_original_height(tmp_path):
    first_image = np.zeros((100, 200, 3), dtype=np.uint8)
    second_image = np.zeros((200, 100, 3), dtype=np.uint8)
    first_points = np.array([[100.0, 50.0]])
    second_points = np.array([[50.0, 100.0]])

    path = tmp_path / "qc_adaptive.png"
    layout = draw_reprojection_qc_overview(
        first_image,
        second_image,
        first_points,
        second_points,
        first_points,
        first_points,
        second_points,
        second_points,
        path,
        tile_size=None,
        title="adaptive test",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    # second image has the largest original height: 200 px, kept unchanged.
    # first image is scaled from 100 px high to 200 px high, width 200 -> 400.
    assert layout["first_scale"] == 2.0
    assert layout["second_scale"] == 1.0
    assert layout["first_width"] == 400
    assert layout["second_width"] == 100
    assert rendered.shape == (400, 500, 3)


def test_reprojection_qc_overview_keeps_a_left_b_right_in_bottom_row(tmp_path):
    first_image = np.zeros((120, 80, 3), dtype=np.uint8)
    first_image[:, :] = (255, 0, 0)
    second_image = np.zeros((120, 80, 3), dtype=np.uint8)
    second_image[:, :] = (0, 255, 0)
    first = np.array([[20.0, 80.0]], dtype=np.float32)
    second = np.array([[30.0, 80.0]], dtype=np.float32)
    path = tmp_path / "overview_ab_order.jpg"

    draw_reprojection_qc_overview(
        first_image,
        second_image,
        first,
        second,
        first,
        first,
        second,
        second,
        path,
        tile_size=None,
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    bottom = rendered[first_image.shape[0] :, :]
    left_half = bottom[50:, : first_image.shape[1]]
    right_half = bottom[50:, first_image.shape[1] :]
    assert left_half[:, :, 0].mean() > 180
    assert left_half[:, :, 1].mean() < 80
    assert right_half[:, :, 1].mean() > 180
    assert right_half[:, :, 0].mean() < 80


def test_reprojection_qc_overview_draws_small_circle_plus_and_x_markers(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first_points = np.array([[50.0, 50.0]])
    second_points = np.array([[50.0, 50.0]])
    before_first = np.array([[80.0, 50.0]])
    after_first = np.array([[50.0, 80.0]])
    before_second = np.array([[80.0, 50.0]])
    after_second = np.array([[50.0, 80.0]])

    path = tmp_path / "qc_markers.png"
    draw_reprojection_qc_overview(
        image,
        image,
        first_points,
        second_points,
        before_first,
        after_first,
        before_second,
        after_second,
        path,
        tile_size=(200, 200),
        title="marker test",
    )

    rendered = cv2.imread(str(path))
    assert rendered is not None
    # Bottom-left tile is A->B on image B. Original coordinates scale by 2.
    y_offset = 200
    observed = (100, y_offset + 100)
    before = (160, y_offset + 100)
    after = (100, y_offset + 160)

    # White hollow circle around the observed point.
    assert rendered[observed[1], observed[0] + 3].min() > 160

    # Red plus marker: horizontal and vertical strokes through before point.
    assert rendered[before[1], before[0] - 3, 2] > 160
    assert rendered[before[1] - 3, before[0], 2] > 160

    # Green x marker: diagonal strokes around after point.
    assert rendered[after[1] - 3, after[0] - 3, 1] > 140
    assert rendered[after[1] + 3, after[0] + 3, 1] > 140
