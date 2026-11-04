import pandas as pd

from dongfeng_registration.match_quality import (
    _annotate_match_quality_thresholds,
    _retained_mask,
)


def test_retained_mask_uses_unified_quality_filter_when_available():
    matches = pd.DataFrame(
        {
            "pose_quality_valid": [True, True, True, False],
            "reprojection_threshold_inlier": [True, True, False, True],
            "quality_filter_valid": [True, False, True, True],
        }
    )

    assert _retained_mask(matches).tolist() == [True, False, True, True]


def test_retained_mask_uses_legacy_pose_reprojection_and_precision_flags():
    matches = pd.DataFrame(
        {
            "pose_quality_valid": [True, True, True, False],
            "reprojection_threshold_inlier": [True, True, False, True],
            "precision_filter_valid": [True, False, True, True],
        }
    )

    assert _retained_mask(matches).tolist() == [True, False, False, False]


def test_annotate_match_quality_thresholds_preserves_expected_validity():
    matched = pd.DataFrame(
        {
            "triangulation_valid_depth": [True, True, False],
            "triangulated_point_distance_m": [20.0, 120.0, 10.0],
            "triangulation_angle_deg": [2.0, 2.0, 2.0],
            "ray_closest_distance_m": [0.01, 0.01, 0.01],
            "reprojection1_px": [5.0, 5.0, 1.0],
            "reprojection2_px": [5.0, 40.0, 1.0],
            "reprojection_rms_px": [5.0, 25.0, 1.0],
        }
    )

    annotated = _annotate_match_quality_thresholds(
        matched,
        motion_pair_valid=True,
        threshold_px=20.0,
        max_triangulated_point_distance_m=100.0,
        min_triangulation_angle_deg=0.5,
        max_ray_closest_distance_m=0.5,
    )

    assert annotated["triangulation_distance_valid"].tolist() == [True, False, True]
    assert annotated["pose_quality_valid"].tolist() == [True, False, False]
    assert annotated["reprojection_threshold_inlier"].tolist() == [True, False, False]


def test_annotate_match_quality_thresholds_can_use_adaptive_mad_ray_threshold():
    matched = pd.DataFrame(
        {
            "triangulation_valid_depth": [True] * 8,
            "triangulated_point_distance_m": [20.0] * 8,
            "triangulation_angle_deg": [2.0] * 8,
            "ray_closest_distance_m": [0.08, 0.09, 0.10, 0.11, 0.12, 0.18, 0.21, 1.5],
            "reprojection1_px": [5.0] * 8,
            "reprojection2_px": [5.0] * 8,
            "reprojection_rms_px": [5.0] * 8,
        }
    )

    annotated = _annotate_match_quality_thresholds(
        matched,
        motion_pair_valid=True,
        threshold_px=20.0,
        max_triangulated_point_distance_m=100.0,
        min_triangulation_angle_deg=0.5,
        max_ray_closest_distance_m=0.1,
        ray_closest_distance_mode="adaptive_mad",
        ray_closest_distance_mad_k=2.0,
        ray_closest_distance_min_candidates=4,
    )

    threshold = float(annotated["ray_closest_distance_threshold_m"].iloc[0])
    assert 0.20 < threshold < 0.21
    assert annotated["ray_closest_distance_valid"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert annotated["ray_closest_distance_threshold_mode"].eq("adaptive_mad").all()


def test_adaptive_mad_thresholds_use_fallbacks_when_candidates_are_insufficient():
    matched = pd.DataFrame(
        {
            "triangulation_valid_depth": [True, True, True],
            "triangulated_point_distance_m": [20.0, 20.0, 20.0],
            "triangulation_angle_deg": [2.0, 2.0, 2.0],
            "ray_closest_distance_m": [0.1, 0.2, 99.0],
            "reprojection1_px": [5.0, 6.0, 999.0],
            "reprojection2_px": [5.0, 6.0, 999.0],
            "reprojection_rms_px": [5.0, 6.0, 999.0],
        }
    )

    annotated = _annotate_match_quality_thresholds(
        matched,
        motion_pair_valid=True,
        threshold_px=20.0,
        max_triangulated_point_distance_m=100.0,
        min_triangulation_angle_deg=0.5,
        max_ray_closest_distance_m=0.1,
        ray_closest_distance_mode="adaptive_mad",
        ray_closest_distance_mad_k=2.0,
        ray_closest_distance_min_candidates=20,
        reprojection_threshold_mode="adaptive_mad",
        reprojection_threshold_mad_k=2.0,
        reprojection_threshold_min_candidates=20,
    )

    assert annotated["reprojection_threshold_px"].iloc[0] == 5.0
    assert annotated["ray_closest_distance_threshold_m"].iloc[0] == 0.10
    assert annotated["reprojection_threshold_inlier"].tolist() == [True, False, False]
    assert annotated["ray_closest_distance_valid"].tolist() == [True, False, False]


def test_ray_adaptive_mad_candidates_match_reprojection_candidate_base():
    matched = pd.DataFrame(
        {
            "triangulation_valid_depth": [True] * 6,
            "triangulated_point_distance_m": [20.0] * 6,
            "triangulation_angle_deg": [2.0] * 6,
            "ray_closest_distance_m": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "reprojection1_px": [1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0],
            "reprojection2_px": [1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0],
            "reprojection_rms_px": [1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0],
        }
    )

    annotated = _annotate_match_quality_thresholds(
        matched,
        motion_pair_valid=True,
        threshold_px=20.0,
        max_triangulated_point_distance_m=100.0,
        min_triangulation_angle_deg=0.5,
        max_ray_closest_distance_m=0.1,
        ray_closest_distance_mode="adaptive_mad",
        ray_closest_distance_mad_k=2.0,
        ray_closest_distance_min_candidates=6,
        reprojection_threshold_mode="fixed",
    )

    assert annotated["ray_closest_distance_candidate_count"].iloc[0] == 6
