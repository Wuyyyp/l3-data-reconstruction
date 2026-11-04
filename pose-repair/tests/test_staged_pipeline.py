from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dongfeng_registration.match_cache import inter_camera_cache_dir, match_cache_key


def test_matching_config_enables_bounded_parallel_filtering_by_default():
    from dongfeng_registration.config import MatchingConfig

    matching = MatchingConfig()

    assert matching.filter_workers == 4
    assert matching.overlap_filtering is True


def test_persistent_matching_overlaps_loma_with_process_filtering(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    completed = []
    executors = []

    class FakeFuture:
        def __init__(self, name):
            self.name = name

        def result(self):
            completed.append(self.name)
            return {"processed_dirs": 1}

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {"filter_workers": 4, "overlap_filtering": True},
            )(),
        },
    )()

    def fake_intra(**kwargs):
        assert kwargs["filter_executor"] is executors[0]
        kwargs["pending_filter_jobs"].append(FakeFuture("intra"))
        return tmp_path / "intra"

    def fake_inter(**kwargs):
        assert kwargs["filter_executor"] is executors[0]
        kwargs["pending_filter_jobs"].append(FakeFuture("inter"))
        return tmp_path / "inter"

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "ProcessPoolExecutor", FakeExecutor, raising=False)
    monkeypatch.setattr(staged_pipeline, "match_dataset_stage", fake_intra)
    monkeypatch.setattr(staged_pipeline, "match_multicamera_dataset_stage", fake_inter)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("matching: {}\n", encoding="utf-8")

    staged_pipeline.run_persistent_matching_stage(
        config_path=config_path,
        dataset_root=tmp_path,
        cameras=("camera_a",),
        semantic_cache_root=tmp_path / "semantic",
        match_cache_root=tmp_path / "matches",
        intra_baselines_m=(2.0,),
        inter_pairs=("camera_a__camera_b",),
    )

    assert len(executors) == 1
    assert executors[0].max_workers == 4
    assert executors[0].mp_context.get_start_method() == "spawn"
    assert completed == ["intra", "inter"]


def test_replace_dataframe_columns_once_preserves_schema_order_and_index():
    from dongfeng_registration import staged_pipeline

    index = pd.Index([7, 9], name="match_index")
    frame = pd.DataFrame(
        {
            "camera": pd.Series(
                ["front", "rear"], index=index, dtype="string[pyarrow]"
            ),
            "score": np.asarray([1.0, 2.0], dtype=np.float64),
        },
        index=index,
    )

    replaced = staged_pipeline._replace_dataframe_columns_once(
        frame,
        {
            "score": np.asarray([3.0, 4.0], dtype=np.float64),
            "retained": np.asarray([True, False], dtype=bool),
        },
    )

    assert replaced.index.equals(frame.index)
    assert list(replaced.columns) == ["camera", "score", "retained"]
    assert str(replaced["camera"].dtype) == str(frame["camera"].dtype)
    assert replaced["camera"].tolist() == ["front", "rear"]
    assert replaced["score"].tolist() == [3.0, 4.0]
    assert replaced["retained"].tolist() == [True, False]
    assert frame["score"].tolist() == [1.0, 2.0]


def test_matching_scope_overrides_deep_score_threshold(tmp_path: Path):
    from dongfeng_registration.config import load_config
    from dongfeng_registration import staged_pipeline

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_root: {tmp_path.as_posix()}",
                f"output_root: {(tmp_path / 'outputs').as_posix()}",
                "cameras: [center_camera_fov30]",
                "matching:",
                "  backend: loma",
                "  deep_score_threshold: 0.7",
                "  deep_score_threshold_intra: 0.3",
                "  deep_score_threshold_inter: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    intra = staged_pipeline._config_for_matching_scope(config, "intra")
    inter = staged_pipeline._config_for_matching_scope(config, "inter")
    unknown = staged_pipeline._config_for_matching_scope(config, "unknown")

    assert intra.matching.deep_score_threshold == 0.3
    assert inter.matching.deep_score_threshold == 0.1
    assert unknown.matching.deep_score_threshold == 0.7


def test_inter_quality_evaluation_does_not_decode_source_images(
    monkeypatch,
    tmp_path: Path,
):
    from dongfeng_registration import staged_pipeline

    class FakeConfig:
        dataset_root = tmp_path
        assumptions = {}

    class FakeFrame:
        index = 0
        timestamp_ns = 123
        path = tmp_path / "missing.jpg"

    calibration = object()
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: FakeConfig())
    monkeypatch.setattr(staged_pipeline, "_navigation_for_config", lambda _config: object())
    monkeypatch.setattr(
        staged_pipeline,
        "_optimized_pose_csv_from_assumptions",
        lambda _assumptions, _root: tmp_path / "missing.csv",
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_load_image_camera_to_world_csv",
        lambda _path: {},
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_calibration_for_config",
        lambda _config, _camera: calibration,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_frames_for_config",
        lambda _config, _camera: [FakeFrame()],
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_recompute_normalized_coordinates_for_calibrations",
        lambda matches, _calibrations: matches,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_evaluate_inter_camera_navigation",
        lambda group, *_args, **_kwargs: (
            group.assign(
                reprojection_rms_px=1.0,
                image_ransac_inlier=True,
                pose_degraded=False,
            ),
            np.eye(3),
            1.0,
        ),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_annotate_match_quality_thresholds",
        lambda evaluated, **_kwargs: evaluated.assign(reprojection_threshold_px=5.0),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_apply_match_quality_filter",
        lambda evaluated, **_kwargs: evaluated.assign(quality_filter_valid=True),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_retained_mask",
        lambda evaluated: pd.Series(True, index=evaluated.index),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "summarize_errors",
        lambda _values: {"count": 1, "median": 1.0, "p95": 1.0},
    )
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inter quality filtering must not decode images")
        ),
    )

    matches = pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_b"],
            "frame1": [0],
            "frame2": [0],
            "x1": [1.0],
            "y1": [1.0],
            "x2": [1.0],
            "y2": [1.0],
            "nx1": [0.0],
            "ny1": [0.0],
            "nx2": [0.0],
            "ny2": [0.0],
        }
    )

    evaluated = staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=matches,
    )

    assert len(evaluated) == 1


def test_pkl_backend_frames_use_result_interpolation_image_root_without_compat_camera(
    tmp_path: Path,
):
    from dongfeng_registration.config import load_config
    from dongfeng_registration import staged_pipeline

    dataset_root = tmp_path / "dataset"
    image_root = tmp_path / "camera_distorted"
    camera_dir = image_root / "front_camera_fov30"
    camera_dir.mkdir(parents=True)
    (camera_dir / "1753002123700312.jpg").write_bytes(b"fake")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_root: {dataset_root.as_posix()}",
                f"output_root: {(tmp_path / 'outputs').as_posix()}",
                "cameras: [front_camera_fov30]",
                "assumptions:",
                "  pose_backend: result_interpolation_pkl",
                "  result_interpolation_pkl_path: result_interpolation.pkl",
                f"  result_interpolation_image_root: {image_root.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    frames = staged_pipeline._frames_for_config(
        load_config(config_path),
        "front_camera_fov30",
    )

    assert len(frames) == 1
    assert frames[0].timestamp_ns == 1753002123700312000
    assert frames[0].path == camera_dir / "1753002123700312.jpg"


def test_recompute_normalized_coordinates_overwrites_cached_values():
    from dongfeng_registration import staged_pipeline
    from dongfeng_registration.camera_models import CameraCalibration

    calibration = CameraCalibration(
        camera="camera_a",
        model="rational",
        image_size=(100, 100),
        intrinsic=np.array(
            [[10.0, 0.0, 50.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=np.eye(4),
        time_lag=0.0,
        images_undistorted=True,
    )
    matches = pd.DataFrame(
        {
            "camera": ["camera_a"],
            "x1": [60.0],
            "y1": [80.0],
            "x2": [70.0],
            "y2": [100.0],
            "nx1": [-999.0],
            "ny1": [-999.0],
            "nx2": [-999.0],
            "ny2": [-999.0],
        }
    )

    updated = staged_pipeline._recompute_normalized_coordinates_for_calibrations(
        matches,
        {"camera_a": calibration},
    )

    assert updated.loc[0, "nx1"] == pytest.approx(1.0)
    assert updated.loc[0, "ny1"] == pytest.approx(2.0)
    assert updated.loc[0, "nx2"] == pytest.approx(2.0)
    assert updated.loc[0, "ny2"] == pytest.approx(3.0)


def test_match_cache_key_ignores_matching_parameter_hash():
    key_a = match_cache_key(
        backend="loma",
        image_size=1600,
        score_threshold=0.3,
        semantic_backend="mask2former",
        semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        semantic_image_size=2048,
        semantic_dilation_pixels=5,
        max_keypoints=1024,
    )
    key_b = match_cache_key(
        backend="loma",
        image_size=1600,
        score_threshold=0.3,
        semantic_backend="mask2former",
        semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        semantic_image_size=2048,
        semantic_dilation_pixels=5,
        max_keypoints=2048,
    )

    assert key_a == "loma__mask2former"
    assert key_b == "loma__mask2former"


def test_apply_match_quality_filter_records_unified_validity_and_reasons(monkeypatch):
    from dongfeng_registration import staged_pipeline

    matches = pd.DataFrame(
        {
            "x1": np.linspace(10.0, 90.0, 9),
            "y1": np.linspace(20.0, 100.0, 9),
            "x2": np.linspace(15.0, 95.0, 9),
            "y2": np.linspace(25.0, 105.0, 9),
            "pose_quality_valid": [True] * 9,
            "reprojection_threshold_inlier": [True] * 9,
            "triangulation_valid_depth": [True] * 6 + [False] + [True] * 2,
            "triangulation_distance_valid": [True] * 9,
            "triangulation_angle_valid": [True] * 9,
            "ray_closest_distance_valid": [True] * 9,
        }
    )
    matches.loc[1, "triangulation_angle_valid"] = False
    matches.loc[2, "reprojection_threshold_inlier"] = False

    ransac_mask = np.ones((9, 1), dtype=np.uint8)
    ransac_mask[3, 0] = 0
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "findFundamentalMat",
        lambda *_args, **_kwargs: (np.eye(3), ransac_mask),
    )

    filtered = staged_pipeline._apply_match_quality_filter(
        matches,
        image_ransac_threshold_px=50.0,
        image_ransac_confidence=0.999,
    )

    assert filtered["quality_filter_valid"].tolist() == [
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert filtered.loc[1, "quality_reject_reason"] == "low_triangulation_angle"
    assert filtered.loc[2, "quality_reject_reason"] == "large_reprojection_error"
    assert filtered.loc[3, "quality_reject_reason"] == "image_ransac_outlier"
    assert filtered.loc[6, "quality_reject_reason"] == "invalid_triangulation_depth"
    assert filtered["precision_filter_valid"].tolist() == filtered[
        "quality_filter_valid"
    ].tolist()


def test_apply_match_quality_filter_uses_ransac_only_for_pose_degraded(monkeypatch):
    from dongfeng_registration import staged_pipeline

    matches = pd.DataFrame(
        {
            "x1": np.linspace(10.0, 90.0, 9),
            "y1": np.linspace(20.0, 100.0, 9),
            "x2": np.linspace(15.0, 95.0, 9),
            "y2": np.linspace(25.0, 105.0, 9),
            "pose_quality_valid": [False] * 9,
            "reprojection_threshold_inlier": [False] * 9,
            "triangulation_valid_depth": [True] * 6 + [False] + [True] * 2,
            "triangulation_distance_valid": [True] * 9,
            "triangulation_angle_valid": [True] * 9,
            "ray_closest_distance_valid": [False] * 9,
            "reprojection_rms_px": [100.0] * 9,
            "ray_closest_distance_m": [1.0] * 9,
        }
    )

    ransac_mask = np.ones((9, 1), dtype=np.uint8)
    ransac_mask[7, 0] = 0
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "findFundamentalMat",
        lambda *_args, **_kwargs: (np.eye(3), ransac_mask),
    )

    filtered = staged_pipeline._apply_match_quality_filter(
        matches,
        image_ransac_threshold_px=50.0,
        image_ransac_confidence=0.999,
        pose_degraded_detection_enabled=True,
        pose_degraded_min_ransac_inliers=5,
        pose_degraded_reprojection_median_px=40.0,
        pose_degraded_reprojection_p75_px=80.0,
        pose_degraded_ray_gap_median_m=0.6,
        pose_degraded_ray_gap_p75_m=1.2,
        pose_degraded_quality_mode="ransac_only",
    )

    assert filtered["pose_degraded"].unique().tolist() == [True]
    assert filtered["quality_filter_valid"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    assert int(staged_pipeline._retained_mask(filtered).sum()) == 7
    assert filtered["pose_degraded_ransac_inliers"].unique().tolist() == [7]
    assert filtered.loc[6, "quality_reject_reason"] == "invalid_triangulation_depth"
    assert filtered.loc[7, "quality_reject_reason"] == "image_ransac_outlier"


def test_pose_degraded_detection_counts_only_ransac_inliers_with_valid_depth(
    monkeypatch,
):
    from dongfeng_registration import staged_pipeline

    matches = pd.DataFrame(
        {
            "x1": np.linspace(10.0, 90.0, 6),
            "y1": np.linspace(20.0, 100.0, 6),
            "x2": np.linspace(15.0, 95.0, 6),
            "y2": np.linspace(25.0, 105.0, 6),
            "pose_quality_valid": [False] * 6,
            "reprojection_threshold_inlier": [False] * 6,
            "triangulation_valid_depth": [True, True, True, True, False, False],
            "triangulation_distance_valid": [True] * 6,
            "triangulation_angle_valid": [True] * 6,
            "ray_closest_distance_valid": [False] * 6,
            "reprojection_rms_px": [120.0] * 6,
            "ray_closest_distance_m": [2.0] * 6,
        }
    )

    ransac_mask = np.ones((6, 1), dtype=np.uint8)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "findFundamentalMat",
        lambda *_args, **_kwargs: (np.eye(3), ransac_mask),
    )

    filtered = staged_pipeline._apply_match_quality_filter(
        matches,
        image_ransac_threshold_px=50.0,
        image_ransac_confidence=0.999,
        pose_degraded_detection_enabled=True,
        pose_degraded_min_ransac_inliers=5,
        pose_degraded_reprojection_median_px=40.0,
        pose_degraded_reprojection_p75_px=80.0,
        pose_degraded_ray_gap_median_m=0.6,
        pose_degraded_ray_gap_p75_m=1.2,
        pose_degraded_quality_mode="ransac_only",
    )

    assert filtered["pose_degraded"].unique().tolist() == [False]
    assert filtered["pose_degraded_ransac_inliers"].unique().tolist() == [4]
    assert filtered["quality_filter_valid"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_annotate_match_quality_thresholds_can_adapt_reprojection_threshold():
    from dongfeng_registration import staged_pipeline

    matches = pd.DataFrame(
        {
            "triangulated_point_distance_m": [10.0] * 7,
            "triangulation_angle_deg": [2.0] * 7,
            "ray_closest_distance_m": [0.01] * 7,
            "reprojection1_px": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 300.0],
            "reprojection2_px": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 300.0],
            "reprojection_rms_px": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 300.0],
            "triangulation_valid_depth": [True] * 7,
        }
    )

    annotated = staged_pipeline._annotate_match_quality_thresholds(
        matches,
        motion_pair_valid=True,
        threshold_px=40.0,
        max_triangulated_point_distance_m=100.0,
        min_triangulation_angle_deg=0.5,
        max_ray_closest_distance_m=0.5,
        reprojection_threshold_mode="adaptive_mad",
        reprojection_threshold_min_px=40.0,
        reprojection_threshold_max_px=120.0,
        reprojection_threshold_mad_k=2.5,
        reprojection_threshold_min_candidates=7,
    )

    assert annotated["reprojection_threshold_mode"].unique().tolist() == [
        "adaptive_mad"
    ]
    assert annotated["reprojection_threshold_px"].iloc[0] > 40.0
    assert annotated["reprojection_threshold_px"].iloc[0] <= 120.0
    assert annotated["reprojection_threshold_inlier"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def test_inter_camera_cache_dir_preserves_declared_pair_order(tmp_path):
    cache_dir = inter_camera_cache_dir(
        tmp_path,
        cache_key="cache",
        first_camera="rear_camera",
        second_camera="left_rear_camera",
        relation="temporal_overlap",
    )

    assert cache_dir == (
        tmp_path
        / "cache"
        / "inter_camera"
        / "rear_camera__left_rear_camera"
        / "temporal_overlap"
    )


def test_inter_camera_same_index_allowed_pair_filter():
    from dongfeng_registration import staged_pipeline

    assumptions = {
        "inter_camera_same_index_allowed_pairs": [
            "front_camera_fov30__left_front_camera",
            "rear_camera__right_rear_camera",
        ]
    }

    assert staged_pipeline._is_same_index_pair_allowed(
        assumptions, "front_camera_fov30", "left_front_camera"
    )
    assert staged_pipeline._is_same_index_pair_allowed(
        assumptions, "left_front_camera", "front_camera_fov30"
    )
    assert not staged_pipeline._is_same_index_pair_allowed(
        assumptions, "front_camera_fov30", "rear_camera"
    )


def test_temporal_overlap_allowed_pairs_are_strict_when_configured():
    from dongfeng_registration import staged_pipeline

    assumptions = {
        "inter_camera_temporal_overlap_allowed_pairs": [
            "front_camera_fov30__left_front_camera"
        ]
    }

    assert staged_pipeline._is_temporal_overlap_direction_allowed(
        assumptions, "front_camera_fov30", "left_front_camera"
    )
    assert staged_pipeline._is_temporal_overlap_direction_allowed(
        assumptions, "left_front_camera", "front_camera_fov30"
    )
    assert not staged_pipeline._is_temporal_overlap_direction_allowed(
        assumptions, "left_front_camera", "rear_camera"
    )


def test_build_baseline_frame_pairs_prefers_future_then_falls_back_to_past():
    from dongfeng_registration.staged_pipeline import _build_baseline_frame_pairs

    centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([3.2, 0.0, 0.0]),
        2: np.array([10.1, 0.0, 0.0]),
        3: np.array([20.2, 0.0, 0.0]),
    }

    pairs = _build_baseline_frame_pairs(
        frame_indices=[0, 1, 2, 3],
        camera_centers_by_index=centers,
        target_baseline_m=10.0,
    )

    assert pairs == [
        (0, 2, 10.1),
        (1, 3, 17.0),
        (2, 3, 10.1),
        (3, 2, 10.1),
    ]


def test_build_baseline_frame_pairs_allows_frames_to_be_reused_as_match_endpoints():
    from dongfeng_registration.staged_pipeline import _build_baseline_frame_pairs

    centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.1, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([3.1, 0.0, 0.0]),
    }

    pairs = _build_baseline_frame_pairs(
        frame_indices=[0, 1, 2, 3],
        camera_centers_by_index=centers,
        target_baseline_m=3.0,
    )

    assert pairs == [
        (0, 2, 3.0),
        (1, 3, 3.0),
        (2, 0, 3.0),
        (3, 1, 3.0),
    ]


def test_deduplicate_baseline_frame_pairs_removes_same_endpoint_pair_regardless_of_direction():
    from dongfeng_registration.staged_pipeline import _deduplicate_baseline_frame_pairs

    pairs_by_baseline = {
        3.0: [(0, 2, 3.0), (1, 3, 3.0)],
        10.0: [(2, 0, 3.0), (3, 1, 3.0), (0, 4, 10.1)],
    }

    deduped = _deduplicate_baseline_frame_pairs(pairs_by_baseline)

    assert deduped == {
        3.0: [(0, 2, 3.0), (1, 3, 3.0)],
        10.0: [(0, 4, 10.1)],
    }


def test_segment_dataset_stage_writes_stage_report(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    fake_report = pd.DataFrame(
        [
            {
                "camera": "center_camera_fov30",
                "frame": 0,
                "timestamp_ns": 100,
                "cache_hit": True,
            }
        ]
    )

    monkeypatch.setattr(
        staged_pipeline,
        "segment_camera",
        lambda *args, **kwargs: fake_report,
    )

    output = staged_pipeline.segment_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cache_root=tmp_path / "semantic_cache",
        camera="center_camera_fov30",
        mask_config=object(),
        start_frame=0,
        frame_count=1,
        force=False,
    )

    report_path = output / "segment_report.csv"
    exported = pd.read_csv(report_path)
    assert report_path.exists()
    assert exported["camera"].tolist() == ["center_camera_fov30"]


def test_segment_dataset_stage_uses_config_frames_for_pkl_backend(
    monkeypatch,
    tmp_path: Path,
):
    from dongfeng_registration import staged_pipeline

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset_root: .\noutput_root: outputs\n", encoding="utf-8")
    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"pose_backend": "result_interpolation_pkl"},
        },
    )()
    fake_frames = [
        type(
            "Frame",
            (),
            {
                "index": 0,
                "timestamp_ns": 123,
                "path": tmp_path / "front_camera_fov30" / "123.jpg",
            },
        )()
    ]
    seen = {}

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "_frames_for_config",
        lambda config, camera: fake_frames,
    )

    def fake_segment_camera(*args, **kwargs):
        seen["frames"] = kwargs.get("frames")
        return pd.DataFrame(
            [
                {
                    "camera": "front_camera_fov30",
                    "frame": 0,
                    "timestamp_ns": 123,
                    "cache_hit": True,
                }
            ]
        )

    monkeypatch.setattr(staged_pipeline, "segment_camera", fake_segment_camera)

    staged_pipeline.segment_dataset_stage(
        config_path=config_path,
        dataset_root=tmp_path,
        cache_root=tmp_path / "semantic_cache",
        camera="front_camera_fov30",
        mask_config=object(),
        start_frame=0,
        frame_count=1,
        force=False,
    )

    assert seen["frames"] is fake_frames


def test_match_dataset_stage_persists_raw_match_artifact(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
        },
    )()
    fake_matches = pd.DataFrame(
        {
            "camera": ["center_camera_fov30"],
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [5],
            "timestamp1_ns": [100],
            "timestamp2_ns": [200],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "backend": ["loma"],
            "backend_score": [1.0],
        }
    )
    fake_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "frame_offset": [5],
            "frame1": [0],
            "frame2": [5],
            "raw_match_count": [1],
            "matching_time_ms": [12.5],
        }
    )

    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_raw_matching",
        lambda *args, **kwargs: fake_matches,
    )
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_intra_camera_match_quality",
        lambda **kwargs: fake_matches.assign(
            pose_quality_valid=True,
            reprojection_threshold_inlier=True,
            precision_filter_valid=True,
            reprojected1_x=1.0,
            reprojected1_y=2.0,
            reprojected2_x=3.0,
            reprojected2_y=4.0,
        ),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "Dataset",
        lambda _root: type(
            "Dataset",
            (),
            {
                "frames": lambda self, _camera: [
                    type("Frame", (), {"index": 0, "path": tmp_path / "0.jpg"})(),
                    type("Frame", (), {"index": 5, "path": tmp_path / "5.jpg"})(),
                ]
            },
        )(),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "cv2",
        type(
            "CV2",
            (),
            {"imread": staticmethod(lambda *_args, **_kwargs: np.full((20, 20, 3), 128, dtype=np.uint8))},
        )(),
    )

    cache_dir = staged_pipeline.match_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        match_cache_root=tmp_path / "matches",
        start_frame=0,
        frame_count=30,
        frame_offsets=(5, 10, 30),
    )

    raw_path = cache_dir / "raw_matches.parquet"
    pair_metrics_path = cache_dir / "pair_metrics.csv"
    retained_path = cache_dir / "retained_matches.parquet"
    manifest_path = cache_dir / "manifest.json"
    cached_path = (
        tmp_path
        / "matches"
        / match_cache_key(
            backend="loma",
            image_size=1600,
            max_keypoints=2048,
            score_threshold=0.0,
            semantic_backend="mask2former",
            semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
            semantic_image_size=2048,
            semantic_dilation_pixels=5,
        )
        / "intra_camera"
        / "center_camera_fov30"
        / "offsets_5"
        / "raw_matches.parquet"
    )
    exported = pd.read_parquet(raw_path)
    assert raw_path.exists()
    assert pair_metrics_path.exists()
    assert retained_path.exists()
    assert manifest_path.exists()
    assert cached_path.exists()
    assert exported["backend"].tolist() == ["loma"]


def test_run_staged_raw_matching_uses_batch_backend_and_persists_pair_metadata(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_paths = [tmp_path / f"frame_{idx}.jpg" for idx in range(3)]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_paths[0]},
                )(),
                type(
                    "Frame",
                    (),
                    {"index": 1, "timestamp_ns": 200, "path": frame_paths[1]},
                )(),
                type(
                    "Frame",
                    (),
                    {"index": 2, "timestamp_ns": 300, "path": frame_paths[2]},
                )(),
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubMasks:
        def __init__(self, *_args, **_kwargs):
            self.paths = {100: tmp_path / "100.png", 200: tmp_path / "200.png", 300: tmp_path / "300.png"}

        def load(self, _frame):
            return np.zeros((4, 4), dtype=np.uint8)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_batch_size": 4,
                },
            )(),
        },
    )()

    seen = {}

    def fake_run_batch(payloads):
        seen["batch_size"] = len(payloads)
        return [
            pd.DataFrame(
                {
                    "x1": [1.0],
                    "y1": [2.0],
                    "x2": [3.0],
                    "y2": [4.0],
                    "nx1": [0.1],
                    "ny1": [0.2],
                    "nx2": [0.3],
                    "ny2": [0.4],
                    "backend": ["loma"],
                    "backend_score": [0.9],
                }
            ),
            pd.DataFrame(
                {
                    "x1": [5.0],
                    "y1": [6.0],
                    "x2": [7.0],
                    "y2": [8.0],
                    "nx1": [0.5],
                    "ny1": [0.6],
                    "nx2": [0.7],
                    "ny2": [0.8],
                    "backend": ["loma"],
                    "backend_score": [0.8],
                }
            ),
        ]

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(staged_pipeline, "SemanticMaskIndex", StubMasks)
    monkeypatch.setattr(staged_pipeline, "run_match_backend_batch", fake_run_batch)
    imread_calls = []

    def fake_imread(path, *_args, **_kwargs):
        imread_calls.append(str(path))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        staged_pipeline,
        "cv2",
        type(
            "CV2",
            (),
            {"imread": staticmethod(fake_imread)},
        )(),
    )
    monkeypatch.setattr(
        "dongfeng_registration.image_store.cv2",
        type(
            "CV2",
            (),
            {"IMREAD_COLOR": 1, "imread": staticmethod(fake_imread)},
        )(),
    )

    result = staged_pipeline.run_staged_raw_matching(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        start_frame=0,
        frame_count=3,
        frame_offsets=(1,),
    )

    assert seen["batch_size"] == 2
    assert len(imread_calls) == 3
    assert result["camera"].tolist() == ["center_camera_fov30", "center_camera_fov30"]
    assert result["frame1"].tolist() == [0, 1]
    assert result["frame2"].tolist() == [1, 2]
    assert result["timestamp1_ns"].tolist() == [100, 200]
    assert result["timestamp2_ns"].tolist() == [200, 300]


def test_run_staged_raw_matching_reuses_preflight_cache_without_backend_call(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline
    from dongfeng_registration.clip_preflight import (
        PreflightMatchCache,
        make_preflight_pair_key,
    )

    frame_paths = [tmp_path / f"frame_{idx}.jpg" for idx in range(3)]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": idx, "timestamp_ns": 100 + idx * 100, "path": path},
                )()
                for idx, path in enumerate(frame_paths)
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubMasks:
        def __init__(self, *_args, **_kwargs):
            self.paths = {
                100: tmp_path / "100.png",
                200: tmp_path / "200.png",
                300: tmp_path / "300.png",
            }

        def load(self, _frame):
            return np.zeros((4, 4), dtype=np.uint8)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_batch_size": 4,
                    "cache_visualization_pair_ratio": 0.01,
                },
            )(),
        },
    )()

    cached_matches = pd.DataFrame(
        {
            "x1": [1.0],
            "y1": [1.0],
            "x2": [2.0],
            "y2": [2.0],
            "nx1": [0.1],
            "ny1": [0.1],
            "nx2": [0.2],
            "ny2": [0.2],
            "backend": ["loma"],
            "backend_score": [0.9],
        }
    )
    cache = PreflightMatchCache(tmp_path / "preflight_cache")
    for frame1, frame2 in ((0, 1), (1, 2)):
        cache.put(
            make_preflight_pair_key(
                pair_type="intra_camera",
                camera1="center_camera_fov30",
                camera2="center_camera_fov30",
                relation="offset_1",
                frame1=frame1,
                frame2=frame2,
            ),
            cached_matches,
            metadata={"matching_source": "preflight_loma"},
        )

    def fail_backend(*_args, **_kwargs):
        raise AssertionError("LoMA backend must not run for cached preflight pairs")

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(staged_pipeline, "SemanticMaskIndex", StubMasks)
    monkeypatch.setattr(staged_pipeline, "run_match_backend", fail_backend)
    fake_cv2 = type(
        "CV2",
        (),
        {"imread": staticmethod(lambda *_args, **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8))},
    )
    monkeypatch.setattr(staged_pipeline, "cv2", fake_cv2)
    monkeypatch.setattr("dongfeng_registration.image_store.cv2", fake_cv2)

    result = staged_pipeline.run_staged_raw_matching(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        start_frame=0,
        frame_count=3,
        frame_offsets=(1,),
        preflight_cache_root=tmp_path / "preflight_cache",
    )

    assert len(result) == 2
    assert result["matching_source"].tolist() == ["preflight_reused", "preflight_reused"]


def test_run_staged_inter_matching_reuses_preflight_cache_without_backend_call(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline
    from dongfeng_registration.clip_preflight import (
        PreflightMatchCache,
        make_preflight_pair_key,
    )

    frame_paths = {
        camera: [tmp_path / f"{camera}_{idx}.jpg" for idx in range(2)]
        for camera in ("camera_a", "camera_b")
    }
    for paths in frame_paths.values():
        for path in paths:
            path.write_bytes(b"stub")

    frames = {
        camera: [
            type(
                "Frame",
                (),
                {"index": idx, "timestamp_ns": 100 + idx * 100, "path": path},
            )()
            for idx, path in enumerate(paths)
        ]
        for camera, paths in frame_paths.items()
    }

    class StubMasks:
        def __init__(self, *_args, **_kwargs):
            self.paths = {100: tmp_path / "100.png", 200: tmp_path / "200.png"}

        def load(self, _frame):
            return np.zeros((4, 4), dtype=np.uint8)

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "matching": type(
                "Matching",
                (),
                {"backend": "loma", "deep_batch_size": 4},
            )(),
            "assumptions": {},
        },
    )()
    cached_matches = pd.DataFrame(
        {
            "x1": [1.0],
            "y1": [1.0],
            "x2": [2.0],
            "y2": [2.0],
            "nx1": [0.1],
            "ny1": [0.1],
            "nx2": [0.2],
            "ny2": [0.2],
            "backend": ["loma"],
            "backend_score": [0.9],
        }
    )
    cache = PreflightMatchCache(tmp_path / "preflight_cache")
    for frame_index in (0, 1):
        cache.put(
            make_preflight_pair_key(
                pair_type="inter_camera",
                camera1="camera_a",
                camera2="camera_b",
                relation="same_index",
                frame1=frame_index,
                frame2=frame_index,
            ),
            cached_matches,
            metadata={"matching_source": "preflight_loma"},
        )

    fake_cv2 = type(
        "CV2",
        (),
        {"imread": staticmethod(lambda *_args, **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8))},
    )

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "_frames_for_config", lambda _config, camera: frames[camera])
    monkeypatch.setattr(staged_pipeline, "_calibration_for_config", lambda *_args: object())
    monkeypatch.setattr(staged_pipeline, "SemanticMaskIndex", StubMasks)
    monkeypatch.setattr(staged_pipeline, "_navigation_for_config", lambda _config: object())
    monkeypatch.setattr(staged_pipeline, "_camera_centers_by_frame_index", lambda **kwargs: {
        int(frame.index): np.zeros(3) for frame in kwargs["frames"]
    })
    monkeypatch.setattr(staged_pipeline, "_camera_view_dirs_by_frame_index", lambda **kwargs: {
        int(frame.index): np.array([1.0, 0.0, 0.0]) for frame in kwargs["frames"]
    })
    monkeypatch.setattr(staged_pipeline, "_is_same_index_pair_allowed", lambda *_args: True)
    monkeypatch.setattr(
        staged_pipeline,
        "_declared_same_index_pair",
        lambda _assumptions, first, second: (first, second),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_config_for_inter_camera_pair",
        lambda config, *_args: config,
    )
    monkeypatch.setattr(staged_pipeline, "undistort_normalized", lambda points, _calibration: points)
    monkeypatch.setattr(staged_pipeline, "run_match_backend", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LoMA backend must not run for cached inter pairs")
    ))
    monkeypatch.setattr(staged_pipeline, "cv2", fake_cv2)
    monkeypatch.setattr("dongfeng_registration.image_store.cv2", fake_cv2)
    visualization_calls = []
    monkeypatch.setattr(
        staged_pipeline,
        "_should_export_match_visualization",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "dongfeng_registration.visualization.draw_tiled_match_overlay",
        lambda *args, **kwargs: visualization_calls.append((args, kwargs)),
    )

    result = staged_pipeline.run_staged_inter_camera_matching(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cameras=("camera_a", "camera_b"),
        start_frame=0,
        frame_count=2,
        relations=("same_index",),
        viz_cache_root=tmp_path / "match_cache",
        viz_cache_key="loma__mask2former",
        preflight_cache_root=tmp_path / "preflight_cache",
    )

    assert len(result) == 2
    assert result["matching_source"].tolist() == ["preflight_reused", "preflight_reused"]
    assert visualization_calls == []


def test_match_multicamera_dataset_stage_persists_inter_camera_cache(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
        },
    )()
    fake_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame_offset": [0],
            "frame1": [0],
            "frame2": [0],
            "timestamp1_ns": [100],
            "timestamp2_ns": [101],
            "temporal_delta_ns": [1],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "backend": ["loma"],
            "backend_score": [1.0],
        }
    )
    fake_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "raw_match_count": [1],
            "matching_time_ms": [12.5],
        }
    )
    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_inter_camera_matching",
        lambda *args, **kwargs: fake_matches,
    )
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_inter_camera_match_quality",
        lambda **kwargs: fake_matches.assign(
            pose_quality_valid=True,
            reprojection_threshold_inlier=True,
            precision_filter_valid=True,
        ),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_export_inter_camera_pair_visualizations",
        lambda **kwargs: None,
    )

    cache_dir = staged_pipeline.match_multicamera_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cameras=("left_front_camera", "right_front_camera"),
        semantic_cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        start_frame=0,
        frame_count=30,
    )

    raw_path = cache_dir / "raw_matches.parquet"
    retained_path = cache_dir / "retained_matches.parquet"
    cached_path = (
        tmp_path
        / "matches"
        / match_cache_key(
            backend="loma",
            image_size=1600,
            max_keypoints=2048,
            score_threshold=0.0,
            semantic_backend="mask2former",
            semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
            semantic_image_size=2048,
            semantic_dilation_pixels=5,
        )
        / "inter_camera"
        / "left_front_camera__right_front_camera"
        / "same_index"
        / "raw_matches.parquet"
    )
    assert raw_path.exists()
    assert retained_path.exists()
    assert cached_path.exists()


def test_match_multicamera_stage_can_defer_filtering_to_executor(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
            "assumptions": {},
        },
    )()
    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "timestamp1_ns": [100],
            "timestamp2_ns": [101],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
        }
    )
    raw_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
        }
    )

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, *args, **kwargs):
            cache_dirs = args[2]
            assert all((Path(path) / "raw_matches.parquet").exists() for path in cache_dirs)
            future = object()
            self.calls.append((fn, args, kwargs, future))
            return future

    executor = FakeExecutor()
    pending = []
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_inter_camera_matching",
        lambda **kwargs: raw_matches,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_inter_camera_match_quality",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("deferred filtering must not run in the LoMA process")
        ),
    )

    staged_pipeline.match_multicamera_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cameras=("left_front_camera", "right_front_camera"),
        semantic_cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        filter_executor=executor,
        pending_filter_jobs=pending,
    )

    assert len(executor.calls) == 1
    assert executor.calls[0][0] is staged_pipeline._filter_inter_cache_group_worker
    assert pending == [executor.calls[0][3]]


def test_match_multicamera_dataset_stage_writes_retained_inter_camera_matches(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
        },
    )()
    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera", "inter_camera"],
            "camera1": ["left_front_camera", "left_front_camera"],
            "camera2": ["right_front_camera", "right_front_camera"],
            "frame_offset": [0, 0],
            "frame1": [0, 0],
            "frame2": [0, 0],
            "timestamp1_ns": [100, 100],
            "timestamp2_ns": [101, 101],
            "temporal_delta_ns": [1, 1],
            "x1": [1.0, 2.0],
            "y1": [2.0, 3.0],
            "x2": [3.0, 4.0],
            "y2": [4.0, 5.0],
            "nx1": [0.1, 0.2],
            "ny1": [0.2, 0.3],
            "nx2": [0.3, 0.4],
            "ny2": [0.4, 0.5],
            "backend": ["loma", "loma"],
            "backend_score": [0.95, 0.40],
        }
    )
    raw_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "raw_match_count": [2],
            "matching_time_ms": [12.5],
        }
    )
    evaluated_matches = raw_matches.copy()
    evaluated_matches["pose_quality_valid"] = [True, True]
    evaluated_matches["reprojection_threshold_inlier"] = [True, True]
    evaluated_matches["precision_filter_valid"] = [True, False]
    evaluated_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "raw_match_count": [2],
            "accepted_matches": [2],
            "retained_matches": [1],
            "matching_time_ms": [12.5],
        }
    )

    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_inter_camera_matching",
        lambda *args, **kwargs: raw_matches,
    )
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_inter_camera_match_quality",
        lambda **kwargs: evaluated_matches,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_export_inter_camera_pair_visualizations",
        lambda **kwargs: None,
    )

    cache_dir = staged_pipeline.match_multicamera_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cameras=("left_front_camera", "right_front_camera"),
        semantic_cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        start_frame=0,
        frame_count=30,
    )

    retained_path = cache_dir / "retained_matches.parquet"
    retained = pd.read_parquet(retained_path)
    assert retained_path.exists()
    assert len(retained) == 1
    assert retained["backend_score"].tolist() == [0.95]


def test_build_temporal_overlap_pairs_selects_forward_baseline_candidates_for_front_to_rear_search():
    from dongfeng_registration.staged_pipeline import _build_temporal_overlap_pairs

    first_frames = [
        type("Frame", (), {"index": 0, "timestamp_ns": 100})(),
        type("Frame", (), {"index": 1, "timestamp_ns": 200})(),
    ]
    second_frames = [
        type("Frame", (), {"index": 5, "timestamp_ns": 1000})(),
        type("Frame", (), {"index": 30, "timestamp_ns": 1100})(),
        type("Frame", (), {"index": 40, "timestamp_ns": 1200})(),
    ]
    first_centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([2.0, 0.0, 0.0]),
    }
    second_centers = {
        5: np.array([5.0, 0.0, 0.0]),
        30: np.array([30.2, 0.0, 0.0]),
        40: np.array([40.0, 0.0, 0.0]),
    }
    first_dirs = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
    }
    second_dirs = {
        5: np.array([-1.0, 0.0, 0.0]),
        30: np.array([-1.0, 0.0, 0.0]),
        40: np.array([-1.0, 0.0, 0.0]),
    }

    pairs = _build_temporal_overlap_pairs(
        first_frames=first_frames,
        second_frames=second_frames,
        first_centers_by_index=first_centers,
        second_centers_by_index=second_centers,
        first_view_dirs_by_index=first_dirs,
        second_view_dirs_by_index=second_dirs,
        target_baseline_m=30.0,
        max_frame_gap=100,
        min_opposite_view_dot=-0.5,
    )

    assert pairs == [
        (0, 30),
        (1, 40),
    ]


def test_temporal_overlap_direction_is_restricted_to_front_to_rear():
    from dongfeng_registration.staged_pipeline import (
        _is_temporal_overlap_direction_allowed,
    )

    assert _is_temporal_overlap_direction_allowed(
        {}, "center_camera_fov30", "rear_camera"
    )
    assert not _is_temporal_overlap_direction_allowed(
        {}, "rear_camera", "center_camera_fov30"
    )
    assert not _is_temporal_overlap_direction_allowed(
        {}, "left_front_camera", "right_front_camera"
    )


def test_temporal_overlap_pair_can_be_explicitly_allowed_for_diagnostic_front_pair():
    from dongfeng_registration.staged_pipeline import (
        _is_temporal_overlap_direction_allowed,
    )

    assumptions = {
        "inter_camera_temporal_overlap_allowed_pairs": [
            "center_camera_fov30__left_front_camera"
        ]
    }

    assert _is_temporal_overlap_direction_allowed(
        assumptions, "center_camera_fov30", "left_front_camera"
    )


def test_build_temporal_overlap_pairs_does_not_fall_back_to_past_when_no_forward_candidate():
    from dongfeng_registration.staged_pipeline import _build_temporal_overlap_pairs

    first_frames = [
        type("Frame", (), {"index": 50, "timestamp_ns": 100})(),
    ]
    second_frames = [
        type("Frame", (), {"index": 10, "timestamp_ns": 1000})(),
        type("Frame", (), {"index": 20, "timestamp_ns": 1100})(),
    ]
    first_centers = {
        50: np.array([50.0, 0.0, 0.0]),
    }
    second_centers = {
        10: np.array([10.0, 0.0, 0.0]),
        20: np.array([20.0, 0.0, 0.0]),
    }
    first_dirs = {
        50: np.array([1.0, 0.0, 0.0]),
    }
    second_dirs = {
        10: np.array([-1.0, 0.0, 0.0]),
        20: np.array([-1.0, 0.0, 0.0]),
    }

    pairs = _build_temporal_overlap_pairs(
        first_frames=first_frames,
        second_frames=second_frames,
        first_centers_by_index=first_centers,
        second_centers_by_index=second_centers,
        first_view_dirs_by_index=first_dirs,
        second_view_dirs_by_index=second_dirs,
        target_baseline_m=30.0,
        max_frame_gap=100,
        min_opposite_view_dot=-0.5,
    )

    assert pairs == []


def test_temporal_overlap_baselines_can_be_configured_per_camera_pair():
    from dongfeng_registration.staged_pipeline import (
        _temporal_overlap_baselines_for_pair,
    )

    assumptions = {
        "inter_camera_temporal_overlap_target_baselines_m": [30.0],
        "inter_camera_temporal_overlap_target_baselines_by_pair": {
            "center_camera_fov30__rear_camera": 30.0,
            "left_front_camera__left_rear_camera": 20.0,
        },
    }

    assert _temporal_overlap_baselines_for_pair(
        assumptions, "center_camera_fov30", "rear_camera"
    ) == (30.0,)
    assert _temporal_overlap_baselines_for_pair(
        assumptions, "left_front_camera", "left_rear_camera"
    ) == (20.0,)
    assert _temporal_overlap_baselines_for_pair(
        assumptions, "right_front_camera", "right_rear_camera"
    ) == (30.0,)


def test_temporal_overlap_view_dot_threshold_can_be_configured_per_camera_pair():
    from dongfeng_registration.staged_pipeline import (
        _temporal_overlap_view_dot_threshold_for_pair,
    )

    assumptions = {
        "inter_camera_temporal_overlap_min_opposite_view_dot": -0.5,
        "inter_camera_temporal_overlap_max_view_dot_by_pair": {
            "left_front_camera__left_rear_camera": 0.5,
        },
    }

    assert _temporal_overlap_view_dot_threshold_for_pair(
        assumptions, "center_camera_fov30", "rear_camera"
    ) == -0.5
    assert _temporal_overlap_view_dot_threshold_for_pair(
        assumptions, "left_front_camera", "left_rear_camera"
    ) == 0.5


def test_camera_pair_assumption_prefers_pair_specific_override_in_both_directions():
    from dongfeng_registration.staged_pipeline import _assumption_for_camera_pair

    assumptions = {
        "ray_closest_distance_mad_k": 2.0,
        "inter_camera_ray_closest_distance_mad_k_by_pair": {
            "left_front_camera__left_rear_camera": 1.0,
        },
    }

    assert _assumption_for_camera_pair(
        assumptions,
        "ray_closest_distance_mad_k",
        "left_front_camera",
        "left_rear_camera",
        2.5,
    ) == 1.0
    assert _assumption_for_camera_pair(
        assumptions,
        "ray_closest_distance_mad_k",
        "left_rear_camera",
        "left_front_camera",
        2.5,
    ) == 1.0
    assert _assumption_for_camera_pair(
        assumptions,
        "ray_closest_distance_mad_k",
        "rear_camera",
        "left_rear_camera",
        2.5,
    ) == 2.0


def test_temporal_overlap_search_direction_can_differ_from_output_pair_name():
    from dongfeng_registration.staged_pipeline import (
        _temporal_overlap_search_direction_for_pair,
    )

    assumptions = {
        "inter_camera_temporal_overlap_search_direction_by_pair": {
            "rear_camera__left_rear_camera": "left_rear_camera__rear_camera",
        }
    }

    assert _temporal_overlap_search_direction_for_pair(
        assumptions, "rear_camera", "left_rear_camera"
    ) == ("left_rear_camera", "rear_camera")
    assert _temporal_overlap_search_direction_for_pair(
        assumptions, "center_camera_fov30", "left_front_camera"
    ) == ("center_camera_fov30", "left_front_camera")


def test_temporal_overlap_relation_label_includes_target_baseline():
    from dongfeng_registration.staged_pipeline import _temporal_overlap_relation_label

    assert _temporal_overlap_relation_label(7.0) == "temporal_overlap_7m"
    assert _temporal_overlap_relation_label(7.5) == "temporal_overlap_7p5m"


def test_match_multicamera_dataset_stage_persists_temporal_overlap_cache(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
        },
    )()
    fake_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "pair_relation": ["temporal_overlap"],
            "camera1": ["center_camera_fov30"],
            "camera2": ["rear_camera"],
            "frame_offset": [25],
            "frame1": [10],
            "frame2": [35],
            "timestamp1_ns": [100],
            "timestamp2_ns": [350],
            "temporal_delta_ns": [250],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "backend": ["loma"],
            "backend_score": [1.0],
        }
    )
    fake_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "pair_relation": ["temporal_overlap"],
            "camera1": ["center_camera_fov30"],
            "camera2": ["rear_camera"],
            "frame1": [10],
            "frame2": [35],
            "raw_match_count": [1],
            "matching_time_ms": [12.5],
        }
    )
    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_inter_camera_matching",
        lambda *args, **kwargs: fake_matches,
    )
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_inter_camera_match_quality",
        lambda **kwargs: fake_matches.assign(
            pose_quality_valid=True,
            reprojection_threshold_inlier=True,
            precision_filter_valid=True,
        ),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_export_inter_camera_pair_visualizations",
        lambda **kwargs: None,
    )

    cache_dir = staged_pipeline.match_multicamera_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        cameras=("center_camera_fov30", "rear_camera"),
        semantic_cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        start_frame=0,
        frame_count=30,
    )

    raw_path = cache_dir / "raw_matches.parquet"
    cached_path = (
        tmp_path
        / "matches"
        / match_cache_key(
            backend="loma",
            image_size=1600,
            max_keypoints=2048,
            score_threshold=0.0,
            semantic_backend="mask2former",
            semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
            semantic_image_size=2048,
            semantic_dilation_pixels=5,
        )
        / "inter_camera"
        / "center_camera_fov30__rear_camera"
        / "temporal_overlap"
        / "raw_matches.parquet"
    )
    assert raw_path.exists()
    assert cached_path.exists()


def test_match_dataset_stage_exports_intra_visualization_by_default(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
            "assumptions": {},
        },
    )()

    raw_matches = pd.DataFrame(
        {
            "camera": ["center_camera_fov30"],
            "frame_offset": [0],
            "baseline_target_m": [2.0],
            "frame1": [0],
            "frame2": [5],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
            "backend": ["loma"],
            "backend_score": [1.0],
        }
    )
    raw_matches.attrs["pair_metrics"] = pd.DataFrame({"baseline_target_m": [2.0]})
    evaluated = raw_matches.assign(
        pose_quality_valid=True,
        reprojection_threshold_inlier=True,
        precision_filter_valid=True,
    )

    exported = {"called": False}

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_raw_matching",
        lambda **kwargs: raw_matches,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_intra_camera_match_quality",
        lambda **kwargs: evaluated,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "_export_intra_camera_pair_visualizations",
        lambda **kwargs: exported.__setitem__("called", True),
    )

    staged_pipeline.match_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        target_baselines_m=(2.0,),
    )

    assert exported["called"] is True


def test_match_dataset_stage_can_defer_filtering_to_executor(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    fake_config = type(
        "Config",
        (),
        {
            "matching": type(
                "Matching",
                (),
                {
                    "backend": "loma",
                    "deep_image_size": 1600,
                    "deep_score_threshold": 0.0,
                    "deep_batch_size": 4,
                },
            )(),
            "assumptions": {},
        },
    )()
    raw_matches = pd.DataFrame(
        {
            "camera": ["center_camera_fov30"],
            "frame_offset": [5],
            "baseline_target_m": [2.0],
            "frame1": [0],
            "frame2": [5],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "nx1": [0.1],
            "ny1": [0.2],
            "nx2": [0.3],
            "ny2": [0.4],
        }
    )
    raw_matches.attrs["pair_metrics"] = pd.DataFrame(
        {"baseline_target_m": [2.0], "frame1": [0], "frame2": [5]}
    )

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, *args, **kwargs):
            raw_paths = args[2]
            assert all((Path(path) / "raw_matches.parquet").exists() for path in raw_paths)
            future = object()
            self.calls.append((fn, args, kwargs, future))
            return future

    executor = FakeExecutor()
    pending = []
    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        staged_pipeline,
        "run_staged_raw_matching",
        lambda **kwargs: raw_matches,
    )
    monkeypatch.setattr(
        staged_pipeline,
        "evaluate_intra_camera_match_quality",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("deferred filtering must not run in the LoMA process")
        ),
    )

    staged_pipeline.match_dataset_stage(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        cache_root=tmp_path / "semantic_cache",
        match_cache_root=tmp_path / "matches",
        target_baselines_m=(2.0,),
        filter_executor=executor,
        pending_filter_jobs=pending,
    )

    assert len(executor.calls) == 1
    assert executor.calls[0][0] is staged_pipeline._filter_intra_cache_group_worker
    assert pending == [executor.calls[0][3]]


def test_refresh_intra_match_cache_recomputes_retained_from_raw_without_matching(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    cache_dir = (
        tmp_path / "intra_camera" / "center_camera_fov30" / "baselines_2m"
    )
    cache_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "camera": ["center_camera_fov30", "center_camera_fov30"],
            "frame1": [0, 0],
            "frame2": [10, 10],
            "frame_offset": [10, 10],
            "baseline_target_m": [2.0, 2.0],
            "x1": [1.0, 2.0],
            "y1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "y2": [3.0, 4.0],
        }
    )
    raw.to_parquet(cache_dir / "raw_matches.parquet", index=False)
    (cache_dir / "visualizations").mkdir()
    (cache_dir / "visualizations" / "stale.txt").write_text("old", encoding="utf-8")

    def fake_evaluate(**kwargs):
        assert len(kwargs["matches"]) == 2
        return kwargs["matches"].assign(
            pose_quality_valid=[True, True],
            reprojection_threshold_inlier=[True, False],
            precision_filter_valid=[True, True],
            reprojected1_x=[1.0, 2.0],
            reprojected1_y=[1.0, 2.0],
            reprojected2_x=[3.0, 4.0],
            reprojected2_y=[3.0, 4.0],
        )

    exported = {}
    monkeypatch.setattr(staged_pipeline, "evaluate_intra_camera_match_quality", fake_evaluate)
    monkeypatch.setattr(
        staged_pipeline,
        "_export_intra_camera_pair_visualizations",
        lambda **kwargs: exported.update(kwargs),
    )

    summary = staged_pipeline.refresh_intra_match_cache(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        match_cache_root=tmp_path,
        regenerate_visualizations=True,
    )

    retained = pd.read_parquet(cache_dir / "retained_matches.parquet")
    assert len(retained) == 1
    assert not (cache_dir / "visualizations" / "stale.txt").exists()
    assert exported["cache_dir"] == cache_dir
    assert summary["processed_dirs"] == 1
    assert summary["raw_matches"] == 2
    assert summary["retained_matches"] == 1


def test_export_intra_visualizations_skips_retained_image_when_no_retained_matches(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    image_path_a = tmp_path / "a.jpg"
    image_path_b = tmp_path / "b.jpg"
    image_path_a.write_bytes(b"stub")
    image_path_b.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type("Frame", (), {"index": 100, "path": image_path_a})(),
                type("Frame", (), {"index": 108, "path": image_path_b})(),
            ]

    evaluated = pd.DataFrame(
        {
            "frame1": [100, 100],
            "frame2": [108, 108],
            "x1": [10.0, 20.0],
            "y1": [10.0, 20.0],
            "x2": [15.0, 25.0],
            "y2": [15.0, 25.0],
            "image_ransac_inlier": [True, True],
            "quality_filter_valid": [False, False],
            "pose_degraded": [False, False],
            "reprojection_rms_px": [np.nan, np.nan],
            "reprojected1_x": [10.0, 20.0],
            "reprojected1_y": [10.0, 20.0],
            "reprojected2_x": [15.0, 25.0],
            "reprojected2_y": [15.0, 25.0],
        }
    )

    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: np.zeros((64, 96, 3), dtype=np.uint8),
    )

    staged_pipeline._export_intra_camera_pair_visualizations(
        cache_dir=tmp_path / "cache",
        dataset_root=tmp_path,
        camera="front_camera_fov30",
        evaluated=evaluated,
        label="baseline_4m",
    )

    retained_images = list((tmp_path / "cache" / "visualizations" / "retained").glob("*.jpg"))
    qc_images = list(
        (tmp_path / "cache" / "visualizations" / "ransac_retained_qc").glob("*.jpg")
    )
    assert retained_images == []
    assert len(qc_images) == 1


def test_export_intra_visualization_metadata_uses_ransac_inlier_reprojection(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    image_path_a = tmp_path / "a.jpg"
    image_path_b = tmp_path / "b.jpg"
    image_path_a.write_bytes(b"stub")
    image_path_b.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type("Frame", (), {"index": 100, "path": image_path_a})(),
                type("Frame", (), {"index": 104, "path": image_path_b})(),
            ]

    evaluated = pd.DataFrame(
        {
            "frame1": [100, 100, 100],
            "frame2": [104, 104, 104],
            "x1": [10.0, 20.0, 30.0],
            "y1": [10.0, 20.0, 30.0],
            "x2": [15.0, 25.0, 35.0],
            "y2": [15.0, 25.0, 35.0],
            "image_ransac_inlier": [True, True, False],
            "quality_filter_valid": [True, False, False],
            "pose_degraded": [False, False, False],
            "reprojection_rms_px": [10.0, 30.0, 1000.0],
            "reprojected1_x": [10.0, 20.0, 30.0],
            "reprojected1_y": [10.0, 20.0, 30.0],
            "reprojected2_x": [15.0, 25.0, 35.0],
            "reprojected2_y": [15.0, 25.0, 35.0],
            "ray_closest_distance_m": [0.1, 0.3, 9.0],
            "reprojection_threshold_px": [60.0, 60.0, 60.0],
            "reprojection_threshold_mad_px": [4.0, 4.0, 4.0],
            "reprojection_threshold_median_px": [20.0, 20.0, 20.0],
            "ray_closest_distance_threshold_m": [0.42, 0.42, 0.42],
            "ray_closest_distance_mad_m": [0.05, 0.05, 0.05],
            "ray_closest_distance_median_m": [0.2, 0.2, 0.2],
        }
    )

    captured = []

    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: np.zeros((64, 96, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        staged_pipeline,
        "draw_ransac_retained_reprojection_qc",
        lambda *args, **kwargs: captured.append(kwargs),
        raising=False,
    )

    # Patch the imported visualization function by intercepting the local import target.
    import dongfeng_registration.visualization as visualization

    monkeypatch.setattr(
        visualization,
        "draw_ransac_retained_reprojection_qc",
        lambda *args, **kwargs: captured.append(kwargs),
    )

    staged_pipeline._export_intra_camera_pair_visualizations(
        cache_dir=tmp_path / "cache",
        dataset_root=tmp_path,
        camera="front_camera_fov30",
        evaluated=evaluated,
        label="baseline_2m",
    )

    retained_call = next(
        call for call in captured if str(call["path"]).endswith("_retained_matches.jpg")
    )
    metadata = retained_call["metadata"]
    assert metadata["reprojection_median_px"] == 20.0
    assert metadata["reprojection_p75_px"] == 25.0
    assert metadata["ray_gap_median_m"] == 0.2
    assert metadata["ray_gap_p75_m"] == 0.25
    assert metadata["reprojection_threshold_px"] == 60.0
    assert metadata["ray_gap_threshold_m"] == 0.42
    assert metadata["reprojection_threshold_median_px"] == 20.0
    assert metadata["reprojection_threshold_mad_px"] == 4.0
    assert metadata["ray_gap_threshold_median_m"] == 0.2
    assert metadata["ray_gap_threshold_mad_m"] == 0.05
    assert round(metadata["reprojection_threshold_calculated_px"], 4) == round(
        20.0 + 2.5 * 1.4826 * 4.0, 4
    )
    assert round(metadata["ray_gap_threshold_calculated_m"], 4) == round(
        0.2 + 2.5 * 1.4826 * 0.05, 4
    )


def test_refresh_inter_match_cache_recomputes_retained_from_raw_without_matching(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    cache_dir = (
        tmp_path
        / "inter_camera"
        / "rear_camera__right_rear_camera"
        / "temporal_overlap_7m"
    )
    cache_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "camera1": ["right_rear_camera", "right_rear_camera"],
            "camera2": ["rear_camera", "rear_camera"],
            "cache_camera1": ["rear_camera", "rear_camera"],
            "cache_camera2": ["right_rear_camera", "right_rear_camera"],
            "pair_relation": ["temporal_overlap_7m", "temporal_overlap_7m"],
            "frame1": [0, 0],
            "frame2": [15, 15],
            "x1": [1.0, 2.0],
            "y1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "y2": [3.0, 4.0],
        }
    )
    raw.to_parquet(cache_dir / "raw_matches.parquet", index=False)

    def fake_evaluate(**kwargs):
        assert len(kwargs["matches"]) == 2
        return kwargs["matches"].assign(
            pose_quality_valid=[True, True],
            reprojection_threshold_inlier=[True, True],
            precision_filter_valid=[False, True],
            reprojected1_x=[1.0, 2.0],
            reprojected1_y=[1.0, 2.0],
            reprojected2_x=[3.0, 4.0],
            reprojected2_y=[3.0, 4.0],
        )

    exported = {}
    monkeypatch.setattr(staged_pipeline, "evaluate_inter_camera_match_quality", fake_evaluate)
    monkeypatch.setattr(
        staged_pipeline,
        "_export_inter_camera_pair_visualizations",
        lambda **kwargs: exported.update(kwargs),
    )

    summary = staged_pipeline.refresh_inter_match_cache(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        match_cache_root=tmp_path,
        regenerate_visualizations=True,
    )

    retained = pd.read_parquet(cache_dir / "retained_matches.parquet")
    assert len(retained) == 1
    assert exported["first_camera"] == "rear_camera"
    assert exported["second_camera"] == "right_rear_camera"
    assert summary["processed_dirs"] == 1
    assert summary["raw_matches"] == 2
    assert summary["retained_matches"] == 1


def test_evaluate_inter_camera_match_quality_accepts_empty_matches(tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    matches = pd.DataFrame(
        columns=["camera1", "camera2", "frame1", "frame2", "x1", "y1", "x2", "y2"]
    )

    evaluated = staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=matches,
        start_frame=0,
        frame_count=10,
    )

    assert evaluated.empty
    assert isinstance(evaluated.attrs["pair_metrics"], pd.DataFrame)


def test_inter_camera_visualization_uses_frame2_for_second_camera(monkeypatch, tmp_path: Path):
    from dongfeng_registration import staged_pipeline

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, camera):
            if camera == "left_front_camera":
                return [
                    type("Frame", (), {"index": 0, "path": tmp_path / "lf_0000.jpg"})(),
                ]
            return [
                type("Frame", (), {"index": 0, "path": tmp_path / "lr_0000.jpg"})(),
                type("Frame", (), {"index": 28, "path": tmp_path / "lr_0028.jpg"})(),
            ]

    read_paths = []

    def fake_imread(path):
        read_paths.append(Path(path).name)
        return np.full((8, 8, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        "dongfeng_registration.visualization.draw_ransac_retained_reprojection_qc",
        lambda *args, **kwargs: None,
    )

    evaluated = pd.DataFrame(
        {
            "camera1": ["left_front_camera"],
            "camera2": ["left_rear_camera"],
            "frame1": [0],
            "frame2": [28],
            "x1": [1.0],
            "y1": [2.0],
            "x2": [3.0],
            "y2": [4.0],
            "reprojected1_x": [1.0],
            "reprojected1_y": [2.0],
            "reprojected2_x": [3.0],
            "reprojected2_y": [4.0],
            "precision_filter_valid": [True],
            "quality_filter_valid": [True],
            "image_ransac_inlier": [True],
        }
    )

    staged_pipeline._export_inter_camera_pair_visualizations(
        cache_dir=tmp_path / "cache",
        dataset_root=tmp_path,
        first_camera="left_front_camera",
        second_camera="left_rear_camera",
        evaluated=evaluated,
    )

    assert read_paths == ["lf_0000.jpg", "lr_0028.jpg"]


def test_match_visualization_sampling_uses_configured_ratio():
    from dongfeng_registration.staged_pipeline import _should_export_match_visualization

    selected = [
        index
        for index in range(101)
        if _should_export_match_visualization(index, 0.02)
    ]

    assert selected == [0, 50, 100]


def test_evaluate_inter_camera_match_quality_uses_unified_quality_filter(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_path},
                )()
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    class StubNavigation:
        pass

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "matching": type("Matching", (), {"backend": "loma"})(),
            "assumptions": {
                "navigation_source": "nav.csv",
                "match_quality_image_ransac_threshold_px": 50.0,
                "max_triangulated_point_distance_m": 100.0,
                "min_triangulation_angle_deg": 0.5,
                "max_ray_closest_distance_m": 0.5,
                "inter_camera_reprojection_threshold_px": 20.0,
            },
        },
    )()

    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera", "inter_camera"],
            "camera1": ["left_front_camera", "left_front_camera"],
            "camera2": ["right_front_camera", "right_front_camera"],
            "frame1": [0, 0],
            "frame2": [0, 0],
            "timestamp1_ns": [100, 100],
            "timestamp2_ns": [100, 100],
            "x1": [50.0, 60.0],
            "y1": [50.0, 60.0],
            "x2": [70.0, 80.0],
            "y2": [70.0, 80.0],
            "nx1": [0.1, 0.2],
            "ny1": [0.1, 0.2],
            "nx2": [0.1, 0.2],
            "ny2": [0.1, 0.2],
            "backend_score": [0.95, 0.40],
        }
    )
    raw_matches.attrs["pair_metrics"] = pd.DataFrame(
        {
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "raw_match_count": [2],
        }
    )

    def fake_eval_navigation(*args, **kwargs):
        group = args[0].copy()
        group["navigation_valid"] = True
        group["triangulation_valid_depth"] = True
        group["triangulated_point_distance_m"] = [10.0, 10.0]
        group["triangulation_angle_deg"] = [1.0, 1.0]
        group["ray_closest_distance_m"] = [0.1, 0.1]
        group["reprojection1_px"] = [1.0, 1.0]
        group["reprojection2_px"] = [1.0, 1.0]
        group["reprojection_rms_px"] = [1.0, 1.0]
        group["reprojected1_x"] = group["x1"]
        group["reprojected1_y"] = group["y1"]
        group["reprojected2_x"] = group["x2"]
        group["reprojected2_y"] = group["y2"]
        return group, np.eye(3), 1.0

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(
        staged_pipeline,
        "NavigationTrajectory",
        type("NavigationTrajectory", (), {"from_csv": staticmethod(lambda _path: StubNavigation())}),
    )
    monkeypatch.setattr(staged_pipeline, "_evaluate_inter_camera_navigation", fake_eval_navigation)
    monkeypatch.setattr(
        staged_pipeline,
        "cv2",
        type(
            "CV2",
            (),
            {"imread": staticmethod(lambda *_args, **_kwargs: np.full((200, 200, 3), 128, dtype=np.uint8))},
        )(),
    )

    evaluated = staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=raw_matches,
        start_frame=0,
        frame_count=1,
    )

    assert evaluated["quality_filter_valid"].tolist() == [True, True]
    assert evaluated["precision_filter_valid"].tolist() == [True, True]
    assert int(staged_pipeline._retained_mask(evaluated).sum()) == 2


def test_evaluate_inter_camera_match_quality_defaults_reprojection_threshold_to_30px(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_path},
                )()
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"],
            "camera1": ["left_front_camera"],
            "camera2": ["right_front_camera"],
            "frame1": [0],
            "frame2": [0],
            "x1": [50.0],
            "y1": [50.0],
            "x2": [70.0],
            "y2": [70.0],
            "nx1": [0.1],
            "ny1": [0.1],
            "nx2": [0.1],
            "ny2": [0.1],
            "backend_score": [0.95],
        }
    )

    def fake_eval_navigation(*args, **kwargs):
        group = args[0].copy()
        group["navigation_valid"] = True
        group["triangulation_valid_depth"] = True
        group["triangulated_point_distance_m"] = [10.0]
        group["triangulation_angle_deg"] = [1.0]
        group["ray_closest_distance_m"] = [0.1]
        group["reprojection1_px"] = [25.0]
        group["reprojection2_px"] = [25.0]
        group["reprojection_rms_px"] = [25.0]
        group["reprojected1_x"] = group["x1"]
        group["reprojected1_y"] = group["y1"]
        group["reprojected2_x"] = group["x2"]
        group["reprojected2_y"] = group["y2"]
        return group, np.eye(3), 1.0

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(
        staged_pipeline,
        "NavigationTrajectory",
        type(
            "NavigationTrajectory",
            (),
            {"from_csv": staticmethod(lambda _path: object())},
        ),
    )
    monkeypatch.setattr(staged_pipeline, "_evaluate_inter_camera_navigation", fake_eval_navigation)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: np.full((200, 200, 3), 128, dtype=np.uint8),
    )

    evaluated = staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=raw_matches,
        start_frame=0,
        frame_count=1,
    )

    assert evaluated["reprojection_threshold_inlier"].tolist() == [True]


def test_evaluate_inter_camera_match_quality_applies_default_image_ransac(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_path},
                )()
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    count = 9
    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"] * count,
            "camera1": ["left_front_camera"] * count,
            "camera2": ["right_front_camera"] * count,
            "frame1": [0] * count,
            "frame2": [0] * count,
            "x1": np.linspace(30.0, 110.0, count),
            "y1": np.linspace(40.0, 120.0, count),
            "x2": np.linspace(35.0, 115.0, count),
            "y2": np.linspace(45.0, 125.0, count),
            "nx1": np.linspace(0.1, 0.9, count),
            "ny1": np.linspace(0.1, 0.9, count),
            "nx2": np.linspace(0.1, 0.9, count),
            "ny2": np.linspace(0.1, 0.9, count),
            "backend_score": [0.95] * count,
        }
    )

    def fake_eval_navigation(*args, **kwargs):
        group = args[0].copy()
        group["navigation_valid"] = True
        group["triangulation_valid_depth"] = True
        group["triangulated_point_distance_m"] = 10.0
        group["triangulation_angle_deg"] = 1.0
        group["ray_closest_distance_m"] = 0.1
        group["reprojection1_px"] = 1.0
        group["reprojection2_px"] = 1.0
        group["reprojection_rms_px"] = 1.0
        group["reprojected1_x"] = group["x1"]
        group["reprojected1_y"] = group["y1"]
        group["reprojected2_x"] = group["x2"]
        group["reprojected2_y"] = group["y2"]
        return group, np.eye(3), 1.0

    ransac_mask = np.ones((count, 1), dtype=np.uint8)
    ransac_mask[-1, 0] = 0

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(
        staged_pipeline,
        "NavigationTrajectory",
        type(
            "NavigationTrajectory",
            (),
            {"from_csv": staticmethod(lambda _path: object())},
        ),
    )
    monkeypatch.setattr(staged_pipeline, "_evaluate_inter_camera_navigation", fake_eval_navigation)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: np.full((200, 200, 3), 128, dtype=np.uint8),
    )
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "findFundamentalMat",
        lambda *_args, **_kwargs: (np.eye(3), ransac_mask),
    )

    evaluated = staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=raw_matches,
        start_frame=0,
        frame_count=1,
    )

    assert evaluated["image_ransac_inlier"].tolist() == [True] * 8 + [False]
    assert evaluated["precision_filter_valid"].tolist() == [True] * 8 + [False]
    assert int(staged_pipeline._retained_mask(evaluated).sum()) == 8


def test_evaluate_navigation_can_use_optimized_image_camera_to_world_poses():
    from dongfeng_registration import staged_pipeline
    from dongfeng_registration.camera_models import CameraCalibration

    class StubNavigation:
        timestamps_ns = np.array([100, 200], dtype=np.int64)

        def pose_at(self, _timestamp_ns):
            return np.eye(4, dtype=np.float64)

    calibration = CameraCalibration(
        camera="camera_a",
        model="rational",
        image_size=(100, 100),
        intrinsic=np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(8, dtype=np.float64),
        sensor_to_car=np.eye(4, dtype=np.float64),
        time_lag=0.0,
        images_undistorted=True,
    )
    matches = pd.DataFrame(
        {
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "frame1": [1],
            "frame2": [2],
            "x1": [50.0],
            "y1": [50.0],
            "x2": [40.0],
            "y2": [50.0],
            "nx1": [0.0],
            "ny1": [0.0],
            "nx2": [-0.1],
            "ny2": [0.0],
        }
    )
    first_pose = np.eye(4, dtype=np.float64)
    second_pose = np.eye(4, dtype=np.float64)
    second_pose[0, 3] = 5.0

    _evaluated, _fundamental, pair_baseline_m = staged_pipeline._evaluate_inter_camera_navigation(
        matches,
        100,
        200,
        calibration,
        calibration,
        StubNavigation(),
        camera_to_world_by_key={
            ("camera_a", 1): first_pose,
            ("camera_a", 2): second_pose,
        },
    )

    assert pair_baseline_m == pytest.approx(5.0)


def test_evaluate_intra_camera_match_quality_applies_default_image_ransac(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_path},
                )(),
                type(
                    "Frame",
                    (),
                    {"index": 5, "timestamp_ns": 200, "path": frame_path},
                )(),
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    count = 9
    raw_matches = pd.DataFrame(
        {
            "camera": ["center_camera_fov30"] * count,
            "frame_offset": [5] * count,
            "frame1": [0] * count,
            "frame2": [5] * count,
            "x1": np.linspace(30.0, 110.0, count),
            "y1": np.linspace(40.0, 120.0, count),
            "x2": np.linspace(35.0, 115.0, count),
            "y2": np.linspace(45.0, 125.0, count),
            "nx1": np.linspace(0.1, 0.9, count),
            "ny1": np.linspace(0.1, 0.9, count),
            "nx2": np.linspace(0.1, 0.9, count),
            "ny2": np.linspace(0.1, 0.9, count),
            "backend_score": [0.95] * count,
        }
    )

    def fake_eval_navigation(*args, **kwargs):
        group = args[0].copy()
        group["navigation_valid"] = True
        group["triangulation_valid_depth"] = True
        group["triangulated_point_distance_m"] = 10.0
        group["triangulation_angle_deg"] = 1.0
        group["ray_closest_distance_m"] = 0.1
        group["reprojection1_px"] = 1.0
        group["reprojection2_px"] = 1.0
        group["reprojection_rms_px"] = 1.0
        group["reprojected1_x"] = group["x1"]
        group["reprojected1_y"] = group["y1"]
        group["reprojected2_x"] = group["x2"]
        group["reprojected2_y"] = group["y2"]
        return group, np.eye(3), 1.0

    ransac_mask = np.ones((count, 1), dtype=np.uint8)
    ransac_mask[-1, 0] = 0

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(
        staged_pipeline,
        "NavigationTrajectory",
        type(
            "NavigationTrajectory",
            (),
            {"from_csv": staticmethod(lambda _path: object())},
        ),
    )
    monkeypatch.setattr(staged_pipeline, "_evaluate_inter_camera_navigation", fake_eval_navigation)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "findFundamentalMat",
        lambda *_args, **_kwargs: (np.eye(3), ransac_mask),
    )

    evaluated = staged_pipeline.evaluate_intra_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=raw_matches,
        start_frame=0,
        frame_count=6,
    )

    assert evaluated["image_ransac_inlier"].tolist() == [True] * 8 + [False]
    assert evaluated["precision_filter_valid"].tolist() == [True] * 8 + [False]
    assert int(staged_pipeline._retained_mask(evaluated).sum()) == 8


def test_evaluate_inter_camera_match_quality_defaults_image_ransac_threshold_to_50px(
    monkeypatch, tmp_path: Path
):
    from dongfeng_registration import staged_pipeline

    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"stub")

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [
                type(
                    "Frame",
                    (),
                    {"index": 0, "timestamp_ns": 100, "path": frame_path},
                )()
            ]

    class StubCalibration:
        @classmethod
        def from_dataset(cls, _root, _camera):
            return cls()

    fake_config = type(
        "Config",
        (),
        {
            "dataset_root": tmp_path,
            "assumptions": {"navigation_source": "nav.csv"},
        },
    )()
    count = 8
    raw_matches = pd.DataFrame(
        {
            "pair_type": ["inter_camera"] * count,
            "camera1": ["left_front_camera"] * count,
            "camera2": ["right_front_camera"] * count,
            "frame1": [0] * count,
            "frame2": [0] * count,
            "x1": np.linspace(30.0, 100.0, count),
            "y1": np.linspace(40.0, 110.0, count),
            "x2": np.linspace(35.0, 105.0, count),
            "y2": np.linspace(45.0, 115.0, count),
            "nx1": np.linspace(0.1, 0.8, count),
            "ny1": np.linspace(0.1, 0.8, count),
            "nx2": np.linspace(0.1, 0.8, count),
            "ny2": np.linspace(0.1, 0.8, count),
            "backend_score": [0.95] * count,
        }
    )

    def fake_eval_navigation(*args, **kwargs):
        group = args[0].copy()
        group["navigation_valid"] = True
        group["triangulation_valid_depth"] = True
        group["triangulated_point_distance_m"] = 10.0
        group["triangulation_angle_deg"] = 1.0
        group["ray_closest_distance_m"] = 0.1
        group["reprojection1_px"] = 1.0
        group["reprojection2_px"] = 1.0
        group["reprojection_rms_px"] = 1.0
        group["reprojected1_x"] = group["x1"]
        group["reprojected1_y"] = group["y1"]
        group["reprojected2_x"] = group["x2"]
        group["reprojected2_y"] = group["y2"]
        return group, np.eye(3), 1.0

    seen = {}

    def fake_find_fundamental_mat(_points1, _points2, _method, threshold, confidence):
        seen["threshold"] = threshold
        seen["confidence"] = confidence
        return np.eye(3), np.ones((count, 1), dtype=np.uint8)

    monkeypatch.setattr(staged_pipeline, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(staged_pipeline, "Dataset", StubDataset)
    monkeypatch.setattr(staged_pipeline, "CameraCalibration", StubCalibration)
    monkeypatch.setattr(
        staged_pipeline,
        "NavigationTrajectory",
        type(
            "NavigationTrajectory",
            (),
            {"from_csv": staticmethod(lambda _path: object())},
        ),
    )
    monkeypatch.setattr(staged_pipeline, "_evaluate_inter_camera_navigation", fake_eval_navigation)
    monkeypatch.setattr(
        staged_pipeline.cv2,
        "imread",
        lambda *_args, **_kwargs: np.full((200, 200, 3), 128, dtype=np.uint8),
    )
    monkeypatch.setattr(staged_pipeline.cv2, "findFundamentalMat", fake_find_fundamental_mat)

    staged_pipeline.evaluate_inter_camera_match_quality(
        config_path=tmp_path / "config.yaml",
        dataset_root=tmp_path,
        matches=raw_matches,
        start_frame=0,
        frame_count=1,
    )

    assert seen["threshold"] == 50.0
    assert seen["confidence"] == 0.999
