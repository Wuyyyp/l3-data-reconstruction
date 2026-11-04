from pathlib import Path

import pandas as pd
import pytest

from dongfeng_registration.clip_preflight import (
    PreflightMatchCache,
    make_preflight_pair_key,
    sample_preflight_pairs,
    summarize_preflight_matches,
)


def test_config_defaults_to_twenty_pairs_per_relation(tmp_path: Path):
    from dongfeng_registration.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_root: {tmp_path.as_posix()}",
                f"output_root: {(tmp_path / 'outputs').as_posix()}",
                "cameras: [center]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.preflight_diagnostic.sample_pairs_per_relation == 20
    assert config.preflight_diagnostic.max_total_pairs == 800
    assert config.preflight_diagnostic.reuse_preflight_matches is True


def test_preflight_always_requires_semantic_masks_and_ignores_legacy_bypass(tmp_path: Path):
    from dongfeng_registration.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_root: {tmp_path.as_posix()}",
                f"output_root: {(tmp_path / 'outputs').as_posix()}",
                "cameras: [center]",
                "preflight_diagnostic:",
                "  run_without_semantic_mask: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert not hasattr(config.preflight_diagnostic, "run_without_semantic_mask")


def test_preflight_semantic_items_are_unique_across_relations():
    from types import SimpleNamespace

    from dongfeng_registration.clip_preflight import _sampled_semantic_items

    sampled = pd.DataFrame(
        [
            {
                "match_camera1": "front",
                "match_camera2": "left",
                "frame1": 3,
                "frame2": 3,
            },
            {
                "match_camera1": "front",
                "match_camera2": "left",
                "frame1": 3,
                "frame2": 8,
            },
        ]
    )
    frames_by_camera = {
        "front": {3: SimpleNamespace(path=Path("front-3.jpg"), timestamp_ns=103)},
        "left": {
            3: SimpleNamespace(path=Path("left-3.jpg"), timestamp_ns=203),
            8: SimpleNamespace(path=Path("left-8.jpg"), timestamp_ns=208),
        },
    }

    items = _sampled_semantic_items(sampled, frames_by_camera)

    assert items == [
        (Path("front-3.jpg"), "front", 103),
        (Path("left-3.jpg"), "left", 203),
        (Path("left-8.jpg"), "left", 208),
    ]


def test_intra_preflight_evaluation_is_split_by_camera():
    from dongfeng_registration.clip_preflight import _split_intra_preflight_matches

    matches = pd.DataFrame(
        [
            {"match_camera1": "front", "match_camera2": "front", "frame1": 0},
            {"match_camera1": "left", "match_camera2": "left", "frame1": 0},
            {"match_camera1": "front", "match_camera2": "front", "frame1": 1},
        ]
    )

    subsets = _split_intra_preflight_matches(matches)

    assert [camera for camera, _ in subsets] == ["front", "left"]
    assert [set(subset["match_camera1"]) for _, subset in subsets] == [
        {"front"},
        {"left"},
    ]


def test_preflight_pair_metrics_report_retained_values_for_quality_decision():
    from dongfeng_registration.clip_preflight import summarize_preflight_pair_metrics

    matches = pd.DataFrame(
        [
            {
                "pair_key": "pair",
                "reprojection_rms_px": 100.0,
                "ray_closest_distance_m": 2.0,
                "quality_filter_valid": False,
            },
            {
                "pair_key": "pair",
                "reprojection_rms_px": 2.0,
                "ray_closest_distance_m": 0.1,
                "quality_filter_valid": True,
            },
        ]
    )

    metrics = summarize_preflight_pair_metrics(matches).iloc[0]

    assert metrics["raw_reprojection_median"] == 51.0
    assert metrics["reprojection_median"] == 2.0
    assert metrics["raw_ray_gap_median"] == 1.05
    assert metrics["ray_gap_median"] == 0.1


def test_preflight_semantic_masks_are_reused_on_second_call(tmp_path: Path, monkeypatch):
    import cv2
    import numpy as np
    from types import SimpleNamespace

    import dongfeng_registration.semantic_masks as semantic_masks
    from dongfeng_registration.clip_preflight import _prepare_preflight_semantic_masks

    image_path = tmp_path / "front-3.jpg"
    cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    sampled = pd.DataFrame(
        [
            {
                "match_camera1": "front",
                "match_camera2": "front",
                "frame1": 3,
                "frame2": 3,
            }
        ]
    )
    frames_by_camera = {
        "front": {3: SimpleNamespace(path=image_path, timestamp_ns=103)},
    }
    base_config = SimpleNamespace(
        semantic_segmentation=SimpleNamespace(
            confidence=0.25,
            processor_image_size=16,
            dilation_pixels=1,
            batch_size=2,
            write_workers=1,
        ),
        assumptions={
            "semantic_backend_default": "mask2former",
            "semantic_mask2former_model": "test-model",
            "semantic_mask_classes": ["car"],
        },
        matching=SimpleNamespace(model_device="cpu"),
    )

    class FakePredictor:
        model_id = "test-model"

        def predict(self, image):
            return [
                semantic_masks.InstancePrediction(
                    class_name="car",
                    confidence=1.0,
                    mask=np.zeros(image.shape[:2], dtype=bool),
                )
            ]

    predictor_creations = []
    monkeypatch.setattr(
        semantic_masks,
        "create_predictor",
        lambda config: predictor_creations.append(config) or FakePredictor(),
    )

    first_masks, first_manifest = _prepare_preflight_semantic_masks(
        sampled=sampled,
        frames_by_camera=frames_by_camera,
        base_config=base_config,
        cache_root=tmp_path / "semantic_cache",
    )
    second_masks, second_manifest = _prepare_preflight_semantic_masks(
        sampled=sampled,
        frames_by_camera=frames_by_camera,
        base_config=base_config,
        cache_root=tmp_path / "semantic_cache",
    )

    assert set(first_masks) == {("front", 103)}
    assert set(second_masks) == {("front", 103)}
    assert len(predictor_creations) == 1
    assert bool(first_manifest.iloc[0]["cache_hit"]) is False
    assert bool(second_manifest.iloc[0]["cache_hit"]) is True


def test_preflight_semantic_generation_respects_batch_size(tmp_path: Path, monkeypatch):
    import cv2
    import numpy as np
    from types import SimpleNamespace

    import dongfeng_registration.semantic_masks as semantic_masks
    from dongfeng_registration.clip_preflight import _prepare_preflight_semantic_masks

    frames = {}
    rows = []
    for index in range(3):
        image_path = tmp_path / f"front-{index}.jpg"
        cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
        frames[index] = SimpleNamespace(path=image_path, timestamp_ns=100 + index)
        rows.append(
            {
                "match_camera1": "front",
                "match_camera2": "front",
                "frame1": index,
                "frame2": index,
            }
        )
    sampled = pd.DataFrame(rows)
    base_config = SimpleNamespace(
        semantic_segmentation=SimpleNamespace(
            confidence=0.25,
            image_size=32,
            processor_image_size=16,
            dilation_pixels=1,
            batch_size=2,
            write_workers=1,
            preview_ratio=0.1,
        ),
        assumptions={
            "semantic_backend_default": "mask2former",
            "semantic_mask2former_model": "test-model",
            "semantic_mask_classes": ["car"],
        },
        matching=SimpleNamespace(model_device="cpu"),
    )

    class FakePredictor:
        model_id = "test-model"

    calls = []
    preview_flags = []

    def fake_create_predictor(config):
        return FakePredictor()

    def fake_generate(items, predictor, cache, **kwargs):
        calls.append(len(items))
        preview_flags.extend(kwargs["write_preview_flags"])
        results = []
        for image_path, camera, timestamp_ns in items:
            mask_path, metadata_path, preview_path = cache.paths(
                camera, timestamp_ns, predictor.model_id
            )
            class_mask_path = cache.class_mask_path(
                camera, timestamp_ns, predictor.model_id
            )
            for path in (mask_path, class_mask_path, metadata_path, preview_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), np.zeros((8, 10), dtype=np.uint8))
            cv2.imwrite(str(class_mask_path), np.zeros((8, 10), dtype=np.uint16))
            cv2.imwrite(str(preview_path), np.zeros((8, 20, 3), dtype=np.uint8))
            metadata_path.write_text(
                __import__("json").dumps(
                    {
                        "schema_version": 4,
                        "inference_dtype": cache.config.inference_dtype,
                        "use_fast_processor": cache.config.use_fast_processor,
                        "cache_key": cache.config.cache_key(predictor.model_id),
                        "backend": cache.config.backend,
                        "camera": camera,
                        "timestamp_ns": timestamp_ns,
                        "model_name": predictor.model_id,
                        "dilation_pixels": cache.config.dilation_pixels,
                        "dynamic_classes": ["car"],
                        "instance_count": 0,
                        "dynamic_pixel_ratio": 0.0,
                        "preview_generated": True,
                    }
                ),
                encoding="utf-8",
            )
            results.append(
                SimpleNamespace(
                    mask_path=mask_path,
                    class_mask_path=class_mask_path,
                        metadata_path=metadata_path,
                    cache_hit=False,
                )
            )
        return results

    monkeypatch.setattr(semantic_masks, "create_predictor", fake_create_predictor)
    monkeypatch.setattr(semantic_masks, "generate_cached_masks_batch", fake_generate)

    _prepare_preflight_semantic_masks(
        sampled=sampled,
        frames_by_camera={"front": frames},
        base_config=base_config,
        cache_root=tmp_path / "semantic_cache",
    )

    assert calls == [2, 1]
    assert preview_flags == [True, False, False]


def test_preflight_match_backend_passes_stable_image_keys(monkeypatch, tmp_path):
    import numpy as np
    from types import SimpleNamespace

    from dongfeng_registration import match_backends
    from dongfeng_registration.clip_preflight import _run_preflight_match_backend

    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return "matches"

    monkeypatch.setattr(match_backends, "run_match_backend", fake_run)
    first = SimpleNamespace(path=tmp_path / "front-0001.jpg")
    second = SimpleNamespace(path=tmp_path / "front-0003.jpg")

    result = _run_preflight_match_backend(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        calibration="calibration",
        matching_config="matching",
        first_mask=None,
        second_mask=None,
        first_frame=first,
        second_frame=second,
    )

    assert result == "matches"
    assert calls[0][1]["first_image_key"] == str(first.path)
    assert calls[0][1]["second_image_key"] == str(second.path)


def test_sample_preflight_pairs_is_temporally_stratified_and_capped():
    records = pd.DataFrame(
        [
            {
                "group_key": "intra:center:baseline_2m",
                "pair_type": "intra_camera",
                "camera1": "center",
                "camera2": "center",
                "relation": "baseline_2m",
                "frame1": frame,
                "frame2": frame + 2,
            }
            for frame in range(0, 100, 2)
        ]
        + [
            {
                "group_key": "inter:center__left:same_index",
                "pair_type": "inter_camera",
                "camera1": "center",
                "camera2": "left",
                "relation": "same_index",
                "frame1": frame,
                "frame2": frame,
            }
            for frame in range(100)
        ]
    )

    sampled = sample_preflight_pairs(
        records,
        pairs_per_group=20,
        max_total_pairs=25,
        seed=20260704,
    )

    assert len(sampled) == 25
    assert sampled["group_key"].value_counts().to_dict() == {
        "intra:center:baseline_2m": 13,
        "inter:center__left:same_index": 12,
    }
    assert sampled.groupby("group_key")["frame1"].min().min() == 0
    assert sampled.groupby("group_key")["frame1"].max().max() >= 98


def test_preflight_pair_key_is_stable_and_distinguishes_relation():
    same = make_preflight_pair_key(
        pair_type="inter_camera",
        camera1="center",
        camera2="left",
        relation="same_index",
        frame1=10,
        frame2=10,
    )
    different = make_preflight_pair_key(
        pair_type="inter_camera",
        camera1="center",
        camera2="left",
        relation="temporal_overlap_12m",
        frame1=10,
        frame2=25,
    )

    assert same == "inter_camera|center|left|same_index|10|10"
    assert different != same


def test_preflight_match_cache_round_trip(tmp_path: Path):
    cache = PreflightMatchCache(tmp_path / "preflight")
    key = make_preflight_pair_key(
        pair_type="intra_camera",
        camera1="center",
        camera2="center",
        relation="baseline_2m",
        frame1=1,
        frame2=3,
    )
    matches = pd.DataFrame({"x1": [1.0], "y1": [2.0], "x2": [3.0], "y2": [4.0]})

    cache.put(key, matches, metadata={"matching_source": "preflight"})

    loaded = cache.get(key)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, matches)
    assert cache.metadata(key)["matching_source"] == "preflight"
    assert cache.row_count(key) == 1
    assert cache.get("missing") is None


def test_pair_metric_plan_reads_cached_row_count_from_index_entry(tmp_path: Path):
    from dongfeng_registration.clip_preflight import _pair_metric_plan

    cache = PreflightMatchCache(tmp_path / "preflight")
    key = make_preflight_pair_key(
        pair_type="intra_camera",
        camera1="center",
        camera2="center",
        relation="baseline_2m",
        frame1=1,
        frame2=3,
    )
    cache.put(
        key,
        pd.DataFrame({"x1": [1.0, 2.0], "y1": [2.0, 3.0]}),
        metadata={"matching_source": "preflight"},
    )

    plan = pd.DataFrame(
        {
            "pair_key": [key],
            "pair_type": ["intra_camera"],
            "camera1": ["center"],
            "camera2": ["center"],
            "relation": ["baseline_2m"],
        }
    )

    result = _pair_metric_plan(plan, cache)

    assert result.loc[0, "raw_match_count"] == 2


def test_summarize_preflight_matches_keeps_intra_and_inter_separate():
    matches = pd.DataFrame(
        {
            "pair_type": ["intra_camera", "intra_camera", "inter_camera"],
            "camera1": ["center", "center", "center"],
            "camera2": ["center", "center", "left"],
            "relation": ["baseline_2m", "baseline_2m", "same_index"],
            "image_ransac_inlier": [True, False, True],
            "reprojection_rms_px": [2.0, 20.0, 8.0],
            "triangulation_valid_depth": [True, True, False],
            "ray_closest_distance_m": [0.1, 0.2, 0.4],
        }
    )

    summary = summarize_preflight_matches(matches)

    assert summary["overall"]["sample_count"] == 3
    assert summary["intra_camera"]["sample_count"] == 2
    assert summary["inter_camera"]["sample_count"] == 1
    assert summary["overall"]["reprojection_median_px"] == pytest.approx(8.0)


def test_classify_preflight_quality_reports_disconnected_camera_and_high_error():
    from types import SimpleNamespace

    from dongfeng_registration.clip_preflight import classify_preflight_quality

    pair_metrics = pd.DataFrame(
        {
            "pair_key": ["a"],
            "pair_type": ["intra_camera"],
            "camera1": ["center"],
            "camera2": ["center"],
            "raw_match_count": [25],
        }
    )
    settings = SimpleNamespace(
        min_raw_matches_per_pair=20,
        min_ransac_inlier_ratio=0.2,
        min_valid_depth_ratio=0.3,
        reprojection_median_warn_px=5.0,
        reprojection_median_fail_px=20.0,
        reprojection_p95_warn_px=15.0,
        reprojection_p95_fail_px=50.0,
        min_intra_valid_pairs=1,
        min_inter_valid_pairs=0,
        min_camera_graph_degree=1,
    )
    config = SimpleNamespace(preflight_diagnostic=settings, cameras=["center", "left"])
    summary = {
        "overall": {
            "ransac_inlier_ratio": 0.5,
            "valid_depth_ratio": 0.5,
            "reprojection_median_px": 25.0,
            "reprojection_p95_px": 60.0,
                "retained_reprojection_median_px": 25.0,
                "retained_reprojection_p95_px": 60.0,
        }
    }

    result = classify_preflight_quality(
        pair_metrics=pair_metrics,
        summary=summary,
        config=config,
    )

    assert result["overall_status"] == "FAIL"
    assert "camera_graph_disconnected:left" in result["failure_reasons"]
    assert "high_overall_reprojection_median" in result["failure_reasons"]


def test_classify_preflight_tolerates_a_few_bad_pairs():
    from types import SimpleNamespace

    from dongfeng_registration.clip_preflight import classify_preflight_quality

    pair_metrics = pd.DataFrame(
        {
            "pair_key": [f"pair-{index}" for index in range(20)],
            "group_key": ["same-group"] * 20,
            "pair_type": ["intra_camera"] * 20,
            "camera1": ["center"] * 20,
            "camera2": ["center"] * 20,
            "raw_match_count": [25] * 20,
            "reprojection_median": [25.0] + [2.0] * 19,
            "reprojection_p95": [60.0] + [5.0] * 19,
        }
    )
    settings = SimpleNamespace(
        min_raw_matches_per_pair=20,
        min_ransac_inlier_ratio=0.2,
        min_valid_depth_ratio=0.3,
        reprojection_median_warn_px=5.0,
        reprojection_median_fail_px=20.0,
        reprojection_p95_warn_px=15.0,
        reprojection_p95_fail_px=50.0,
        max_bad_pair_fraction=0.25,
        min_bad_pairs_for_failure=3,
        min_intra_valid_pairs=3,
        min_inter_valid_pairs=0,
        min_camera_graph_degree=0,
    )
    config = SimpleNamespace(preflight_diagnostic=settings, cameras=["center"])
    summary = {
        "overall": {
            "ransac_inlier_ratio": 0.9,
            "valid_depth_ratio": 0.9,
            "retained_reprojection_median_px": 2.0,
            "retained_reprojection_p95_px": 5.0,
        }
    }

    result = classify_preflight_quality(
        pair_metrics=pair_metrics,
        summary=summary,
        config=config,
    )

    assert result["overall_status"] == "WARN"
    assert result["failure_reasons"] == []
    assert "relation_reprojection_median_outliers_within_tolerance" in result["warnings"]


def test_classify_preflight_tolerates_a_failing_relation_below_clip_fraction():
    from types import SimpleNamespace

    from dongfeng_registration.clip_preflight import classify_preflight_quality

    rows = []
    for relation_index in range(10):
        for pair_index in range(4):
            rows.append(
                {
                    "pair_key": f"relation-{relation_index}-pair-{pair_index}",
                    "group_key": f"relation-{relation_index}",
                    "pair_type": "inter_camera",
                    "camera1": "camera-a",
                    "camera2": f"camera-{relation_index}",
                    "raw_match_count": 25,
                    "reprojection_median": 25.0 if relation_index == 0 else 2.0,
                    "reprojection_p95": 60.0 if relation_index == 0 else 5.0,
                }
            )
    settings = SimpleNamespace(
        min_raw_matches_per_pair=20,
        min_ransac_inlier_ratio=0.2,
        min_valid_depth_ratio=0.3,
        reprojection_median_warn_px=5.0,
        reprojection_median_fail_px=20.0,
        reprojection_p95_warn_px=15.0,
        reprojection_p95_fail_px=50.0,
        max_bad_pair_fraction=0.25,
        min_bad_pairs_for_failure=3,
        min_failed_relation_fraction_for_failure=0.20,
        min_intra_valid_pairs=0,
        min_inter_valid_pairs=0,
        min_camera_graph_degree=0,
    )
    config = SimpleNamespace(
        preflight_diagnostic=settings,
        cameras=["camera-a", *[f"camera-{index}" for index in range(10)]],
    )
    summary = {
        "overall": {
            "ransac_inlier_ratio": 0.9,
            "valid_depth_ratio": 0.9,
            "retained_reprojection_median_px": 2.0,
            "retained_reprojection_p95_px": 5.0,
        }
    }

    result = classify_preflight_quality(
        pair_metrics=pd.DataFrame(rows),
        summary=summary,
        config=config,
    )

    assert result["overall_status"] == "WARN"
    assert result["failure_reasons"] == []
    assert "relation_reprojection_median_outliers_within_tolerance" in result["warnings"]


def test_cached_matches_apply_semantic_masks_after_preflight():
    from dongfeng_registration.match_backends import apply_semantic_masks

    matches = pd.DataFrame(
        {
            "x1": [0.0, 1.0],
            "y1": [0.0, 0.0],
            "x2": [0.0, 1.0],
            "y2": [0.0, 0.0],
        }
    )
    first_mask = pd.NA
    first_mask = __import__("numpy").array([[0, 255], [0, 0]], dtype="uint8")
    second_mask = __import__("numpy").zeros((2, 2), dtype="uint8")

    filtered = apply_semantic_masks(
        matches,
        first_mask=first_mask,
        second_mask=second_mask,
        first_shape=(2, 2),
        second_shape=(2, 2),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["x1"] == 0.0
