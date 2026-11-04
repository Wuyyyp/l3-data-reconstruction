import importlib
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from dongfeng_registration.camera_models import CameraCalibration
from dongfeng_registration.config import MatchingConfig
from dongfeng_registration.match_backends import (
    UNIFIED_MATCH_COLUMNS,
    normalize_match_records,
    run_match_backend,
)


def _import_deep_matchers():
    spec = importlib.util.find_spec("dongfeng_registration.deep_matchers")
    assert spec is not None, "dongfeng_registration.deep_matchers module must exist"
    return importlib.import_module("dongfeng_registration.deep_matchers")


def _dummy_calibration(width: int = 32, height: int = 24) -> CameraCalibration:
    kwargs = {
        "camera": "test",
        "model": "rational",
        "image_size": (width, height),
        "intrinsic": np.eye(3, dtype=np.float64),
        "distortion": np.zeros(8, dtype=np.float64),
        "sensor_to_car": np.eye(4, dtype=np.float64),
        "time_lag": 0.0,
    }
    if "images_undistorted" in getattr(CameraCalibration, "__dataclass_fields__", {}):
        kwargs["images_undistorted"] = False
    return CameraCalibration(
        **kwargs,
    )


def _image(width: int = 32, height: int = 24) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 60)
    return image


def test_loma_matches_passes_feature_cache_context_to_single_pair_runner(monkeypatch):
    deep_matchers = _import_deep_matchers()
    calls = []
    bundle = {
        "model": object(),
        "model_name": "fake_loma",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "device_fallback": False,
    }
    monkeypatch.setattr(deep_matchers, "_get_loma_backend_bundle", lambda *_args: bundle)

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return np.empty((0, 2)), np.empty((0, 2))

    monkeypatch.setattr(deep_matchers, "_run_loma_matching", fake_run)
    monkeypatch.setattr(
        deep_matchers,
        "loma_matches_from_points",
        lambda *_args, **_kwargs: (
            pd.DataFrame(columns=UNIFIED_MATCH_COLUMNS),
            {},
        ),
    )

    config = MatchingConfig(
        backend="loma",
        model_device="cpu",
        deep_image_size=32,
        deep_max_keypoints=32,
        loma_feature_cache_mode="model",
        loma_feature_cache_gpu_max_items=17,
    )
    feature_cache = {}
    gpu_cache = {}
    deep_matchers.loma_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        config,
        first_image_key="a.jpg",
        second_image_key="b.jpg",
        feature_cache=feature_cache,
        gpu_cache=gpu_cache,
    )

    assert len(calls) == 1
    assert calls[0]["first_image_key"] == "a.jpg"
    assert calls[0]["second_image_key"] == "b.jpg"
    assert calls[0]["feature_cache"] is feature_cache
    assert calls[0]["gpu_cache"] is gpu_cache
    assert calls[0]["cache_device"] == "model"
    assert calls[0]["cache_max_items"] == 17


def test_loma_matches_maps_cached_coordinates_back_to_original_image_shape(monkeypatch):
    deep_matchers = _import_deep_matchers()
    bundle = {
        "model": object(),
        "model_name": "fake_loma",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "device_fallback": False,
    }
    monkeypatch.setattr(deep_matchers, "_get_loma_backend_bundle", lambda *_args: bundle)
    monkeypatch.setattr(
        deep_matchers,
        "_run_loma_matching",
        lambda *_args, **_kwargs: (
            np.array([[25.0, 15.0]], dtype=np.float64),
            np.array([[30.0, 20.0]], dtype=np.float64),
        ),
    )

    matches, _diagnostics = deep_matchers.loma_matches(
        _image(width=100, height=80),
        _image(width=100, height=80),
        _dummy_calibration(width=200, height=160),
        MatchingConfig(
            backend="loma",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=100,
            deep_max_keypoints=32,
        ),
        first_original_shape=(160, 200),
        second_original_shape=(160, 200),
    )

    np.testing.assert_allclose(
        matches[["x1", "y1", "x2", "y2"]].to_numpy(),
        [[50.0, 30.0, 60.0, 40.0]],
    )


def test_loma_cached_features_reports_gpu_miss_then_hit(monkeypatch):
    deep_matchers = _import_deep_matchers()
    detect_calls = []

    def fake_detect(*_args, **_kwargs):
        detect_calls.append(1)
        return (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            24,
            32,
        )

    monkeypatch.setattr(
        deep_matchers,
        "_loma_detect_and_describe_from_cv2_image",
        fake_detect,
    )
    gpu_cache = {}
    stats = {}
    kwargs = {
        "model": object(),
        "image": _image(),
        "image_key": "frame.jpg",
        "num_keypoints": 32,
        "cache": {},
        "cache_device": "model",
        "cache_max_items": 10,
        "gpu_cache": gpu_cache,
        "model_device": "cpu",
        "cache_stats": stats,
    }

    deep_matchers._loma_cached_features(**kwargs)
    deep_matchers._loma_cached_features(**kwargs)

    assert len(detect_calls) == 1
    assert stats["feature_cache_miss_count"] == 1
    assert stats["feature_cache_hit_count"] == 1
    assert stats["feature_cache_gpu_hit_count"] == 1


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(
            max_keypoints=None,
            keypoint_threshold=None,
            filter_threshold=None,
        )
        self.keypoint_detector = types.SimpleNamespace(
            config=types.SimpleNamespace(
                max_keypoints=None,
                keypoint_threshold=None,
            )
        )
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return types.SimpleNamespace(kwargs=kwargs)


class _FakeSuperPointProcessor:
    def __init__(self, detections):
        self._detections = detections
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append(
            {
                "images": [np.array(image, copy=True) for image in images],
                "kwargs": dict(kwargs),
            }
        )
        return {"pixel_values": torch.zeros((len(images), 1, 4, 4), dtype=torch.float32)}

    def post_process_keypoint_detection(self, outputs, target_sizes):
        return self._detections


class _FakeLightGlueProcessor:
    def __init__(self, match_result):
        self._match_result = match_result
        self.thresholds = []
        self.calls = []

    def __call__(self, images, **kwargs):
        pair_count = len(images)
        self.calls.append(
            {
                "images": [[np.array(item, copy=True) for item in pair] for pair in images],
                "kwargs": dict(kwargs),
            }
        )
        return {"pixel_values": torch.zeros((pair_count, 2, 1, 4, 4), dtype=torch.float32)}

    def post_process_keypoint_matching(self, outputs, target_sizes, threshold=0.0):
        self.thresholds.append(threshold)
        if isinstance(self._match_result, list):
            return self._match_result
        return [self._match_result]


class _FakeEfficientLoFTRProcessor:
    def __init__(self, match_result):
        self._match_result = match_result
        self.calls = []
        self.thresholds = []

    def __call__(self, images, **kwargs):
        pair_count = len(images)
        self.calls.append(
            {
                "images": [[np.array(item, copy=True) for item in pair] for pair in images],
                "kwargs": dict(kwargs),
            }
        )
        return {"pixel_values": torch.zeros((pair_count, 2, 1, 4, 4), dtype=torch.float32)}

    def post_process_keypoint_matching(self, outputs, target_sizes, threshold=0.0):
        self.thresholds.append(threshold)
        if isinstance(self._match_result, list):
            return self._match_result
        return [self._match_result]


def _install_fake_bundle(monkeypatch, deep_matchers, *, keypoints0, keypoints1, scores):
    detections = [
        {
            "keypoints": torch.tensor(keypoints0, dtype=torch.int32),
            "scores": torch.ones((len(keypoints0),), dtype=torch.float32),
            "descriptors": torch.zeros((len(keypoints0), 2), dtype=torch.float32),
        },
        {
            "keypoints": torch.tensor(keypoints1, dtype=torch.int32),
            "scores": torch.ones((len(keypoints1),), dtype=torch.float32),
            "descriptors": torch.zeros((len(keypoints1), 2), dtype=torch.float32),
        },
    ]
    lightglue_processor = _FakeLightGlueProcessor(
        {
            "keypoints0": torch.tensor(keypoints0, dtype=torch.int32),
            "keypoints1": torch.tensor(keypoints1, dtype=torch.int32),
            "matching_scores": torch.tensor(scores, dtype=torch.float32),
        }
    )
    bundle = {
        "superpoint_model": _FakeModel(),
        "superpoint_processor": _FakeSuperPointProcessor(detections),
        "lightglue_model": _FakeModel(),
        "lightglue_processor": lightglue_processor,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "device_fallback": False,
    }
    monkeypatch.setattr(deep_matchers, "_get_backend_bundle", lambda *args, **kwargs: bundle)
    return bundle


def _install_fake_batch_bundle(monkeypatch, deep_matchers, *, pair_specs):
    detections = []
    for spec in pair_specs:
        detections.append(
            {
                "keypoints": torch.tensor(spec["keypoints0"], dtype=torch.int32),
                "scores": torch.ones((len(spec["keypoints0"]),), dtype=torch.float32),
                "descriptors": torch.zeros((len(spec["keypoints0"]), 2), dtype=torch.float32),
            }
        )
        detections.append(
            {
                "keypoints": torch.tensor(spec["keypoints1"], dtype=torch.int32),
                "scores": torch.ones((len(spec["keypoints1"]),), dtype=torch.float32),
                "descriptors": torch.zeros((len(spec["keypoints1"]), 2), dtype=torch.float32),
            }
        )
    lightglue_processor = _FakeLightGlueProcessor(
        [
            {
                "keypoints0": torch.tensor(spec["keypoints0"], dtype=torch.int32),
                "keypoints1": torch.tensor(spec["keypoints1"], dtype=torch.int32),
                "matching_scores": torch.tensor(spec["scores"], dtype=torch.float32),
            }
            for spec in pair_specs
        ]
    )
    bundle = {
        "superpoint_model": _FakeModel(),
        "superpoint_processor": _FakeSuperPointProcessor(detections),
        "lightglue_model": _FakeModel(),
        "lightglue_processor": lightglue_processor,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "device_fallback": False,
    }
    monkeypatch.setattr(deep_matchers, "_get_backend_bundle", lambda *args, **kwargs: bundle)
    return bundle


def _install_fake_loftr_bundle(monkeypatch, deep_matchers, *, keypoints0, keypoints1, scores):
    processor = _FakeEfficientLoFTRProcessor(
        {
            "keypoints0": torch.tensor(keypoints0, dtype=torch.int32),
            "keypoints1": torch.tensor(keypoints1, dtype=torch.int32),
            "matching_scores": torch.tensor(scores, dtype=torch.float32),
        }
    )
    bundle = {
        "model": _FakeModel(),
        "processor": processor,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "device_fallback": False,
    }
    monkeypatch.setattr(deep_matchers, "_get_loftr_backend_bundle", lambda *args, **kwargs: bundle)
    return bundle


def test_superpoint_lightglue_matches_returns_dataframe_and_diagnostics(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[2, 3], [7, 9]],
        keypoints1=[[3, 4], [8, 10]],
        scores=[0.9, 0.6],
    )

    matches, diagnostics = deep_matchers.superpoint_lightglue_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_max_keypoints=128,
            deep_score_threshold=0.25,
        ),
    )

    assert isinstance(matches, pd.DataFrame)
    assert diagnostics["backend"] == "superpoint_lightglue"
    assert list(matches.columns) == UNIFIED_MATCH_COLUMNS
    assert matches["backend"].eq("superpoint_lightglue").all()
    assert matches["backend_score"].tolist() == pytest.approx([0.9, 0.6])


def test_superpoint_lightglue_explicitly_expects_cv2_bgr_or_grayscale_input(monkeypatch):
    deep_matchers = _import_deep_matchers()

    with pytest.raises(ValueError, match="cv2-style BGR"):
        deep_matchers.superpoint_lightglue_matches(
            np.array([1, 2, 3], dtype=np.uint8),
            _image(),
            _dummy_calibration(),
            MatchingConfig(
                backend="superpoint_lightglue",
                lk_mode="off",
                model_device="cpu",
            ),
        )


def test_superpoint_lightglue_converts_cv2_bgr_input_to_rgb_for_transformers(monkeypatch):
    deep_matchers = _import_deep_matchers()
    bundle = _install_fake_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[2, 3]],
        keypoints1=[[3, 4]],
        scores=[0.9],
    )
    first_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    second_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    first_bgr[0, 0] = (5, 17, 99)
    second_bgr[0, 0] = (7, 19, 101)

    deep_matchers.superpoint_lightglue_matches(
        first_bgr,
        second_bgr,
        _dummy_calibration(width=5, height=4),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=16,
            deep_max_keypoints=128,
            deep_score_threshold=0.0,
        ),
    )

    detection_images = bundle["superpoint_processor"].calls[0]["images"]
    matching_images = bundle["lightglue_processor"].calls[0]["images"][0]
    assert tuple(detection_images[0][0, 0]) == (99, 17, 5)
    assert tuple(detection_images[1][0, 0]) == (101, 19, 7)
    assert tuple(matching_images[0][0, 0]) == (99, 17, 5)
    assert tuple(matching_images[1][0, 0]) == (101, 19, 7)


def test_superpoint_lightglue_matches_applies_semantic_filtering(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[1, 1], [5, 5], [8, 8]],
        keypoints1=[[1, 1], [5, 5], [8, 8]],
        scores=[0.9, 0.8, 0.7],
    )
    source_mask = np.zeros((24, 32), dtype=np.uint8)
    destination_mask = np.zeros((24, 32), dtype=np.uint8)
    source_mask[5, 5] = 255
    destination_mask[8, 8] = 255

    matches, diagnostics = deep_matchers.superpoint_lightglue_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_max_keypoints=128,
            deep_score_threshold=0.0,
        ),
        first_dynamic_mask=source_mask,
        second_dynamic_mask=destination_mask,
    )

    assert len(matches) == 1
    assert matches.loc[0, "x1"] == 1
    assert matches.loc[0, "x2"] == 1
    assert diagnostics["semantic_source_rejected"] == 1
    assert diagnostics["semantic_destination_rejected"] == 1
    assert diagnostics["post_semantic_matches"] == 1


def test_superpoint_lightglue_matches_accept_three_channel_semantic_masks(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[1, 1], [5, 5], [8, 8]],
        keypoints1=[[1, 1], [5, 5], [8, 8]],
        scores=[0.9, 0.8, 0.7],
    )
    source_mask = np.zeros((24, 32, 3), dtype=np.uint8)
    destination_mask = np.zeros((24, 32, 3), dtype=np.uint8)
    source_mask[5, 5] = (255, 255, 255)
    destination_mask[8, 8] = (255, 255, 255)

    matches, diagnostics = deep_matchers.superpoint_lightglue_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_max_keypoints=128,
            deep_score_threshold=0.0,
        ),
        first_dynamic_mask=source_mask,
        second_dynamic_mask=destination_mask,
    )

    assert len(matches) == 1
    assert matches.loc[0, "x1"] == 1
    assert diagnostics["semantic_source_rejected"] == 1
    assert diagnostics["semantic_destination_rejected"] == 1


def test_superpoint_lightglue_matches_exercises_lk_mode(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[1, 1], [2, 2]],
        keypoints1=[[10, 10], [20, 20]],
        scores=[0.95, 0.55],
    )
    calls = {"count": 0}

    def fake_lk(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            predicted = np.array([[[10.0, 10.0]], [[20.0, 20.0]]], dtype=np.float32)
            status = np.array([[1], [1]], dtype=np.uint8)
            return predicted, status, None
        returned = np.array([[[1.0, 1.0]], [[99.0, 99.0]]], dtype=np.float32)
        status = np.array([[1], [1]], dtype=np.uint8)
        return returned, status, None

    monkeypatch.setattr(deep_matchers.cv2, "calcOpticalFlowPyrLK", fake_lk)

    matches, diagnostics = deep_matchers.superpoint_lightglue_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="on",
            model_device="cpu",
            deep_image_size=64,
            deep_max_keypoints=128,
            deep_score_threshold=0.0,
            lk_forward_backward_threshold=1.5,
        ),
    )

    assert calls["count"] == 2
    assert len(matches) == 1
    assert matches["lk_valid"].all()
    assert matches.loc[0, "lk_fb_error"] == pytest.approx(0.0)
    assert diagnostics["lk_mode"] == "on"
    assert diagnostics["lk_verified_matches"] == 1
    assert diagnostics["lk_rejected"] == 1


def test_superpoint_lightglue_batch_matches_preserves_pair_order(monkeypatch):
    deep_matchers = _import_deep_matchers()
    bundle = _install_fake_batch_bundle(
        monkeypatch,
        deep_matchers,
        pair_specs=[
            {"keypoints0": [[2, 3]], "keypoints1": [[3, 4]], "scores": [0.9]},
            {"keypoints0": [[5, 6], [7, 8]], "keypoints1": [[6, 7], [8, 9]], "scores": [0.8, 0.7]},
        ],
    )
    config = MatchingConfig(
        backend="superpoint_lightglue",
        lk_mode="off",
        model_device="cpu",
        deep_image_size=64,
        deep_max_keypoints=128,
        deep_score_threshold=0.0,
    )

    results = deep_matchers.superpoint_lightglue_matches_batch(
        [
            {
                "first_image": _image(),
                "second_image": _image(),
                "calibration": _dummy_calibration(),
                "config": config,
            },
            {
                "first_image": _image(),
                "second_image": _image(),
                "calibration": _dummy_calibration(),
                "config": config,
            },
        ]
    )

    assert len(results) == 2
    assert results[0][0]["backend_score"].tolist() == pytest.approx([0.9])
    assert results[1][0]["backend_score"].tolist() == pytest.approx([0.8, 0.7])
    assert results[0][1]["raw_matches"] == 1
    assert results[1][1]["raw_matches"] == 2
    assert len(bundle["superpoint_processor"].calls) == 1
    assert len(bundle["lightglue_processor"].calls) == 1


def test_get_backend_bundle_isolates_cached_models_by_inference_settings(monkeypatch):
    deep_matchers = _import_deep_matchers()
    deep_matchers._BACKEND_CACHE.clear()

    class FakeSuperPointForKeypointDetection:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeModel()

    class FakeLightGlueForKeypointMatching:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeModel()

    fake_transformers = types.SimpleNamespace(
        SuperPointForKeypointDetection=FakeSuperPointForKeypointDetection,
        SuperPointImageProcessor=lambda: _FakeSuperPointProcessor(
            [
                {"keypoints": torch.zeros((0, 2), dtype=torch.int32), "scores": torch.zeros((0,)), "descriptors": torch.zeros((0, 2))},
                {"keypoints": torch.zeros((0, 2), dtype=torch.int32), "scores": torch.zeros((0,)), "descriptors": torch.zeros((0, 2))},
            ]
        ),
        LightGlueForKeypointMatching=FakeLightGlueForKeypointMatching,
        LightGlueImageProcessor=lambda: _FakeLightGlueProcessor(
            {"keypoints0": torch.zeros((0, 2), dtype=torch.int32), "keypoints1": torch.zeros((0, 2), dtype=torch.int32), "matching_scores": torch.zeros((0,))}
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    bundle_a = deep_matchers._get_backend_bundle(
        model_device="cpu",
        max_keypoints=128,
        score_threshold=0.1,
    )
    bundle_b = deep_matchers._get_backend_bundle(
        model_device="cpu",
        max_keypoints=512,
        score_threshold=0.25,
    )

    assert bundle_a["superpoint_model"] is not bundle_b["superpoint_model"]
    assert bundle_a["lightglue_model"] is not bundle_b["lightglue_model"]
    assert bundle_a["superpoint_model"].config.max_keypoints == 128
    assert bundle_b["superpoint_model"].config.max_keypoints == 512
    assert bundle_a["superpoint_model"].config.keypoint_threshold == pytest.approx(0.1)
    assert bundle_b["superpoint_model"].config.keypoint_threshold == pytest.approx(0.25)
    assert bundle_a["lightglue_model"].config.filter_threshold == pytest.approx(0.1)
    assert bundle_b["lightglue_model"].config.filter_threshold == pytest.approx(0.25)


def test_run_match_backend_dispatches_superpoint_lightglue_and_attaches_diagnostics(monkeypatch):
    deep_matchers = _import_deep_matchers()
    expected = normalize_match_records(
        [
            {
                "kp1": 0,
                "kp2": 0,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 0.1,
                "ny1": 0.2,
                "nx2": 0.3,
                "ny2": 0.4,
                "backend": "superpoint_lightglue",
                "backend_score": 0.75,
            }
        ]
    )

    def fake_wrapper(*args, **kwargs):
        return expected.copy(), {"backend": "superpoint_lightglue", "marker": 123}

    monkeypatch.setattr(deep_matchers, "superpoint_lightglue_matches", fake_wrapper)

    matches = run_match_backend(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="superpoint_lightglue",
            lk_mode="off",
            model_device="cpu",
        ),
    )

    assert list(matches.columns) == UNIFIED_MATCH_COLUMNS
    assert matches.attrs["backend"] == "superpoint_lightglue"
    assert matches.attrs["marker"] == 123


def test_loftr_matches_returns_normalized_dataframe_and_diagnostics(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_loftr_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[2, 3], [7, 9]],
        keypoints1=[[3, 4], [8, 10]],
        scores=[0.9, 0.6],
    )

    matches, diagnostics = deep_matchers.loftr_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="loftr",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_score_threshold=0.25,
        ),
    )

    assert isinstance(matches, pd.DataFrame)
    assert list(matches.columns) == UNIFIED_MATCH_COLUMNS
    assert diagnostics["backend"] == "loftr"
    assert matches["backend"].eq("loftr").all()
    assert matches["backend_score"].tolist() == pytest.approx([0.9, 0.6])


def test_loftr_matches_applies_semantic_filtering(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_loftr_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[1, 1], [5, 5], [8, 8]],
        keypoints1=[[1, 1], [5, 5], [8, 8]],
        scores=[0.9, 0.8, 0.7],
    )
    source_mask = np.zeros((24, 32), dtype=np.uint8)
    destination_mask = np.zeros((24, 32), dtype=np.uint8)
    source_mask[5, 5] = 255
    destination_mask[8, 8] = 255

    matches, diagnostics = deep_matchers.loftr_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="loftr",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_score_threshold=0.0,
        ),
        first_dynamic_mask=source_mask,
        second_dynamic_mask=destination_mask,
    )

    assert len(matches) == 1
    assert matches.loc[0, "x1"] == 1
    assert matches.loc[0, "x2"] == 1
    assert diagnostics["semantic_source_rejected"] == 1
    assert diagnostics["semantic_destination_rejected"] == 1
    assert diagnostics["post_semantic_matches"] == 1


def test_loftr_matches_rejects_out_of_bounds_points_via_semantic_filtering(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_loftr_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[-3, 1], [5, 5], [8, 8]],
        keypoints1=[[1, 1], [40, 5], [8, 8]],
        scores=[0.9, 0.8, 0.7],
    )
    source_mask = np.zeros((24, 32), dtype=np.uint8)
    destination_mask = np.zeros((24, 32), dtype=np.uint8)

    matches, diagnostics = deep_matchers.loftr_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="loftr",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=64,
            deep_score_threshold=0.0,
        ),
        first_dynamic_mask=source_mask,
        second_dynamic_mask=destination_mask,
    )

    assert len(matches) == 1
    assert matches.loc[0, "x1"] == 8
    assert matches.loc[0, "x2"] == 8
    assert diagnostics["semantic_source_rejected"] == 1
    assert diagnostics["semantic_destination_rejected"] == 1
    assert diagnostics["post_semantic_matches"] == 1


def test_loftr_matches_ignores_lk_mode_with_explicit_diagnostics(monkeypatch):
    deep_matchers = _import_deep_matchers()
    _install_fake_loftr_bundle(
        monkeypatch,
        deep_matchers,
        keypoints0=[[1, 1], [2, 2]],
        keypoints1=[[10, 10], [20, 20]],
        scores=[0.95, 0.55],
    )

    matches, diagnostics = deep_matchers.loftr_matches(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="loftr",
            lk_mode="on",
            model_device="cpu",
            deep_image_size=64,
            deep_score_threshold=0.0,
        ),
    )

    assert len(matches) == 2
    assert diagnostics["backend"] == "loftr"
    assert diagnostics["lk_mode"] == "off"
    assert diagnostics["lk_mode_requested"] == "on"
    assert diagnostics["lk_ignored"] is True
    assert "ignored" in diagnostics["lk_note"]
    assert diagnostics["lk_verified_matches"] == 2
    assert matches["lk_valid"].all()


def test_loma_matches_uses_no_io_tensors_and_configured_thresholds(monkeypatch):
    deep_matchers = _import_deep_matchers()
    calls = []

    class FakeDetector:
        def detect(self, batch, *, num_keypoints):
            image = batch["image"]
            calls.append(
                {
                    "stage": "detect",
                    "shape": tuple(image.shape),
                    "num_keypoints": num_keypoints,
                }
            )
            assert isinstance(image, torch.Tensor)
            assert tuple(image.shape[:2]) == (1, 3)
            assert image.shape[-1] % 8 == 0
            assert image.shape[-2] % 8 == 0
            assert max(image.shape[-2:]) <= 1024
            return {
                "keypoints": torch.tensor(
                    [[[0.0, 0.0], [0.5, -0.5]]],
                    dtype=torch.float32,
                )
            }

    class FakeDescriptor:
        def describe_keypoints(self, image, keypoints):
            calls.append(
                {
                    "stage": "describe",
                    "shape": tuple(image.shape),
                    "keypoints_shape": tuple(keypoints.shape),
                }
            )
            assert isinstance(image, torch.Tensor)
            assert tuple(image.shape) == (1, 3, 784, 784)
            return {"descriptions": torch.ones((1, keypoints.shape[1], 256))}

    class FakeLoMaModel:
        def __init__(self):
            self._detector = FakeDetector()
            self._descriptor = FakeDescriptor()

        def __call__(
            self,
            keypoints_first,
            keypoints_second,
            descriptors_first,
            descriptors_second,
        ):
            calls.append(
                {
                    "stage": "match",
                    "keypoints_first_shape": tuple(keypoints_first.shape),
                    "keypoints_second_shape": tuple(keypoints_second.shape),
                    "descriptors_first_shape": tuple(descriptors_first.shape),
                    "descriptors_second_shape": tuple(descriptors_second.shape),
                }
            )
            return {
                "scores": torch.tensor(
                    [[[0.95, 0.05], [0.05, 0.90]]],
                    dtype=torch.float32,
                )
            }

    monkeypatch.setattr(
        deep_matchers,
        "_get_loma_backend_bundle",
        lambda model_device: {
            "model": FakeLoMaModel(),
            "model_name": "fake_loma",
            "requested_device": model_device,
            "resolved_device": "cpu",
            "device_fallback": False,
        },
    )

    matches, diagnostics = deep_matchers.loma_matches(
        _image(width=32, height=24),
        _image(width=32, height=24),
        _dummy_calibration(width=32, height=24),
        MatchingConfig(
            backend="loma",
            lk_mode="off",
            model_device="cpu",
            deep_image_size=16,
            deep_max_keypoints=1024,
            deep_score_threshold=0.7,
        ),
    )

    detect_calls = [call for call in calls if call["stage"] == "detect"]
    describe_calls = [call for call in calls if call["stage"] == "describe"]
    match_calls = [call for call in calls if call["stage"] == "match"]
    assert len(detect_calls) == 2
    assert len(describe_calls) == 2
    assert len(match_calls) == 1
    assert {call["num_keypoints"] for call in detect_calls} == {1024}
    assert diagnostics["first_scale"] == pytest.approx(0.5)
    assert diagnostics["second_scale"] == pytest.approx(0.5)
    assert matches["backend"].eq("loma").all()
    assert len(matches) == 2


def test_loma_batch_reuses_features_for_same_image_key(monkeypatch):
    deep_matchers = _import_deep_matchers()
    calls = {"detect": 0, "describe": 0, "match": 0}

    class FakeDetector:
        def detect(self, batch, *, num_keypoints):
            calls["detect"] += 1
            return {"keypoints": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)}

    class FakeDescriptor:
        def describe_keypoints(self, image, keypoints):
            calls["describe"] += 1
            return {"descriptions": torch.ones((1, 1, 4), dtype=torch.float32)}

    class FakeLoMaModel:
        def __init__(self):
            self._detector = FakeDetector()
            self._descriptor = FakeDescriptor()

        def __call__(self, *args):
            calls["match"] += 1
            return {"scores": torch.tensor([[[0.9]]], dtype=torch.float32)}

    monkeypatch.setattr(
        deep_matchers,
        "_get_loma_backend_bundle",
        lambda model_device: {
            "model": FakeLoMaModel(),
            "model_name": "fake_loma",
            "requested_device": model_device,
            "resolved_device": "cpu",
            "device_fallback": False,
        },
    )
    config = MatchingConfig(
        backend="loma",
        lk_mode="off",
        model_device="cpu",
        deep_image_size=16,
        deep_max_keypoints=128,
        deep_score_threshold=0.7,
    )
    payload = {
        "first_image": _image(width=32, height=24),
        "second_image": _image(width=32, height=24),
        "first_image_key": "camera_a/0001.jpg",
        "second_image_key": "camera_a/0002.jpg",
        "calibration": _dummy_calibration(width=32, height=24),
        "config": config,
    }

    results = deep_matchers.loma_matches_batch([payload, dict(payload)])

    assert len(results) == 2
    assert calls["detect"] == 2
    assert calls["describe"] == 2
    assert calls["match"] == 2


def test_loma_batch_skips_deep_resize_for_cached_features(monkeypatch):
    deep_matchers = _import_deep_matchers()
    calls = {"detect": 0, "describe": 0, "match": 0, "resize": 0}

    class FakeDetector:
        def detect(self, batch, *, num_keypoints):
            calls["detect"] += 1
            return {"keypoints": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)}

    class FakeDescriptor:
        def describe_keypoints(self, image, keypoints):
            calls["describe"] += 1
            return {"descriptions": torch.ones((1, 1, 4), dtype=torch.float32)}

    class FakeLoMaModel:
        def __init__(self):
            self._detector = FakeDetector()
            self._descriptor = FakeDescriptor()

        def __call__(self, *args):
            calls["match"] += 1
            return {"scores": torch.tensor([[[0.9]]], dtype=torch.float32)}

    monkeypatch.setattr(
        deep_matchers,
        "_get_loma_backend_bundle",
        lambda model_device: {
            "model": FakeLoMaModel(),
            "model_name": "fake_loma",
            "requested_device": model_device,
            "resolved_device": "cpu",
            "device_fallback": False,
        },
    )
    original_resize = deep_matchers._resize_for_deep_with_scale

    def counting_resize(image, max_dimension):
        calls["resize"] += 1
        return original_resize(image, max_dimension)

    monkeypatch.setattr(
        deep_matchers,
        "_resize_for_deep_with_scale",
        counting_resize,
    )
    deep_matchers._LOMA_FEATURE_CACHE.clear()
    deep_matchers._LOMA_FEATURE_GPU_CACHE.clear()
    config = MatchingConfig(
        backend="loma",
        lk_mode="off",
        model_device="cpu",
        deep_image_size=16,
        deep_max_keypoints=128,
        deep_score_threshold=0.7,
        loma_feature_cache_mode="model",
    )
    payload = {
        "first_image": _image(width=64, height=48),
        "second_image": _image(width=64, height=48),
        "first_image_key": "camera_resize/0001.jpg",
        "second_image_key": "camera_resize/0002.jpg",
        "calibration": _dummy_calibration(width=64, height=48),
        "config": config,
    }

    results = deep_matchers.loma_matches_batch([payload, dict(payload)])

    assert len(results) == 2
    assert calls == {"detect": 2, "describe": 2, "match": 2, "resize": 2}
    deep_matchers._LOMA_FEATURE_CACHE.clear()
    deep_matchers._LOMA_FEATURE_GPU_CACHE.clear()


def test_loma_cudagraph_matcher_compiles_once_and_marks_each_call(monkeypatch):
    deep_matchers = _import_deep_matchers()
    compile_calls = []
    mark_calls = []
    inference_states = []

    class FakeModel:
        pass

    class FakeCompiled:
        def __call__(self, *args, **kwargs):
            inference_states.append(torch.is_inference_mode_enabled())
            return {"scores": torch.ones((1, 1, 1))}

    monkeypatch.setattr(
        torch,
        "compile",
        lambda model, *, backend: compile_calls.append((model, backend))
        or FakeCompiled(),
    )
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: mark_calls.append(1),
    )
    deep_matchers._LOMA_MATCHER_RUNNER_CACHE.clear()
    model = FakeModel()

    first, first_info = deep_matchers._resolve_loma_matcher_runner(
        model,
        requested_backend="cudagraphs",
        resolved_device="cuda",
    )
    second, second_info = deep_matchers._resolve_loma_matcher_runner(
        model,
        requested_backend="cudagraphs",
        resolved_device="cuda",
    )
    first(None, None, None, None)
    second(None, None, None, None)

    assert first is second
    assert compile_calls == [(model, "cudagraphs")]
    assert mark_calls == [1, 1]
    assert inference_states == [True, True]
    assert first_info["matcher_compile_backend"] == "cudagraphs"
    assert second_info["matcher_compile_fallback"] is False


def test_loma_cudagraph_matcher_falls_back_after_runtime_failure(monkeypatch):
    deep_matchers = _import_deep_matchers()
    calls = {"compiled": 0, "eager": 0}

    class FakeModel:
        def __call__(self, *args, **kwargs):
            calls["eager"] += 1
            return {"scores": torch.zeros((1, 1, 1))}

    class FailingCompiled:
        def __call__(self, *args, **kwargs):
            calls["compiled"] += 1
            raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", lambda *_args, **_kwargs: FailingCompiled())
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: None,
    )
    deep_matchers._LOMA_MATCHER_RUNNER_CACHE.clear()
    model = FakeModel()
    runner, _info = deep_matchers._resolve_loma_matcher_runner(
        model,
        requested_backend="cudagraphs",
        resolved_device="cuda:0",
    )

    with pytest.warns(RuntimeWarning, match="compile failed"):
        runner(None, None, None, None)
    runner(None, None, None, None)

    assert calls == {"compiled": 1, "eager": 2}
    assert runner.fallback is True


def test_loma_cudagraph_matcher_config_is_explicit():
    config = MatchingConfig(loma_matcher_compile_backend="cudagraphs")

    assert config.loma_matcher_compile_backend == "cudagraphs"
    assert MatchingConfig().loma_matcher_compile_backend == "off"


def test_loma_feature_cache_stores_features_on_cpu_and_moves_them_for_matching(monkeypatch):
    deep_matchers = _import_deep_matchers()
    import torch

    cached = (
        torch.ones((1, 1, 2), dtype=torch.float32),
        torch.ones((1, 1, 4), dtype=torch.float32),
        24,
        32,
    )
    monkeypatch.setattr(
        deep_matchers,
        "_loma_detect_and_describe_from_cv2_image",
        lambda *args, **kwargs: cached,
    )
    cache = deep_matchers.OrderedDict()

    first = deep_matchers._loma_cached_features(
        object(),
        _image(width=32, height=24),
        image_key="camera/0001.jpg",
        num_keypoints=128,
        cache=cache,
        cache_device="cpu",
        cache_max_items=None,
    )
    assert first[-1] is False
    assert all(
        not hasattr(value, "device") or value.device.type == "cpu"
        for value in cache[next(iter(cache))][:2]
    )

    second = deep_matchers._loma_cached_features(
        object(),
        _image(width=32, height=24),
        image_key="camera/0001.jpg",
        num_keypoints=128,
        cache=cache,
        cache_device="cpu",
        cache_max_items=None,
    )
    assert second[-1] is True

    moved = deep_matchers._move_loma_features_to_device(second[:2], "cpu")
    assert all(value.device.type == "cpu" for value in moved)


def test_loma_feature_cache_honors_configured_item_limit(monkeypatch):
    deep_matchers = _import_deep_matchers()
    import torch

    monkeypatch.setattr(
        deep_matchers,
        "_loma_detect_and_describe_from_cv2_image",
        lambda *args, **kwargs: (
            torch.zeros((1, 1, 2)),
            torch.zeros((1, 1, 4)),
            24,
            32,
        ),
    )
    cache = deep_matchers.OrderedDict()
    for image_key in ("camera/0001.jpg", "camera/0002.jpg"):
        deep_matchers._loma_cached_features(
            object(),
            _image(width=32, height=24),
            image_key=image_key,
            num_keypoints=128,
            cache=cache,
            cache_device="cpu",
            cache_max_items=1,
            cache_cpu_max_items=1,
        )
    assert list(cache)[0][0] == "camera/0002.jpg"


def test_loma_two_tier_cache_refills_gpu_from_cpu(monkeypatch):
    deep_matchers = _import_deep_matchers()
    import torch

    calls = {"detect": 0}

    def detect(*args, **kwargs):
        calls["detect"] += 1
        return (
            torch.zeros((1, 1, 2)),
            torch.zeros((1, 1, 4)),
            24,
            32,
        )

    monkeypatch.setattr(
        deep_matchers,
        "_loma_detect_and_describe_from_cv2_image",
        detect,
    )
    cpu_cache = deep_matchers.OrderedDict()
    gpu_cache = deep_matchers.OrderedDict()
    kwargs = {
        "model": object(),
        "image": _image(width=32, height=24),
        "image_key": "camera/0001.jpg",
        "num_keypoints": 128,
        "cache": cpu_cache,
        "gpu_cache": gpu_cache,
        "cache_device": "two_tier",
        "cache_max_items": 1,
        "cache_cpu_max_items": None,
        "model_device": "cpu",
    }

    first = deep_matchers._loma_cached_features(**kwargs)
    assert first[-1] is False
    assert calls["detect"] == 1
    assert len(cpu_cache) == 1
    assert len(gpu_cache) == 1

    second = deep_matchers._loma_cached_features(**kwargs)
    assert second[-1] is True
    assert calls["detect"] == 1

    gpu_cache.clear()
    third = deep_matchers._loma_cached_features(**kwargs)
    assert third[-1] is True
    assert calls["detect"] == 1
    assert len(gpu_cache) == 1


def test_real_superpoint_and_lightglue_processors_accept_backend_image_formats():
    transformers = pytest.importorskip("transformers")
    LightGlueImageProcessor = getattr(transformers, "LightGlueImageProcessor", None)
    SuperPointImageProcessor = getattr(transformers, "SuperPointImageProcessor", None)
    if LightGlueImageProcessor is None or SuperPointImageProcessor is None:
        pytest.skip("installed transformers does not provide LightGlue processors")

    deep_matchers = _import_deep_matchers()
    first_rgb = deep_matchers._as_transformers_rgb_from_cv2_image(_image(width=5, height=4))
    second_rgb = deep_matchers._as_transformers_rgb_from_cv2_image(_image(width=5, height=4))

    superpoint_batch = SuperPointImageProcessor()(
        [first_rgb, second_rgb],
        do_resize=False,
        return_tensors="pt",
    )
    lightglue_batch = LightGlueImageProcessor()(
        [[first_rgb, second_rgb]],
        do_resize=False,
        return_tensors="pt",
    )

    assert tuple(superpoint_batch["pixel_values"].shape) == (2, 3, 4, 5)
    assert tuple(lightglue_batch["pixel_values"].shape) == (1, 2, 3, 4, 5)


def test_run_loftr_matching_with_real_processor_accepts_backend_grayscale_format():
    transformers = pytest.importorskip("transformers")
    EfficientLoFTRImageProcessor = getattr(
        transformers, "EfficientLoFTRImageProcessor", None
    )
    if EfficientLoFTRImageProcessor is None:
        pytest.skip("installed transformers does not provide EfficientLoFTR processor")

    deep_matchers = _import_deep_matchers()

    class FakeEfficientLoFTRModel:
        def __call__(self, **kwargs):
            assert tuple(kwargs["pixel_values"].shape) == (1, 2, 1, 4, 5)
            return types.SimpleNamespace(
                matches=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64),
                matching_scores=torch.tensor([[[0.9, 0.8], [0.9, 0.8]]], dtype=torch.float32),
                keypoints=torch.tensor(
                    [[[[0.2, 0.25], [0.6, 0.75]], [[0.4, 0.25], [0.8, 0.75]]]],
                    dtype=torch.float32,
                ),
            )

    result = deep_matchers._run_loftr_matching(
        {
            "processor": EfficientLoFTRImageProcessor(),
            "model": FakeEfficientLoFTRModel(),
            "resolved_device": "cpu",
        },
        deep_matchers._as_gray_from_cv2_image(_image(width=5, height=4)),
        deep_matchers._as_gray_from_cv2_image(_image(width=5, height=4)),
        (4, 5),
        (4, 5),
        0.0,
    )

    assert result["matching_scores"].tolist() == pytest.approx([0.9, 0.8])
    assert result["keypoints0"].tolist() == [[1, 1], [3, 3]]
    assert result["keypoints1"].tolist() == [[2, 1], [4, 3]]


def test_run_match_backend_dispatches_loftr_and_attaches_diagnostics(monkeypatch):
    deep_matchers = _import_deep_matchers()
    expected = normalize_match_records(
        [
            {
                "kp1": 0,
                "kp2": 0,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 0.1,
                "ny1": 0.2,
                "nx2": 0.3,
                "ny2": 0.4,
                "backend": "loftr",
                "backend_score": 0.75,
            }
        ]
    )

    def fake_wrapper(*args, **kwargs):
        return expected.copy(), {"backend": "loftr", "marker": 456}

    monkeypatch.setattr(deep_matchers, "loftr_matches", fake_wrapper)

    matches = run_match_backend(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="loftr",
            lk_mode="off",
            model_device="cpu",
        ),
    )

    assert list(matches.columns) == UNIFIED_MATCH_COLUMNS
    assert matches.attrs["backend"] == "loftr"
    assert matches.attrs["marker"] == 456


def test_run_match_backend_batch_dispatches_superpoint_lightglue(monkeypatch):
    deep_matchers = _import_deep_matchers()
    expected_first = normalize_match_records(
        [
            {
                "kp1": 0,
                "kp2": 0,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 0.1,
                "ny1": 0.2,
                "nx2": 0.3,
                "ny2": 0.4,
                "backend": "superpoint_lightglue",
                "backend_score": 0.75,
            }
        ]
    )
    expected_second = expected_first.copy()

    def fake_batch(pairs):
        assert len(pairs) == 2
        return [
            (expected_first.copy(), {"backend": "superpoint_lightglue", "marker": 1}),
            (expected_second.copy(), {"backend": "superpoint_lightglue", "marker": 2}),
        ]

    monkeypatch.setattr(deep_matchers, "superpoint_lightglue_matches_batch", fake_batch)
    from dongfeng_registration.match_backends import run_match_backend_batch

    results = run_match_backend_batch(
        [
            {
                "first_image": _image(),
                "second_image": _image(),
                "calibration": _dummy_calibration(),
                "config": MatchingConfig(backend="superpoint_lightglue", lk_mode="off", model_device="cpu"),
            },
            {
                "first_image": _image(),
                "second_image": _image(),
                "calibration": _dummy_calibration(),
                "config": MatchingConfig(backend="superpoint_lightglue", lk_mode="off", model_device="cpu"),
            },
        ]
    )

    assert len(results) == 2
    assert results[0].attrs["marker"] == 1
    assert results[1].attrs["marker"] == 2


def test_run_match_backend_keeps_empty_sift_lk_outputs_on_unified_columns(monkeypatch):
    from dongfeng_registration import match_backends

    empty = pd.DataFrame()
    empty.attrs["marker"] = "empty-sift"
    monkeypatch.setattr(match_backends, "match_frame_pair", lambda *args, **kwargs: empty)

    matches = run_match_backend(
        _image(),
        _image(),
        _dummy_calibration(),
        MatchingConfig(
            backend="sift_lk",
            lk_mode="off",
            model_device="cpu",
        ),
    )

    assert matches.empty
    assert list(matches.columns) == UNIFIED_MATCH_COLUMNS
    assert matches.attrs["marker"] == "empty-sift"


def test_mask_point_sampler_reuses_resized_mask_for_same_target_shape(monkeypatch):
    deep_matchers = _import_deep_matchers()
    resize_calls = {"count": 0}
    original_resize = deep_matchers.cv2.resize

    def counted_resize(*args, **kwargs):
        resize_calls["count"] += 1
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(deep_matchers.cv2, "resize", counted_resize)
    sampler = deep_matchers.MaskPointSampler(np.zeros((8, 8), dtype=np.uint8))
    points = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float64)

    first = deep_matchers._static_points_mask(sampler, (16, 16), points)
    second = deep_matchers._static_points_mask(sampler, (16, 16), points)

    assert first.tolist() == second.tolist()
    assert resize_calls["count"] == 1
