from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import types

import cv2
import numpy as np
import torch

from dongfeng_registration.semantic_masks import (
    ClassLabelPrediction,
    DEFAULT_MASK2FORMER_MODEL,
    DynamicMaskConfig,
    InstancePrediction,
    MaskCache,
    generate_cached_masks_batch,
    build_dynamic_mask,
    build_dynamic_mask_from_class_labels,
    build_class_label_mask,
    class_preview_color_bgr,
    canonicalize_label,
    generate_cached_mask,
    load_dynamic_mask,
    predictor_uses_confidence_threshold,
    image_processor_loading_kwargs,
    resolve_model_name,
    clear_predictor_cache,
    create_predictor,
)


@dataclass
class FakePredictor:
    calls: int = 0

    @property
    def model_id(self) -> str:
        return "fake-seg-v1"

    def predict(self, image: np.ndarray) -> list[InstancePrediction]:
        self.calls += 1
        person = np.zeros(image.shape[:2], dtype=bool)
        person[2:5, 3:6] = True
        car = np.zeros(image.shape[:2], dtype=bool)
        car[6:9, 7:11] = True
        sign = np.zeros(image.shape[:2], dtype=bool)
        sign[0:2, 0:2] = True
        return [
            InstancePrediction("person", 0.9, person),
            InstancePrediction("car", 0.8, car),
            InstancePrediction("stop sign", 0.95, sign),
        ]


def test_create_predictor_reuses_runtime_equivalent_model(monkeypatch, tmp_path):
    import dongfeng_registration.semantic_masks as semantic_masks

    creations = []

    class StubPredictor:
        def __init__(self, config):
            creations.append(config)

    clear_predictor_cache()
    try:
        monkeypatch.setattr(
            semantic_masks,
            "Mask2FormerMapillaryPredictor",
            StubPredictor,
        )
        first = create_predictor(
            DynamicMaskConfig(
                cache_root=tmp_path / "preflight",
                preview_ratio=1.0,
                device="cuda:0",
            )
        )
        second = create_predictor(
            DynamicMaskConfig(
                cache_root=tmp_path / "full",
                preview_ratio=0.1,
                image_cache_max_items=5000,
                device="cuda:0",
            )
        )

        assert first is second
        assert len(creations) == 1
    finally:
        clear_predictor_cache()


def test_build_dynamic_mask_merges_only_configured_classes():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    predictions = FakePredictor().predict(image)
    mask, accepted = build_dynamic_mask(
        predictions,
        dynamic_classes=frozenset({"person", "car"}),
        confidence_threshold=0.5,
        dilation_pixels=0,
    )
    assert mask.dtype == np.uint8
    assert mask[3, 4] == 255
    assert mask[7, 9] == 255
    assert mask[0, 0] == 0
    assert [item.class_name for item in accepted] == ["person", "car"]


def test_build_class_label_mask_preserves_per_pixel_class_identity():
    person = np.zeros((6, 8), dtype=bool)
    person[1:3, 1:3] = True
    car = np.zeros((6, 8), dtype=bool)
    car[3:5, 4:7] = True

    class_mask, class_id_to_name = build_class_label_mask(
        [
            InstancePrediction("person", 1.0, person),
            InstancePrediction("car", 1.0, car),
        ]
    )

    name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}
    assert class_mask.dtype == np.uint8
    assert class_mask[1, 1] == name_to_id["person"]
    assert class_mask[3, 4] == name_to_id["car"]
    assert class_mask[0, 0] == 0


def test_dynamic_mask_is_rebuilt_exactly_from_class_labels():
    labels = np.zeros((12, 16), dtype=np.uint8)
    labels[2:5, 3:7] = 1
    labels[7:10, 9:13] = 2

    rebuilt = build_dynamic_mask_from_class_labels(
        labels,
        {1: "car", 2: "building"},
        dynamic_classes=frozenset({"car"}),
        dilation_pixels=2,
    )

    expected = np.zeros_like(labels)
    expected[2:5, 3:7] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expected = cv2.dilate(expected, kernel)
    assert np.array_equal(rebuilt, expected)


def test_class_preview_color_is_stable_across_per_image_label_ids():
    car_color = class_preview_color_bgr("car")

    assert np.array_equal(car_color, class_preview_color_bgr("car"))
    assert not np.array_equal(car_color, class_preview_color_bgr("truck"))


def test_build_dynamic_mask_accepts_mapillary_vegetation():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    vegetation = np.zeros(image.shape[:2], dtype=bool)
    vegetation[1:4, 2:6] = True
    mask, accepted = build_dynamic_mask(
        [InstancePrediction("vegetation", 0.95, vegetation)],
        dynamic_classes=frozenset({"vegetation"}),
        confidence_threshold=0.5,
        dilation_pixels=0,
    )
    assert mask[2, 3] == 255
    assert [item.class_name for item in accepted] == ["vegetation"]


def test_build_dynamic_mask_accepts_mapillary_sky():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    sky = np.zeros(image.shape[:2], dtype=bool)
    sky[0:3, :] = True
    mask, accepted = build_dynamic_mask(
        [InstancePrediction("sky", 0.95, sky)],
        dynamic_classes=frozenset({"sky"}),
        confidence_threshold=0.5,
        dilation_pixels=0,
    )
    assert mask[1, 5] == 255
    assert [item.class_name for item in accepted] == ["sky"]


def test_dynamic_mask_config_cache_key_changes_with_backend():
    first = DynamicMaskConfig(cache_root=Path("cache"), backend="mask2former")
    second = DynamicMaskConfig(cache_root=Path("cache"), backend="mask2former")
    assert first.cache_key("model-a") == second.cache_key("model-a")


def test_canonicalize_label_maps_mapillary_vegetation_and_vehicle_names():
    assert canonicalize_label("nature--vegetation") == "vegetation"
    assert canonicalize_label("object--vehicle--car") == "car"
    assert canonicalize_label("human--person") == "person"
    assert canonicalize_label("object--vehicle--trailer") == "trailer"
    assert canonicalize_label("object--vehicle--caravan") == "caravan"
    assert canonicalize_label("object--vehicle--train") == "train"
    assert canonicalize_label("animal--ground-animal") == "animal"


def test_resolve_model_name_switches_default_for_mask2former_backend():
    assert resolve_model_name("mask2former", "") == DEFAULT_MASK2FORMER_MODEL


def test_predictor_does_not_use_confidence_threshold_for_semantic_mainline():
    assert predictor_uses_confidence_threshold("mask2former") is False
    assert predictor_uses_confidence_threshold("unsupported") is False


def test_dynamic_mask_defaults_follow_mask2former_mainline():
    config = DynamicMaskConfig(cache_root=Path("cache"))
    assert config.backend == "mask2former"
    assert config.model_name == DEFAULT_MASK2FORMER_MODEL
    assert "sky" in config.dynamic_classes
    assert "animal" in config.dynamic_classes
    assert "trailer" in config.dynamic_classes
    assert "caravan" in config.dynamic_classes
    assert "train" in config.dynamic_classes
    assert not hasattr(config, "image_size")
    assert config.inference_dtype == "fp16"
    assert config.use_fast_processor is True
    assert config.batch_size == 4


def test_semantic_cache_key_separates_precision_and_processor_modes():
    base = DynamicMaskConfig(cache_root=Path("cache"))
    fp32 = DynamicMaskConfig(cache_root=Path("cache"), inference_dtype="fp32")
    slow = DynamicMaskConfig(cache_root=Path("cache"), use_fast_processor=False)

    assert base.cache_key("model") != fp32.cache_key("model")
    assert base.cache_key("model") != slow.cache_key("model")


def test_class_cache_key_does_not_change_with_dynamic_mask_policy():
    base = DynamicMaskConfig(cache_root=Path("cache"))
    changed = DynamicMaskConfig(
        cache_root=Path("cache"),
        dynamic_classes=frozenset({"car"}),
        dilation_pixels=17,
    )

    assert base.cache_key("model") == changed.cache_key("model")
    assert base.legacy_cache_key("model") != changed.legacy_cache_key("model")


def test_image_processor_loading_kwargs_support_transformers_four_and_five():
    assert image_processor_loading_kwargs(True, "4.57.6") == {
        "local_files_only": True,
        "use_fast": True,
    }
    assert image_processor_loading_kwargs(True, "5.16.1") == {
        "local_files_only": True,
        "backend": "torchvision",
    }
    assert image_processor_loading_kwargs(False, "5.16.1") == {
        "local_files_only": True,
        "backend": "pil",
    }


def test_cached_mask_is_reused_without_repeating_inference(tmp_path):
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(
        cache_root=tmp_path / "cache",
        dilation_pixels=0,
    )
    predictor = FakePredictor()
    cache = MaskCache(config)

    first = generate_cached_mask(image_path, "camera", 100, predictor, cache)
    second = generate_cached_mask(image_path, "camera", 100, predictor, cache)

    assert predictor.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.mask_path == second.mask_path
    assert first.metadata_path.exists()
    assert first.preview_path.exists()


def test_class_mask_is_the_only_persisted_semantic_mask(tmp_path):
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(cache_root=tmp_path / "cache", dilation_pixels=1)
    predictor = FakePredictor()
    cache = MaskCache(config)

    first = generate_cached_mask(image_path, "camera", 100, predictor, cache)
    second = generate_cached_mask(image_path, "camera", 100, predictor, cache)
    rebuilt = load_dynamic_mask(
        second,
        dynamic_classes=config.dynamic_classes,
        dilation_pixels=config.dilation_pixels,
    )

    assert predictor.calls == 1
    assert first.class_mask_path.exists()
    assert not first.mask_path.exists()
    assert second.cache_hit is True
    assert rebuilt.dtype == np.uint8
    assert rebuilt[3, 4] == 255
    assert not second.mask_path.exists()


def test_default_cache_validation_reuses_mask_without_hashing_source_image(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(
        cache_root=tmp_path / "cache",
        dilation_pixels=0,
    )
    predictor = FakePredictor()
    cache = MaskCache(config)

    first = generate_cached_mask(image_path, "camera", 100, predictor, cache)
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert "sha256" not in metadata["source"]

    def fail_read_bytes(self):
        raise AssertionError("default cache validation should not hash source images")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    second = generate_cached_mask(image_path, "camera", 100, predictor, cache)

    assert predictor.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_mask_cache_writes_backend_model_camera_directory_structure(tmp_path):
    image_path = tmp_path / "101.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(cache_root=tmp_path / "cache", dilation_pixels=0)
    predictor = FakePredictor()
    cache = MaskCache(config)

    result = generate_cached_mask(image_path, "center_camera_fov30", 101, predictor, cache)

    parts = result.mask_path.relative_to(config.cache_root).parts
    assert parts[0] == "backends"
    assert parts[1] == config.backend
    assert parts[3] == "center_camera_fov30"
    assert parts[4] == "masks"
    assert result.metadata_path.relative_to(config.cache_root).parts[4] == "metadata"
    assert result.preview_path.relative_to(config.cache_root).parts[4] == "previews"


def test_strict_cache_validation_is_invalidated_when_source_image_changes(tmp_path):
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(
        cache_root=tmp_path / "cache",
        dilation_pixels=0,
        source_validation="strict",
    )
    predictor = FakePredictor()
    cache = MaskCache(config)
    generate_cached_mask(image_path, "camera", 100, predictor, cache)

    cv2.imwrite(str(image_path), np.ones((20, 30, 3), dtype=np.uint8) * 255)
    generate_cached_mask(image_path, "camera", 100, predictor, cache)

    assert predictor.calls == 2


def test_generate_cached_mask_records_backend_and_classes(tmp_path):
    image_path = tmp_path / "100.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(
        cache_root=tmp_path / "cache",
        backend="mask2former",
        model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        dynamic_classes=frozenset({"person", "car", "vegetation"}),
        dilation_pixels=0,
    )
    predictor = FakePredictor()
    cache = MaskCache(config)

    result = generate_cached_mask(image_path, "camera", 100, predictor, cache)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["backend"] == "mask2former"
    assert metadata["dynamic_classes"] == ["car", "person", "vegetation"]
    class_mask = cv2.imread(str(result.class_mask_path), cv2.IMREAD_UNCHANGED)
    assert class_mask.dtype == np.uint8
    assert metadata["schema_version"] == 5
    assert "image_size" not in metadata
    assert set(metadata["class_id_to_name"].values()) == {"person", "car", "stop sign"}


def test_generate_cached_masks_batch_calls_batch_predictor_once(tmp_path):
    image_a = tmp_path / "100.jpg"
    image_b = tmp_path / "200.jpg"
    cv2.imwrite(str(image_a), np.zeros((20, 30, 3), dtype=np.uint8))
    cv2.imwrite(str(image_b), np.zeros((20, 30, 3), dtype=np.uint8))

    class BatchPredictor:
        def __init__(self):
            self.predict_calls = 0
            self.predict_batch_calls = 0

        @property
        def model_id(self) -> str:
            return "fake-batch-seg-v1"

        def predict(self, image: np.ndarray) -> list[InstancePrediction]:
            self.predict_calls += 1
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[2:5, 3:6] = True
            return [InstancePrediction("car", 0.9, mask)]

        def predict_batch(
            self, images: list[np.ndarray]
        ) -> list[list[InstancePrediction]]:
            self.predict_batch_calls += 1
            outputs = []
            for image in images:
                mask = np.zeros(image.shape[:2], dtype=bool)
                mask[2:5, 3:6] = True
                outputs.append([InstancePrediction("car", 0.9, mask)])
            return outputs

    predictor = BatchPredictor()
    cache = MaskCache(
        DynamicMaskConfig(cache_root=tmp_path / "cache", dilation_pixels=0, write_workers=2)
    )
    results = generate_cached_masks_batch(
        [
            (image_a, "camera", 100),
            (image_b, "camera", 200),
        ],
        predictor,
        cache,
    )

    assert len(results) == 2
    assert predictor.predict_batch_calls == 1
    assert predictor.predict_calls == 0
    assert [result.mask_path.stem for result in results] == ["100", "200"]
    assert all(result.class_mask_path.exists() for result in results)
    assert all(not result.mask_path.exists() for result in results)


def test_mask2former_predictor_moves_model_to_device_once(monkeypatch, tmp_path):
    from dongfeng_registration.semantic_masks import Mask2FormerMapillaryPredictor

    class FakeTensor:
        def __init__(self):
            self.to_calls: list[str] = []

        def to(self, device: str):
            self.to_calls.append(device)
            return self

    class FakeSegmentation:
        def __init__(self, shape: tuple[int, int]):
            self._array = np.ones(shape, dtype=np.int64)

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class FakeProcessor:
        last_instance = None
        from_pretrained_kwargs = None

        def __init__(self):
            self.inputs = FakeTensor()
            self.size = {"height": 384, "width": 384}
            self.image_sizes: list[tuple[int, int]] = []
            self.target_sizes: list[tuple[int, int]] = []
            FakeProcessor.last_instance = self

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.from_pretrained_kwargs = kwargs
            return cls()

        def __call__(self, *, images, return_tensors: str):
            self.image_sizes = [image.size for image in images]
            return {"pixel_values": self.inputs}

        def post_process_semantic_segmentation(self, outputs, target_sizes):
            self.target_sizes = list(target_sizes)
            return [FakeSegmentation(tuple(size)) for size in target_sizes]

    class FakeModel:
        last_instance = None
        from_pretrained_kwargs = None

        def __init__(self):
            self.to_calls: list[str] = []
            self.config = types.SimpleNamespace(id2label={1: "object--vehicle--car"})
            FakeModel.last_instance = self

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.from_pretrained_kwargs = kwargs
            return cls()

        def eval(self):
            return self

        def to(self, device: str):
            self.to_calls.append(device)
            return self

        def __call__(self, **inputs):
            return types.SimpleNamespace()

    fake_transformers = types.SimpleNamespace(
        AutoImageProcessor=FakeProcessor,
        Mask2FormerForUniversalSegmentation=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    autocast_calls = []

    class FakeAutocast:
        def __init__(self, device_type, dtype):
            autocast_calls.append((device_type, dtype))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", FakeAutocast)

    predictor = Mask2FormerMapillaryPredictor(
        DynamicMaskConfig(
            cache_root=tmp_path / "cache",
            device="cuda:0",
            processor_image_size=1024,
        )
    )
    image = np.zeros((700, 1000, 3), dtype=np.uint8)

    predictions = predictor.predict_batch([image])
    predictor.predict_batch([image])

    assert FakeModel.last_instance.to_calls == ["cuda:0"]
    assert FakeProcessor.last_instance.inputs.to_calls == ["cuda:0", "cuda:0"]
    assert FakeProcessor.last_instance.size == {"height": 1024, "width": 1024}
    assert FakeProcessor.last_instance.image_sizes == [(1000, 700)]
    assert isinstance(predictions[0], ClassLabelPrediction)
    assert predictions[0].labels.dtype == np.uint8
    assert FakeProcessor.from_pretrained_kwargs == {
        "local_files_only": True,
        "use_fast": True,
    }
    assert FakeModel.from_pretrained_kwargs == {"local_files_only": True}
    assert autocast_calls == [
        ("cuda", torch.float16),
        ("cuda", torch.float16),
    ]


def test_cached_mask_without_preview_is_reused_without_preview_file(tmp_path):
    image_path = tmp_path / "300.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(
        cache_root=tmp_path / "cache",
        dilation_pixels=0,
    )
    predictor = FakePredictor()
    cache = MaskCache(config)

    first = generate_cached_mask(
        image_path,
        "camera",
        300,
        predictor,
        cache,
        write_preview=False,
    )
    second = generate_cached_mask(
        image_path,
        "camera",
        300,
        predictor,
        cache,
        write_preview=False,
    )

    assert predictor.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert not first.preview_path.exists()


def test_missing_preview_is_rebuilt_from_class_mask_without_inference(tmp_path):
    image_path = tmp_path / "301.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    config = DynamicMaskConfig(cache_root=tmp_path / "cache", dilation_pixels=0)
    predictor = FakePredictor()
    cache = MaskCache(config)

    first = generate_cached_mask(
        image_path, "camera", 301, predictor, cache, write_preview=False
    )
    second = generate_cached_mask(
        image_path, "camera", 301, predictor, cache, write_preview=True
    )

    assert predictor.calls == 1
    assert second.cache_hit is True
    assert first.preview_path.exists()


def test_semantic_mask_index_ignores_report_config_and_wrong_precision_cache(tmp_path):
    from dongfeng_registration.match_quality import SemanticMaskIndex

    cache_root = tmp_path / "semantic_masks"
    camera = "center_camera_fov30"
    cache_key = "mask2former-swin-large-mapillary-vistas-semantic-af445a831fc47800"
    metadata_dir = cache_root / "backends" / "mask2former" / cache_key / camera / "metadata"
    mask_dir = cache_root / "backends" / "mask2former" / cache_key / camera / "masks"
    report_dir = cache_root / "reports" / camera
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    class_mask_dir = metadata_dir.parent / "class_masks"
    class_mask_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = 1772699910107617000
    (metadata_dir / f"{timestamp}.json").write_text(
        json.dumps(
            {
                "backend": "mask2former",
                "model_name": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
                "confidence_threshold": 0.25,
                "schema_version": 4,
                "inference_dtype": "fp16",
                "use_fast_processor": True,
                "dilation_pixels": 5,
                "timestamp_ns": timestamp,
                "dynamic_classes": [
                    "person",
                    "rider",
                    "bicycle",
                    "motorcycle",
                        "car",
                        "bus",
                        "truck",
                        "train",
                        "trailer",
                        "caravan",
                        "animal",
                        "vegetation",
                        "sky",
                    ],
                    "class_id_to_name": {"1": "car"},
            }
        ),
        encoding="utf-8",
    )
    class_labels = np.zeros((4, 4), dtype=np.uint8)
    class_labels[1, 1] = 1
    cv2.imwrite(
        str(class_mask_dir / f"{timestamp}.png"), class_labels
    )
    stale_metadata_dir = (
        cache_root / "backends" / "mask2former" / "zz-stale" / camera / "metadata"
    )
    stale_mask_dir = stale_metadata_dir.parent / "masks"
    stale_metadata_dir.mkdir(parents=True, exist_ok=True)
    stale_mask_dir.mkdir(parents=True, exist_ok=True)
    stale_class_mask_dir = stale_metadata_dir.parent / "class_masks"
    stale_class_mask_dir.mkdir(parents=True, exist_ok=True)
    stale_payload = json.loads((metadata_dir / f"{timestamp}.json").read_text())
    stale_payload["inference_dtype"] = "fp32"
    stale_payload["use_fast_processor"] = False
    (stale_metadata_dir / f"{timestamp}.json").write_text(
        json.dumps(stale_payload), encoding="utf-8"
    )
    cv2.imwrite(
        str(stale_mask_dir / f"{timestamp}.png"),
        np.full((4, 4), 255, dtype=np.uint8),
    )
    cv2.imwrite(
        str(stale_class_mask_dir / f"{timestamp}.png"),
        np.full((4, 4), 1, dtype=np.uint8),
    )
    (report_dir / "config.json").write_text(
        json.dumps({"backend": "mask2former", "image_size": 2048}),
        encoding="utf-8",
    )

    index = SemanticMaskIndex(
        cache_root,
        camera,
        "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        "mask2former",
        0.25,
        5,
    )

    assert list(index.paths.keys()) == [timestamp]
    assert index.paths[timestamp] == class_mask_dir / f"{timestamp}.png"
    frame = type("Frame", (), {"timestamp_ns": timestamp})()
    loaded = index.load(frame)
    assert loaded[1, 1] == 255
    assert not (mask_dir / f"{timestamp}.png").exists()
