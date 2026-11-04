from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .semantic_labels import DEFAULT_DYNAMIC_CLASSES, LABEL_ALIASES, canonicalize_label

DEFAULT_MASK2FORMER_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"


def resolve_model_name(backend: str, model_name: str) -> str:
    if not model_name:
        return DEFAULT_MASK2FORMER_MODEL
    return model_name


def predictor_uses_confidence_threshold(backend: str) -> bool:
    return False


def image_processor_loading_kwargs(
    use_fast_processor: bool,
    transformers_version: str,
) -> dict[str, object]:
    try:
        major_version = int(str(transformers_version).split(".", maxsplit=1)[0])
    except ValueError:
        major_version = 4
    if major_version >= 5:
        return {
            "local_files_only": True,
            "backend": "torchvision" if use_fast_processor else "pil",
        }
    return {
        "local_files_only": True,
        "use_fast": bool(use_fast_processor),
    }


@dataclass(frozen=True)
class DynamicMaskConfig:
    cache_root: Path
    backend: str = "mask2former"
    model_name: str = DEFAULT_MASK2FORMER_MODEL
    confidence_threshold: float = 0.25
    processor_image_size: int | None = None
    inference_dtype: str = "fp16"
    use_fast_processor: bool = True
    dilation_pixels: int = 5
    batch_size: int = 4
    write_workers: int = 1
    preview_ratio: float = 1.0
    dynamic_classes: frozenset[str] = DEFAULT_DYNAMIC_CLASSES
    device: str = "cpu"
    source_validation: str = "none"
    image_cache_mode: str = "preload"
    image_cache_enabled: bool = True
    image_cache_max_long_side: int | None = 1600
    image_cache_max_items: int | None = 1024

    def cache_key(self, model_id: str) -> str:
        payload = {
            "class_cache_schema": 5,
            "backend": self.backend,
            "model_id": model_id,
            "processor_image_size": self.processor_image_size,
            "inference_dtype": self.inference_dtype,
            "use_fast_processor": self.use_fast_processor,
            "image_cache_enabled": self.image_cache_enabled,
            "image_cache_max_long_side": self.image_cache_max_long_side,
        }
        if predictor_uses_confidence_threshold(self.backend):
            payload["confidence_threshold"] = self.confidence_threshold
        return self._cache_key_from_payload(model_id, payload)

    def legacy_cache_key(self, model_id: str) -> str:
        payload = {
            "backend": self.backend,
            "model_id": model_id,
            "processor_image_size": self.processor_image_size,
            "inference_dtype": self.inference_dtype,
            "use_fast_processor": self.use_fast_processor,
            "image_cache_enabled": self.image_cache_enabled,
            "image_cache_max_long_side": self.image_cache_max_long_side,
            "dilation_pixels": self.dilation_pixels,
            "dynamic_classes": sorted(self.dynamic_classes),
        }
        if predictor_uses_confidence_threshold(self.backend):
            payload["confidence_threshold"] = self.confidence_threshold
        return self._cache_key_from_payload(model_id, payload)

    @staticmethod
    def _cache_key_from_payload(model_id: str, payload: dict[str, object]) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{Path(model_id).stem}-{digest}"


@dataclass(frozen=True)
class InstancePrediction:
    class_name: str
    confidence: float
    mask: np.ndarray


@dataclass(frozen=True)
class ClassLabelPrediction:
    """Dense semantic result used by backends that already predict class IDs."""

    labels: np.ndarray
    class_id_to_name: dict[int, str]


class SegmentationPredictor(Protocol):
    @property
    def model_id(self) -> str: ...

    def predict(
        self, image: np.ndarray
    ) -> list[InstancePrediction] | ClassLabelPrediction: ...

    def predict_batch(
        self, images: list[np.ndarray]
    ) -> list[list[InstancePrediction] | ClassLabelPrediction]: ...


@dataclass(frozen=True)
class CachedMaskResult:
    mask_path: Path
    class_mask_path: Path
    metadata_path: Path
    preview_path: Path
    cache_hit: bool
    instance_count: int
    dynamic_pixel_ratio: float


class Mask2FormerMapillaryPredictor:
    def __init__(self, config: DynamicMaskConfig):
        try:
            import transformers
            from transformers import (
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is required for backend='mask2former'. "
                "Update the Conda environment before running this backend."
            ) from exc

        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            config.model_name,
            **image_processor_loading_kwargs(
                config.use_fast_processor,
                getattr(transformers, "__version__", "4"),
            ),
        )
        if config.processor_image_size is not None:
            size = int(config.processor_image_size)
            if size < 1:
                raise ValueError("processor_image_size must be positive")
            processor_size = getattr(self.processor, "size", None)
            if processor_size is not None and type(processor_size).__name__ == "SizeDict":
                self.processor.size = type(processor_size)(height=size, width=size)
            elif isinstance(processor_size, dict) and (
                "height" in processor_size or "width" in processor_size
            ):
                self.processor.size = {"height": size, "width": size}
            else:
                self.processor.size = {"shortest_edge": size}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            config.model_name,
            local_files_only=True,
        )
        self.model.eval()
        self.device = str(config.device or "cpu")
        self.device_type = self.device.split(":", maxsplit=1)[0].lower()
        dtype_name = str(config.inference_dtype).lower()
        if dtype_name not in {"fp32", "fp16", "bf16"}:
            raise ValueError("inference_dtype must be one of: fp32, fp16, bf16")
        self.inference_dtype = dtype_name
        if self.device != "cpu":
            self.model.to(self.device)

    @property
    def model_id(self) -> str:
        return str(self.config.model_name)

    def predict(self, image: np.ndarray) -> ClassLabelPrediction:
        return self.predict_batch([image])[0]

    def predict_batch(
        self, images: list[np.ndarray]
    ) -> list[ClassLabelPrediction]:
        from PIL import Image
        import torch

        rgb_images = [
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for image in images
        ]
        inputs = self.processor(
            images=rgb_images, return_tensors="pt"
        )
        if self.device != "cpu":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.inference_dtype)
        autocast = (
            torch.autocast(device_type=self.device_type, dtype=autocast_dtype)
            if self.device_type != "cpu" and autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(**inputs)
        segmentations = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2] for image in images]
        )
        id2label = self.model.config.id2label
        batch_predictions: list[ClassLabelPrediction] = []
        for segmentation_tensor in segmentations:
            segmentation = segmentation_tensor.cpu().numpy()
            source_ids = np.unique(segmentation)
            names = sorted(
                {
                    canonicalize_label(str(id2label.get(int(class_id), class_id)))
                    for class_id in source_ids
                }
            )
            dtype = np.uint8 if len(names) <= np.iinfo(np.uint8).max else np.uint16
            name_to_id = {name: index + 1 for index, name in enumerate(names)}
            source_to_local = np.zeros(int(source_ids.max()) + 1, dtype=dtype)
            for class_id in source_ids:
                name = canonicalize_label(str(id2label.get(int(class_id), class_id)))
                source_to_local[int(class_id)] = name_to_id[name]
            labels = source_to_local[segmentation]
            batch_predictions.append(
                ClassLabelPrediction(
                    labels=labels,
                    class_id_to_name={value: key for key, value in name_to_id.items()},
                )
            )
        return batch_predictions


_PREDICTOR_CACHE: dict[tuple[object, ...], SegmentationPredictor] = {}


def _predictor_cache_key(config: DynamicMaskConfig) -> tuple[object, ...]:
    return (
        str(config.backend),
        str(config.model_name),
        config.processor_image_size,
        str(config.inference_dtype).lower(),
        bool(config.use_fast_processor),
        str(config.device or "cpu"),
    )


def clear_predictor_cache() -> None:
    _PREDICTOR_CACHE.clear()


def create_predictor(config: DynamicMaskConfig) -> SegmentationPredictor:
    key = _predictor_cache_key(config)
    cached = _PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if config.backend == "mask2former":
        predictor = Mask2FormerMapillaryPredictor(config)
    else:
        raise ValueError(f"unsupported semantic backend: {config.backend}")
    _PREDICTOR_CACHE[key] = predictor
    return predictor


class MaskCache:
    def __init__(self, config: DynamicMaskConfig):
        self.config = config

    def base_dir(self, camera: str, model_id: str) -> Path:
        return self.base_dir_for_key(camera, self.config.cache_key(model_id))

    def base_dir_for_key(self, camera: str, cache_key: str) -> Path:
        return (
            self.config.cache_root
            / "backends"
            / self.config.backend
            / cache_key
            / camera
        )

    def paths(self, camera: str, timestamp_ns: int, model_id: str):
        base = self.base_dir(camera, model_id)
        return (
            base / "masks" / f"{timestamp_ns}.png",
            base / "metadata" / f"{timestamp_ns}.json",
            base / "previews" / f"{timestamp_ns}.jpg",
        )

    def class_mask_path(self, camera: str, timestamp_ns: int, model_id: str) -> Path:
        return self.base_dir(camera, model_id) / "class_masks" / f"{timestamp_ns}.png"

    @staticmethod
    def source_signature(path: Path, *, include_hash: bool = True) -> dict[str, int | str]:
        stat = path.stat()
        signature: dict[str, int | str] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_hash:
            signature["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return signature

    @staticmethod
    def _source_matches(
        metadata_source: object,
        image_path: Path,
        *,
        fast: bool,
    ) -> bool:
        if not isinstance(metadata_source, dict):
            return False
        if not fast:
            return metadata_source == MaskCache.source_signature(
                image_path, include_hash=True
            )
        current = MaskCache.source_signature(image_path, include_hash=False)
        for key in ("path", "size", "mtime_ns"):
            if metadata_source.get(key) != current[key]:
                return False
        return True

    def load_if_valid(
        self,
        image_path: Path,
        camera: str,
        timestamp_ns: int,
        model_id: str,
        require_preview: bool = True,
    ) -> CachedMaskResult | None:
        del require_preview
        accepted_cache_keys = (
            self.config.cache_key(model_id),
            self.config.legacy_cache_key(model_id),
        )
        backend_root = self.config.cache_root / "backends" / self.config.backend
        bases = [
            self.base_dir_for_key(camera, cache_key)
            for cache_key in dict.fromkeys(accepted_cache_keys)
        ]
        if backend_root.is_dir():
            bases.extend(
                candidate / camera
                for candidate in sorted(backend_root.iterdir())
                if candidate.is_dir()
            )
        artifacts = None
        metadata = None
        seen_bases: set[Path] = set()
        source_validation = str(self.config.source_validation).lower()
        if source_validation not in {"none", "fast", "strict"}:
            raise ValueError("source_validation must be one of: none, fast, strict")
        for base in bases:
            if base in seen_bases:
                continue
            seen_bases.add(base)
            candidate = (
                base / "masks" / f"{timestamp_ns}.png",
                base / "class_masks" / f"{timestamp_ns}.png",
                base / "metadata" / f"{timestamp_ns}.json",
                base / "previews" / f"{timestamp_ns}.jpg",
            )
            if not candidate[1].exists() or not candidate[2].exists():
                continue
            candidate_metadata = json.loads(
                candidate[2].read_text(encoding="utf-8")
            )
            if int(candidate_metadata.get("schema_version", 1)) < 4:
                continue
            if candidate_metadata.get("backend", self.config.backend) != self.config.backend:
                continue
            candidate_model = candidate_metadata.get(
                "model_id", candidate_metadata.get("model_name", "")
            )
            if Path(str(candidate_model)).name != Path(model_id).name:
                continue
            if str(candidate_metadata.get("inference_dtype", "")) != str(
                self.config.inference_dtype
            ):
                continue
            if bool(candidate_metadata.get("use_fast_processor", False)) != bool(
                self.config.use_fast_processor
            ):
                continue
            processor_size = candidate_metadata.get("processor_image_size")
            if processor_size is not None and int(processor_size) != int(
                self.config.processor_image_size or 0
            ):
                continue
            cache_key = str(candidate_metadata.get("cache_key", ""))
            if cache_key not in accepted_cache_keys:
                source = candidate_metadata.get("source", {})
                source_path = source.get("path") if isinstance(source, dict) else None
                if not source_path or Path(str(source_path)).resolve() != image_path.resolve():
                    continue
            if source_validation != "none" and not self._source_matches(
                candidate_metadata.get("source"),
                image_path,
                fast=source_validation == "fast",
            ):
                continue
            artifacts = candidate
            metadata = candidate_metadata
            break
        if artifacts is None:
            return None
        mask_path, class_mask_path, metadata_path, preview_path = artifacts
        if metadata is None:
            raise RuntimeError("semantic cache metadata resolution failed")
        mask_path.unlink(missing_ok=True)
        return CachedMaskResult(
            mask_path=mask_path,
            class_mask_path=class_mask_path,
            metadata_path=metadata_path,
            preview_path=preview_path,
            cache_hit=True,
            instance_count=int(metadata["instance_count"]),
            dynamic_pixel_ratio=float(metadata["dynamic_pixel_ratio"]),
        )


def build_dynamic_mask(
    predictions: list[InstancePrediction],
    dynamic_classes: frozenset[str],
    confidence_threshold: float,
    dilation_pixels: int,
) -> tuple[np.ndarray, list[InstancePrediction]]:
    if not predictions:
        raise ValueError("predictions must include image-sized masks")
    combined = np.zeros(predictions[0].mask.shape, dtype=np.uint8)
    accepted = []
    for prediction in predictions:
        if (
            prediction.class_name in dynamic_classes
            and prediction.confidence >= confidence_threshold
        ):
            combined[prediction.mask.astype(bool)] = 255
            accepted.append(prediction)
    if dilation_pixels > 0 and np.any(combined):
        size = 2 * dilation_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        combined = cv2.dilate(combined, kernel)
    return combined, accepted


def build_class_label_mask(
    predictions: list[InstancePrediction],
) -> tuple[np.ndarray, dict[int, str]]:
    """Encode every semantic class as the smallest lossless integer label map.

    The per-image mapping is persisted in metadata. Mask2Former semantic masks are
    mutually exclusive; for other backends, later predictions retain their existing
    predictor ordering if masks overlap.
    """

    if not predictions:
        raise ValueError("predictions must include image-sized masks")
    names = sorted({str(item.class_name) for item in predictions})
    if len(names) >= np.iinfo(np.uint16).max:
        raise ValueError("too many semantic classes for uint16 class label mask")
    dtype = np.uint8 if len(names) <= np.iinfo(np.uint8).max else np.uint16
    class_name_to_id = {name: index + 1 for index, name in enumerate(names)}
    class_id_to_name = {class_id: name for name, class_id in class_name_to_id.items()}
    labels = np.zeros(predictions[0].mask.shape, dtype=dtype)
    for prediction in predictions:
        labels[prediction.mask.astype(bool)] = class_name_to_id[str(prediction.class_name)]
    return labels, class_id_to_name


def build_dynamic_mask_from_class_labels(
    labels: np.ndarray,
    class_id_to_name: dict[int, str],
    *,
    dynamic_classes: frozenset[str] | set[str],
    dilation_pixels: int,
) -> np.ndarray:
    """Derive a binary dynamic mask from the canonical class-label image."""

    class_labels = np.asarray(labels)
    if class_labels.ndim != 2:
        raise ValueError("semantic class mask must be single-channel")
    normalized_dynamic = {canonicalize_label(value) for value in dynamic_classes}
    max_label = int(class_labels.max()) if class_labels.size else 0
    lookup = np.zeros(max_label + 1, dtype=np.uint8)
    for class_id, class_name in class_id_to_name.items():
        numeric_id = int(class_id)
        if (
            0 <= numeric_id <= max_label
            and canonicalize_label(class_name) in normalized_dynamic
        ):
            lookup[numeric_id] = 255
    combined = lookup[class_labels]
    if dilation_pixels > 0 and np.any(combined):
        size = 2 * int(dilation_pixels) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        combined = cv2.dilate(combined, kernel)
    return combined


def load_dynamic_mask(
    result: CachedMaskResult,
    *,
    dynamic_classes: frozenset[str] | set[str],
    dilation_pixels: int,
) -> np.ndarray:
    """Build a dynamic binary mask in memory from the persisted class labels."""

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    result.mask_path.unlink(missing_ok=True)
    labels = cv2.imread(str(result.class_mask_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"could not read semantic class mask: {result.class_mask_path}")
    class_id_to_name = {
        int(class_id): str(class_name)
        for class_id, class_name in metadata.get("class_id_to_name", {}).items()
    }
    return build_dynamic_mask_from_class_labels(
        labels,
        class_id_to_name,
        dynamic_classes=dynamic_classes,
        dilation_pixels=dilation_pixels,
    )


def class_preview_color_bgr(class_name: str) -> np.ndarray:
    """Return a stable preview color determined solely by semantic class name."""
    canonical_name = canonicalize_label(str(class_name))
    digest = hashlib.blake2b(canonical_name.encode("utf-8"), digest_size=2).digest()
    hue = int.from_bytes(digest, byteorder="big") % 180
    hsv = np.asarray([[[hue, 210, 255]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def _preview(
    image: np.ndarray,
    class_mask: np.ndarray,
    class_id_to_name: dict[int, str],
) -> np.ndarray:
    overlay = image.copy()
    for class_id in class_id_to_name:
        pixels = class_mask == class_id
        if np.any(pixels):
            color = class_preview_color_bgr(class_id_to_name[class_id])
            overlay[pixels] = cv2.addWeighted(
                image[pixels], 0.35, np.broadcast_to(color, image[pixels].shape), 0.65, 0
            )
    source = image.copy()
    for panel, title in ((source, "SOURCE"), (overlay, "SEMANTIC CLASSES")):
        cv2.rectangle(panel, (10, 8), (250, 48), (0, 0, 0), -1)
        cv2.putText(
            panel,
            title,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    x, y = 18, 76
    font_scale = max(0.4, min(0.7, overlay.shape[1] / 3600.0))
    for class_id, name in sorted(class_id_to_name.items()):
        if y > overlay.shape[0] - 12:
            break
        color = class_preview_color_bgr(name).tolist()
        cv2.rectangle(overlay, (x, y - 14), (x + 16, y + 2), color, -1)
        cv2.putText(
            overlay,
            name,
            (x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += max(18, int(round(28 * font_scale)))
    return np.hstack([source, overlay])


def write_cached_preview(
    result: CachedMaskResult,
    image: np.ndarray,
) -> None:
    """Materialize a missing preview from cached classes without model inference."""

    labels = cv2.imread(str(result.class_mask_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"could not read semantic class mask: {result.class_mask_path}")
    source = np.asarray(image)
    if source.shape[:2] != labels.shape:
        source = cv2.resize(
            source,
            (int(labels.shape[1]), int(labels.shape[0])),
            interpolation=cv2.INTER_AREA,
        )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    class_id_to_name = {
        int(class_id): str(class_name)
        for class_id, class_name in metadata.get("class_id_to_name", {}).items()
    }
    result.preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(result.preview_path),
        _preview(source, labels, class_id_to_name),
    )
    if not bool(metadata.get("preview_generated", False)):
        metadata["preview_generated"] = True
        result.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _predict_batch(
    predictor: SegmentationPredictor,
    images: list[np.ndarray],
) -> list[list[InstancePrediction] | ClassLabelPrediction]:
    predict_batch = getattr(predictor, "predict_batch", None)
    if callable(predict_batch):
        return predict_batch(images)
    return [predictor.predict(image) for image in images]


def _write_cached_mask_result(
    image_path: Path,
    camera: str,
    timestamp_ns: int,
    predictor: SegmentationPredictor,
    cache: MaskCache,
    image: np.ndarray,
    predictions: list[InstancePrediction] | ClassLabelPrediction,
    write_preview: bool = True,
) -> CachedMaskResult:
    if isinstance(predictions, ClassLabelPrediction):
        class_mask = predictions.labels
        class_id_to_name = predictions.class_id_to_name
        normalized_dynamic = {
            canonicalize_label(value) for value in cache.config.dynamic_classes
        }
        mask = build_dynamic_mask_from_class_labels(
            class_mask,
            class_id_to_name,
            dynamic_classes=cache.config.dynamic_classes,
            dilation_pixels=cache.config.dilation_pixels,
        )
        class_counts = np.bincount(class_mask.reshape(-1).astype(np.int64))
        accepted_records = [
            {
                "class_name": class_name,
                "confidence": 1.0,
                "pixel_count_before_dilation": int(class_counts[class_id]),
            }
            for class_id, class_name in class_id_to_name.items()
            if canonicalize_label(class_name) in normalized_dynamic
        ]
    elif predictions:
        mask, accepted = build_dynamic_mask(
            predictions,
            cache.config.dynamic_classes,
            cache.config.confidence_threshold,
            cache.config.dilation_pixels,
        )
        accepted_records = [
            {
                "class_name": item.class_name,
                "confidence": item.confidence,
                "pixel_count_before_dilation": int(np.count_nonzero(item.mask)),
            }
            for item in accepted
        ]
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        accepted_records = []
    if isinstance(predictions, ClassLabelPrediction):
        pass
    elif predictions:
        class_mask, class_id_to_name = build_class_label_mask(predictions)
    else:
        class_mask, class_id_to_name = np.zeros(image.shape[:2], dtype=np.uint8), {}
    mask_path, metadata_path, preview_path = cache.paths(
        camera, timestamp_ns, predictor.model_id
    )
    class_mask_path = cache.class_mask_path(camera, timestamp_ns, predictor.model_id)
    mask_path.unlink(missing_ok=True)
    class_mask_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(class_mask_path), class_mask)
    if write_preview:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), _preview(image, class_mask, class_id_to_name))
    metadata = {
        "schema_version": 5,
        "cache_key": cache.config.cache_key(predictor.model_id),
        "source": cache.source_signature(
            image_path,
            include_hash=str(cache.config.source_validation).lower() == "strict",
        ),
        "camera": camera,
        "timestamp_ns": int(timestamp_ns),
        "model_id": predictor.model_id,
        "model_name": cache.config.model_name,
        "backend": cache.config.backend,
        "confidence_threshold": cache.config.confidence_threshold,
        "processor_image_size": cache.config.processor_image_size,
        "inference_dtype": cache.config.inference_dtype,
        "use_fast_processor": cache.config.use_fast_processor,
        "image_cache_enabled": cache.config.image_cache_enabled,
        "image_cache_max_long_side": cache.config.image_cache_max_long_side,
        "dilation_pixels": cache.config.dilation_pixels,
        "dynamic_classes": sorted(cache.config.dynamic_classes),
        "preview_generated": bool(write_preview),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "instance_count": len(accepted_records),
        "dynamic_pixel_ratio": float(np.mean(mask > 0)),
        "class_mask_format": f"{class_mask.dtype}_png",
        "class_mask_path": str(class_mask_path),
        "class_id_to_name": {str(key): value for key, value in class_id_to_name.items()},
        "instances": accepted_records,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CachedMaskResult(
        mask_path=mask_path,
        class_mask_path=class_mask_path,
        metadata_path=metadata_path,
        preview_path=preview_path,
        cache_hit=False,
        instance_count=len(accepted_records),
        dynamic_pixel_ratio=float(np.mean(mask > 0)),
    )


def generate_cached_mask(
    image_path: Path,
    camera: str,
    timestamp_ns: int,
    predictor: SegmentationPredictor,
    cache: MaskCache,
    force: bool = False,
    write_preview: bool = True,
) -> CachedMaskResult:
    image_path = Path(image_path)
    if not force:
        cached = cache.load_if_valid(
            image_path,
            camera,
            timestamp_ns,
            predictor.model_id,
            require_preview=write_preview,
        )
        if cached is not None:
            if write_preview and not cached.preview_path.exists():
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"image could not be read: {image_path}")
                write_cached_preview(cached, image)
            return cached
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image could not be read: {image_path}")
    predictions = predictor.predict(image)
    return _write_cached_mask_result(
        image_path,
        camera,
        timestamp_ns,
        predictor,
        cache,
        image,
        predictions,
        write_preview=write_preview,
    )


def generate_cached_masks_batch(
    items: list[tuple[Path, str, int]],
    predictor: SegmentationPredictor,
    cache: MaskCache,
    force: bool = False,
    write_preview_flags: list[bool] | None = None,
    image_store: object | None = None,
    write_workers: int | None = None,
) -> list[CachedMaskResult]:
    results: list[CachedMaskResult | None] = [None] * len(items)
    uncached_images: list[np.ndarray] = []
    uncached_meta: list[tuple[int, Path, str, int, bool]] = []
    if write_preview_flags is None:
        write_preview_flags = [True] * len(items)
    for index, (image_path_raw, camera, timestamp_ns) in enumerate(items):
        image_path = Path(image_path_raw)
        write_preview = bool(write_preview_flags[index])
        if not force:
            cached = cache.load_if_valid(
                image_path,
                camera,
                timestamp_ns,
                predictor.model_id,
                require_preview=write_preview,
            )
            if cached is not None:
                if write_preview and not cached.preview_path.exists():
                    image = (
                        image_store.get(image_path)
                        if image_store is not None and hasattr(image_store, "get")
                        else cv2.imread(str(image_path))
                    )
                    if image is None:
                        raise ValueError(f"image could not be read: {image_path}")
                    write_cached_preview(cached, image)
                results[index] = cached
                continue
        image = (
            image_store.get(image_path)
            if image_store is not None and hasattr(image_store, "get")
            else cv2.imread(str(image_path))
        )
        if image is None:
            raise ValueError(f"image could not be read: {image_path}")
        uncached_images.append(image)
        uncached_meta.append((index, image_path, camera, timestamp_ns, write_preview))
    if uncached_images:
        predictions_batch = _predict_batch(predictor, uncached_images)
        write_items = list(zip(
            uncached_meta,
            uncached_images,
            predictions_batch,
            strict=True,
        ))

        def write_result(
            item: tuple[tuple[int, Path, str, int, bool], np.ndarray, list[InstancePrediction]]
        ) -> tuple[int, CachedMaskResult]:
            (result_index, image_path, camera, timestamp_ns, write_preview), image, predictions = item
            return result_index, _write_cached_mask_result(
                image_path,
                camera,
                timestamp_ns,
                predictor,
                cache,
                image,
                predictions,
                write_preview=write_preview,
            )
        workers = int(cache.config.write_workers if write_workers is None else write_workers)
        if workers < 1:
            raise ValueError("write_workers must be at least 1")
        if workers == 1:
            written_results = map(write_result, write_items)
            for result_index, result in written_results:
                results[result_index] = result
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="semantic-write"
            ) as executor:
                for result_index, result in executor.map(write_result, write_items):
                    results[result_index] = result
    return [result for result in results if result is not None]
