from __future__ import annotations

import math
import warnings
from collections import OrderedDict

import cv2
import numpy as np

from .camera_models import undistort_normalized
from .match_backends import normalize_match_records


DEFAULT_SUPERPOINT_MODEL_ID = "magic-leap-community/superpoint"
DEFAULT_LIGHTGLUE_MODEL_ID = "ETH-CVG/lightglue_superpoint"
DEFAULT_EFFICIENT_LOFTR_MODEL_ID = "zju-community/efficientloftr"

_BACKEND_CACHE: dict[tuple[str, str, str, int | None, float | None], dict[str, object]] = {}
_LOFTR_BACKEND_CACHE: dict[tuple[str, str], dict[str, object]] = {}
_LOMA_BACKEND_CACHE: dict[str, dict[str, object]] = {}
_LOMA_FEATURE_CACHE: OrderedDict[
    tuple[object, ...], tuple[object, object, int, int]
] = OrderedDict()
_LOMA_FEATURE_GPU_CACHE: OrderedDict[
    tuple[object, ...], tuple[object, object, int, int]
] = OrderedDict()
_LOMA_FEATURE_CACHE_MAX_ITEMS = 5000
_LOMA_FEATURE_CACHE_CPU_MAX_ITEMS: int | None = None
_LOMA_FEATURE_CACHE_MODE = "model"
_LOMA_MATCHER_RUNNER_CACHE: dict[tuple[int, str, str], object] = {}


class _CudagraphLoMaMatcher:
    def __init__(self, model: object):
        import torch

        self.original = model
        self.compiled = None
        self.fallback = False
        self.error = ""
        try:
            self.compiled = torch.compile(model, backend="cudagraphs")
        except Exception as exc:  # pragma: no cover - backend-specific failure
            self._activate_fallback(exc)

    def _activate_fallback(self, exc: Exception) -> None:
        self.fallback = True
        self.error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"LoMa cudagraph matcher failed; falling back to eager: {self.error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def __call__(self, *args, **kwargs):
        if self.fallback or self.compiled is None:
            return self.original(*args, **kwargs)
        import torch

        try:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                return self.compiled(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            self._activate_fallback(exc)
            return self.original(*args, **kwargs)


def _resolve_loma_matcher_runner(
    model: object,
    *,
    requested_backend: str,
    resolved_device: str,
) -> tuple[object, dict[str, object]]:
    backend = str(requested_backend).lower()
    if backend == "off":
        return model, {
            "matcher_compile_backend": "off",
            "matcher_compile_fallback": False,
        }
    if backend != "cudagraphs":
        raise ValueError(f"unsupported LoMa matcher compile backend: {backend}")
    if not str(resolved_device).startswith("cuda"):
        return model, {
            "matcher_compile_backend": "off",
            "matcher_compile_fallback": True,
            "matcher_compile_error": "cudagraphs requires a CUDA device",
        }
    key = (id(model), backend, str(resolved_device))
    runner = _LOMA_MATCHER_RUNNER_CACHE.get(key)
    if runner is None:
        runner = _CudagraphLoMaMatcher(model)
        _LOMA_MATCHER_RUNNER_CACHE[key] = runner
    return runner, {
        "matcher_compile_backend": backend,
        "matcher_compile_fallback": bool(getattr(runner, "fallback", False)),
        "matcher_compile_error": str(getattr(runner, "error", "")),
    }


def _loma_matcher_diagnostics(
    runner: object,
    *,
    requested_backend: str,
) -> dict[str, object]:
    fallback = bool(getattr(runner, "fallback", False))
    return {
        "matcher_compile_backend_requested": str(requested_backend),
        "matcher_compile_backend": (
            "off" if fallback else str(requested_backend)
        ),
        "matcher_compile_fallback": fallback,
        "matcher_compile_error": str(getattr(runner, "error", "")),
    }


class MaskPointSampler:
    def __init__(self, mask: np.ndarray | None):
        self._original_mask = None if mask is None else np.asarray(mask)
        self._prepared_cache: dict[tuple[int, int], np.ndarray | None] = {}

    def prepared_mask(self, target_shape: tuple[int, int]) -> np.ndarray | None:
        key = (int(target_shape[0]), int(target_shape[1]))
        if key not in self._prepared_cache:
            self._prepared_cache[key] = _resize_mask(self._original_mask, target_shape)
        return self._prepared_cache[key]


def _resolve_model_device(requested_device: str) -> tuple[str, bool]:
    import torch

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", True
    try:
        torch.device(requested_device)
    except (RuntimeError, ValueError):
        return "cpu", True
    return requested_device, False


def _get_backend_bundle(
    model_device: str,
    superpoint_model_id: str = DEFAULT_SUPERPOINT_MODEL_ID,
    lightglue_model_id: str = DEFAULT_LIGHTGLUE_MODEL_ID,
    max_keypoints: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, object]:
    from transformers import (
        LightGlueForKeypointMatching,
        LightGlueImageProcessor,
        SuperPointForKeypointDetection,
        SuperPointImageProcessor,
    )

    resolved_device, device_fallback = _resolve_model_device(model_device)
    cache_key = (
        resolved_device,
        superpoint_model_id,
        lightglue_model_id,
        int(max_keypoints) if max_keypoints is not None else None,
        float(score_threshold) if score_threshold is not None else None,
    )
    bundle = _BACKEND_CACHE.get(cache_key)
    if bundle is None:
        superpoint_model = SuperPointForKeypointDetection.from_pretrained(superpoint_model_id)
        lightglue_model = LightGlueForKeypointMatching.from_pretrained(lightglue_model_id)
        superpoint_processor = SuperPointImageProcessor()
        lightglue_processor = LightGlueImageProcessor()
        if max_keypoints is not None:
            superpoint_model.config.max_keypoints = int(max_keypoints)
            lightglue_model.keypoint_detector.config.max_keypoints = int(max_keypoints)
        if score_threshold is not None:
            superpoint_model.config.keypoint_threshold = float(score_threshold)
            lightglue_model.config.filter_threshold = float(score_threshold)
            lightglue_model.keypoint_detector.config.keypoint_threshold = float(score_threshold)
        bundle = {
            "superpoint_model": superpoint_model.to(resolved_device).eval(),
            "superpoint_processor": superpoint_processor,
            "lightglue_model": lightglue_model.to(resolved_device).eval(),
            "lightglue_processor": lightglue_processor,
        }
        _BACKEND_CACHE[cache_key] = bundle

    return {
        **bundle,
        "requested_device": model_device,
        "resolved_device": resolved_device,
        "device_fallback": device_fallback,
    }


def _get_loftr_backend_bundle(
    model_device: str,
    model_id: str = DEFAULT_EFFICIENT_LOFTR_MODEL_ID,
) -> dict[str, object]:
    from transformers import EfficientLoFTRForKeypointMatching, EfficientLoFTRImageProcessor

    resolved_device, device_fallback = _resolve_model_device(model_device)
    cache_key = (resolved_device, model_id)
    bundle = _LOFTR_BACKEND_CACHE.get(cache_key)
    if bundle is None:
        model = EfficientLoFTRForKeypointMatching.from_pretrained(model_id)
        processor = EfficientLoFTRImageProcessor()
        bundle = {
            "model": model.to(resolved_device).eval(),
            "processor": processor,
        }
        _LOFTR_BACKEND_CACHE[cache_key] = bundle

    return {
        **bundle,
        "requested_device": model_device,
        "resolved_device": resolved_device,
        "device_fallback": device_fallback,
        "model_id": model_id,
    }


def _get_loma_backend_bundle(model_device: str) -> dict[str, object]:
    try:
        from loma import LoMa, LoMaB
        from loma.device import device as loma_device
    except ImportError as exc:
        raise ImportError(
            "LoMa backend requires the official davnords/LoMa package. "
            "Install it with: python -m pip install -e .tools/loma"
        ) from exc

    cache_key = "loma_B"
    bundle = _LOMA_BACKEND_CACHE.get(cache_key)
    if bundle is None:
        model = LoMa(LoMaB())
        bundle = {
            "model": model,
            "model_name": "loma_B",
            "resolved_device": str(loma_device),
        }
        _LOMA_BACKEND_CACHE[cache_key] = bundle
    return {
        **bundle,
        "requested_device": model_device,
        "device_fallback": model_device.startswith("cuda")
        and not str(bundle["resolved_device"]).startswith("cuda"),
    }


def _move_batch_to_device(batch: object, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return batch


def _as_transformers_rgb_from_cv2_image(image: np.ndarray) -> np.ndarray:
    """Convert pipeline image input to the RGB layout expected by transformers.

    The registration pipeline provides images loaded via cv2, so this helper
    explicitly accepts grayscale or cv2-style BGR input.
    """

    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    raise ValueError("image must be grayscale or cv2-style BGR")


def _as_gray_from_cv2_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError("image must be grayscale or cv2-style BGR")


def _as_loftr_single_channel_image(image: np.ndarray) -> np.ndarray:
    gray = _as_gray_from_cv2_image(image)
    return gray[:, :, None]


def _resize_for_deep(image: np.ndarray, max_dimension: int) -> np.ndarray:
    if max_dimension <= 0:
        return image
    height, width = image.shape[:2]
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _deep_resize_shape_and_scale(
    image_shape: tuple[int, int],
    max_dimension: int,
) -> tuple[tuple[int, int], float]:
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if max_dimension <= 0:
        return (height, width), 1.0
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    if scale >= 1.0:
        return (height, width), 1.0
    return (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    ), scale


def _resize_for_deep_with_scale(
    image: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    (target_height, target_width), scale = _deep_resize_shape_and_scale(
        (int(height), int(width)), max_dimension
    )
    if scale >= 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _to_numpy_points(value: object) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return array.reshape(-1, 2).astype(np.float64)


def _map_points_to_shape(
    points: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Map pixel coordinates between image shapes using the resize convention."""
    points = _to_numpy_points(points)
    source_height, source_width = (int(source_shape[0]), int(source_shape[1]))
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if points.size == 0 or (source_height, source_width) == (target_height, target_width):
        return points
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source image shape must be positive")
    mapped = points.copy()
    mapped[:, 0] *= float(target_width) / float(source_width)
    mapped[:, 1] *= float(target_height) / float(source_height)
    return mapped


def _to_numpy_scores(value: object) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _resize_mask(mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray | None:
    if isinstance(mask, MaskPointSampler):
        return mask.prepared_mask(target_shape)
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.max(array, axis=2)
    elif array.ndim != 2:
        raise ValueError("mask must be a 2D single-channel array or a 3-channel image-like array")
    if array.shape != target_shape:
        array = cv2.resize(
            array.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(array, dtype=np.uint8)


def _static_points_mask(
    dynamic_mask: np.ndarray | None,
    image_shape: tuple[int, int],
    points: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    mask = _resize_mask(dynamic_mask, image_shape)
    if mask is None:
        return np.ones((len(points),), dtype=bool)
    in_bounds = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(image_shape[1] - 1))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(image_shape[0] - 1))
    )
    keep = np.zeros((len(points),), dtype=bool)
    if not np.any(in_bounds):
        return keep
    x = np.rint(points[in_bounds, 0]).astype(int)
    y = np.rint(points[in_bounds, 1]).astype(int)
    keep[in_bounds] = mask[y, x] == 0
    return keep


def _run_superpoint_detections(
    bundle: dict[str, object],
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
) -> list[dict[str, object]]:
    import torch

    processor = bundle["superpoint_processor"]
    model = bundle["superpoint_model"]
    inputs = processor(
        [first_image, second_image],
        do_resize=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**_move_batch_to_device(inputs, bundle["resolved_device"]))
    return processor.post_process_keypoint_detection(outputs, [first_shape, second_shape])


def _run_superpoint_detections_batch(
    bundle: dict[str, object],
    images: list[np.ndarray],
    target_sizes: list[tuple[int, int]],
) -> list[dict[str, object]]:
    import torch

    processor = bundle["superpoint_processor"]
    model = bundle["superpoint_model"]
    inputs = processor(
        images,
        do_resize=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**_move_batch_to_device(inputs, bundle["resolved_device"]))
    return processor.post_process_keypoint_detection(outputs, target_sizes)


def _run_lightglue_matching(
    bundle: dict[str, object],
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
    score_threshold: float,
) -> dict[str, object]:
    import torch

    processor = bundle["lightglue_processor"]
    model = bundle["lightglue_model"]
    inputs = processor(
        [[first_image, second_image]],
        do_resize=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**_move_batch_to_device(inputs, bundle["resolved_device"]))
    return processor.post_process_keypoint_matching(
        outputs,
        [(first_shape, second_shape)],
        threshold=score_threshold,
    )[0]


def _run_lightglue_matching_batch(
    bundle: dict[str, object],
    pair_images: list[list[np.ndarray]],
    target_sizes: list[tuple[tuple[int, int], tuple[int, int]]],
    score_threshold: float,
) -> list[dict[str, object]]:
    import torch

    processor = bundle["lightglue_processor"]
    model = bundle["lightglue_model"]
    inputs = processor(
        pair_images,
        do_resize=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**_move_batch_to_device(inputs, bundle["resolved_device"]))
    return processor.post_process_keypoint_matching(
        outputs,
        target_sizes,
        threshold=score_threshold,
    )


def _run_loftr_matching(
    bundle: dict[str, object],
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
    score_threshold: float,
) -> dict[str, object]:
    import torch

    processor = bundle["processor"]
    model = bundle["model"]
    first_input = _as_loftr_single_channel_image(first_image)
    second_input = _as_loftr_single_channel_image(second_image)
    inputs = processor(
        [[first_input, second_input]],
        do_resize=False,
        do_grayscale=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**_move_batch_to_device(inputs, bundle["resolved_device"]))
    return processor.post_process_keypoint_matching(
        outputs,
        [(first_shape, second_shape)],
        threshold=score_threshold,
    )[0]


def _lk_verify_matches(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(first_points) == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            0,
        )

    first_gray = _as_gray_from_cv2_image(first_image)
    second_gray = _as_gray_from_cv2_image(second_image)
    first_input = first_points.astype(np.float32).reshape(-1, 1, 2)
    second_initial = second_points.astype(np.float32).reshape(-1, 1, 2)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        first_gray,
        second_gray,
        first_input,
        second_initial,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    if forward is None or forward_status is None:
        return (
            np.zeros((len(first_points),), dtype=bool),
            np.full((len(first_points),), np.nan, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            1,
        )

    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        second_gray,
        first_gray,
        forward,
        first_input.copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    if backward is None or backward_status is None:
        return (
            np.zeros((len(first_points),), dtype=bool),
            np.full((len(first_points),), np.nan, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            1,
        )

    backward_points = backward.reshape(-1, 2).astype(np.float64)
    forward_points = forward.reshape(-1, 2).astype(np.float64)
    errors = np.linalg.norm(backward_points - first_points, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (errors <= threshold)
    )
    return valid, errors, forward_points, 0


def superpoint_lightglue_matches(
    first_image,
    second_image,
    calibration,
    config,
    first_dynamic_mask=None,
    second_dynamic_mask=None,
):
    """Match a pair of cv2-loaded images with SuperPoint + LightGlue.

    This wrapper expects grayscale images or color images in cv2's default BGR
    channel order. Color inputs are converted to RGB before calling the
    transformers image processors.
    """

    first_rgb = _as_transformers_rgb_from_cv2_image(first_image)
    second_rgb = _as_transformers_rgb_from_cv2_image(second_image)
    first_shape = tuple(int(v) for v in first_rgb.shape[:2])
    second_shape = tuple(int(v) for v in second_rgb.shape[:2])
    first_resized = _resize_for_deep(first_rgb, int(config.deep_image_size))
    second_resized = _resize_for_deep(second_rgb, int(config.deep_image_size))
    bundle = _get_backend_bundle(
        model_device=config.model_device,
        max_keypoints=int(config.deep_max_keypoints),
        score_threshold=float(config.deep_score_threshold),
    )

    detections = _run_superpoint_detections(
        bundle,
        first_resized,
        second_resized,
        first_shape,
        second_shape,
    )
    match_result = _run_lightglue_matching(
        bundle,
        first_resized,
        second_resized,
        first_shape,
        second_shape,
        float(config.deep_score_threshold),
    )

    first_points = _to_numpy_points(match_result.get("keypoints0"))
    second_points = _to_numpy_points(match_result.get("keypoints1"))
    scores = _to_numpy_scores(match_result.get("matching_scores"))
    diagnostics = {
        "backend": "superpoint_lightglue",
        "requested_device": bundle["requested_device"],
        "resolved_device": bundle["resolved_device"],
        "device_fallback": bool(bundle["device_fallback"]),
        "first_keypoints": len(_to_numpy_points(detections[0].get("keypoints"))),
        "second_keypoints": len(_to_numpy_points(detections[1].get("keypoints"))),
        "raw_matches": int(min(len(first_points), len(second_points), len(scores))),
        "semantic_source_rejected": 0,
        "semantic_destination_rejected": 0,
        "post_semantic_matches": 0,
        "lk_mode": config.lk_mode,
        "lk_input_matches": 0,
        "lk_verified_matches": 0,
        "lk_rejected": 0,
        "lk_failure_count": 0,
    }

    raw_count = diagnostics["raw_matches"]
    if raw_count == 0:
        return normalize_match_records([]), diagnostics

    first_points = first_points[:raw_count]
    second_points = second_points[:raw_count]
    scores = scores[:raw_count]
    keep_source = _static_points_mask(first_dynamic_mask, first_shape, first_points)
    keep_destination = _static_points_mask(second_dynamic_mask, second_shape, second_points)
    diagnostics["semantic_source_rejected"] = int(np.count_nonzero(~keep_source))
    diagnostics["semantic_destination_rejected"] = int(np.count_nonzero(~keep_destination))
    keep = keep_source & keep_destination
    if not np.any(keep):
        return normalize_match_records([]), diagnostics

    match_indices = np.arange(raw_count, dtype=int)[keep]
    first_points = first_points[keep]
    second_points = second_points[keep]
    scores = scores[keep]
    diagnostics["post_semantic_matches"] = int(len(first_points))

    lk_errors = np.full((len(first_points),), np.nan, dtype=np.float64)
    lk_valid = np.ones((len(first_points),), dtype=bool)
    if config.lk_mode == "on":
        diagnostics["lk_input_matches"] = int(len(first_points))
        lk_valid, lk_errors, refined_second_points, lk_failure_count = _lk_verify_matches(
            first_image,
            second_image,
            first_points,
            second_points,
            float(config.lk_forward_backward_threshold),
        )
        diagnostics["lk_failure_count"] = int(lk_failure_count)
        diagnostics["lk_verified_matches"] = int(np.count_nonzero(lk_valid))
        diagnostics["lk_rejected"] = int(len(lk_valid) - np.count_nonzero(lk_valid))
        if not np.any(lk_valid):
            return normalize_match_records([]), diagnostics
        match_indices = match_indices[lk_valid]
        first_points = first_points[lk_valid]
        second_points = refined_second_points[lk_valid]
        scores = scores[lk_valid]
        lk_errors = lk_errors[lk_valid]
        lk_valid = lk_valid[lk_valid]
    else:
        diagnostics["lk_input_matches"] = int(len(first_points))
        diagnostics["lk_verified_matches"] = int(len(first_points))

    normalized_first = undistort_normalized(first_points, calibration)
    normalized_second = undistort_normalized(second_points, calibration)
    records = []
    for row_index, match_index in enumerate(match_indices):
        records.append(
            {
                "kp1": int(match_index),
                "kp2": int(match_index),
                "x1": float(first_points[row_index, 0]),
                "y1": float(first_points[row_index, 1]),
                "x2": float(second_points[row_index, 0]),
                "y2": float(second_points[row_index, 1]),
                "nx1": float(normalized_first[row_index, 0]),
                "ny1": float(normalized_first[row_index, 1]),
                "nx2": float(normalized_second[row_index, 0]),
                "ny2": float(normalized_second[row_index, 1]),
                "descriptor_distance": math.nan,
                "ratio": math.nan,
                "lk_fb_error": float(lk_errors[row_index]) if not math.isnan(lk_errors[row_index]) else math.nan,
                "lk_valid": bool(lk_valid[row_index]),
                "ransac_inlier": False,
                "backend": "superpoint_lightglue",
                "backend_score": float(scores[row_index]),
            }
        )
    return normalize_match_records(records), diagnostics


def loftr_matches(
    first_image,
    second_image,
    calibration,
    config,
    first_dynamic_mask=None,
    second_dynamic_mask=None,
):
    """Match cv2-style BGR or grayscale inputs with EfficientLoFTR.

    Public project backend naming remains ``loftr`` even though the
    implementation uses Hugging Face EfficientLoFTR.
    """

    first_gray = _as_gray_from_cv2_image(first_image)
    second_gray = _as_gray_from_cv2_image(second_image)
    first_shape = tuple(int(v) for v in first_gray.shape[:2])
    second_shape = tuple(int(v) for v in second_gray.shape[:2])
    first_resized = _resize_for_deep(first_gray, int(config.deep_image_size))
    second_resized = _resize_for_deep(second_gray, int(config.deep_image_size))
    bundle = _get_loftr_backend_bundle(model_device=config.model_device)
    match_result = _run_loftr_matching(
        bundle,
        first_resized,
        second_resized,
        first_shape,
        second_shape,
        float(config.deep_score_threshold),
    )

    first_points = _to_numpy_points(match_result.get("keypoints0"))
    second_points = _to_numpy_points(match_result.get("keypoints1"))
    scores = _to_numpy_scores(match_result.get("matching_scores"))
    raw_count = int(min(len(first_points), len(second_points), len(scores)))
    lk_requested = config.lk_mode == "on"
    diagnostics = {
        "backend": "loftr",
        "requested_device": bundle["requested_device"],
        "resolved_device": bundle["resolved_device"],
        "device_fallback": bool(bundle["device_fallback"]),
        "model_id": str(bundle.get("model_id", DEFAULT_EFFICIENT_LOFTR_MODEL_ID)),
        "raw_matches": raw_count,
        "semantic_source_rejected": 0,
        "semantic_destination_rejected": 0,
        "post_semantic_matches": 0,
        "lk_mode_requested": config.lk_mode,
        "lk_mode": "off",
        "lk_ignored": lk_requested,
        "lk_note": (
            "LoFTR backend does not use LK verification; requested lk_mode='on' was ignored."
            if lk_requested
            else "LoFTR backend does not use LK verification."
        ),
        "lk_input_matches": 0,
        "lk_verified_matches": 0,
        "lk_rejected": 0,
        "lk_failure_count": 0,
    }

    if raw_count == 0:
        return normalize_match_records([]), diagnostics

    first_points = first_points[:raw_count]
    second_points = second_points[:raw_count]
    scores = scores[:raw_count]
    keep_source = _static_points_mask(first_dynamic_mask, first_shape, first_points)
    keep_destination = _static_points_mask(second_dynamic_mask, second_shape, second_points)
    diagnostics["semantic_source_rejected"] = int(np.count_nonzero(~keep_source))
    diagnostics["semantic_destination_rejected"] = int(np.count_nonzero(~keep_destination))
    keep = keep_source & keep_destination
    if not np.any(keep):
        return normalize_match_records([]), diagnostics

    match_indices = np.arange(raw_count, dtype=int)[keep]
    first_points = first_points[keep]
    second_points = second_points[keep]
    scores = scores[keep]
    diagnostics["post_semantic_matches"] = int(len(first_points))
    diagnostics["lk_input_matches"] = int(len(first_points))
    diagnostics["lk_verified_matches"] = int(len(first_points))

    normalized_first = undistort_normalized(first_points, calibration)
    normalized_second = undistort_normalized(second_points, calibration)
    records = []
    for row_index, match_index in enumerate(match_indices):
        records.append(
            {
                "kp1": int(match_index),
                "kp2": int(match_index),
                "x1": float(first_points[row_index, 0]),
                "y1": float(first_points[row_index, 1]),
                "x2": float(second_points[row_index, 0]),
                "y2": float(second_points[row_index, 1]),
                "nx1": float(normalized_first[row_index, 0]),
                "ny1": float(normalized_first[row_index, 1]),
                "nx2": float(normalized_second[row_index, 0]),
                "ny2": float(normalized_second[row_index, 1]),
                "descriptor_distance": math.nan,
                "ratio": math.nan,
                "lk_fb_error": math.nan,
                "lk_valid": True,
                "ransac_inlier": False,
                "backend": "loftr",
                "backend_score": float(scores[row_index]),
            }
        )
    return normalize_match_records(records), diagnostics


def loma_matches_from_points(
    first_points: np.ndarray,
    second_points: np.ndarray,
    calibration,
    config,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
    first_dynamic_mask=None,
    second_dynamic_mask=None,
):
    first_points = _to_numpy_points(first_points)
    second_points = _to_numpy_points(second_points)
    raw_count = int(min(len(first_points), len(second_points)))
    diagnostics = {
        "backend": "loma",
        "raw_matches": raw_count,
        "semantic_source_rejected": 0,
        "semantic_destination_rejected": 0,
        "post_semantic_matches": 0,
        "lk_mode_requested": config.lk_mode,
        "lk_mode": "off",
        "lk_ignored": config.lk_mode == "on",
        "lk_note": (
            "LoMa backend does not use LK verification; requested lk_mode='on' was ignored."
            if config.lk_mode == "on"
            else "LoMa backend does not use LK verification."
        ),
        "lk_input_matches": 0,
        "lk_verified_matches": 0,
        "lk_rejected": 0,
        "lk_failure_count": 0,
    }
    if raw_count == 0:
        return normalize_match_records([]), diagnostics

    first_points = first_points[:raw_count]
    second_points = second_points[:raw_count]
    diagnostics["raw_first_points"] = first_points.copy()
    diagnostics["raw_second_points"] = second_points.copy()
    keep_source = _static_points_mask(first_dynamic_mask, first_shape, first_points)
    keep_destination = _static_points_mask(second_dynamic_mask, second_shape, second_points)
    diagnostics["semantic_source_rejected"] = int(np.count_nonzero(~keep_source))
    diagnostics["semantic_destination_rejected"] = int(np.count_nonzero(~keep_destination))
    keep = keep_source & keep_destination
    if not np.any(keep):
        return normalize_match_records([]), diagnostics

    match_indices = np.arange(raw_count, dtype=int)[keep]
    first_points = first_points[keep]
    second_points = second_points[keep]
    diagnostics["post_semantic_matches"] = int(len(first_points))
    diagnostics["lk_input_matches"] = int(len(first_points))
    diagnostics["lk_verified_matches"] = int(len(first_points))

    normalized_first = undistort_normalized(first_points, calibration)
    normalized_second = undistort_normalized(second_points, calibration)
    records = []
    for row_index, match_index in enumerate(match_indices):
        records.append(
            {
                "kp1": int(match_index),
                "kp2": int(match_index),
                "x1": float(first_points[row_index, 0]),
                "y1": float(first_points[row_index, 1]),
                "x2": float(second_points[row_index, 0]),
                "y2": float(second_points[row_index, 1]),
                "nx1": float(normalized_first[row_index, 0]),
                "ny1": float(normalized_first[row_index, 1]),
                "nx2": float(normalized_second[row_index, 0]),
                "ny2": float(normalized_second[row_index, 1]),
                "descriptor_distance": math.nan,
                "ratio": math.nan,
                "lk_fb_error": math.nan,
                "lk_valid": True,
                "ransac_inlier": False,
                "backend": "loma",
                "backend_score": 1.0,
            }
        )
    return normalize_match_records(records), diagnostics


def _finalize_dense_matches(
    *,
    backend: str,
    first_points: np.ndarray,
    second_points: np.ndarray,
    scores: np.ndarray | None,
    calibration,
    config,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_dynamic_mask=None,
    second_dynamic_mask=None,
    supports_lk: bool,
    base_diagnostics: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    first_points = _to_numpy_points(first_points)
    second_points = _to_numpy_points(second_points)
    scores = (
        np.ones((len(first_points),), dtype=np.float64)
        if scores is None
        else _to_numpy_scores(scores)
    )
    raw_count = int(min(len(first_points), len(second_points), len(scores)))
    diagnostics: dict[str, object] = {
        "backend": backend,
        "raw_matches": raw_count,
        "semantic_source_rejected": 0,
        "semantic_destination_rejected": 0,
        "post_semantic_matches": 0,
        "lk_mode_requested": config.lk_mode,
        "lk_mode": config.lk_mode if supports_lk else "off",
        "lk_ignored": (config.lk_mode == "on") and (not supports_lk),
        "lk_note": (
            f"{backend} backend does not use LK verification; requested lk_mode='on' was ignored."
            if (config.lk_mode == "on") and (not supports_lk)
            else (
                f"{backend} backend does not use LK verification."
                if not supports_lk
                else ""
            )
        ),
        "lk_input_matches": 0,
        "lk_verified_matches": 0,
        "lk_rejected": 0,
        "lk_failure_count": 0,
    }
    if base_diagnostics:
        diagnostics.update(base_diagnostics)
    if raw_count == 0:
        return normalize_match_records([]), diagnostics

    first_points = first_points[:raw_count]
    second_points = second_points[:raw_count]
    scores = scores[:raw_count]
    keep_source = _static_points_mask(first_dynamic_mask, first_shape, first_points)
    keep_destination = _static_points_mask(second_dynamic_mask, second_shape, second_points)
    diagnostics["semantic_source_rejected"] = int(np.count_nonzero(~keep_source))
    diagnostics["semantic_destination_rejected"] = int(np.count_nonzero(~keep_destination))
    keep = keep_source & keep_destination
    if not np.any(keep):
        return normalize_match_records([]), diagnostics

    match_indices = np.arange(raw_count, dtype=int)[keep]
    first_points = first_points[keep]
    second_points = second_points[keep]
    scores = scores[keep]
    diagnostics["post_semantic_matches"] = int(len(first_points))

    lk_errors = np.full((len(first_points),), np.nan, dtype=np.float64)
    lk_valid = np.ones((len(first_points),), dtype=bool)
    if supports_lk and config.lk_mode == "on":
        diagnostics["lk_input_matches"] = int(len(first_points))
        lk_valid, lk_errors, refined_second_points, lk_failure_count = _lk_verify_matches(
            first_image,
            second_image,
            first_points,
            second_points,
            float(config.lk_forward_backward_threshold),
        )
        diagnostics["lk_failure_count"] = int(lk_failure_count)
        diagnostics["lk_verified_matches"] = int(np.count_nonzero(lk_valid))
        diagnostics["lk_rejected"] = int(len(lk_valid) - np.count_nonzero(lk_valid))
        if not np.any(lk_valid):
            return normalize_match_records([]), diagnostics
        match_indices = match_indices[lk_valid]
        first_points = first_points[lk_valid]
        second_points = refined_second_points[lk_valid]
        scores = scores[lk_valid]
        lk_errors = lk_errors[lk_valid]
        lk_valid = lk_valid[lk_valid]
    else:
        diagnostics["lk_input_matches"] = int(len(first_points))
        diagnostics["lk_verified_matches"] = int(len(first_points))

    normalized_first = undistort_normalized(first_points, calibration)
    normalized_second = undistort_normalized(second_points, calibration)
    records = []
    for row_index, match_index in enumerate(match_indices):
        records.append(
            {
                "kp1": int(match_index),
                "kp2": int(match_index),
                "x1": float(first_points[row_index, 0]),
                "y1": float(first_points[row_index, 1]),
                "x2": float(second_points[row_index, 0]),
                "y2": float(second_points[row_index, 1]),
                "nx1": float(normalized_first[row_index, 0]),
                "ny1": float(normalized_first[row_index, 1]),
                "nx2": float(normalized_second[row_index, 0]),
                "ny2": float(normalized_second[row_index, 1]),
                "descriptor_distance": math.nan,
                "ratio": math.nan,
                "lk_fb_error": float(lk_errors[row_index]) if not math.isnan(lk_errors[row_index]) else math.nan,
                "lk_valid": bool(lk_valid[row_index]),
                "ransac_inlier": False,
                "backend": backend,
                "backend_score": float(scores[row_index]),
            }
        )
    return normalize_match_records(records), diagnostics


def _as_loma_rgb_tensor_from_cv2_image(
    image: np.ndarray,
    *,
    target_size: tuple[int, int] | None = None,
):
    import torch

    rgb = _as_transformers_rgb_from_cv2_image(image)
    if target_size is not None:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )


def _as_loma_detector_tensor_from_cv2_image(image: np.ndarray, resize: int = 1024):
    height, width = image.shape[:2]
    scale = float(resize) / float(max(width, height))
    target_width = max(8, int((scale * width) // 8 * 8))
    target_height = max(8, int((scale * height) // 8 * 8))
    return _as_loma_rgb_tensor_from_cv2_image(
        image,
        target_size=(target_width, target_height),
    )


def _as_loma_descriptor_tensor_from_cv2_image(
    image: np.ndarray,
    size: int = 784,
):
    return _as_loma_rgb_tensor_from_cv2_image(image, target_size=(size, size))


def _loma_detect_and_describe_from_cv2_image(
    model: object,
    image: np.ndarray,
    *,
    num_keypoints: int | None,
):
    detector_tensor = _as_loma_detector_tensor_from_cv2_image(image)
    descriptor_tensor = _as_loma_descriptor_tensor_from_cv2_image(image)
    keypoints = model._detector.detect(  # noqa: SLF001 - official LoMa exposes no no-IO public API.
        {"image": detector_tensor},
        num_keypoints=num_keypoints,
    )["keypoints"]
    descriptions = model._descriptor.describe_keypoints(  # noqa: SLF001
        descriptor_tensor,
        keypoints,
    )["descriptions"]
    height, width = image.shape[:2]
    return keypoints, descriptions, int(height), int(width)


def _loma_feature_cache_key(
    image_key: object,
    image: np.ndarray,
    *,
    num_keypoints: int | None,
) -> tuple[object, ...] | None:
    """Build a stable cache key for one resized source image."""
    return _loma_feature_cache_key_from_shape(
        image_key,
        image.shape[:2],
        num_keypoints=num_keypoints,
    )


def _loma_feature_cache_key_from_shape(
    image_key: object,
    image_shape: tuple[int, int],
    *,
    num_keypoints: int | None,
) -> tuple[object, ...] | None:
    if image_key is None:
        return None
    height, width = (int(image_shape[0]), int(image_shape[1]))
    return (
        str(image_key),
        int(width),
        int(height),
        int(num_keypoints) if num_keypoints is not None else None,
        1024,
        784,
    )


def _lookup_loma_cached_features(
    key: tuple[object, ...] | None,
    *,
    cache: OrderedDict[tuple[object, ...], tuple[object, object, int, int]] | None,
    cache_device: str,
    cache_max_items: int | None,
    gpu_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None,
    model_device: str,
    cache_stats: dict[str, int] | None,
) -> tuple[object, object, int, int] | None:
    if key is None:
        return None
    if cache_device in {"two_tier", "model"} and gpu_cache is not None and key in gpu_cache:
        features = gpu_cache.pop(key)
        gpu_cache[key] = features
        if cache_stats is not None:
            cache_stats["feature_cache_hit_count"] = (
                cache_stats.get("feature_cache_hit_count", 0) + 1
            )
            cache_stats["feature_cache_gpu_hit_count"] = (
                cache_stats.get("feature_cache_gpu_hit_count", 0) + 1
            )
        return features
    if cache_device in {"two_tier", "cpu"} and cache is not None and key in cache:
        features = cache.pop(key)
        cache[key] = features
        if cache_device == "two_tier" and gpu_cache is not None:
            gpu_features = (
                _move_loma_feature_to_device(features[0], model_device),
                _move_loma_feature_to_device(features[1], model_device),
                features[2],
                features[3],
            )
            gpu_cache[key] = gpu_features
            while cache_max_items is not None and len(gpu_cache) > int(cache_max_items):
                gpu_cache.popitem(last=False)
            features = gpu_features
        if cache_stats is not None:
            cache_stats["feature_cache_hit_count"] = (
                cache_stats.get("feature_cache_hit_count", 0) + 1
            )
            cache_stats["feature_cache_cpu_hit_count"] = (
                cache_stats.get("feature_cache_cpu_hit_count", 0) + 1
            )
        return features
    return None


def _loma_cached_features(
    model: object,
    image: np.ndarray,
    *,
    image_key: object,
    num_keypoints: int | None,
    cache: OrderedDict[tuple[object, ...], tuple[object, object, int, int]] | None,
    cache_device: str = "model",
    cache_max_items: int | None = 5000,
    gpu_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None = None,
    cache_cpu_max_items: int | None = None,
    model_device: str = "cpu",
    cache_stats: dict[str, int] | None = None,
) -> tuple[object, object, int, int, bool]:
    if cache_device not in {"two_tier", "cpu", "model"}:
        raise ValueError("cache_device must be 'two_tier', 'cpu', or 'model'")
    if cache_device == "model" and gpu_cache is None:
        gpu_cache = cache
        cache = None
    key = _loma_feature_cache_key(image_key, image, num_keypoints=num_keypoints)
    cached = _lookup_loma_cached_features(
        key,
        cache=cache,
        cache_device=cache_device,
        cache_max_items=cache_max_items,
        gpu_cache=gpu_cache,
        model_device=model_device,
        cache_stats=cache_stats,
    )
    if cached is not None:
        return (*cached, True)
    if key is not None:
        if cache_stats is not None:
            cache_stats["feature_cache_miss_count"] = (
                cache_stats.get("feature_cache_miss_count", 0) + 1
            )

    features = _loma_detect_and_describe_from_cv2_image(
        model,
        image,
        num_keypoints=num_keypoints,
    )
    if key is not None:
        if cache_device in {"two_tier", "cpu"} and cache is not None:
            cpu_features = (
                _move_loma_feature_to_device(features[0], "cpu"),
                _move_loma_feature_to_device(features[1], "cpu"),
                features[2],
                features[3],
            )
            cache[key] = cpu_features
            while cache_cpu_max_items is not None and len(cache) > int(cache_cpu_max_items):
                cache.popitem(last=False)
        if cache_device in {"two_tier", "model"} and gpu_cache is not None:
            gpu_features = (
                _move_loma_feature_to_device(features[0], model_device),
                _move_loma_feature_to_device(features[1], model_device),
                features[2],
                features[3],
            )
            gpu_cache[key] = gpu_features
            while cache_max_items is not None and len(gpu_cache) > int(cache_max_items):
                gpu_cache.popitem(last=False)
            features = gpu_features
        elif cache_device == "cpu" and cache is not None:
            features = cpu_features
    return (*features, False)


def _move_loma_feature_to_device(feature: object, device: str):
    return feature.to(device) if hasattr(feature, "to") else feature


def _move_loma_features_to_device(
    features: tuple[object, object],
    device: str,
) -> tuple[object, object]:
    return (
        _move_loma_feature_to_device(features[0], device),
        _move_loma_feature_to_device(features[1], device),
    )


def _run_loma_matching_from_features(
    model: object,
    first_features: tuple[object, object, int, int],
    second_features: tuple[object, object, int, int],
    *,
    model_device: str,
    scale_first: float,
    scale_second: float,
    filter_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    from loma.loma import filter_matches, to_pixel_coords

    keypoints_first, descriptors_first, height_first, width_first = first_features
    keypoints_second, descriptors_second, height_second, width_second = second_features
    keypoints_first, descriptors_first = _move_loma_features_to_device(
        (keypoints_first, descriptors_first), model_device
    )
    keypoints_second, descriptors_second = _move_loma_features_to_device(
        (keypoints_second, descriptors_second), model_device
    )
    scores = model(
        keypoints_first,
        keypoints_second,
        descriptors_first,
        descriptors_second,
    )["scores"]
    m0, _, _, _ = filter_matches(scores, float(filter_threshold))
    valid = m0[0] > -1
    matched_first = keypoints_first[0][valid]
    matched_second = keypoints_second[0][m0[0][valid]]
    first_points = to_pixel_coords(matched_first, height_first, width_first).cpu().numpy()
    second_points = (
        to_pixel_coords(matched_second, height_second, width_second).cpu().numpy()
    )
    first_points = _to_numpy_points(first_points)
    second_points = _to_numpy_points(second_points)
    if scale_first != 1.0:
        first_points = first_points / float(scale_first)
    if scale_second != 1.0:
        second_points = second_points / float(scale_second)
    return first_points, second_points


def _run_loma_matching(
    bundle: dict[str, object],
    first_image: np.ndarray,
    second_image: np.ndarray,
    scale_first: float,
    scale_second: float,
    filter_threshold: float,
    num_keypoints: int | None,
    first_image_key: object = None,
    second_image_key: object = None,
    feature_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None = None,
    cache_device: str = "two_tier",
    cache_max_items: int | None = 5000,
    gpu_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None = None,
    cache_cpu_max_items: int | None = None,
    cache_stats: dict[str, int] | None = None,
    matcher_model: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    loma_num_keypoints = None if num_keypoints is None else int(num_keypoints)
    first_features = _loma_cached_features(
        model,
        first_image,
        image_key=first_image_key,
        num_keypoints=loma_num_keypoints,
        cache=feature_cache,
        cache_device=cache_device,
        cache_max_items=cache_max_items,
        gpu_cache=gpu_cache,
        cache_cpu_max_items=cache_cpu_max_items,
        model_device=str(bundle["resolved_device"]),
        cache_stats=cache_stats,
    )
    second_features = _loma_cached_features(
        model,
        second_image,
        image_key=second_image_key,
        num_keypoints=loma_num_keypoints,
        cache=feature_cache,
        cache_device=cache_device,
        cache_max_items=cache_max_items,
        gpu_cache=gpu_cache,
        cache_cpu_max_items=cache_cpu_max_items,
        model_device=str(bundle["resolved_device"]),
        cache_stats=cache_stats,
    )
    return _run_loma_matching_from_features(
        model if matcher_model is None else matcher_model,
        first_features[:4],
        second_features[:4],
        model_device=str(bundle["resolved_device"]),
        scale_first=scale_first,
        scale_second=scale_second,
        filter_threshold=filter_threshold,
    )


def _loma_features_for_batch_image(
    bundle: dict[str, object],
    image: np.ndarray,
    *,
    image_key: object,
    max_dimension: int,
    num_keypoints: int | None,
    feature_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None,
    cache_device: str,
    cache_max_items: int | None,
    gpu_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None,
    cache_cpu_max_items: int | None,
    cache_stats: dict[str, int] | None,
) -> tuple[tuple[object, object, int, int], float]:
    target_shape, scale = _deep_resize_shape_and_scale(
        tuple(int(value) for value in image.shape[:2]),
        max_dimension,
    )
    key = _loma_feature_cache_key_from_shape(
        image_key,
        target_shape,
        num_keypoints=num_keypoints,
    )
    normalized_feature_cache = feature_cache
    normalized_gpu_cache = gpu_cache
    if cache_device == "model" and normalized_gpu_cache is None:
        normalized_gpu_cache = normalized_feature_cache
        normalized_feature_cache = None
    cached = _lookup_loma_cached_features(
        key,
        cache=normalized_feature_cache,
        cache_device=cache_device,
        cache_max_items=cache_max_items,
        gpu_cache=normalized_gpu_cache,
        model_device=str(bundle["resolved_device"]),
        cache_stats=cache_stats,
    )
    if cached is not None:
        return cached, scale

    resized, actual_scale = _resize_for_deep_with_scale(image, max_dimension)
    extracted = _loma_cached_features(
        bundle["model"],
        resized,
        image_key=image_key,
        num_keypoints=num_keypoints,
        cache=feature_cache,
        cache_device=cache_device,
        cache_max_items=cache_max_items,
        gpu_cache=gpu_cache,
        cache_cpu_max_items=cache_cpu_max_items,
        model_device=str(bundle["resolved_device"]),
        cache_stats=cache_stats,
    )
    return extracted[:4], actual_scale


def loma_matches(
    first_image,
    second_image,
    calibration,
    config,
    first_dynamic_mask=None,
    second_dynamic_mask=None,
    *,
    first_image_key: object = None,
    second_image_key: object = None,
    feature_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None = None,
    cache_device: str | None = None,
    cache_max_items: int | None = None,
    gpu_cache: OrderedDict[
        tuple[object, ...], tuple[object, object, int, int]
    ] | None = None,
    cache_cpu_max_items: int | None = None,
    first_original_shape: tuple[int, int] | None = None,
    second_original_shape: tuple[int, int] | None = None,
):
    """Match cv2-style BGR or grayscale inputs with official davnords/LoMa."""

    first_bgr = np.asarray(first_image)
    second_bgr = np.asarray(second_image)
    first_shape = tuple(int(v) for v in first_bgr.shape[:2])
    second_shape = tuple(int(v) for v in second_bgr.shape[:2])
    first_resized, first_scale = _resize_for_deep_with_scale(
        first_bgr,
        int(config.deep_image_size),
    )
    second_resized, second_scale = _resize_for_deep_with_scale(
        second_bgr,
        int(config.deep_image_size),
    )
    bundle = _get_loma_backend_bundle(config.model_device)
    compile_backend = str(
        getattr(config, "loma_matcher_compile_backend", "off")
    )
    matcher_model, _matcher_info = _resolve_loma_matcher_runner(
        bundle["model"],
        requested_backend=compile_backend,
        resolved_device=str(bundle["resolved_device"]),
    )
    cache_stats: dict[str, int] = {
        "feature_cache_hit_count": 0,
        "feature_cache_miss_count": 0,
        "feature_cache_gpu_hit_count": 0,
        "feature_cache_cpu_hit_count": 0,
    }
    cache_enabled = first_image_key is not None or second_image_key is not None
    if cache_enabled:
        feature_cache = _LOMA_FEATURE_CACHE if feature_cache is None else feature_cache
        gpu_cache = _LOMA_FEATURE_GPU_CACHE if gpu_cache is None else gpu_cache
        cache_device = (
            getattr(config, "loma_feature_cache_mode", "model")
            if cache_device is None
            else cache_device
        )
        cache_max_items = (
            getattr(config, "loma_feature_cache_gpu_max_items", 5000)
            if cache_max_items is None
            else cache_max_items
        )
        cache_cpu_max_items = (
            getattr(config, "loma_feature_cache_cpu_max_items", None)
            if cache_cpu_max_items is None
            else cache_cpu_max_items
        )
    else:
        feature_cache = None
        gpu_cache = None
        cache_device = "model"
        cache_max_items = 5000
    run_kwargs = {
        "first_image_key": first_image_key,
        "second_image_key": second_image_key,
        "feature_cache": feature_cache,
        "cache_device": str(cache_device),
        "cache_max_items": cache_max_items,
        "gpu_cache": gpu_cache,
        "cache_cpu_max_items": cache_cpu_max_items,
    }
    if cache_enabled:
        run_kwargs["cache_stats"] = cache_stats
    if compile_backend != "off":
        run_kwargs["matcher_model"] = matcher_model
    first_points, second_points = _run_loma_matching(
        bundle,
        first_resized,
        second_resized,
        first_scale,
        second_scale,
        float(config.deep_score_threshold),
        int(config.deep_max_keypoints) if config.deep_max_keypoints is not None else None,
        **run_kwargs,
    )
    if first_original_shape is not None:
        first_points = _map_points_to_shape(
            first_points, first_shape, first_original_shape
        )
        first_output_shape = tuple(int(v) for v in first_original_shape)
    else:
        first_output_shape = first_shape
    if second_original_shape is not None:
        second_points = _map_points_to_shape(
            second_points, second_shape, second_original_shape
        )
        second_output_shape = tuple(int(v) for v in second_original_shape)
    else:
        second_output_shape = second_shape
    matches, diagnostics = loma_matches_from_points(
        first_points,
        second_points,
        calibration,
        config,
        first_output_shape,
        second_output_shape,
        first_dynamic_mask=first_dynamic_mask,
        second_dynamic_mask=second_dynamic_mask,
    )
    diagnostics.update(
        {
            "requested_device": bundle["requested_device"],
            "resolved_device": bundle["resolved_device"],
            "device_fallback": bool(bundle["device_fallback"]),
            "model_id": bundle["model_name"],
            "first_scale": float(first_scale),
            "second_scale": float(second_scale),
            "first_input_shape": first_shape,
            "second_input_shape": second_shape,
            "first_original_shape": first_output_shape,
            "second_original_shape": second_output_shape,
            **_loma_matcher_diagnostics(
                matcher_model,
                requested_backend=compile_backend,
            ),
            **cache_stats,
        }
    )
    return matches, diagnostics


def superpoint_lightglue_matches_batch(
    pair_payloads: list[dict[str, object]],
) -> list[tuple[object, dict[str, object]]]:
    if not pair_payloads:
        return []
    config = pair_payloads[0]["config"]
    bundle = _get_backend_bundle(
        model_device=config.model_device,
        max_keypoints=int(config.deep_max_keypoints),
        score_threshold=float(config.deep_score_threshold),
    )
    prepared = []
    detection_images: list[np.ndarray] = []
    pair_images: list[list[np.ndarray]] = []
    pair_target_sizes: list[tuple[tuple[int, int], tuple[int, int]]] = []
    target_sizes: list[tuple[int, int]] = []
    for payload in pair_payloads:
        first_image = np.asarray(payload["first_image"])
        second_image = np.asarray(payload["second_image"])
        first_rgb = _as_transformers_rgb_from_cv2_image(first_image)
        second_rgb = _as_transformers_rgb_from_cv2_image(second_image)
        first_shape = tuple(int(v) for v in first_rgb.shape[:2])
        second_shape = tuple(int(v) for v in second_rgb.shape[:2])
        first_resized = _resize_for_deep(first_rgb, int(config.deep_image_size))
        second_resized = _resize_for_deep(second_rgb, int(config.deep_image_size))
        prepared.append((first_rgb, second_rgb, first_shape, second_shape, first_resized, second_resized, payload))
        detection_images.extend([first_resized, second_resized])
        target_sizes.extend([first_shape, second_shape])
        pair_images.append([first_resized, second_resized])
        pair_target_sizes.append((first_shape, second_shape))

    detections = _run_superpoint_detections_batch(bundle, detection_images, target_sizes)
    match_results = _run_lightglue_matching_batch(
        bundle,
        pair_images,
        pair_target_sizes,
        float(config.deep_score_threshold),
    )
    results = []
    for index, (first_rgb, second_rgb, first_shape, second_shape, _first_resized, _second_resized, payload) in enumerate(prepared):
        first_det = detections[index * 2]
        second_det = detections[index * 2 + 1]
        match_result = match_results[index]
        matches, diagnostics = _finalize_dense_matches(
            backend="superpoint_lightglue",
            first_points=match_result.get("keypoints0"),
            second_points=match_result.get("keypoints1"),
            scores=match_result.get("matching_scores"),
            calibration=payload["calibration"],
            config=payload["config"],
            first_shape=first_shape,
            second_shape=second_shape,
            first_image=np.asarray(payload["first_image"]),
            second_image=np.asarray(payload["second_image"]),
            first_dynamic_mask=payload.get("first_dynamic_mask"),
            second_dynamic_mask=payload.get("second_dynamic_mask"),
            supports_lk=True,
            base_diagnostics={
                "requested_device": bundle["requested_device"],
                "resolved_device": bundle["resolved_device"],
                "device_fallback": bool(bundle["device_fallback"]),
                "first_keypoints": len(_to_numpy_points(first_det.get("keypoints"))),
                "second_keypoints": len(_to_numpy_points(second_det.get("keypoints"))),
            },
        )
        results.append((matches, diagnostics))
    return results


def loma_matches_batch(
    pair_payloads: list[dict[str, object]],
) -> list[tuple[object, dict[str, object]]]:
    if not pair_payloads:
        return []
    config = pair_payloads[0]["config"]
    bundle = _get_loma_backend_bundle(config.model_device)
    compile_backend = str(
        getattr(config, "loma_matcher_compile_backend", "off")
    )
    matcher_model, _matcher_info = _resolve_loma_matcher_runner(
        bundle["model"],
        requested_backend=compile_backend,
        resolved_device=str(bundle["resolved_device"]),
    )
    feature_cache = _LOMA_FEATURE_CACHE
    global _LOMA_FEATURE_CACHE_MAX_ITEMS, _LOMA_FEATURE_CACHE_CPU_MAX_ITEMS, _LOMA_FEATURE_CACHE_MODE
    _LOMA_FEATURE_CACHE_MAX_ITEMS = getattr(
        config, "loma_feature_cache_gpu_max_items", 5000
    )
    _LOMA_FEATURE_CACHE_CPU_MAX_ITEMS = getattr(
        config, "loma_feature_cache_cpu_max_items", None
    )
    _LOMA_FEATURE_CACHE_MODE = getattr(config, "loma_feature_cache_mode", "model")
    results = []
    for payload in pair_payloads:
        first_bgr = np.asarray(payload["first_image"])
        second_bgr = np.asarray(payload["second_image"])
        first_shape = tuple(int(v) for v in first_bgr.shape[:2])
        second_shape = tuple(int(v) for v in second_bgr.shape[:2])
        deep_max_keypoints = getattr(payload["config"], "deep_max_keypoints", None)
        cache_stats = {
            "feature_cache_hit_count": 0,
            "feature_cache_miss_count": 0,
            "feature_cache_gpu_hit_count": 0,
            "feature_cache_cpu_hit_count": 0,
        }
        num_keypoints = (
            int(deep_max_keypoints) if deep_max_keypoints is not None else None
        )
        cache_enabled = (
            "first_image_key" in payload or "second_image_key" in payload
        )
        if cache_enabled:
            common_feature_kwargs = {
                "max_dimension": int(payload["config"].deep_image_size),
                "num_keypoints": num_keypoints,
                "feature_cache": feature_cache,
                "cache_device": _LOMA_FEATURE_CACHE_MODE,
                "cache_max_items": _LOMA_FEATURE_CACHE_MAX_ITEMS,
                "gpu_cache": _LOMA_FEATURE_GPU_CACHE,
                "cache_cpu_max_items": _LOMA_FEATURE_CACHE_CPU_MAX_ITEMS,
                "cache_stats": cache_stats,
            }
            first_features, first_scale = _loma_features_for_batch_image(
                bundle,
                first_bgr,
                image_key=payload.get("first_image_key"),
                **common_feature_kwargs,
            )
            second_features, second_scale = _loma_features_for_batch_image(
                bundle,
                second_bgr,
                image_key=payload.get("second_image_key"),
                **common_feature_kwargs,
            )
            first_points, second_points = _run_loma_matching_from_features(
                matcher_model,
                first_features,
                second_features,
                model_device=str(bundle["resolved_device"]),
                scale_first=first_scale,
                scale_second=second_scale,
                filter_threshold=float(payload["config"].deep_score_threshold),
            )
        else:
            first_resized, first_scale = _resize_for_deep_with_scale(
                first_bgr, int(payload["config"].deep_image_size)
            )
            second_resized, second_scale = _resize_for_deep_with_scale(
                second_bgr, int(payload["config"].deep_image_size)
            )
            first_points, second_points = _run_loma_matching(
                bundle,
                first_resized,
                second_resized,
                first_scale,
                second_scale,
                float(payload["config"].deep_score_threshold),
                num_keypoints,
                **(
                    {"matcher_model": matcher_model}
                    if compile_backend != "off"
                    else {}
                ),
            )
        first_original_shape = payload.get("first_original_shape")
        second_original_shape = payload.get("second_original_shape")
        if first_original_shape is not None:
            first_points = _map_points_to_shape(
                first_points, first_shape, first_original_shape
            )
            first_output_shape = tuple(int(v) for v in first_original_shape)
        else:
            first_output_shape = first_shape
        if second_original_shape is not None:
            second_points = _map_points_to_shape(
                second_points, second_shape, second_original_shape
            )
            second_output_shape = tuple(int(v) for v in second_original_shape)
        else:
            second_output_shape = second_shape
        matches, diagnostics = loma_matches_from_points(
            first_points,
            second_points,
            payload["calibration"],
            payload["config"],
            first_output_shape,
            second_output_shape,
            first_dynamic_mask=payload.get("first_dynamic_mask"),
            second_dynamic_mask=payload.get("second_dynamic_mask"),
        )
        diagnostics.update(
            {
                "requested_device": bundle["requested_device"],
                "resolved_device": bundle["resolved_device"],
                "device_fallback": bool(bundle["device_fallback"]),
                "model_id": bundle["model_name"],
                "first_scale": float(first_scale),
                "second_scale": float(second_scale),
                "first_input_shape": first_shape,
                "second_input_shape": second_shape,
                "first_original_shape": first_output_shape,
                "second_original_shape": second_output_shape,
                **_loma_matcher_diagnostics(
                    matcher_model,
                    requested_backend=compile_backend,
                ),
                **cache_stats,
            }
        )
        results.append((matches, diagnostics))
    return results
