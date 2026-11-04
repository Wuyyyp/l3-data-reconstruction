from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import math
import numbers

import numpy as np
import pandas as pd

from .camera_models import undistort_normalized
from .config import BACKEND_NAMES
from .features import match_frame_pair

UNIFIED_MATCH_COLUMNS = [
    "kp1",
    "kp2",
    "x1",
    "y1",
    "x2",
    "y2",
    "nx1",
    "ny1",
    "nx2",
    "ny2",
    "descriptor_distance",
    "ratio",
    "lk_fb_error",
    "lk_valid",
    "ransac_inlier",
    "backend",
    "backend_score",
]


def backend_names() -> list[str]:
    return list(BACKEND_NAMES)


def _is_numeric(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _validate_backend(record: dict) -> str:
    backend = record.get("backend")
    if backend is None:
        raise ValueError("backend is required")
    if not isinstance(backend, str):
        raise ValueError("backend must be a string")
    if backend not in backend_names():
        raise ValueError(f"unknown backend: {backend}")
    return backend


def _validate_required_numeric(record: dict, field: str) -> object:
    if field not in record:
        raise ValueError(f"{field} is required")
    value = record[field]
    if not _is_numeric(value):
        raise ValueError(f"{field} must be numeric")
    return value


def _validate_optional_integer(record: dict, field: str, default: int) -> int:
    if field not in record:
        return default
    value = record[field]
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _validate_optional_numeric(record: dict, field: str) -> object:
    if field not in record:
        return math.nan
    value = record[field]
    if not _is_numeric(value):
        raise ValueError(f"{field} must be numeric")
    return value


def _validate_optional_bool(record: dict, field: str, default: bool) -> bool:
    if field not in record:
        return default
    value = record[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be bool")
    return value


def _normalize_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("each record must be a mapping")

    backend = _validate_backend(record)

    return {
        "kp1": _validate_optional_integer(record, "kp1", -1),
        "kp2": _validate_optional_integer(record, "kp2", -1),
        "x1": _validate_required_numeric(record, "x1"),
        "y1": _validate_required_numeric(record, "y1"),
        "x2": _validate_required_numeric(record, "x2"),
        "y2": _validate_required_numeric(record, "y2"),
        "nx1": _validate_required_numeric(record, "nx1"),
        "ny1": _validate_required_numeric(record, "ny1"),
        "nx2": _validate_required_numeric(record, "nx2"),
        "ny2": _validate_required_numeric(record, "ny2"),
        "descriptor_distance": _validate_optional_numeric(record, "descriptor_distance"),
        "ratio": _validate_optional_numeric(record, "ratio"),
        "lk_fb_error": _validate_optional_numeric(record, "lk_fb_error"),
        "lk_valid": _validate_optional_bool(record, "lk_valid", True),
        "ransac_inlier": _validate_optional_bool(record, "ransac_inlier", False),
        "backend": backend,
        "backend_score": _validate_optional_numeric(record, "backend_score"),
    }


def normalize_match_records(records: list[dict]) -> pd.DataFrame:
    normalized_records = [_normalize_record(record) for record in records]
    return pd.DataFrame.from_records(normalized_records, columns=UNIFIED_MATCH_COLUMNS)


def _restore_original_match_coordinates(
    matches: pd.DataFrame,
    *,
    first_image,
    second_image,
    calibration,
    first_original_shape,
    second_original_shape,
) -> pd.DataFrame:
    """Restore cached-image matches to the calibration/original-image frame."""
    if matches.empty:
        return matches
    from .deep_matchers import _map_points_to_shape

    first_shape = tuple(int(v) for v in np.asarray(first_image).shape[:2])
    second_shape = tuple(int(v) for v in np.asarray(second_image).shape[:2])
    first_target = (
        first_shape
        if first_original_shape is None
        else tuple(int(v) for v in first_original_shape)
    )
    second_target = (
        second_shape
        if second_original_shape is None
        else tuple(int(v) for v in second_original_shape)
    )
    result = matches.copy()
    first_points = _map_points_to_shape(
        result[["x1", "y1"]].to_numpy(dtype="float64"),
        first_shape,
        first_target,
    )
    second_points = _map_points_to_shape(
        result[["x2", "y2"]].to_numpy(dtype="float64"),
        second_shape,
        second_target,
    )
    result[["x1", "y1"]] = first_points
    result[["x2", "y2"]] = second_points
    result[["nx1", "ny1"]] = undistort_normalized(first_points, calibration)
    result[["nx2", "ny2"]] = undistort_normalized(second_points, calibration)
    result.attrs.update(matches.attrs)
    return result


def run_match_backend(
    first_image,
    second_image,
    calibration,
    config,
    first_semantic_mask=None,
    second_semantic_mask=None,
    first_image_key=None,
    second_image_key=None,
    feature_cache=None,
    cache_device=None,
    cache_max_items=None,
    gpu_cache=None,
    cache_cpu_max_items=None,
    first_original_shape=None,
    second_original_shape=None,
) -> pd.DataFrame:
    diagnostics = {}
    if config.backend == "sift_lk":
        matches = match_frame_pair(
            first_image,
            second_image,
            calibration,
            config,
            first_dynamic_mask=first_semantic_mask,
            second_dynamic_mask=second_semantic_mask,
        )
        diagnostics.update(matches.attrs)
    elif config.backend == "superpoint_lightglue":
        from .deep_matchers import superpoint_lightglue_matches

        matches, diagnostics = superpoint_lightglue_matches(
            first_image,
            second_image,
            calibration,
            config,
            first_dynamic_mask=first_semantic_mask,
            second_dynamic_mask=second_semantic_mask,
        )
    elif config.backend == "loftr":
        from .deep_matchers import loftr_matches

        matches, diagnostics = loftr_matches(
            first_image,
            second_image,
            calibration,
            config,
            first_dynamic_mask=first_semantic_mask,
            second_dynamic_mask=second_semantic_mask,
        )
    elif config.backend == "loma":
        from .deep_matchers import loma_matches

        matches, diagnostics = loma_matches(
            first_image,
            second_image,
            calibration,
            config,
            first_dynamic_mask=first_semantic_mask,
            second_dynamic_mask=second_semantic_mask,
            first_image_key=first_image_key,
            second_image_key=second_image_key,
            feature_cache=feature_cache,
            cache_device=cache_device,
            cache_max_items=cache_max_items,
            gpu_cache=gpu_cache,
            cache_cpu_max_items=cache_cpu_max_items,
            first_original_shape=first_original_shape,
            second_original_shape=second_original_shape,
        )
    else:
        raise ValueError(f"backend not implemented yet: {config.backend}")

    if config.backend != "loma":
        matches = _restore_original_match_coordinates(
            matches,
            first_image=first_image,
            second_image=second_image,
            calibration=calibration,
            first_original_shape=first_original_shape,
            second_original_shape=second_original_shape,
        )
    matches = matches.reindex(columns=UNIFIED_MATCH_COLUMNS)
    if config.backend == "sift_lk" and not matches.empty:
        matches["backend"] = "sift_lk"
        matches["backend_score"] = math.nan
    matches.attrs.update(diagnostics)
    return matches


def apply_semantic_masks(
    matches: pd.DataFrame,
    *,
    first_mask,
    second_mask,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
) -> pd.DataFrame:
    """Apply post-hoc dynamic masks to backend matches reused from preflight."""
    if matches.empty:
        return matches.copy()
    from .deep_matchers import _static_points_mask

    first_keep = _static_points_mask(
        first_mask,
        first_shape,
        matches[["x1", "y1"]].to_numpy(dtype="float64"),
    )
    second_keep = _static_points_mask(
        second_mask,
        second_shape,
        matches[["x2", "y2"]].to_numpy(dtype="float64"),
    )
    result = matches.loc[first_keep & second_keep].copy()
    result.attrs.update(matches.attrs)
    return result.reset_index(drop=True)


def _run_match_payload(payload: dict[str, object]) -> pd.DataFrame:
    return run_match_backend(
        payload["first_image"],
        payload["second_image"],
        payload["calibration"],
        payload["config"],
        first_semantic_mask=payload.get("first_dynamic_mask"),
        second_semantic_mask=payload.get("second_dynamic_mask"),
        first_image_key=payload.get("first_image_key"),
        second_image_key=payload.get("second_image_key"),
        first_original_shape=payload.get("first_original_shape"),
        second_original_shape=payload.get("second_original_shape"),
    )


def run_match_backend_batch(
    pair_payloads: list[dict[str, object]],
) -> list[pd.DataFrame]:
    if not pair_payloads:
        return []
    backend = pair_payloads[0]["config"].backend
    if any(payload["config"].backend != backend for payload in pair_payloads):
        raise ValueError("all batch payloads must use the same backend")
    if backend == "superpoint_lightglue":
        from .deep_matchers import superpoint_lightglue_matches_batch

        raw_results = superpoint_lightglue_matches_batch(pair_payloads)
    elif backend == "loma":
        process_count = max(
            1, int(getattr(pair_payloads[0]["config"], "loma_process_count", 1))
        )
        if process_count > 1:
            # CUDA/PPU contexts must be created in spawned children, never forked.
            with ProcessPoolExecutor(
                max_workers=process_count,
                mp_context=get_context("spawn"),
            ) as executor:
                return list(executor.map(_run_match_payload, pair_payloads))
        from .deep_matchers import loma_matches_batch

        raw_results = loma_matches_batch(pair_payloads)
    else:
        return [
            run_match_backend(
                payload["first_image"],
                payload["second_image"],
                payload["calibration"],
                payload["config"],
                first_semantic_mask=payload.get("first_dynamic_mask"),
                second_semantic_mask=payload.get("second_dynamic_mask"),
                first_original_shape=payload.get("first_original_shape"),
                second_original_shape=payload.get("second_original_shape"),
            )
            for payload in pair_payloads
        ]

    results: list[pd.DataFrame] = []
    for index, (matches, diagnostics) in enumerate(raw_results):
        if backend != "loma":
            payload = pair_payloads[index]
            matches = _restore_original_match_coordinates(
                matches,
                first_image=payload["first_image"],
                second_image=payload["second_image"],
                calibration=payload["calibration"],
                first_original_shape=payload.get("first_original_shape"),
                second_original_shape=payload.get("second_original_shape"),
            )
        matches = matches.reindex(columns=UNIFIED_MATCH_COLUMNS)
        matches.attrs.update(diagnostics)
        results.append(matches)
    return results
