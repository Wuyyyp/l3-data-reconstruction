from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .dataset import Frame
from .semantic_masks import (
    DEFAULT_DYNAMIC_CLASSES,
    CachedMaskResult,
    canonicalize_label,
    load_dynamic_mask,
    predictor_uses_confidence_threshold,
)


MIN_OBSERVABLE_BASELINE_M = 0.05
DEFAULT_MAX_TRIANGULATED_POINT_DISTANCE_M = 100.0


def _replace_dataframe_columns_once(
    frame: pd.DataFrame,
    columns: dict[str, object],
) -> pd.DataFrame:
    """Replace and append columns without rebuilding every existing column."""
    if not columns:
        return frame.copy()
    ordered_columns = list(frame.columns)
    ordered_columns.extend(name for name in columns if name not in frame.columns)
    replaced = [name for name in columns if name in frame.columns]
    base = frame.drop(columns=replaced) if replaced else frame
    additions = pd.DataFrame(columns, index=frame.index)
    result = pd.concat([base, additions], axis=1)
    if replaced:
        result = result.loc[:, ordered_columns]
    return result


def _retained_mask(matches: pd.DataFrame) -> pd.Series:
    if "quality_filter_valid" in matches.columns:
        return matches["quality_filter_valid"].astype(bool)
    mask = (
        matches["pose_quality_valid"].astype(bool)
        & matches["reprojection_threshold_inlier"].astype(bool)
    )
    if "precision_filter_valid" in matches.columns:
        mask = mask & matches["precision_filter_valid"].astype(bool)
    return mask


class SemanticMaskIndex:
    def __init__(
        self,
        cache_root: Path,
        camera: str,
        model_name: str,
        backend: str,
        confidence: float,
        dilation_pixels: int,
        inference_dtype: str = "fp16",
        use_fast_processor: bool = True,
        dynamic_classes: frozenset[str] | set[str] = DEFAULT_DYNAMIC_CLASSES,
    ):
        self.paths: dict[int, Path] = {}
        self._results: dict[int, CachedMaskResult] = {}
        self._scores: dict[int, int] = {}
        self.dynamic_classes = frozenset(dynamic_classes)
        self.dilation_pixels = int(dilation_pixels)
        expected_model = Path(model_name).name
        metadata_candidates = list(
            Path(cache_root).glob(f"backends/*/*/{camera}/metadata/*.json")
        ) + list(Path(cache_root).glob(f"*/{camera}/*.json"))
        for metadata_path in metadata_candidates:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "timestamp_ns" not in metadata:
                continue
            confidence_ok = (
                True
                if not predictor_uses_confidence_threshold(backend)
                else float(metadata.get("confidence_threshold", -1)) == confidence
            )
            if (
                metadata.get("backend", "mask2former") == backend
                and Path(str(metadata.get("model_name", ""))).name == expected_model
                and confidence_ok
                and int(metadata.get("schema_version", 1)) >= 4
                and str(metadata.get("inference_dtype", "")) == str(inference_dtype)
                and bool(metadata.get("use_fast_processor", False))
                == bool(use_fast_processor)
            ):
                timestamp = int(metadata["timestamp_ns"])
                if metadata_path.parent.name == "metadata":
                    camera_root = metadata_path.parent.parent
                    mask_path = camera_root / "masks" / metadata_path.with_suffix(".png").name
                    class_mask_path = (
                        camera_root / "class_masks" / metadata_path.with_suffix(".png").name
                    )
                else:
                    mask_path = metadata_path.with_suffix(".png")
                    class_mask_path = Path(str(metadata.get("class_mask_path", "")))
                if not class_mask_path.exists():
                    continue
                score = int(
                    int(metadata.get("dilation_pixels", -1)) == self.dilation_pixels
                    and {
                        canonicalize_label(value)
                        for value in metadata.get("dynamic_classes", [])
                    }
                    == {canonicalize_label(value) for value in self.dynamic_classes}
                )
                if score < self._scores.get(timestamp, -1):
                    continue
                result = CachedMaskResult(
                    mask_path=mask_path,
                    class_mask_path=class_mask_path,
                    metadata_path=metadata_path,
                    preview_path=metadata_path.parent.parent / "previews" / f"{timestamp}.jpg",
                    cache_hit=True,
                    instance_count=int(metadata.get("instance_count", 0)),
                    dynamic_pixel_ratio=float(metadata.get("dynamic_pixel_ratio", 0.0)),
                )
                self._scores[timestamp] = score
                self._results[timestamp] = result
                self.paths[timestamp] = class_mask_path

    def load(self, frame: Frame) -> np.ndarray:
        result = self._results.get(frame.timestamp_ns)
        if result is None:
            raise FileNotFoundError(
                f"missing semantic mask for timestamp {frame.timestamp_ns}"
            )
        return load_dynamic_mask(
            result,
            dynamic_classes=self.dynamic_classes,
            dilation_pixels=self.dilation_pixels,
        )


def _annotate_match_quality_thresholds(
    matched: pd.DataFrame,
    *,
    motion_pair_valid: bool,
    threshold_px: float,
    max_triangulated_point_distance_m: float,
    min_triangulation_angle_deg: float,
    max_ray_closest_distance_m: float,
    ray_closest_distance_mode: str = "fixed",
    ray_closest_distance_min_m: float | None = None,
    ray_closest_distance_max_m: float | None = None,
    ray_closest_distance_mad_k: float = 2.5,
    ray_closest_distance_min_candidates: int = 20,
    ray_closest_distance_insufficient_candidates_threshold_m: float = 0.10,
    reprojection_threshold_mode: str = "fixed",
    reprojection_threshold_min_px: float | None = None,
    reprojection_threshold_max_px: float | None = None,
    reprojection_threshold_mad_k: float = 2.5,
    reprojection_threshold_min_candidates: int = 30,
    reprojection_insufficient_candidates_threshold_px: float = 5.0,
) -> pd.DataFrame:
    result = matched.copy()
    default_metric_columns: dict[str, object] = {}
    if "triangulated_point_distance_m" not in result.columns:
        default_metric_columns["triangulated_point_distance_m"] = 0.0
    if "triangulation_angle_deg" not in result.columns:
        default_metric_columns["triangulation_angle_deg"] = 180.0
    if "ray_closest_distance_m" not in result.columns:
        default_metric_columns["ray_closest_distance_m"] = 0.0
    if "reprojection1_px" not in result.columns:
        default_metric_columns["reprojection1_px"] = result.get(
            "reprojection_rms_px", np.nan
        )
    if "reprojection2_px" not in result.columns:
        default_metric_columns["reprojection2_px"] = result.get(
            "reprojection_rms_px", np.nan
        )
    if default_metric_columns:
        result = _replace_dataframe_columns_once(result, default_metric_columns)

    distance = result["triangulated_point_distance_m"].to_numpy(dtype=np.float64)
    angle = result["triangulation_angle_deg"].to_numpy(dtype=np.float64)
    ray_gap = result["ray_closest_distance_m"].to_numpy(dtype=np.float64)
    reproj1 = result["reprojection1_px"].to_numpy(dtype=np.float64)
    reproj2 = result["reprojection2_px"].to_numpy(dtype=np.float64)
    reproj_rms = result["reprojection_rms_px"].to_numpy(dtype=np.float64)
    valid_depth = result["triangulation_valid_depth"].astype(bool).to_numpy()

    distance_valid = np.isfinite(distance) & (
        distance <= max_triangulated_point_distance_m
    )
    angle_valid = np.isfinite(angle) & (angle >= min_triangulation_angle_deg)
    reprojection_threshold = float(threshold_px)
    reprojection_median = np.nan
    reprojection_mad = np.nan
    reprojection_candidate_count = 0
    finite_reprojection = (
        np.isfinite(reproj_rms) & np.isfinite(reproj1) & np.isfinite(reproj2)
    )
    adaptive_candidate_base = (
        valid_depth & distance_valid & angle_valid & finite_reprojection
    )
    if str(reprojection_threshold_mode) == "adaptive_mad":
        candidate_reprojection = reproj_rms[adaptive_candidate_base]
        reprojection_candidate_count = int(candidate_reprojection.size)
        if reprojection_candidate_count >= int(reprojection_threshold_min_candidates):
            reprojection_median = float(np.median(candidate_reprojection))
            reprojection_mad = float(
                np.median(np.abs(candidate_reprojection - reprojection_median))
            )
            adaptive_threshold = reprojection_median + (
                float(reprojection_threshold_mad_k) * 1.4826 * reprojection_mad
            )
            if reprojection_threshold_min_px is not None:
                adaptive_threshold = max(
                    adaptive_threshold, float(reprojection_threshold_min_px)
                )
            if reprojection_threshold_max_px is not None:
                adaptive_threshold = min(
                    adaptive_threshold, float(reprojection_threshold_max_px)
                )
            reprojection_threshold = float(adaptive_threshold)
        else:
            reprojection_threshold = float(
                reprojection_insufficient_candidates_threshold_px
            )
    elif str(reprojection_threshold_mode) != "fixed":
        raise ValueError("reprojection_threshold_mode must be 'fixed' or 'adaptive_mad'")
    reprojection_values_valid = (
        finite_reprojection
        & (reproj1 <= reprojection_threshold)
        & (reproj2 <= reprojection_threshold)
        & (reproj_rms <= reprojection_threshold)
    )
    ray_threshold = float(max_ray_closest_distance_m)
    ray_median = np.nan
    ray_mad = np.nan
    ray_candidate_count = 0
    if str(ray_closest_distance_mode) == "adaptive_mad":
        adaptive_candidates = adaptive_candidate_base & np.isfinite(ray_gap)
        candidate_ray = ray_gap[adaptive_candidates]
        ray_candidate_count = int(candidate_ray.size)
        if ray_candidate_count >= int(ray_closest_distance_min_candidates):
            ray_median = float(np.median(candidate_ray))
            ray_mad = float(np.median(np.abs(candidate_ray - ray_median)))
            adaptive_threshold = ray_median + (
                float(ray_closest_distance_mad_k) * 1.4826 * ray_mad
            )
            if ray_closest_distance_min_m is not None:
                adaptive_threshold = max(
                    adaptive_threshold, float(ray_closest_distance_min_m)
                )
            if ray_closest_distance_max_m is not None:
                adaptive_threshold = min(
                    adaptive_threshold, float(ray_closest_distance_max_m)
                )
            ray_threshold = float(adaptive_threshold)
        else:
            ray_threshold = float(
                ray_closest_distance_insufficient_candidates_threshold_m
            )
    elif str(ray_closest_distance_mode) != "fixed":
        raise ValueError("ray_closest_distance_mode must be 'fixed' or 'adaptive_mad'")

    ray_gap_valid = np.isfinite(ray_gap) & (ray_gap <= ray_threshold)
    pose_quality_valid = (
        valid_depth
        & distance_valid
        & angle_valid
        & ray_gap_valid
        & bool(motion_pair_valid)
    )
    reprojection_threshold_inlier = (
        pose_quality_valid
        & np.isfinite(reproj_rms)
        & np.isfinite(reproj1)
        & np.isfinite(reproj2)
        & (reproj1 <= reprojection_threshold)
        & (reproj2 <= reprojection_threshold)
        & (reproj_rms <= reprojection_threshold)
    )
    return _replace_dataframe_columns_once(
        result,
        {
            "triangulation_distance_valid": distance_valid,
            "triangulation_angle_valid": angle_valid,
            "ray_closest_distance_valid": ray_gap_valid,
            "ray_closest_distance_threshold_m": ray_threshold,
            "ray_closest_distance_threshold_mode": str(ray_closest_distance_mode),
            "ray_closest_distance_candidate_count": ray_candidate_count,
            "ray_closest_distance_median_m": ray_median,
            "ray_closest_distance_mad_m": ray_mad,
            "ray_closest_distance_mad_k": float(ray_closest_distance_mad_k),
            "reprojection_threshold_px": reprojection_threshold,
            "reprojection_threshold_mode": str(reprojection_threshold_mode),
            "reprojection_threshold_candidate_count": reprojection_candidate_count,
            "reprojection_threshold_median_px": reprojection_median,
            "reprojection_threshold_mad_px": reprojection_mad,
            "reprojection_threshold_mad_k": float(reprojection_threshold_mad_k),
            "pose_quality_valid": pose_quality_valid,
            "reprojection_threshold_inlier": reprojection_threshold_inlier,
        },
    )
