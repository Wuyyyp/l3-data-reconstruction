from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

from .camera_models import CameraCalibration, project_points, undistort_normalized
from .config import load_config
from .dataset import Dataset
from .evaluation import summarize_errors
from .geometry import (
    epipolar_errors,
    essential_from_relative_pose,
    fundamental_from_essential,
    ray_geometry_metrics,
    triangulate_and_reproject,
)
from .match_cache import (
    inter_camera_cache_dir,
    intra_camera_baseline_cache_dir,
    intra_camera_cache_dir,
    match_cache_key,
    write_match_cache_artifacts,
)
from .match_backends import run_match_backend, run_match_backend_batch
from .image_store import ImageStore, image_store_options
from .navigation import NavigationTrajectory, navigation_time_offset_ns
from .pkl_pose import (
    calibration_from_result_interpolation,
    calibrations_from_result_interpolation,
    frames_from_image_root,
    frame_metadata_from_result_interpolation,
    navigation_from_result_interpolation,
    resolve_result_interpolation_image_root,
    resolve_result_interpolation_path,
)
from .match_quality import (
    DEFAULT_MAX_TRIANGULATED_POINT_DISTANCE_M,
    MIN_OBSERVABLE_BASELINE_M,
    SemanticMaskIndex,
    _annotate_match_quality_thresholds,
    _replace_dataframe_columns_once,
    _retained_mask,
)
from .semantic_masks import DEFAULT_DYNAMIC_CLASSES
from .semantic_pipeline import segment_camera


def _should_export_match_visualization(frame_index: int, sample_ratio: float) -> bool:
    """Deterministically select approximately ``sample_ratio`` of frame pairs."""
    stride = max(1, round(1.0 / float(sample_ratio)))
    return int(frame_index) % stride == 0


def _config_for_matching_scope(config, scope: str):
    matching = getattr(config, "matching", None)
    if matching is None:
        return config
    scoped_threshold = getattr(
        matching,
        f"deep_score_threshold_{scope}",
        None,
    )
    if scoped_threshold is None:
        return config
    if hasattr(matching, "model_copy"):
        scoped_matching = matching.model_copy(
            update={"deep_score_threshold": float(scoped_threshold)}
        )
        if hasattr(config, "model_copy"):
            return config.model_copy(update={"matching": scoped_matching})
        config.matching = scoped_matching
        return config
    try:
        matching.deep_score_threshold = float(scoped_threshold)
    except Exception:
        pass
    return config


def _result_interpolation_path_for_config(config) -> Path | None:
    return resolve_result_interpolation_path(
        dict(getattr(config, "assumptions", {}) or {}),
        Path(config.dataset_root),
    )


def _navigation_for_config(config) -> NavigationTrajectory:
    pkl_path = _result_interpolation_path_for_config(config)
    if pkl_path is not None:
        assumptions = dict(getattr(config, "assumptions", {}) or {})
        return navigation_from_result_interpolation(
            pkl_path,
            ego_field=str(
                assumptions.get(
                    "result_interpolation_ego_field",
                    "ego2global_transformation_matrix",
                )
            ),
        )
    assumptions = dict(getattr(config, "assumptions", {}) or {})
    return NavigationTrajectory.from_csv(
        Path(config.dataset_root) / assumptions["navigation_source"]
    )


def _calibration_for_config(config, camera: str) -> CameraCalibration:
    pkl_path = _result_interpolation_path_for_config(config)
    if pkl_path is not None:
        assumptions = dict(getattr(config, "assumptions", {}) or {})
        return calibration_from_result_interpolation(
            pkl_path,
            camera,
            extrinsic_direction=str(
                assumptions.get(
                    "result_interpolation_extrinsic_direction",
                    "camera_to_car",
                )
            ),
            images_undistorted=bool(
                assumptions.get("result_interpolation_images_undistorted", False)
            ),
        )
    return CameraCalibration.from_dataset(Path(config.dataset_root), camera)


def _calibrations_for_config(config, cameras: tuple[str, ...] | list[str]):
    pkl_path = _result_interpolation_path_for_config(config)
    if pkl_path is not None:
        assumptions = dict(getattr(config, "assumptions", {}) or {})
        return calibrations_from_result_interpolation(
            pkl_path,
            tuple(cameras),
            extrinsic_direction=str(
                assumptions.get(
                    "result_interpolation_extrinsic_direction",
                    "camera_to_car",
                )
            ),
            images_undistorted=bool(
                assumptions.get("result_interpolation_images_undistorted", False)
            ),
        )
    return {
        camera: CameraCalibration.from_dataset(Path(config.dataset_root), camera)
        for camera in cameras
    }


def _frames_for_config(config, camera: str):
    pkl_path = _result_interpolation_path_for_config(config)
    if pkl_path is not None:
        image_root = resolve_result_interpolation_image_root(
            dict(getattr(config, "assumptions", {}) or {}),
            Path(config.dataset_root),
        )
        if image_root is not None:
            return frames_from_image_root(image_root, str(camera))
    return Dataset(Path(config.dataset_root)).frames(str(camera))


def _recompute_normalized_coordinates_for_calibrations(
    matches: pd.DataFrame,
    calibrations: dict[str, CameraCalibration],
) -> pd.DataFrame:
    result = matches.copy()
    if result.empty:
        return result

    def _can_recompute(calibration: object) -> bool:
        return all(
            hasattr(calibration, name)
            for name in ("intrinsic", "distortion", "images_undistorted", "model")
        )

    if {"camera1", "camera2"}.issubset(result.columns):
        camera_pairs = (
            ("camera1", "x1", "y1", "nx1", "ny1"),
            ("camera2", "x2", "y2", "nx2", "ny2"),
        )
        for camera_col, x_col, y_col, nx_col, ny_col in camera_pairs:
            for camera, index in result.groupby(camera_col, sort=False).groups.items():
                calibration = calibrations.get(str(camera))
                if calibration is None or not _can_recompute(calibration):
                    continue
                points = result.loc[index, [x_col, y_col]].to_numpy(dtype=np.float64)
                result.loc[index, [nx_col, ny_col]] = undistort_normalized(
                    points,
                    calibration,
                )
        return result
    if "camera" in result.columns and len(calibrations) == 1:
        calibration = next(iter(calibrations.values()))
        if not _can_recompute(calibration):
            return result
        result[["nx1", "ny1"]] = undistort_normalized(
            result[["x1", "y1"]].to_numpy(dtype=np.float64),
            calibration,
        )
        result[["nx2", "ny2"]] = undistort_normalized(
            result[["x2", "y2"]].to_numpy(dtype=np.float64),
            calibration,
        )
    return result


def _assumption_for_camera_pair(
    assumptions: dict[str, object],
    key: str,
    first_camera: str,
    second_camera: str,
    default: object,
) -> object:
    pair_key = f"{first_camera}__{second_camera}"
    reverse_pair_key = f"{second_camera}__{first_camera}"
    override_key = f"{key}_by_pair" if key.startswith("inter_camera_") else f"inter_camera_{key}_by_pair"
    pair_overrides = assumptions.get(override_key, {})
    if isinstance(pair_overrides, dict):
        if pair_key in pair_overrides:
            return pair_overrides[pair_key]
        if reverse_pair_key in pair_overrides:
            return pair_overrides[reverse_pair_key]
    return assumptions.get(key, default)


def _config_for_inter_camera_pair(config, first_camera: str, second_camera: str):
    assumptions = dict(getattr(config, "assumptions", {}) or {})
    threshold = _assumption_for_camera_pair(
        assumptions,
        "deep_score_threshold",
        first_camera,
        second_camera,
        None,
    )
    if threshold is None:
        return config
    matching = getattr(config, "matching", None)
    if matching is None:
        return config
    if hasattr(matching, "model_copy"):
        scoped_matching = matching.model_copy(
            update={"deep_score_threshold": float(threshold)}
        )
        if hasattr(config, "model_copy"):
            return config.model_copy(update={"matching": scoped_matching})
        config.matching = scoped_matching
        return config
    try:
        matching.deep_score_threshold = float(threshold)
    except Exception:
        pass
    return config


def segment_dataset_stage(
    *,
    config_path: Path,
    dataset_root: Path,
    cache_root: Path,
    camera: str,
    mask_config,
    start_frame: int = 0,
    frame_count: int | None = None,
    force: bool = False,
) -> Path:
    stage_root = Path(cache_root) / "reports" / camera
    stage_root.mkdir(parents=True, exist_ok=True)
    frames = None
    if Path(config_path).exists():
        frames = _frames_for_config(load_config(config_path), str(camera))
    report = segment_camera(
        dataset_root=dataset_root,
        camera=camera,
        config=mask_config,
        start_frame=start_frame,
        frame_count=frame_count,
        force=force,
        frames=frames,
    )
    report.to_csv(stage_root / "segment_report.csv", index=False)
    return stage_root


def _supports_batch_backend(backend: str) -> bool:
    return backend in {"superpoint_lightglue", "loma"}


def _resolve_window(
    total_frames: int,
    *,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> tuple[int, int]:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if start_frame > total_frames:
        raise ValueError("start_frame exceeds available frames")
    end_frame = (
        total_frames
        if frame_count is None
        else min(total_frames, start_frame + frame_count)
    )
    return start_frame, end_frame


def _build_baseline_frame_pairs(
    *,
    frame_indices: list[int],
    camera_centers_by_index: dict[int, np.ndarray],
    target_baseline_m: float,
) -> list[tuple[int, int, float]]:
    if target_baseline_m <= 0:
        raise ValueError("target_baseline_m must be positive")
    ordered = [int(value) for value in frame_indices]
    pairs: list[tuple[int, int, float]] = []
    for anchor_pos, anchor_index in enumerate(ordered):
        first_center = camera_centers_by_index.get(anchor_index)
        if first_center is None:
            continue

        def best_candidate(candidates: list[int]) -> tuple[int, float] | None:
            best: tuple[int, float, float] | None = None
            for candidate_index in candidates:
                second_center = camera_centers_by_index.get(candidate_index)
                if second_center is None:
                    continue
                baseline = float(np.linalg.norm(second_center - first_center))
                if baseline < float(target_baseline_m):
                    continue
                distance_to_target = abs(baseline - float(target_baseline_m))
                if best is None or distance_to_target < best[0]:
                    best = (distance_to_target, candidate_index, baseline)
            if best is None:
                return None
            return int(best[1]), float(best[2])

        future = ordered[anchor_pos + 1 :]
        past = ordered[:anchor_pos]
        selected = best_candidate(future) or best_candidate(past)
        if selected is None:
            continue
        second_index, baseline = selected
        pairs.append((anchor_index, second_index, baseline))
    return pairs


def _deduplicate_baseline_frame_pairs(
    pairs_by_baseline: dict[float, list[tuple[int, int, float]]],
) -> dict[float, list[tuple[int, int, float]]]:
    seen_endpoint_sets: set[tuple[int, int]] = set()
    deduped: dict[float, list[tuple[int, int, float]]] = {}
    for target_baseline_m in sorted(pairs_by_baseline):
        kept: list[tuple[int, int, float]] = []
        for first_index, second_index, pair_baseline_m in pairs_by_baseline[
            target_baseline_m
        ]:
            key = tuple(sorted((int(first_index), int(second_index))))
            if key in seen_endpoint_sets:
                continue
            seen_endpoint_sets.add(key)
            kept.append((int(first_index), int(second_index), float(pair_baseline_m)))
        deduped[float(target_baseline_m)] = kept
    return deduped


def _camera_centers_by_frame_index(
    *,
    frames: list,
    calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    navigation_offset_ns: int = 0,
) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for frame in frames:
        try:
            body_to_world = navigation.pose_at(
                int(frame.timestamp_ns) + int(navigation_offset_ns)
            )
        except ValueError:
            continue
        camera_to_world = body_to_world @ calibration.sensor_to_car
        centers[int(frame.index)] = camera_to_world[:3, 3].copy()
    return centers


def _camera_view_dirs_by_frame_index(
    *,
    frames: list,
    calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    navigation_offset_ns: int = 0,
) -> dict[int, np.ndarray]:
    view_dirs: dict[int, np.ndarray] = {}
    optical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for frame in frames:
        try:
            body_to_world = navigation.pose_at(
                int(frame.timestamp_ns) + int(navigation_offset_ns)
            )
        except ValueError:
            continue
        camera_to_world = body_to_world @ calibration.sensor_to_car
        direction = camera_to_world[:3, :3] @ optical_axis
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        view_dirs[int(frame.index)] = direction / norm
    return view_dirs


def _build_temporal_overlap_pairs(
    *,
    first_frames: list,
    second_frames: list,
    first_centers_by_index: dict[int, np.ndarray],
    second_centers_by_index: dict[int, np.ndarray],
    first_view_dirs_by_index: dict[int, np.ndarray],
    second_view_dirs_by_index: dict[int, np.ndarray],
    target_baseline_m: float,
    max_frame_gap: int,
    min_opposite_view_dot: float,
) -> list[tuple[int, int]]:
    if target_baseline_m <= 0:
        raise ValueError("target_baseline_m must be positive")
    if max_frame_gap <= 0:
        raise ValueError("max_frame_gap must be positive")

    second_frame_by_index = {int(frame.index): frame for frame in second_frames}
    seen_endpoint_sets: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    ordered_first = sorted(first_frames, key=lambda frame: int(frame.index))
    ordered_second_indices = sorted(int(frame.index) for frame in second_frames)

    for first in ordered_first:
        first_index = int(first.index)
        first_center = first_centers_by_index.get(first_index)
        first_dir = first_view_dirs_by_index.get(first_index)
        if first_center is None or first_dir is None:
            continue

        def _select_candidate(candidate_indices: list[int]) -> tuple[int, float] | None:
            best: tuple[float, int, float] | None = None
            for second_index in candidate_indices:
                if abs(int(second_index) - first_index) > int(max_frame_gap):
                    continue
                second_center = second_centers_by_index.get(int(second_index))
                second_dir = second_view_dirs_by_index.get(int(second_index))
                if second_center is None or second_dir is None:
                    continue
                view_dot = float(np.dot(first_dir, second_dir))
                if view_dot > float(min_opposite_view_dot):
                    continue
                baseline = float(np.linalg.norm(second_center - first_center))
                if baseline < float(target_baseline_m):
                    continue
                distance_to_target = abs(baseline - float(target_baseline_m))
                if best is None or distance_to_target < best[0]:
                    best = (distance_to_target, int(second_index), baseline)
            if best is None:
                return None
            return int(best[1]), float(best[2])

        future = [idx for idx in ordered_second_indices if idx > first_index]
        selected = _select_candidate(future)
        if selected is None:
            continue
        second_index, _baseline = selected
        endpoint_key = tuple(sorted((first_index, second_index)))
        if endpoint_key in seen_endpoint_sets:
            continue
        if second_index not in second_frame_by_index:
            continue
        seen_endpoint_sets.add(endpoint_key)
        pairs.append((first_index, int(second_index)))
    return pairs


def _is_temporal_overlap_direction_allowed(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> bool:
    allowed_pairs = assumptions.get("inter_camera_temporal_overlap_allowed_pairs", [])
    if isinstance(allowed_pairs, (list, tuple, set)):
        pair_key = f"{first_camera}__{second_camera}"
        reverse_pair_key = f"{second_camera}__{first_camera}"
        allowed = {str(value) for value in allowed_pairs}
        if pair_key in allowed or reverse_pair_key in allowed:
            return True
        if len(allowed_pairs) > 0:
            return False
    front_group = {
        "center_camera_fov30",
        "center_camera_fov120",
        "front_camera_fov195",
        "left_front_camera",
        "right_front_camera",
    }
    rear_group = {
        "rear_camera",
        "rear_camera_fov195",
        "left_rear_camera",
        "right_rear_camera",
    }
    return str(first_camera) in front_group and str(second_camera) in rear_group


def _declared_temporal_overlap_pair(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> tuple[str, str]:
    allowed_pairs = assumptions.get("inter_camera_temporal_overlap_allowed_pairs", [])
    if isinstance(allowed_pairs, (list, tuple, set)):
        pair_key = f"{first_camera}__{second_camera}"
        reverse_pair_key = f"{second_camera}__{first_camera}"
        allowed = {str(value) for value in allowed_pairs}
        if pair_key in allowed:
            return str(first_camera), str(second_camera)
        if reverse_pair_key in allowed:
            return str(second_camera), str(first_camera)
    return str(first_camera), str(second_camera)


def _is_same_index_pair_allowed(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> bool:
    allowed_pairs = assumptions.get("inter_camera_same_index_allowed_pairs")
    if not allowed_pairs:
        return True
    if not isinstance(allowed_pairs, (list, tuple, set)):
        return True
    allowed = {str(value) for value in allowed_pairs}
    pair_key = f"{first_camera}__{second_camera}"
    reverse_pair_key = f"{second_camera}__{first_camera}"
    return pair_key in allowed or reverse_pair_key in allowed


def _declared_same_index_pair(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> tuple[str, str]:
    allowed_pairs = assumptions.get("inter_camera_same_index_allowed_pairs")
    if isinstance(allowed_pairs, (list, tuple, set)):
        pair_key = f"{first_camera}__{second_camera}"
        reverse_pair_key = f"{second_camera}__{first_camera}"
        allowed = {str(value) for value in allowed_pairs}
        if pair_key in allowed:
            return str(first_camera), str(second_camera)
        if reverse_pair_key in allowed:
            return str(second_camera), str(first_camera)
    return str(first_camera), str(second_camera)


def _temporal_overlap_baselines_for_pair(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> tuple[float, ...]:
    default_values = tuple(
        float(value)
        for value in assumptions.get(
            "inter_camera_temporal_overlap_target_baselines_m", [30.0]
        )
    )
    by_pair = assumptions.get("inter_camera_temporal_overlap_target_baselines_by_pair", {})
    if isinstance(by_pair, dict):
        pair_key = f"{first_camera}__{second_camera}"
        reverse_pair_key = f"{second_camera}__{first_camera}"
        value = by_pair.get(pair_key, by_pair.get(reverse_pair_key))
        if value is not None:
            if isinstance(value, (list, tuple)):
                return tuple(float(item) for item in value)
            return (float(value),)
    return default_values


def _temporal_overlap_search_direction_for_pair(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> tuple[str, str]:
    pair_key = f"{first_camera}__{second_camera}"
    by_pair = assumptions.get(
        "inter_camera_temporal_overlap_search_direction_by_pair", {}
    )
    if isinstance(by_pair, dict) and pair_key in by_pair:
        value = str(by_pair[pair_key])
        parts = value.split("__", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "inter_camera_temporal_overlap_search_direction_by_pair values "
                "must use '<first_camera>__<second_camera>' format"
            )
        return parts[0], parts[1]
    return str(first_camera), str(second_camera)


def _temporal_overlap_relation_label(target_baseline_m: float) -> str:
    label = f"{float(target_baseline_m):g}m".replace(".", "p")
    return f"temporal_overlap_{label}"


def _temporal_overlap_view_dot_threshold_for_pair(
    assumptions: dict,
    first_camera: str,
    second_camera: str,
) -> float:
    default_value = float(
        assumptions.get("inter_camera_temporal_overlap_min_opposite_view_dot", -0.5)
    )
    by_pair = assumptions.get("inter_camera_temporal_overlap_max_view_dot_by_pair", {})
    if isinstance(by_pair, dict):
        pair_key = f"{first_camera}__{second_camera}"
        reverse_pair_key = f"{second_camera}__{first_camera}"
        value = by_pair.get(pair_key, by_pair.get(reverse_pair_key))
        if value is not None:
            return float(value)
    return default_value


def run_staged_raw_matching(
    *,
    config_path: Path,
    dataset_root: Path,
    camera: str,
    cache_root: Path = Path("data_cache/semantic_masks"),
    start_frame: int = 0,
    frame_count: int | None = None,
    frame_offsets: tuple[int, ...] = (5, 10, 30),
    target_baselines_m: tuple[float, ...] | None = None,
    mask_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    mask_backend: str = "mask2former",
    mask_confidence: float = 0.15,
    mask_image_size: int = 1280,
    mask_dilation_pixels: int = 5,
    viz_cache_root: Path | None = None,
    viz_cache_key: str = "",
    viz_offset: int = 0,
    matching_scope: str = "intra",
    image_cache_mode: str = "preload",
    preflight_cache_root: Path | None = None,
    reuse_preflight_matches: bool = True,
) -> pd.DataFrame:
    config = _config_for_matching_scope(load_config(config_path), matching_scope)
    config_dataset_root = Path(getattr(config, "dataset_root", dataset_root)).resolve()
    if Path(dataset_root).resolve() != config_dataset_root and hasattr(config, "model_copy"):
        config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    frames = _frames_for_config(config, camera)
    window_start, window_end = _resolve_window(
        len(frames),
        start_frame=start_frame,
        frame_count=frame_count,
    )
    selected_frames = frames[window_start:window_end]
    image_store = ImageStore.from_paths(
        [frame.path for frame in selected_frames],
        **image_store_options(config),
    )
    calibration = _calibration_for_config(config, camera)
    semantic_config = getattr(config, "semantic_segmentation", None)
    semantic_dynamic_classes = frozenset(
        str(item)
        for item in dict(getattr(config, "assumptions", {}) or {}).get(
            "semantic_mask_classes", DEFAULT_DYNAMIC_CLASSES
        )
    )
    masks = SemanticMaskIndex(
        Path(cache_root),
        camera,
        mask_model,
        mask_backend,
        mask_confidence,
        mask_dilation_pixels,
        inference_dtype=str(getattr(semantic_config, "inference_dtype", "fp16")),
        use_fast_processor=bool(
            getattr(semantic_config, "use_fast_processor", True)
        ),
        dynamic_classes=semantic_dynamic_classes,
    )
    missing = [
        frame.timestamp_ns
        for frame in selected_frames
        if frame.timestamp_ns not in masks.paths
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} semantic masks are missing; first timestamp: {missing[0]}"
        )
    semantic_masks = {int(frame.index): masks.load(frame) for frame in selected_frames}
    preflight_cache = None
    if preflight_cache_root is not None and reuse_preflight_matches:
        from .clip_preflight import PreflightMatchCache

        candidate_cache = PreflightMatchCache(Path(preflight_cache_root))
        if candidate_cache.has_entries:
            preflight_cache = candidate_cache
    frame_lookup = {int(frame.index): frame for frame in selected_frames}
    baseline_pairs_by_target: dict[float, list[tuple[int, int, float]]] = {}
    if target_baselines_m:
        assumptions = dict(config.assumptions)
        navigation = _navigation_for_config(config)
        centers_by_index = _camera_centers_by_frame_index(
            frames=selected_frames,
            calibration=calibration,
            navigation=navigation,
            navigation_offset_ns=navigation_time_offset_ns(assumptions),
        )
        frame_indices = [int(frame.index) for frame in selected_frames]
        for target in tuple(float(value) for value in target_baselines_m):
            baseline_pairs_by_target[float(target)] = _build_baseline_frame_pairs(
                frame_indices=frame_indices,
                camera_centers_by_index=centers_by_index,
                target_baseline_m=float(target),
            )
        baseline_pairs_by_target = _deduplicate_baseline_frame_pairs(
            baseline_pairs_by_target
        )
    records: list[pd.DataFrame] = []
    pair_records: list[dict[str, object]] = []
    batch_size = max(1, int(getattr(config.matching, "deep_batch_size", 1)))
    if (
        config.matching.backend == "loma"
        and int(getattr(config.matching, "loma_process_count", 1)) > 1
    ):
        # Keep workers alive for an entire matching group so each loads LoMa once.
        batch_size = max(batch_size, len(selected_frames))
    # Cached preflight pairs are selectively loaded below. Keeping this path
    # sequential avoids sending cached and uncached pairs through one batch.
    use_batch = (
        _supports_batch_backend(config.matching.backend)
        and batch_size > 1
        and preflight_cache is None
    )

    _viz_all_root: Path | None = None
    if viz_cache_root is not None and viz_cache_key:
        if target_baselines_m:
            _viz_all_root = None
        else:
            _viz_all_root = (
                intra_camera_cache_dir(
                    viz_cache_root,
                    cache_key=viz_cache_key,
                    camera=camera,
                    frame_offset=viz_offset,
                )
                / "visualizations"
                / "all"
            )
            _viz_all_root.mkdir(parents=True, exist_ok=True)

    def _maybe_viz(
        offset: int, first_idx: int, matched: pd.DataFrame,
        img1: np.ndarray, img2: np.ndarray,
    ) -> None:
        if _viz_all_root is None or matched.empty:
            return
        sample_ratio = float(
            getattr(config.matching, "cache_visualization_pair_ratio", 0.01)
        )
        if not _should_export_match_visualization(first_idx, sample_ratio):
            return
        from .visualization import draw_tiled_match_overlay
        stem = f"offset_{int(offset)}_{int(first_idx):04d}_{int(first_idx + offset):04d}"
        draw_tiled_match_overlay(
            img1, img2,
            matched[["x1", "y1"]].to_numpy(),
            matched[["x2", "y2"]].to_numpy(),
            _viz_all_root / f"{stem}_all_matches.jpg",
            title_prefix=f"{camera} offset {offset}",
        )

    def append_result(
        *,
        offset: int,
        first,
        second,
        matched: pd.DataFrame,
        elapsed_ms: float,
    ) -> None:
        result = matched.copy()
        matching_source = str(result.attrs.get("matching_source", "full_loma"))
        cache_metrics = _feature_cache_metrics(result)
        result.attrs.clear()
        result["camera"] = camera
        result["frame_offset"] = int(offset)
        result["frame1"] = int(first.index)
        result["frame2"] = int(second.index)
        result["timestamp1_ns"] = int(first.timestamp_ns)
        result["timestamp2_ns"] = int(second.timestamp_ns)
        result["matching_source"] = matching_source
        result.attrs.clear()
        records.append(result)
        pair_records.append(
            {
                "camera": camera,
                "frame_offset": int(offset),
                "frame1": int(first.index),
                "frame2": int(second.index),
                "timestamp1_ns": int(first.timestamp_ns),
                "timestamp2_ns": int(second.timestamp_ns),
                "raw_match_count": int(len(result)),
                "matching_time_ms": float(elapsed_ms),
                "matching_source": matching_source,
                **cache_metrics,
            }
        )

    def append_baseline_result(
        *,
        target_baseline_m: float,
        pair_baseline_m: float,
        first,
        second,
        matched: pd.DataFrame,
        elapsed_ms: float,
    ) -> None:
        result = matched.copy()
        matching_source = str(result.attrs.get("matching_source", "full_loma"))
        cache_metrics = _feature_cache_metrics(result)
        result.attrs.clear()
        result["camera"] = camera
        result["frame_offset"] = int(second.index) - int(first.index)
        result["baseline_target_m"] = float(target_baseline_m)
        result["baseline_pair_m"] = float(pair_baseline_m)
        result["pair_strategy"] = "baseline"
        result["frame1"] = int(first.index)
        result["frame2"] = int(second.index)
        result["timestamp1_ns"] = int(first.timestamp_ns)
        result["timestamp2_ns"] = int(second.timestamp_ns)
        result["matching_source"] = matching_source
        result.attrs.clear()
        records.append(result)
        pair_records.append(
            {
                "camera": camera,
                "frame_offset": int(second.index) - int(first.index),
                "baseline_target_m": float(target_baseline_m),
                "baseline_pair_m": float(pair_baseline_m),
                "pair_strategy": "baseline",
                "frame1": int(first.index),
                "frame2": int(second.index),
                "timestamp1_ns": int(first.timestamp_ns),
                "timestamp2_ns": int(second.timestamp_ns),
                "raw_match_count": int(len(result)),
                "matching_time_ms": float(elapsed_ms),
                "matching_source": matching_source,
                **cache_metrics,
            }
        )

    def _cached_pair(
        *,
        relation: str,
        first,
        second,
        image1: np.ndarray,
        image2: np.ndarray,
        first_mask: np.ndarray,
        second_mask: np.ndarray,
    ) -> pd.DataFrame | None:
        if preflight_cache is None:
            return None
        from .clip_preflight import make_preflight_pair_key
        from .match_backends import apply_semantic_masks

        key = make_preflight_pair_key(
            pair_type="intra_camera",
            camera1=camera,
            camera2=camera,
            relation=relation,
            frame1=int(first.index),
            frame2=int(second.index),
        )
        cached = preflight_cache.get(key)
        if cached is None:
            return None
        cached = apply_semantic_masks(
            cached,
            first_mask=first_mask,
            second_mask=second_mask,
            first_shape=image_store.original_shape(first.path),
            second_shape=image_store.original_shape(second.path),
        )
        cached.attrs["matching_source"] = "preflight_reused"
        return cached

    if target_baselines_m:
        for target_baseline_m, baseline_pairs in baseline_pairs_by_target.items():
            if use_batch:
                payloads: list[dict[str, object]] = []
                contexts: list[tuple[object, object, float]] = []
                for first_idx, second_idx, pair_baseline_m in baseline_pairs:
                    first = frame_lookup[int(first_idx)]
                    second = frame_lookup[int(second_idx)]
                    img1 = image_store.get(first.path)
                    img2 = image_store.get(second.path)
                    payloads.append(
                        {
                            "first_image": img1,
                            "second_image": img2,
                            "first_image_key": str(first.path),
                            "second_image_key": str(second.path),
                            "calibration": calibration,
                            "config": config.matching,
                            "first_dynamic_mask": semantic_masks[int(first.index)],
                            "second_dynamic_mask": semantic_masks[int(second.index)],
                            "first_original_shape": image_store.original_shape(first.path),
                            "second_original_shape": image_store.original_shape(second.path),
                        }
                    )
                    contexts.append((first, second, float(pair_baseline_m)))
                    if len(payloads) < batch_size:
                        continue
                    started = time.perf_counter()
                    batch_matches = run_match_backend_batch(payloads)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                    for matched, (first_ctx, second_ctx, pair_baseline_ctx) in zip(
                        batch_matches, contexts, strict=True
                    ):
                        append_baseline_result(
                            target_baseline_m=target_baseline_m,
                            pair_baseline_m=pair_baseline_ctx,
                            first=first_ctx,
                            second=second_ctx,
                            matched=matched,
                            elapsed_ms=elapsed_ms,
                        )
                    payloads = []
                    contexts = []
                if payloads:
                    started = time.perf_counter()
                    batch_matches = run_match_backend_batch(payloads)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                    for matched, (first_ctx, second_ctx, pair_baseline_ctx) in zip(
                        batch_matches, contexts, strict=True
                    ):
                        append_baseline_result(
                            target_baseline_m=target_baseline_m,
                            pair_baseline_m=pair_baseline_ctx,
                            first=first_ctx,
                            second=second_ctx,
                            matched=matched,
                            elapsed_ms=elapsed_ms,
                        )
                continue

            for first_idx, second_idx, pair_baseline_m in baseline_pairs:
                first = frame_lookup[int(first_idx)]
                second = frame_lookup[int(second_idx)]
                img1 = image_store.get(first.path)
                img2 = image_store.get(second.path)
                started = time.perf_counter()
                matched = _cached_pair(
                    relation=f"baseline_{float(target_baseline_m):g}m",
                    first=first,
                    second=second,
                    image1=img1,
                    image2=img2,
                    first_mask=semantic_masks[int(first.index)],
                    second_mask=semantic_masks[int(second.index)],
                )
                if matched is None:
                    matched = run_match_backend(
                        img1,
                        img2,
                        calibration,
                        config.matching,
                        first_semantic_mask=semantic_masks[int(first.index)],
                        second_semantic_mask=semantic_masks[int(second.index)],
                        first_image_key=str(first.path),
                        second_image_key=str(second.path),
                        first_original_shape=image_store.original_shape(first.path),
                        second_original_shape=image_store.original_shape(second.path),
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                append_baseline_result(
                    target_baseline_m=target_baseline_m,
                    pair_baseline_m=float(pair_baseline_m),
                    first=first,
                    second=second,
                    matched=matched,
                    elapsed_ms=elapsed_ms,
                )
        if pair_records:
            pairs = pd.DataFrame.from_records(pair_records)
        else:
            pairs = pd.DataFrame(
                columns=[
                    "camera",
                    "frame_offset",
                    "baseline_target_m",
                    "baseline_pair_m",
                    "pair_strategy",
                    "frame1",
                    "frame2",
                    "timestamp1_ns",
                    "timestamp2_ns",
                    "raw_match_count",
                    "matching_time_ms",
                ]
            )
        matches = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        matches.attrs["pair_metrics"] = pairs
        return matches

    for offset in tuple(int(v) for v in frame_offsets):
        if offset <= 0:
            raise ValueError("frame offsets must be positive")
        if len(selected_frames) <= offset:
            continue
        if use_batch:
            payloads: list[dict[str, object]] = []
            contexts: list[tuple[object, object]] = []
            for first_pos in range(len(selected_frames) - offset):
                first = selected_frames[first_pos]
                second = selected_frames[first_pos + offset]
                img1 = image_store.get(first.path)
                img2 = image_store.get(second.path)
                payloads.append(
                    {
                        "first_image": img1,
                        "second_image": img2,
                        "first_image_key": str(first.path),
                        "second_image_key": str(second.path),
                        "calibration": calibration,
                        "config": config.matching,
                        "first_dynamic_mask": semantic_masks[int(first.index)],
                        "second_dynamic_mask": semantic_masks[int(second.index)],
                        "first_original_shape": image_store.original_shape(first.path),
                        "second_original_shape": image_store.original_shape(second.path),
                    }
                )
                contexts.append((first, second, img1, img2))
                if len(payloads) < batch_size:
                    continue
                started = time.perf_counter()
                batch_matches = run_match_backend_batch(payloads)
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                for matched, (first_ctx, second_ctx, img1_ctx, img2_ctx) in zip(batch_matches, contexts, strict=True):
                    append_result(
                        offset=offset,
                        first=first_ctx,
                        second=second_ctx,
                        matched=matched,
                        elapsed_ms=elapsed_ms,
                    )
                    _maybe_viz(offset, int(first_ctx.index), matched, img1_ctx, img2_ctx)
                payloads = []
                contexts = []
            if payloads:
                started = time.perf_counter()
                batch_matches = run_match_backend_batch(payloads)
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                for matched, (first_ctx, second_ctx, img1_ctx, img2_ctx) in zip(batch_matches, contexts, strict=True):
                    append_result(
                        offset=offset,
                        first=first_ctx,
                        second=second_ctx,
                        matched=matched,
                        elapsed_ms=elapsed_ms,
                    )
                    _maybe_viz(offset, int(first_ctx.index), matched, img1_ctx, img2_ctx)
            continue

        for first_pos in range(len(selected_frames) - offset):
            first = frame_lookup[int(selected_frames[first_pos].index)]
            second = frame_lookup[int(selected_frames[first_pos + offset].index)]
            img1 = image_store.get(first.path)
            img2 = image_store.get(second.path)
            started = time.perf_counter()
            matched = _cached_pair(
                relation=f"offset_{int(offset)}",
                first=first,
                second=second,
                image1=img1,
                image2=img2,
                first_mask=semantic_masks[int(first.index)],
                second_mask=semantic_masks[int(second.index)],
            )
            if matched is None:
                matched = run_match_backend(
                    img1, img2,
                    calibration,
                    config.matching,
                    first_semantic_mask=semantic_masks[int(first.index)],
                    second_semantic_mask=semantic_masks[int(second.index)],
                    first_image_key=str(first.path),
                    second_image_key=str(second.path),
                    first_original_shape=image_store.original_shape(first.path),
                    second_original_shape=image_store.original_shape(second.path),
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            append_result(
                offset=offset,
                first=first,
                second=second,
                matched=matched,
                elapsed_ms=elapsed_ms,
            )
            _maybe_viz(offset, int(first.index), matched, img1, img2)

    if pair_records:
        pairs = pd.DataFrame.from_records(pair_records)
    else:
        pairs = pd.DataFrame(
            columns=[
                "camera",
                "frame_offset",
                "frame1",
                "frame2",
                "timestamp1_ns",
                "timestamp2_ns",
                "raw_match_count",
                "matching_time_ms",
            ]
        )
    matches = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    matches.attrs["pair_metrics"] = pairs
    return matches


def evaluate_intra_camera_match_quality(
    *,
    config_path: Path,
    dataset_root: Path,
    matches: pd.DataFrame,
    start_frame: int = 0,
    frame_count: int | None = None,
    pose_pkl_path: Path | None = None,
    pose_field: str | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    config_dataset_root = Path(getattr(config, "dataset_root", dataset_root)).resolve()
    if Path(dataset_root).resolve() != config_dataset_root and hasattr(config, "model_copy"):
        config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    assumptions = dict(config.assumptions)
    if pose_pkl_path is not None:
        assumptions["result_interpolation_pkl_path"] = str(Path(pose_pkl_path).resolve())
    if pose_field is not None:
        assumptions["result_interpolation_ego_field"] = str(pose_field)
    if pose_pkl_path is not None or pose_field is not None:
        config = config.model_copy(update={"assumptions": assumptions})
    navigation = _navigation_for_config(config)
    navigation_offset_ns = navigation_time_offset_ns(assumptions)
    max_triangulated_point_distance_m = float(
        assumptions.get("max_triangulated_point_distance_m", DEFAULT_MAX_TRIANGULATED_POINT_DISTANCE_M)
    )
    min_triangulation_angle_deg = float(assumptions.get("min_triangulation_angle_deg", 0.5))
    quality_threshold_mode = str(assumptions.get("quality_threshold_mode", "fixed"))
    quality_threshold_mad_k = float(assumptions.get("quality_threshold_mad_k", 2.0))
    quality_threshold_min_candidates = int(
        assumptions.get("quality_threshold_min_candidates", 20)
    )
    quality_threshold_insufficient_reprojection_px = float(
        assumptions.get("quality_threshold_insufficient_reprojection_px", 5.0)
    )
    quality_threshold_insufficient_ray_closest_distance_m = float(
        assumptions.get("quality_threshold_insufficient_ray_closest_distance_m", 0.10)
    )
    max_ray_closest_distance_m = float(assumptions.get("max_ray_closest_distance_m", 0.5))
    ray_closest_distance_mode = str(
        assumptions.get("ray_closest_distance_mode", quality_threshold_mode)
    )
    ray_closest_distance_min_m = assumptions.get("ray_closest_distance_min_m")
    ray_closest_distance_max_m = assumptions.get("ray_closest_distance_max_m")
    ray_closest_distance_mad_k = float(
        assumptions.get("ray_closest_distance_mad_k", quality_threshold_mad_k)
    )
    ray_closest_distance_min_candidates = int(
        assumptions.get(
            "ray_closest_distance_min_candidates",
            quality_threshold_min_candidates,
        )
    )
    reprojection_threshold_mode = str(
        assumptions.get("reprojection_threshold_mode", quality_threshold_mode)
    )
    reprojection_threshold_mad_k = float(
        assumptions.get("reprojection_threshold_mad_k", quality_threshold_mad_k)
    )
    reprojection_threshold_min_candidates = int(
        assumptions.get(
            "reprojection_threshold_min_candidates",
            quality_threshold_min_candidates,
        )
    )
    image_ransac_threshold_px = float(
        assumptions.get(
            "intra_camera_image_ransac_threshold_px",
            assumptions.get("match_quality_image_ransac_threshold_px", 50.0),
        )
    )
    image_ransac_confidence = float(
        assumptions.get(
            "intra_camera_image_ransac_confidence",
            assumptions.get("match_quality_image_ransac_confidence", 0.999),
        )
    )
    pose_degraded_detection_enabled = bool(
        assumptions.get("pose_degraded_detection_enabled", False)
    )
    pose_degraded_min_ransac_inliers = int(
        assumptions.get("pose_degraded_min_ransac_inliers", 30)
    )
    pose_degraded_reprojection_median_px = float(
        assumptions.get("pose_degraded_reprojection_median_px", 80.0)
    )
    pose_degraded_reprojection_p75_px = float(
        assumptions.get("pose_degraded_reprojection_p75_px", 120.0)
    )
    pose_degraded_ray_gap_median_m = float(
        assumptions.get("pose_degraded_ray_gap_median_m", 0.8)
    )
    pose_degraded_ray_gap_p75_m = float(
        assumptions.get("pose_degraded_ray_gap_p75_m", 1.2)
    )
    pose_degraded_quality_mode = str(
        assumptions.get("pose_degraded_quality_mode", "standard")
    )
    camera_to_world_by_key = _load_image_camera_to_world_csv(
        _optimized_pose_csv_from_assumptions(assumptions, config.dataset_root)
    )
    camera = str(matches["camera"].iloc[0]) if "camera" in matches.columns else ""
    calibration = _calibration_for_config(config, camera) if camera else None
    if camera and calibration is not None:
        matches = _recompute_normalized_coordinates_for_calibrations(
            matches,
            {camera: calibration},
        )
    frame_offsets = sorted(matches["frame_offset"].drop_duplicates().astype(int).tolist())

    frames = _frames_for_config(config, camera) if camera else []
    window_start, window_end = _resolve_window(len(frames), start_frame=start_frame, frame_count=frame_count)
    frame_lookup = {int(frame.index): frame for frame in frames[window_start:window_end]}

    evaluated_groups: list[pd.DataFrame] = []
    evaluation_pair_records: list[dict[str, object]] = []
    for offset in frame_offsets:
        offset_matches = matches.loc[matches["frame_offset"].astype(int) == offset]

        for (frame1, frame2), group in offset_matches.groupby(["frame1", "frame2"], sort=False):
            pair_evaluation_start = time.perf_counter()
            if "baseline_target_m" in group.columns and group["baseline_target_m"].notna().any():
                baseline_target = float(group["baseline_target_m"].dropna().iloc[0])
                baseline_label = f"{baseline_target:g}".replace(".", "p")
                default_by_baseline = {2.0: 20.0, 5.0: 30.0, 10.0: 50.0}
                threshold_px = float(
                    assumptions.get(
                        f"baseline_reprojection_threshold_px_{baseline_label}",
                        default_by_baseline.get(baseline_target, 30.0),
                    )
                )
            else:
                threshold_px = float(
                    assumptions.get(
                        f"match_quality_reprojection_threshold_px_{offset}", 20.0
                    )
                )
            first_frame = frame_lookup.get(int(frame1))
            second_frame = frame_lookup.get(int(frame2))
            if first_frame is None or second_frame is None or calibration is None:
                continue

            navigation_geometry_start = time.perf_counter()
            evaluated, _fundamental, pair_baseline_m = _evaluate_inter_camera_navigation(
                group,
                int(first_frame.timestamp_ns),
                int(second_frame.timestamp_ns),
                calibration,
                calibration,  # same calibration for both views (intra-camera)
                navigation,
                navigation_offset_ns=navigation_offset_ns,
                camera_to_world_by_key=camera_to_world_by_key,
            )
            navigation_geometry_time_ms = (
                time.perf_counter() - navigation_geometry_start
            ) * 1000.0
            evaluated["pair_baseline_m"] = pair_baseline_m
            threshold_annotation_start = time.perf_counter()
            evaluated = _annotate_match_quality_thresholds(
                evaluated,
                motion_pair_valid=bool(pair_baseline_m >= MIN_OBSERVABLE_BASELINE_M),
                threshold_px=threshold_px,
                max_triangulated_point_distance_m=max_triangulated_point_distance_m,
                min_triangulation_angle_deg=min_triangulation_angle_deg,
                max_ray_closest_distance_m=max_ray_closest_distance_m,
                ray_closest_distance_mode=ray_closest_distance_mode,
                ray_closest_distance_min_m=ray_closest_distance_min_m,
                ray_closest_distance_max_m=ray_closest_distance_max_m,
                ray_closest_distance_mad_k=ray_closest_distance_mad_k,
                ray_closest_distance_min_candidates=ray_closest_distance_min_candidates,
                ray_closest_distance_insufficient_candidates_threshold_m=(
                    quality_threshold_insufficient_ray_closest_distance_m
                ),
                reprojection_threshold_mode=reprojection_threshold_mode,
                reprojection_threshold_min_px=(
                    threshold_px if str(reprojection_threshold_mode) == "fixed" else None
                ),
                reprojection_threshold_max_px=None,
                reprojection_threshold_mad_k=reprojection_threshold_mad_k,
                reprojection_threshold_min_candidates=reprojection_threshold_min_candidates,
                reprojection_insufficient_candidates_threshold_px=(
                    quality_threshold_insufficient_reprojection_px
                ),
            )
            threshold_annotation_time_ms = (
                time.perf_counter() - threshold_annotation_start
            ) * 1000.0

            quality_filter_start = time.perf_counter()
            evaluated = _apply_match_quality_filter(
                evaluated,
                image_ransac_threshold_px=image_ransac_threshold_px,
                image_ransac_confidence=image_ransac_confidence,
                pose_degraded_detection_enabled=pose_degraded_detection_enabled,
                pose_degraded_min_ransac_inliers=pose_degraded_min_ransac_inliers,
                pose_degraded_reprojection_median_px=pose_degraded_reprojection_median_px,
                pose_degraded_reprojection_p75_px=pose_degraded_reprojection_p75_px,
                pose_degraded_ray_gap_median_m=pose_degraded_ray_gap_median_m,
                pose_degraded_ray_gap_p75_m=pose_degraded_ray_gap_p75_m,
                pose_degraded_quality_mode=pose_degraded_quality_mode,
            )
            quality_filter_time_ms = (
                time.perf_counter() - quality_filter_start
            ) * 1000.0
            retained_mask = _retained_mask(evaluated)
            valid_reprojection = (
                retained_mask
                & np.isfinite(evaluated["reprojection_rms_px"].astype(float))
            )
            record: dict[str, object] = {
                "camera": camera,
                "frame_offset": int(offset),
                "frame1": int(frame1),
                "frame2": int(frame2),
                "timestamp1_ns": int(first_frame.timestamp_ns),
                "timestamp2_ns": int(second_frame.timestamp_ns),
                "pair_baseline_m": float(pair_baseline_m),
                "reprojection_threshold_px": float(
                    evaluated["reprojection_threshold_px"].iloc[0]
                    if "reprojection_threshold_px" in evaluated.columns
                    else threshold_px
                ),
                "reprojection_threshold_mode": str(reprojection_threshold_mode),
                "image_ransac_threshold_px": float(image_ransac_threshold_px),
                "accepted_matches": int(len(evaluated)),
                "retained_matches": int(retained_mask.sum()),
                "valid_triangulations": int(valid_reprojection.sum()),
                "image_ransac_rejected": int(
                    (~evaluated["image_ransac_inlier"].astype(bool)).sum()
                ),
                "pose_degraded": bool(
                    evaluated["pose_degraded"].iloc[0]
                    if "pose_degraded" in evaluated.columns
                    else False
                ),
                "pose_degraded_reprojection_median_px": float(
                    evaluated["pose_degraded_reprojection_median_px"].iloc[0]
                    if "pose_degraded_reprojection_median_px" in evaluated.columns
                    else np.nan
                ),
                "pose_degraded_reprojection_p75_px": float(
                    evaluated["pose_degraded_reprojection_p75_px"].iloc[0]
                    if "pose_degraded_reprojection_p75_px" in evaluated.columns
                    else np.nan
                ),
                "pose_degraded_ray_gap_median_m": float(
                    evaluated["pose_degraded_ray_gap_median_m"].iloc[0]
                    if "pose_degraded_ray_gap_median_m" in evaluated.columns
                    else np.nan
                ),
                "pose_degraded_ray_gap_p75_m": float(
                    evaluated["pose_degraded_ray_gap_p75_m"].iloc[0]
                    if "pose_degraded_ray_gap_p75_m" in evaluated.columns
                    else np.nan
                ),
                "navigation_geometry_time_ms": float(navigation_geometry_time_ms),
                "threshold_annotation_time_ms": float(threshold_annotation_time_ms),
                "quality_filter_time_ms": float(quality_filter_time_ms),
                "evaluation_total_time_ms": float(
                    (time.perf_counter() - pair_evaluation_start) * 1000.0
                ),
            }
            if "baseline_target_m" in group.columns and group["baseline_target_m"].notna().any():
                record["baseline_target_m"] = float(
                    group["baseline_target_m"].dropna().iloc[0]
                )
            if "baseline_pair_m" in group.columns and group["baseline_pair_m"].notna().any():
                record["baseline_pair_m"] = float(
                    group["baseline_pair_m"].dropna().iloc[0]
                )
            evaluation_pair_records.append(record)

            evaluated.attrs.clear()
            evaluated_groups.append(evaluated)

    result = pd.concat(evaluated_groups, ignore_index=True) if evaluated_groups else matches.copy()
    evaluation_pair_metrics = pd.DataFrame.from_records(evaluation_pair_records)
    base_pair_metrics = matches.attrs.get("pair_metrics")
    if (
        isinstance(base_pair_metrics, pd.DataFrame)
        and not base_pair_metrics.empty
        and not evaluation_pair_metrics.empty
    ):
        merge_keys = [
            key
            for key in ("camera", "frame_offset", "baseline_target_m", "frame1", "frame2")
            if key in base_pair_metrics.columns and key in evaluation_pair_metrics.columns
        ]
        if merge_keys:
            evaluation_pair_metrics = base_pair_metrics.merge(
                evaluation_pair_metrics,
                on=merge_keys,
                how="right",
                suffixes=("", "_evaluation"),
            )
    result.attrs["pair_metrics"] = evaluation_pair_metrics
    return result


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _filter_intra_cache_group_worker(
    config_path: Path,
    dataset_root: Path,
    cache_dirs: tuple[Path, ...],
    start_frame: int,
    frame_count: int | None,
    export_visualizations: bool,
) -> dict[str, int]:
    raw_parts: list[pd.DataFrame] = []
    pair_metric_parts: list[pd.DataFrame] = []
    raw_by_cache: list[tuple[Path, pd.DataFrame]] = []
    for cache_dir_value in cache_dirs:
        cache_dir = Path(cache_dir_value)
        raw = pd.read_parquet(cache_dir / "raw_matches.parquet")
        raw.attrs.clear()
        raw_parts.append(raw)
        raw_by_cache.append((cache_dir, raw))
        pair_metrics_path = cache_dir / "pair_metrics.csv"
        if pair_metrics_path.exists():
            pair_metric_parts.append(pd.read_csv(pair_metrics_path))

    all_matches = (
        pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    )
    if pair_metric_parts:
        all_matches.attrs["pair_metrics"] = pd.concat(
            pair_metric_parts, ignore_index=True
        )
    evaluated_all = (
        evaluate_intra_camera_match_quality(
            config_path=Path(config_path),
            dataset_root=Path(dataset_root),
            matches=all_matches,
            start_frame=int(start_frame),
            frame_count=frame_count,
        )
        if not all_matches.empty
        else all_matches.copy()
    )
    evaluated_pair_metrics = evaluated_all.attrs.get("pair_metrics")
    retained_count = 0
    for cache_dir, raw in raw_by_cache:
        if raw.empty:
            evaluated = raw.copy()
        elif "baseline_target_m" in raw.columns:
            target = float(raw["baseline_target_m"].dropna().iloc[0])
            evaluated = evaluated_all.loc[
                np.isclose(
                    evaluated_all["baseline_target_m"].astype(float), target
                )
            ].copy()
        else:
            evaluated = evaluated_all.copy()
        retained = (
            evaluated.loc[_retained_mask(evaluated)].copy()
            if not evaluated.empty
            else evaluated.copy()
        )
        retained.attrs.clear()
        _atomic_write_parquet(retained, cache_dir / "retained_matches.parquet")
        retained_count += int(len(retained))

        if isinstance(evaluated_pair_metrics, pd.DataFrame):
            quality_metrics = evaluated_pair_metrics
            if (
                not raw.empty
                and "baseline_target_m" in raw.columns
                and "baseline_target_m" in quality_metrics.columns
            ):
                target = float(raw["baseline_target_m"].dropna().iloc[0])
                quality_metrics = quality_metrics.loc[
                    np.isclose(
                        quality_metrics["baseline_target_m"].astype(float), target
                    )
                ]
            _atomic_write_csv(
                quality_metrics.copy(), cache_dir / "pair_metrics_quality.csv"
            )

        if export_visualizations and not evaluated.empty:
            camera = str(evaluated["camera"].iloc[0])
            _export_intra_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=Path(dataset_root),
                config_path=Path(config_path),
                camera=camera,
                evaluated=evaluated,
                label=_label_from_intra_cache_dir(cache_dir),
            )
    return {
        "processed_dirs": len(raw_by_cache),
        "raw_matches": int(len(all_matches)),
        "retained_matches": int(retained_count),
    }


def match_dataset_stage(
    *,
    config_path: Path,
    dataset_root: Path,
    camera: str,
    cache_root: Path = Path("data_cache/semantic_masks"),
    match_cache_root: Path = Path("data_cache/matches"),
    mask_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    mask_backend: str = "mask2former",
    mask_confidence: float = 0.25,
    mask_image_size: int = 2048,
    mask_dilation_pixels: int = 5,
    start_frame: int = 0,
    frame_count: int | None = None,
    frame_offsets: tuple[int, ...] = (5, 10, 30),
    target_baselines_m: tuple[float, ...] | None = None,
    image_cache_mode: str = "preload",
    preflight_cache_root: Path | None = None,
    filter_executor=None,
    pending_filter_jobs: list[object] | None = None,
) -> Path:
    config = _config_for_matching_scope(load_config(config_path), "intra")
    assumptions = dict(getattr(config, "assumptions", {}) or {})
    export_cache_visualizations = bool(
        assumptions.get("export_match_cache_visualizations", True)
    )
    cache_key = match_cache_key(
        backend=str(config.matching.backend),
        image_size=int(config.matching.deep_image_size),
        max_keypoints=int(getattr(config.matching, "deep_max_keypoints", 2048)),
        score_threshold=float(config.matching.deep_score_threshold),
        semantic_backend="mask2former",
        semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        semantic_image_size=mask_image_size,
        semantic_dilation_pixels=mask_dilation_pixels,
    )
    first_cache_dir: Path | None = None
    if target_baselines_m:
        all_matches = run_staged_raw_matching(
            config_path=config_path,
            dataset_root=dataset_root,
            camera=camera,
            cache_root=cache_root,
            start_frame=start_frame,
            frame_count=frame_count,
            frame_offsets=(),
            target_baselines_m=tuple(float(value) for value in target_baselines_m),
            mask_model=mask_model,
            mask_backend=mask_backend,
            mask_confidence=mask_confidence,
            mask_image_size=mask_image_size,
            mask_dilation_pixels=mask_dilation_pixels,
            matching_scope="intra",
            image_cache_mode=image_cache_mode,
            preflight_cache_root=preflight_cache_root,
        )
        all_pair_metrics = all_matches.attrs.get("pair_metrics")
        evaluated_all = (
            evaluate_intra_camera_match_quality(
                config_path=config_path,
                dataset_root=dataset_root,
                matches=all_matches,
                start_frame=start_frame,
                frame_count=frame_count,
            )
            if filter_executor is None and not all_matches.empty
            else all_matches.copy()
        )
        evaluated_pair_metrics = evaluated_all.attrs.get("pair_metrics")
        deferred_cache_dirs: list[Path] = []
        for target in tuple(float(value) for value in target_baselines_m):
            cache_dir = intra_camera_baseline_cache_dir(
                Path(match_cache_root),
                cache_key=cache_key,
                camera=camera,
                target_baseline_m=target,
            )
            target_matches = (
                all_matches.loc[
                    np.isclose(
                        all_matches["baseline_target_m"].astype(float), float(target)
                    )
                ].copy()
                if "baseline_target_m" in all_matches.columns
                else all_matches.iloc[0:0].copy()
            )
            matches_for_storage = target_matches.copy()
            matches_for_storage.attrs.clear()
            pair_metrics = (
                all_pair_metrics.loc[
                    np.isclose(
                        all_pair_metrics["baseline_target_m"].astype(float),
                        float(target),
                    )
                ].copy()
                if isinstance(all_pair_metrics, pd.DataFrame)
                and "baseline_target_m" in all_pair_metrics.columns
                else None
            )
            quality_pair_metrics = (
                evaluated_pair_metrics.loc[
                    np.isclose(
                        evaluated_pair_metrics["baseline_target_m"].astype(float),
                        float(target),
                    )
                ].copy()
                if isinstance(evaluated_pair_metrics, pd.DataFrame)
                and "baseline_target_m" in evaluated_pair_metrics.columns
                else None
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            if first_cache_dir is None:
                first_cache_dir = cache_dir
            deferred_cache_dirs.append(cache_dir)
            matches_for_storage.to_parquet(
                cache_dir / "raw_matches.parquet", index=False
            )
            write_match_cache_artifacts(
                cache_dir,
                matches=matches_for_storage,
                pair_metrics=pair_metrics if isinstance(pair_metrics, pd.DataFrame) else None,
                manifest={
                    "pair_type": "intra_camera",
                    "pair_strategy": "baseline",
                    "camera": camera,
                    "baseline_target_m": float(target),
                    "start_frame": int(start_frame),
                    "frame_count": None if frame_count is None else int(frame_count),
                },
            )
            if matches_for_storage.empty:
                continue
            if filter_executor is not None:
                continue
            evaluated = (
                evaluated_all.loc[
                    np.isclose(
                        evaluated_all["baseline_target_m"].astype(float),
                        float(target),
                    )
                ].copy()
                if "baseline_target_m" in evaluated_all.columns
                else evaluated_all.iloc[0:0].copy()
            )
            retained_mask = _retained_mask(evaluated)
            retained = evaluated.loc[retained_mask].copy()
            retained.attrs.clear()
            retained.to_parquet(cache_dir / "retained_matches.parquet", index=False)
            if isinstance(quality_pair_metrics, pd.DataFrame):
                quality_pair_metrics.to_csv(
                    cache_dir / "pair_metrics_quality.csv", index=False
                )
            if export_cache_visualizations:
                _export_intra_camera_pair_visualizations(
                    cache_dir=cache_dir,
                    dataset_root=dataset_root,
                    config_path=config_path,
                    camera=camera,
                    evaluated=evaluated,
                    label=f"baseline_{target:g}m",
                )
        if filter_executor is not None and deferred_cache_dirs:
            future = filter_executor.submit(
                _filter_intra_cache_group_worker,
                Path(config_path),
                Path(dataset_root),
                tuple(deferred_cache_dirs),
                int(start_frame),
                frame_count,
                bool(export_cache_visualizations),
            )
            if pending_filter_jobs is not None:
                pending_filter_jobs.append(future)
        return first_cache_dir or Path(match_cache_root)

    for offset in frame_offsets:
        matches = run_staged_raw_matching(
            config_path=config_path,
            dataset_root=dataset_root,
            camera=camera,
            cache_root=cache_root,
            start_frame=start_frame,
            frame_count=frame_count,
            frame_offsets=(offset,),
            mask_model=mask_model,
            mask_backend=mask_backend,
            mask_confidence=mask_confidence,
            mask_image_size=mask_image_size,
            mask_dilation_pixels=mask_dilation_pixels,
            viz_cache_root=Path(match_cache_root),
            viz_cache_key=cache_key,
            viz_offset=offset,
            matching_scope="intra",
            image_cache_mode=image_cache_mode,
            preflight_cache_root=preflight_cache_root,
        )
        matches_for_storage = matches.copy()
        matches_for_storage.attrs.clear()
        pair_metrics = matches.attrs.get("pair_metrics")

        cache_dir = intra_camera_cache_dir(
            Path(match_cache_root),
            cache_key=cache_key,
            camera=camera,
            frame_offset=offset,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        if first_cache_dir is None:
            first_cache_dir = cache_dir

        matches_for_storage.to_parquet(
            cache_dir / "raw_matches.parquet", index=False
        )
        write_match_cache_artifacts(
            cache_dir,
            matches=matches_for_storage,
            pair_metrics=pair_metrics if isinstance(pair_metrics, pd.DataFrame) else None,
            manifest={
                "pair_type": "intra_camera",
                "camera": camera,
                "frame_offset": int(offset),
                "start_frame": int(start_frame),
                "frame_count": None if frame_count is None else int(frame_count),
            },
        )

        if matches_for_storage.empty:
            continue

        evaluated = evaluate_intra_camera_match_quality(
            config_path=config_path,
            dataset_root=dataset_root,
            matches=matches,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        retained_mask = _retained_mask(evaluated)
        retained = evaluated.loc[retained_mask].copy()
        retained.attrs.clear()
        retained.to_parquet(cache_dir / "retained_matches.parquet", index=False)

        evaluated_pair_metrics = evaluated.attrs.get("pair_metrics")
        if isinstance(evaluated_pair_metrics, pd.DataFrame):
            retained_pm = (
                evaluated_pair_metrics.loc[
                    evaluated_pair_metrics["frame_offset"].astype(int) == offset
                ].copy()
                if "frame_offset" in evaluated_pair_metrics.columns
                else evaluated_pair_metrics.copy()
            )
            if not retained_pm.empty:
                retained_pm.to_csv(cache_dir / "pair_metrics_quality.csv", index=False)

        if export_cache_visualizations:
            _export_intra_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=dataset_root,
                config_path=config_path,
                camera=camera,
                evaluated=evaluated,
                label=f"offset_{offset}",
            )

    return first_cache_dir or Path(match_cache_root)


def _export_intra_camera_pair_visualizations(
    *,
    cache_dir: Path,
    dataset_root: Path,
    config_path: Path | None = None,
    camera: str,
    evaluated: pd.DataFrame,
    label: str,
) -> None:
    if evaluated.empty:
        return
    from .visualization import draw_ransac_retained_reprojection_qc

    _viz_root = cache_dir / "visualizations"
    _retained_root = _viz_root / "retained"
    _ransac_qc_root = _viz_root / "ransac_retained_qc"
    _retained_root.mkdir(parents=True, exist_ok=True)
    _ransac_qc_root.mkdir(parents=True, exist_ok=True)
    config = None
    if config_path is not None:
        config = load_config(config_path)
        config_dataset_root = Path(getattr(config, "dataset_root", dataset_root)).resolve()
        if Path(dataset_root).resolve() != config_dataset_root and hasattr(config, "model_copy"):
            config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
        if hasattr(config, "dataset_root"):
            frames = _frames_for_config(config, camera)
        else:
            frames = Dataset(dataset_root).frames(camera)
    else:
        frames = Dataset(dataset_root).frames(camera)
    sample_ratio = float(
        getattr(
            getattr(config, "matching", None),
            "cache_visualization_pair_ratio",
            0.01,
        )
    )
    idx_to_path = {int(f.index): f.path for f in frames}
    for idx in sorted(evaluated["frame1"].drop_duplicates().astype(int).tolist()):
        if not _should_export_match_visualization(idx, sample_ratio) or idx not in idx_to_path:
            continue
        pair_eval = evaluated.loc[evaluated["frame1"].astype(int) == int(idx)].copy()
        if pair_eval.empty:
            continue
        second_idx = int(pair_eval["frame2"].iloc[0])
        if second_idx not in idx_to_path:
            continue
        img1 = cv2.imread(str(idx_to_path[idx]))
        img2 = cv2.imread(str(idx_to_path[second_idx]))
        if img1 is None or img2 is None:
            continue
        stem = f"{label}_{idx:04d}_{second_idx:04d}"
        retained_mask = _retained_mask(pair_eval).to_numpy(dtype=bool)
        ransac_inlier = (
            pair_eval["image_ransac_inlier"].astype(bool).to_numpy()
            if "image_ransac_inlier" in pair_eval.columns
            else np.ones((len(pair_eval),), dtype=bool)
        )
        reprojection_rms = (
            pair_eval["reprojection_rms_px"].astype(float).to_numpy()
            if "reprojection_rms_px" in pair_eval.columns
            else np.full((len(pair_eval),), np.nan)
        )
        ransac_reprojection_mask = ransac_inlier & np.isfinite(reprojection_rms)
        reprojection_values = (
            pair_eval.loc[ransac_reprojection_mask, "reprojection_rms_px"].to_numpy(
                dtype=np.float64
            )
            if "reprojection_rms_px" in pair_eval.columns
            else np.array([], dtype=np.float64)
        )
        ray_gap_values = (
            pair_eval.loc[
                ransac_inlier
                & np.isfinite(
                    pair_eval["ray_closest_distance_m"].astype(float).to_numpy()
                ),
                "ray_closest_distance_m",
            ].to_numpy(dtype=np.float64)
            if "ray_closest_distance_m" in pair_eval.columns
            else np.array([], dtype=np.float64)
        )
        metadata = {
            "pose_degraded": bool(
                pair_eval["pose_degraded"].iloc[0]
                if "pose_degraded" in pair_eval.columns
                else False
            ),
            "reprojection_median_px": float(np.median(reprojection_values))
            if reprojection_values.size
            else np.nan,
            "reprojection_p75_px": float(np.percentile(reprojection_values, 75))
            if reprojection_values.size
            else np.nan,
            "ray_gap_median_m": float(np.median(ray_gap_values))
            if ray_gap_values.size
            else np.nan,
            "ray_gap_p75_m": float(np.percentile(ray_gap_values, 75))
            if ray_gap_values.size
            else np.nan,
            "reprojection_threshold_px": float(
                pair_eval["reprojection_threshold_px"].iloc[0]
                if "reprojection_threshold_px" in pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_median_px": float(
                pair_eval["reprojection_threshold_median_px"].iloc[0]
                if "reprojection_threshold_median_px" in pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_mad_px": float(
                pair_eval["reprojection_threshold_mad_px"].iloc[0]
                if "reprojection_threshold_mad_px" in pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_k": float(
                pair_eval["reprojection_threshold_mad_k"].iloc[0]
                if "reprojection_threshold_mad_k" in pair_eval.columns
                else 2.5
            ),
            "ray_gap_threshold_m": float(
                pair_eval["ray_closest_distance_threshold_m"].iloc[0]
                if "ray_closest_distance_threshold_m" in pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_median_m": float(
                pair_eval["ray_closest_distance_median_m"].iloc[0]
                if "ray_closest_distance_median_m" in pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_mad_m": float(
                pair_eval["ray_closest_distance_mad_m"].iloc[0]
                if "ray_closest_distance_mad_m" in pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_k": float(
                pair_eval["ray_closest_distance_mad_k"].iloc[0]
                if "ray_closest_distance_mad_k" in pair_eval.columns
                else 2.5
            ),
        }
        reproj_median = metadata["reprojection_threshold_median_px"]
        reproj_k = metadata["reprojection_threshold_k"]
        reproj_mad = metadata["reprojection_threshold_mad_px"]
        metadata["reprojection_threshold_calculated_px"] = (
            float(reproj_median + reproj_k * 1.4826 * reproj_mad)
            if np.isfinite(reproj_median)
            and np.isfinite(reproj_k)
            and np.isfinite(reproj_mad)
            else np.nan
        )
        ray_median = metadata["ray_gap_threshold_median_m"]
        ray_k = metadata["ray_gap_threshold_k"]
        ray_mad = metadata["ray_gap_threshold_mad_m"]
        metadata["ray_gap_threshold_calculated_m"] = (
            float(ray_median + ray_k * 1.4826 * ray_mad)
            if np.isfinite(ray_median) and np.isfinite(ray_k) and np.isfinite(ray_mad)
            else np.nan
        )
        filter_steps: list[dict[str, int | str]] = []
        step_keep = np.ones((len(pair_eval),), dtype=bool)
        for step_column in (
            "navigation_valid",
            "triangulation_valid_depth",
            "triangulation_distance_valid",
            "triangulation_angle_valid",
            "ray_closest_distance_valid",
            "reprojection_threshold_inlier",
            "image_ransac_inlier",
        ):
            step_condition = (
                pair_eval[step_column].astype(bool).to_numpy()
                if step_column in pair_eval.columns
                else np.ones((len(pair_eval),), dtype=bool)
            )
            step_deleted = step_keep & ~step_condition
            step_keep = step_keep & step_condition
            filter_steps.append(
                {
                    "stage": step_column,
                    "deleted_this_step": int(step_deleted.sum()),
                    "remaining_after": int(step_keep.sum()),
                }
            )
        metadata["filter_steps"] = filter_steps
        if retained_mask.any():
            draw_ransac_retained_reprojection_qc(
                img1,
                img2,
                pair_eval[["x1", "y1"]].to_numpy(),
                pair_eval[["x2", "y2"]].to_numpy(),
                ransac_inlier=ransac_inlier,
                retained_mask=retained_mask,
                reprojected_first=pair_eval[["reprojected1_x", "reprojected1_y"]].to_numpy(),
                reprojected_second=pair_eval[["reprojected2_x", "reprojected2_y"]].to_numpy(),
                metadata=metadata,
                path=_retained_root / f"{stem}_retained_matches.jpg",
                include_rejected_ransac=False,
                max_per_row=max(1, len(pair_eval)),
                reprojection_layout="side_by_side",
                reprojection_points="retained",
                retained_match_color_mode="index",
                text_scale_multiplier=1.5,
            )
        draw_ransac_retained_reprojection_qc(
            img1,
            img2,
            pair_eval[["x1", "y1"]].to_numpy(),
            pair_eval[["x2", "y2"]].to_numpy(),
            ransac_inlier=ransac_inlier,
            retained_mask=retained_mask,
            reprojected_first=pair_eval[["reprojected1_x", "reprojected1_y"]].to_numpy(),
            reprojected_second=pair_eval[["reprojected2_x", "reprojected2_y"]].to_numpy(),
            metadata=metadata,
            path=_ransac_qc_root / f"{stem}_ransac_retained_qc.jpg",
            include_rejected_ransac=True,
            reprojection_layout="side_by_side",
            reprojection_points="ransac_inliers",
            retained_match_color_mode="green",
            text_scale_multiplier=1.5,
        )


def run_staged_inter_camera_matching(
    *,
    config_path: Path,
    dataset_root: Path,
    cameras: tuple[str, ...],
    semantic_cache_root: Path = Path("data_cache/semantic_masks"),
    start_frame: int = 0,
    frame_count: int | None = None,
    mask_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    mask_backend: str = "mask2former",
    mask_confidence: float = 0.15,
    mask_image_size: int = 1280,
    mask_dilation_pixels: int = 5,
    relations: tuple[str, ...] = ("same_index",),
    viz_cache_root: Path | None = None,
    viz_cache_key: str = "",
    matching_scope: str = "inter",
    image_cache_mode: str = "preload",
    preflight_cache_root: Path | None = None,
    reuse_preflight_matches: bool = True,
) -> pd.DataFrame:
    config = _config_for_matching_scope(load_config(config_path), matching_scope)
    if Path(dataset_root).resolve() != Path(config.dataset_root).resolve():
        config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    assumptions = dict(getattr(config, "assumptions", {}) or {})
    preflight_cache = None
    if preflight_cache_root is not None and reuse_preflight_matches:
        from .clip_preflight import PreflightMatchCache

        candidate_cache = PreflightMatchCache(Path(preflight_cache_root))
        if candidate_cache.has_entries:
            preflight_cache = candidate_cache
    navigation = _navigation_for_config(config)
    navigation_offset_ns = navigation_time_offset_ns(assumptions)
    temporal_overlap_max_frame_gap = int(
        assumptions.get("inter_camera_temporal_overlap_max_frame_gap", 300)
    )
    selected_by_camera: dict[str, list] = {}
    image_store_by_camera: dict[str, ImageStore] = {}
    calibration_by_camera: dict[str, CameraCalibration] = {}
    semantic_index_by_camera: dict[str, SemanticMaskIndex] = {}
    masks_by_camera_frame: dict[str, dict[int, object]] = {}
    centers_by_camera_frame: dict[str, dict[int, np.ndarray]] = {}
    view_dirs_by_camera_frame: dict[str, dict[int, np.ndarray]] = {}
    semantic_config = getattr(config, "semantic_segmentation", None)
    semantic_dynamic_classes = frozenset(
        str(item)
        for item in assumptions.get("semantic_mask_classes", DEFAULT_DYNAMIC_CLASSES)
    )
    for camera in cameras:
        frames = _frames_for_config(config, camera)
        window_start, window_end = _resolve_window(
            len(frames), start_frame=start_frame, frame_count=frame_count
        )
        selected = frames[window_start:window_end]
        selected_by_camera[camera] = selected
        image_store_by_camera[camera] = ImageStore.from_paths(
            [frame.path for frame in selected],
            **image_store_options(config),
        )
        calibration_by_camera[camera] = _calibration_for_config(config, camera)
        semantic_index = SemanticMaskIndex(
            Path(semantic_cache_root),
            camera,
            mask_model,
            mask_backend,
            mask_confidence,
            mask_dilation_pixels,
            inference_dtype=str(getattr(semantic_config, "inference_dtype", "fp16")),
            use_fast_processor=bool(
                getattr(semantic_config, "use_fast_processor", True)
            ),
            dynamic_classes=semantic_dynamic_classes,
        )
        missing = [
            frame.timestamp_ns
            for frame in selected
            if frame.timestamp_ns not in semantic_index.paths
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} semantic masks are missing for {camera}; first timestamp: {missing[0]}"
            )
        semantic_index_by_camera[camera] = semantic_index
        masks_by_camera_frame[camera] = {
            int(frame.index): semantic_index.load(frame) for frame in selected
        }
        centers_by_camera_frame[camera] = _camera_centers_by_frame_index(
            frames=selected,
            calibration=calibration_by_camera[camera],
            navigation=navigation,
            navigation_offset_ns=navigation_offset_ns,
        )
        view_dirs_by_camera_frame[camera] = _camera_view_dirs_by_frame_index(
            frames=selected,
            calibration=calibration_by_camera[camera],
            navigation=navigation,
            navigation_offset_ns=navigation_offset_ns,
        )

    records: list[pd.DataFrame] = []
    pair_records: list[dict[str, object]] = []
    for first_camera, second_camera in combinations(cameras, 2):
        second_by_index = {
            int(frame.index): frame for frame in selected_by_camera[second_camera]
        }
        first_by_index = {
            int(frame.index): frame for frame in selected_by_camera[first_camera]
        }
        cam_pair_label = f"{first_camera}__{second_camera}"
        relation_to_pairs: dict[str, list[tuple[str, str, object, object, str, str]]] = {}
        if "same_index" in relations and _is_same_index_pair_allowed(
            assumptions, first_camera, second_camera
        ):
            cache_first_camera, cache_second_camera = _declared_same_index_pair(
                assumptions, first_camera, second_camera
            )
            relation_to_pairs["same_index"] = [
                (
                    first_camera,
                    second_camera,
                    first,
                    second_by_index[int(first.index)],
                    cache_first_camera,
                    cache_second_camera,
                )
                for first in selected_by_camera[first_camera]
                if int(first.index) in second_by_index
            ]
        if "temporal_overlap" in relations and _is_temporal_overlap_direction_allowed(
            assumptions, first_camera, second_camera
        ):
            cache_first_camera, cache_second_camera = _declared_temporal_overlap_pair(
                assumptions, first_camera, second_camera
            )
            search_first_camera, search_second_camera = (
                _temporal_overlap_search_direction_for_pair(
                    assumptions, cache_first_camera, cache_second_camera
                )
            )
            if (
                search_first_camera not in selected_by_camera
                or search_second_camera not in selected_by_camera
            ):
                continue
            search_first_by_index = {
                int(frame.index): frame
                for frame in selected_by_camera[search_first_camera]
            }
            search_second_by_index = {
                int(frame.index): frame
                for frame in selected_by_camera[search_second_camera]
            }
            for target_baseline_m in _temporal_overlap_baselines_for_pair(
                assumptions, cache_first_camera, cache_second_camera
            ):
                relation_label = _temporal_overlap_relation_label(float(target_baseline_m))
                temporal_frame_pairs: list[tuple[str, str, object, object, str, str]] = []
                seen_pairs: set[tuple[int, int]] = set()
                temporal_pairs = _build_temporal_overlap_pairs(
                    first_frames=selected_by_camera[search_first_camera],
                    second_frames=selected_by_camera[search_second_camera],
                    first_centers_by_index=centers_by_camera_frame[search_first_camera],
                    second_centers_by_index=centers_by_camera_frame[search_second_camera],
                    first_view_dirs_by_index=view_dirs_by_camera_frame[search_first_camera],
                    second_view_dirs_by_index=view_dirs_by_camera_frame[search_second_camera],
                    target_baseline_m=float(target_baseline_m),
                    max_frame_gap=temporal_overlap_max_frame_gap,
                    min_opposite_view_dot=_temporal_overlap_view_dot_threshold_for_pair(
                        assumptions, cache_first_camera, cache_second_camera
                    ),
                )
                for first_index, second_index in temporal_pairs:
                    key = tuple(sorted((int(first_index), int(second_index))))
                    if key in seen_pairs:
                        continue
                    first_frame = search_first_by_index.get(int(first_index))
                    second_frame = search_second_by_index.get(int(second_index))
                    if first_frame is None or second_frame is None:
                        continue
                    seen_pairs.add(key)
                    temporal_frame_pairs.append(
                        (
                            search_first_camera,
                            search_second_camera,
                            first_frame,
                            second_frame,
                            cache_first_camera,
                            cache_second_camera,
                        )
                    )
                if temporal_frame_pairs:
                    relation_to_pairs[relation_label] = temporal_frame_pairs
        for relation, frame_pairs in relation_to_pairs.items():
            _viz_all_root: Path | None = None
            if viz_cache_root is not None and viz_cache_key:
                first_cache_camera = frame_pairs[0][4]
                second_cache_camera = frame_pairs[0][5]
                _viz_all_root = (
                    inter_camera_cache_dir(
                        viz_cache_root,
                        cache_key=viz_cache_key,
                        first_camera=first_cache_camera,
                        second_camera=second_cache_camera,
                        relation=relation,
                    )
                    / "visualizations"
                    / "all"
                )
                _viz_all_root.mkdir(parents=True, exist_ok=True)
            parallel_results: dict[int, tuple[object, pd.DataFrame, float]] = {}
            batch_size = max(1, int(getattr(config.matching, "deep_batch_size", 1)))
            use_batch = (
                _supports_batch_backend(config.matching.backend)
                and batch_size > 1
                and preflight_cache is None
            )
            if use_batch:
                payloads: list[dict[str, object]] = []
                contexts: list[tuple[int, object, object, object, object]] = []
                for pair_index, (
                    match_first_camera,
                    match_second_camera,
                    first,
                    second,
                    _cache_first_camera,
                    _cache_second_camera,
                ) in enumerate(frame_pairs):
                    img1 = image_store_by_camera[match_first_camera].get(first.path)
                    img2 = image_store_by_camera[match_second_camera].get(second.path)
                    if img1 is None or img2 is None:
                        continue
                    pair_config = _config_for_inter_camera_pair(
                        config, match_first_camera, match_second_camera
                    )
                    payloads.append(
                        {
                            "first_image": img1,
                            "second_image": img2,
                            "first_image_key": str(first.path),
                            "second_image_key": str(second.path),
                            "calibration": calibration_by_camera[match_first_camera],
                            "config": pair_config.matching,
                            "first_dynamic_mask": masks_by_camera_frame[match_first_camera][int(first.index)],
                            "second_dynamic_mask": masks_by_camera_frame[match_second_camera][int(second.index)],
                            "first_original_shape": image_store_by_camera[match_first_camera].original_shape(first.path),
                            "second_original_shape": image_store_by_camera[match_second_camera].original_shape(second.path),
                        }
                    )
                    contexts.append((pair_index, img1, img2, pair_config, first))
                    if len(payloads) < batch_size:
                        continue
                    started = time.perf_counter()
                    batch_matches = run_match_backend_batch(payloads)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                    for (context, matched) in zip(contexts, batch_matches, strict=True):
                        pair_index, img1_ctx, img2_ctx, pair_config_ctx, _ = context
                        parallel_results[pair_index] = (
                            pair_config_ctx, matched, elapsed_ms
                        )
                    payloads = []
                    contexts = []
                if payloads:
                    started = time.perf_counter()
                    batch_matches = run_match_backend_batch(payloads)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(payloads)
                    for (pair_index, _img1, _img2, pair_config, _), matched in zip(
                        contexts, batch_matches, strict=True
                    ):
                        parallel_results[pair_index] = (
                            pair_config, matched, elapsed_ms
                        )
            for pair_index, (
                match_first_camera,
                match_second_camera,
                first,
                second,
                cache_first_camera,
                cache_second_camera,
            ) in enumerate(frame_pairs):
                matching_source = "full_loma"
                if pair_index in parallel_results:
                    pair_config, matched, elapsed_ms = parallel_results[pair_index]
                    img1 = image_store_by_camera[match_first_camera].get(first.path)
                    img2 = image_store_by_camera[match_second_camera].get(second.path)
                else:
                    img1 = image_store_by_camera[match_first_camera].get(first.path)
                    img2 = image_store_by_camera[match_second_camera].get(second.path)
                    if img1 is None or img2 is None:
                        continue
                    pair_config = _config_for_inter_camera_pair(
                        config,
                        match_first_camera,
                        match_second_camera,
                    )
                    started = time.perf_counter()
                    matching_source = "full_loma"
                    matched = None
                    if preflight_cache is not None:
                        from .clip_preflight import make_preflight_pair_key
                        from .match_backends import apply_semantic_masks

                        pair_key = make_preflight_pair_key(
                            pair_type="inter_camera",
                            camera1=match_first_camera,
                            camera2=match_second_camera,
                            relation=relation,
                            frame1=int(first.index),
                            frame2=int(second.index),
                        )
                        cached = preflight_cache.get(pair_key)
                        if cached is not None:
                            matched = apply_semantic_masks(
                                cached,
                                first_mask=masks_by_camera_frame[match_first_camera][int(first.index)],
                                second_mask=masks_by_camera_frame[match_second_camera][int(second.index)],
                                first_shape=image_store_by_camera[match_first_camera].original_shape(first.path),
                                second_shape=image_store_by_camera[match_second_camera].original_shape(second.path),
                            )
                            matching_source = "preflight_reused"
                    if matched is None:
                        matched = run_match_backend(
                            img1, img2,
                            calibration_by_camera[match_first_camera],
                            pair_config.matching,
                            first_semantic_mask=masks_by_camera_frame[match_first_camera][int(first.index)],
                            second_semantic_mask=masks_by_camera_frame[match_second_camera][int(second.index)],
                            first_image_key=str(first.path),
                            second_image_key=str(second.path),
                            first_original_shape=image_store_by_camera[match_first_camera].original_shape(first.path),
                            second_original_shape=image_store_by_camera[match_second_camera].original_shape(second.path),
                        )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                cache_metrics = _feature_cache_metrics(matched)
                matched = matched.copy()
                if not matched.empty:
                    matched[["nx1", "ny1"]] = undistort_normalized(
                        matched[["x1", "y1"]].to_numpy(dtype="float64"),
                        calibration_by_camera[match_first_camera],
                    )
                    matched[["nx2", "ny2"]] = undistort_normalized(
                        matched[["x2", "y2"]].to_numpy(dtype="float64"),
                        calibration_by_camera[match_second_camera],
                    )
                matched["pair_type"] = "inter_camera"
                matched["pair_relation"] = relation
                matched["camera1"] = match_first_camera
                matched["camera2"] = match_second_camera
                matched["cache_camera1"] = cache_first_camera
                matched["cache_camera2"] = cache_second_camera
                matched["frame1"] = int(first.index)
                matched["frame2"] = int(second.index)
                matched["timestamp1_ns"] = int(first.timestamp_ns)
                matched["timestamp2_ns"] = int(second.timestamp_ns)
                matched["frame_offset"] = int(second.index) - int(first.index)
                matched["temporal_delta_ns"] = int(second.timestamp_ns - first.timestamp_ns)
                matched["matching_source"] = matching_source
                records.append(matched)
                sample_ratio = float(
                    getattr(config.matching, "cache_visualization_pair_ratio", 0.01)
                )
                if (
                    _viz_all_root is not None
                    and not matched.empty
                    and _should_export_match_visualization(int(first.index), sample_ratio)
                ):
                    from .visualization import draw_tiled_match_overlay
                    stem = (
                        f"{cache_first_camera}__{cache_second_camera}_{relation}_"
                        f"{int(first.index):04d}_{int(second.index):04d}"
                    )
                    draw_tiled_match_overlay(
                        img1, img2,
                        matched[["x1", "y1"]].to_numpy(),
                        matched[["x2", "y2"]].to_numpy(),
                        _viz_all_root / f"{stem}_all_matches.jpg",
                        title_prefix=f"{match_first_camera} vs {match_second_camera} {relation}",
                    )
                pair_records.append(
                    {
                        "pair_type": "inter_camera",
                        "pair_relation": relation,
                        "camera1": match_first_camera,
                        "camera2": match_second_camera,
                        "cache_camera1": cache_first_camera,
                        "cache_camera2": cache_second_camera,
                        "frame1": int(first.index),
                        "frame2": int(second.index),
                        "timestamp1_ns": int(first.timestamp_ns),
                        "timestamp2_ns": int(second.timestamp_ns),
                        "temporal_delta_ns": int(second.timestamp_ns - first.timestamp_ns),
                        "raw_match_count": int(len(matched)),
                        "matching_time_ms": float(elapsed_ms),
                        "matching_source": matching_source,
                        **cache_metrics,
                    }
                )
    matches = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    matches.attrs["pair_metrics"] = (
        pd.DataFrame.from_records(pair_records)
        if pair_records
        else pd.DataFrame(
            columns=[
                "pair_type",
                "pair_relation",
                "camera1",
                "camera2",
                "frame1",
                "frame2",
                "timestamp1_ns",
                "timestamp2_ns",
                "temporal_delta_ns",
                "raw_match_count",
                "matching_time_ms",
            ]
        )
    )
    return matches


def _inter_camera_image_ransac_mask(
    matches: pd.DataFrame,
    *,
    threshold_px: float,
    confidence: float,
) -> np.ndarray:
    count = len(matches)
    if count < 8 or float(threshold_px) <= 0.0:
        return np.ones((count,), dtype=bool)
    try:
        _fundamental, mask = cv2.findFundamentalMat(
            matches[["x1", "y1"]].to_numpy(dtype=np.float32),
            matches[["x2", "y2"]].to_numpy(dtype=np.float32),
            cv2.FM_RANSAC,
            float(threshold_px),
            float(confidence),
        )
    except cv2.error:
        mask = None
    if mask is None:
        return np.ones((count,), dtype=bool)
    return mask.reshape(-1).astype(bool)


def _boolean_quality_column(
    matches: pd.DataFrame,
    column: str,
    default: bool = True,
) -> np.ndarray:
    if column not in matches.columns:
        return np.full((len(matches),), bool(default), dtype=bool)
    return matches[column].astype(bool).to_numpy()


def _load_image_camera_to_world_csv(
    csv_path: Path | str | None,
) -> dict[tuple[str, int], np.ndarray]:
    if csv_path in (None, ""):
        return {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"camera", "frame"} | {f"T_cw_{row}{column}" for row in range(4) for column in range(4)}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"optimized image pose csv missing columns: {', '.join(missing)}")
    poses: dict[tuple[str, int], np.ndarray] = {}
    matrix_columns = [f"T_cw_{row}{column}" for row in range(4) for column in range(4)]
    for row in table[["camera", "frame", *matrix_columns]].itertuples(index=False):
        camera = str(row.camera)
        frame = int(row.frame)
        values = np.asarray([getattr(row, column) for column in matrix_columns], dtype=np.float64)
        poses[(camera, frame)] = values.reshape(4, 4)
    return poses


def _optimized_pose_csv_from_assumptions(
    assumptions: dict[str, object],
    dataset_root: Path,
) -> Path | None:
    value = assumptions.get("optimized_image_camera_to_world_csv")
    if value in (None, ""):
        value = assumptions.get("match_quality_optimized_image_camera_to_world_csv")
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(dataset_root) / path
    return path


def _apply_match_quality_filter(
    matches: pd.DataFrame,
    *,
    image_ransac_threshold_px: float,
    image_ransac_confidence: float,
    pose_degraded_detection_enabled: bool = False,
    pose_degraded_min_ransac_inliers: int = 30,
    pose_degraded_reprojection_median_px: float = 80.0,
    pose_degraded_reprojection_p75_px: float = 120.0,
    pose_degraded_ray_gap_median_m: float = 0.8,
    pose_degraded_ray_gap_p75_m: float = 1.2,
    pose_degraded_quality_mode: str = "standard",
) -> pd.DataFrame:
    result = matches.copy()
    if result.empty:
        return _replace_dataframe_columns_once(
            result,
            {
                "image_ransac_inlier": pd.Series(dtype="bool"),
                "quality_filter_valid": pd.Series(dtype="bool"),
                "quality_reject_reason": pd.Series(dtype="object"),
                "precision_filter_valid": pd.Series(dtype="bool"),
            },
        )

    conditions: list[tuple[str, str, np.ndarray]] = [
        (
            "pose_quality_valid",
            "invalid_pose_quality",
            _boolean_quality_column(result, "pose_quality_valid"),
        ),
        (
            "triangulation_valid_depth",
            "invalid_triangulation_depth",
            _boolean_quality_column(result, "triangulation_valid_depth"),
        ),
        (
            "triangulation_distance_valid",
            "far_triangulated_point",
            _boolean_quality_column(result, "triangulation_distance_valid"),
        ),
        (
            "triangulation_angle_valid",
            "low_triangulation_angle",
            _boolean_quality_column(result, "triangulation_angle_valid"),
        ),
        (
            "ray_closest_distance_valid",
            "large_ray_gap",
            _boolean_quality_column(result, "ray_closest_distance_valid"),
        ),
        (
            "reprojection_threshold_inlier",
            "large_reprojection_error",
            _boolean_quality_column(result, "reprojection_threshold_inlier"),
        ),
    ]

    keep = np.ones((len(result),), dtype=bool)
    reasons = np.full((len(result),), "kept", dtype=object)
    output_columns: dict[str, object] = {}
    for column, reason, condition in conditions:
        output_columns[f"quality_stage_{column}"] = condition
        rejected = keep & ~condition
        reasons[rejected] = reason
        keep[rejected] = False

    image_ransac_inlier = _inter_camera_image_ransac_mask(
        result,
        threshold_px=image_ransac_threshold_px,
        confidence=image_ransac_confidence,
    )
    output_columns["image_ransac_inlier"] = image_ransac_inlier
    output_columns["quality_stage_image_ransac"] = image_ransac_inlier

    depth_valid_for_pose_degraded = _boolean_quality_column(
        result,
        "triangulation_valid_depth",
    )
    pose_degraded_reference = image_ransac_inlier & depth_valid_for_pose_degraded
    ransac_count = int(pose_degraded_reference.sum())
    reprojection_values = (
        result.loc[pose_degraded_reference, "reprojection_rms_px"].to_numpy(dtype=np.float64)
        if "reprojection_rms_px" in result.columns
        else np.array([], dtype=np.float64)
    )
    ray_gap_values = (
        result.loc[pose_degraded_reference, "ray_closest_distance_m"].to_numpy(dtype=np.float64)
        if "ray_closest_distance_m" in result.columns
        else np.array([], dtype=np.float64)
    )
    reprojection_values = reprojection_values[np.isfinite(reprojection_values)]
    ray_gap_values = ray_gap_values[np.isfinite(ray_gap_values)]
    reprojection_median = (
        float(np.median(reprojection_values)) if reprojection_values.size else np.nan
    )
    reprojection_p75 = (
        float(np.percentile(reprojection_values, 75)) if reprojection_values.size else np.nan
    )
    ray_gap_median = float(np.median(ray_gap_values)) if ray_gap_values.size else np.nan
    ray_gap_p75 = float(np.percentile(ray_gap_values, 75)) if ray_gap_values.size else np.nan
    trigger_reprojection = bool(
        np.isfinite(reprojection_median)
        and np.isfinite(reprojection_p75)
        and reprojection_median > float(pose_degraded_reprojection_median_px)
        and reprojection_p75 > float(pose_degraded_reprojection_p75_px)
    )
    trigger_ray_gap = bool(
        np.isfinite(ray_gap_median)
        and np.isfinite(ray_gap_p75)
        and ray_gap_median > float(pose_degraded_ray_gap_median_m)
        and ray_gap_p75 > float(pose_degraded_ray_gap_p75_m)
    )
    pose_degraded = bool(
        pose_degraded_detection_enabled
        and ransac_count >= int(pose_degraded_min_ransac_inliers)
        and (trigger_reprojection or trigger_ray_gap)
    )
    output_columns["pose_degraded"] = pose_degraded
    output_columns["pose_degraded_quality_mode"] = (
        str(pose_degraded_quality_mode) if pose_degraded else "standard"
    )
    output_columns["pose_degraded_ransac_inliers"] = ransac_count
    output_columns["pose_degraded_reprojection_median_px"] = reprojection_median
    output_columns["pose_degraded_reprojection_p75_px"] = reprojection_p75
    output_columns["pose_degraded_ray_gap_median_m"] = ray_gap_median
    output_columns["pose_degraded_ray_gap_p75_m"] = ray_gap_p75
    output_columns["pose_degraded_trigger_reprojection"] = trigger_reprojection
    output_columns["pose_degraded_trigger_ray_gap"] = trigger_ray_gap

    if pose_degraded and str(pose_degraded_quality_mode) == "ransac_only":
        depth_valid = depth_valid_for_pose_degraded
        keep = image_ransac_inlier & depth_valid
        reasons = np.full((len(result),), "kept", dtype=object)
        reasons[~image_ransac_inlier] = "image_ransac_outlier"
        reasons[image_ransac_inlier & ~depth_valid] = "invalid_triangulation_depth"
        output_columns["quality_filter_valid"] = keep
        output_columns["quality_reject_reason"] = reasons
        output_columns["precision_filter_valid"] = keep
        return _replace_dataframe_columns_once(result, output_columns)

    rejected = keep & ~image_ransac_inlier
    reasons[rejected] = "image_ransac_outlier"
    keep[rejected] = False

    output_columns["quality_filter_valid"] = keep
    output_columns["quality_reject_reason"] = reasons
    output_columns["precision_filter_valid"] = keep
    return _replace_dataframe_columns_once(result, output_columns)


def _evaluate_inter_camera_navigation(
    matches: pd.DataFrame,
    first_timestamp_ns: int,
    second_timestamp_ns: int,
    first_calibration: CameraCalibration,
    second_calibration: CameraCalibration,
    navigation: NavigationTrajectory,
    navigation_offset_ns: int = 0,
    camera_to_world_by_key: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, float]:
    result = matches.copy()
    numeric_inputs = {
        column: pd.to_numeric(result[column], errors="coerce").astype("float64")
        for column in ("x1", "y1", "x2", "y2", "nx1", "ny1", "nx2", "ny2")
        if column in result.columns
    }
    result = _replace_dataframe_columns_once(result, numeric_inputs)
    numeric_columns = (
        "nav_point_to_epiline_px",
        "nav_symmetric_epipolar_px",
        "nav_sampson_px2",
        "normalized_algebraic_residual",
        "reprojection1_px",
        "reprojection2_px",
        "reprojection_rms_px",
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "triangulated_point_distance_m",
        "triangulation_angle_deg",
        "ray_closest_distance_m",
    )
    default_navigation_columns: dict[str, object] = {
        "navigation_valid": False,
        **{column: np.nan for column in numeric_columns},
        "triangulation_valid_depth": False,
    }
    camera_to_world_by_key = camera_to_world_by_key or {}
    first_camera_name = str(
        result["camera1"].iloc[0]
        if "camera1" in result.columns
        else result["camera"].iloc[0]
        if "camera" in result.columns
        else first_calibration.camera
    )
    second_camera_name = str(
        result["camera2"].iloc[0]
        if "camera2" in result.columns
        else result["camera"].iloc[0]
        if "camera" in result.columns
        else second_calibration.camera
    )
    first_frame_index = (
        int(result["frame1"].iloc[0]) if "frame1" in result.columns else None
    )
    second_frame_index = (
        int(result["frame2"].iloc[0]) if "frame2" in result.columns else None
    )
    first_camera_to_world = (
        camera_to_world_by_key.get((first_camera_name, first_frame_index))
        if first_frame_index is not None
        else None
    )
    second_camera_to_world = (
        camera_to_world_by_key.get((second_camera_name, second_frame_index))
        if second_frame_index is not None
        else None
    )
    try:
        if first_camera_to_world is None:
            first_body = navigation.pose_at(int(first_timestamp_ns) + int(navigation_offset_ns))
            first_camera_to_world = first_body @ first_calibration.sensor_to_car
        if second_camera_to_world is None:
            second_body = navigation.pose_at(int(second_timestamp_ns) + int(navigation_offset_ns))
            second_camera_to_world = second_body @ second_calibration.sensor_to_car
    except ValueError:
        return (
            _replace_dataframe_columns_once(result, default_navigation_columns),
            None,
            0.0,
        )
    first_camera_to_world = np.asarray(first_camera_to_world, dtype=np.float64)
    second_camera_to_world = np.asarray(second_camera_to_world, dtype=np.float64)
    pair_baseline_m = float(
        np.linalg.norm(second_camera_to_world[:3, 3] - first_camera_to_world[:3, 3])
    )
    first_to_second = np.linalg.inv(second_camera_to_world) @ first_camera_to_world
    essential = essential_from_relative_pose(first_to_second)
    fundamental = fundamental_from_essential(
        essential,
        first_calibration.intrinsic,
        second_calibration.intrinsic,
    )
    pixel_errors = epipolar_errors(
        result[["x1", "y1"]].to_numpy(),
        result[["x2", "y2"]].to_numpy(),
        fundamental,
    )
    normalized_errors = epipolar_errors(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        essential,
    )
    first_world_to_camera = np.linalg.inv(first_camera_to_world)
    second_world_to_camera = np.linalg.inv(second_camera_to_world)
    triangulation = triangulate_and_reproject(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        first_world_to_camera,
        second_world_to_camera,
    )
    ray_metrics = ray_geometry_metrics(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        first_camera_to_world,
        second_camera_to_world,
    )
    first_camera_points = triangulation["first_camera_points"]
    second_camera_points = triangulation["second_camera_points"]
    first_pixels = project_points(first_camera_points, first_calibration)
    second_pixels = project_points(second_camera_points, second_calibration)
    first_reprojection = np.linalg.norm(
        first_pixels - result[["x1", "y1"]].to_numpy(), axis=1
    )
    second_reprojection = np.linalg.norm(
        second_pixels - result[["x2", "y2"]].to_numpy(), axis=1
    )
    first_camera_center_world = first_camera_to_world[:3, 3]
    result = _replace_dataframe_columns_once(
        result,
        {
            "navigation_valid": True,
            "nav_point_to_epiline_px": pixel_errors["point_to_epiline"],
            "nav_symmetric_epipolar_px": pixel_errors["symmetric_epipolar"],
            "nav_sampson_px2": pixel_errors["sampson"],
            "normalized_algebraic_residual": normalized_errors["algebraic"],
            "reprojection1_px": first_reprojection,
            "reprojection2_px": second_reprojection,
            "reprojection_rms_px": np.sqrt(
                (first_reprojection**2 + second_reprojection**2) / 2.0
            ),
            "reprojected1_x": first_pixels[:, 0],
            "reprojected1_y": first_pixels[:, 1],
            "reprojected2_x": second_pixels[:, 0],
            "reprojected2_y": second_pixels[:, 1],
            "triangulated_point_distance_m": np.linalg.norm(
                triangulation["world_points"] - first_camera_center_world,
                axis=1,
            ),
            "triangulation_angle_deg": ray_metrics["triangulation_angle_deg"],
            "ray_closest_distance_m": ray_metrics["ray_closest_distance_m"],
            "triangulation_valid_depth": triangulation["valid_depth"],
        },
    )
    return result, fundamental, pair_baseline_m


def evaluate_inter_camera_match_quality(
    *,
    config_path: Path,
    dataset_root: Path,
    matches: pd.DataFrame,
    start_frame: int = 0,
    frame_count: int | None = None,
    pose_pkl_path: Path | None = None,
    pose_field: str | None = None,
) -> pd.DataFrame:
    if matches.empty:
        result = matches.copy()
        result.attrs["pair_metrics"] = pd.DataFrame()
        return result

    config = load_config(config_path)
    config_dataset_root = Path(getattr(config, "dataset_root", dataset_root)).resolve()
    if Path(dataset_root).resolve() != config_dataset_root and hasattr(config, "model_copy"):
        config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    assumptions = dict(config.assumptions)
    if pose_pkl_path is not None:
        assumptions["result_interpolation_pkl_path"] = str(Path(pose_pkl_path).resolve())
    if pose_field is not None:
        assumptions["result_interpolation_ego_field"] = str(pose_field)
    if pose_pkl_path is not None or pose_field is not None:
        config = config.model_copy(update={"assumptions": assumptions})
    navigation = _navigation_for_config(config)
    navigation_offset_ns = navigation_time_offset_ns(assumptions)
    camera_to_world_by_key = _load_image_camera_to_world_csv(
        _optimized_pose_csv_from_assumptions(assumptions, config.dataset_root)
    )
    max_triangulated_point_distance_m = float(
        assumptions.get(
            "max_triangulated_point_distance_m",
            DEFAULT_MAX_TRIANGULATED_POINT_DISTANCE_M,
        )
    )
    min_triangulation_angle_deg = float(
        assumptions.get("min_triangulation_angle_deg", 0.5)
    )
    quality_threshold_mode = str(assumptions.get("quality_threshold_mode", "fixed"))
    quality_threshold_mad_k = float(assumptions.get("quality_threshold_mad_k", 2.0))
    quality_threshold_min_candidates = int(
        assumptions.get("quality_threshold_min_candidates", 20)
    )
    quality_threshold_insufficient_reprojection_px = float(
        assumptions.get("quality_threshold_insufficient_reprojection_px", 5.0)
    )
    quality_threshold_insufficient_ray_closest_distance_m = float(
        assumptions.get("quality_threshold_insufficient_ray_closest_distance_m", 0.10)
    )
    default_max_ray_closest_distance_m = float(
        assumptions.get("max_ray_closest_distance_m", 0.5)
    )
    ray_closest_distance_mode = str(
        assumptions.get("ray_closest_distance_mode", quality_threshold_mode)
    )
    default_ray_closest_distance_min_m = assumptions.get("ray_closest_distance_min_m")
    default_ray_closest_distance_max_m = assumptions.get("ray_closest_distance_max_m")
    default_ray_closest_distance_mad_k = float(
        assumptions.get("ray_closest_distance_mad_k", quality_threshold_mad_k)
    )
    ray_closest_distance_min_candidates = int(
        assumptions.get(
            "ray_closest_distance_min_candidates",
            quality_threshold_min_candidates,
        )
    )
    reprojection_threshold_mode = str(
        assumptions.get("reprojection_threshold_mode", quality_threshold_mode)
    )
    default_reprojection_threshold_min_px = assumptions.get(
        "inter_camera_reprojection_threshold_min_px"
    )
    default_reprojection_threshold_max_px = assumptions.get(
        "inter_camera_reprojection_threshold_max_px"
    )
    reprojection_threshold_mad_k = float(
        assumptions.get("reprojection_threshold_mad_k", quality_threshold_mad_k)
    )
    reprojection_threshold_min_candidates = int(
        assumptions.get(
            "reprojection_threshold_min_candidates",
            quality_threshold_min_candidates,
        )
    )
    default_threshold_px = float(assumptions.get("inter_camera_reprojection_threshold_px", 30.0))
    default_image_ransac_threshold_px = float(
        assumptions.get(
            "inter_camera_image_ransac_threshold_px",
            assumptions.get("match_quality_image_ransac_threshold_px", 50.0),
        )
    )
    default_image_ransac_confidence = float(
        assumptions.get(
            "inter_camera_image_ransac_confidence",
            assumptions.get("match_quality_image_ransac_confidence", 0.999),
        )
    )
    pose_degraded_detection_enabled = bool(
        assumptions.get("pose_degraded_detection_enabled", False)
    )
    pose_degraded_min_ransac_inliers = int(
        assumptions.get("pose_degraded_min_ransac_inliers", 30)
    )
    pose_degraded_reprojection_median_px = float(
        assumptions.get("pose_degraded_reprojection_median_px", 80.0)
    )
    pose_degraded_reprojection_p75_px = float(
        assumptions.get("pose_degraded_reprojection_p75_px", 120.0)
    )
    pose_degraded_ray_gap_median_m = float(
        assumptions.get("pose_degraded_ray_gap_median_m", 0.8)
    )
    pose_degraded_ray_gap_p75_m = float(
        assumptions.get("pose_degraded_ray_gap_p75_m", 1.2)
    )
    pose_degraded_quality_mode = str(
        assumptions.get("pose_degraded_quality_mode", "standard")
    )
    camera_names = sorted(
        set(matches.get("camera1", pd.Series(dtype=str)).astype(str))
        | set(matches.get("camera2", pd.Series(dtype=str)).astype(str))
    )
    calibration_by_camera = {
        camera: _calibration_for_config(config, camera)
        for camera in camera_names
        if camera and camera != "nan"
    }
    matches = _recompute_normalized_coordinates_for_calibrations(
        matches,
        calibration_by_camera,
    )
    frames_by_camera_index: dict[str, dict[int, object]] = {}
    for camera in camera_names:
        if camera not in calibration_by_camera:
            continue
        frames = _frames_for_config(config, camera)
        window_start, window_end = _resolve_window(
            len(frames), start_frame=start_frame, frame_count=frame_count
        )
        frames_by_camera_index[camera] = {
            int(frame.index): frame for frame in frames[window_start:window_end]
        }

    evaluated_groups: list[pd.DataFrame] = []
    pair_records: list[dict[str, object]] = []
    pair_metrics = matches.attrs.get("pair_metrics")
    for (camera1, camera2, frame1, frame2), group in matches.groupby(
        ["camera1", "camera2", "frame1", "frame2"], sort=False
    ):
        first_frame = frames_by_camera_index.get(str(camera1), {}).get(int(frame1))
        second_frame = frames_by_camera_index.get(str(camera2), {}).get(int(frame2))
        if first_frame is None or second_frame is None:
            continue
        evaluated, _fundamental, pair_baseline_m = _evaluate_inter_camera_navigation(
            group,
            int(first_frame.timestamp_ns),
            int(second_frame.timestamp_ns),
            calibration_by_camera[str(camera1)],
            calibration_by_camera[str(camera2)],
            navigation,
            navigation_offset_ns=navigation_offset_ns,
            camera_to_world_by_key=camera_to_world_by_key,
        )
        evaluated["pair_baseline_m"] = pair_baseline_m
        threshold_px = float(
            _assumption_for_camera_pair(
                assumptions,
                "inter_camera_reprojection_threshold_px",
                str(camera1),
                str(camera2),
                default_threshold_px,
            )
        )
        reprojection_threshold_min_px = float(
            _assumption_for_camera_pair(
                assumptions,
                "inter_camera_reprojection_threshold_min_px",
                str(camera1),
                str(camera2),
                default_reprojection_threshold_min_px,
            )
        ) if default_reprojection_threshold_min_px is not None else None
        reprojection_threshold_max_px = float(
            _assumption_for_camera_pair(
                assumptions,
                "inter_camera_reprojection_threshold_max_px",
                str(camera1),
                str(camera2),
                default_reprojection_threshold_max_px,
            )
        ) if default_reprojection_threshold_max_px is not None else None
        max_ray_closest_distance_m = float(
            _assumption_for_camera_pair(
                assumptions,
                "max_ray_closest_distance_m",
                str(camera1),
                str(camera2),
                default_max_ray_closest_distance_m,
            )
        )
        ray_closest_distance_min_m = float(
            _assumption_for_camera_pair(
                assumptions,
                "ray_closest_distance_min_m",
                str(camera1),
                str(camera2),
                default_ray_closest_distance_min_m,
            )
        ) if default_ray_closest_distance_min_m is not None else None
        ray_closest_distance_max_m = float(
            _assumption_for_camera_pair(
                assumptions,
                "ray_closest_distance_max_m",
                str(camera1),
                str(camera2),
                default_ray_closest_distance_max_m,
            )
        ) if default_ray_closest_distance_max_m is not None else None
        ray_closest_distance_mad_k = float(
            _assumption_for_camera_pair(
                assumptions,
                "ray_closest_distance_mad_k",
                str(camera1),
                str(camera2),
                default_ray_closest_distance_mad_k,
            )
        )
        image_ransac_threshold_px = float(
            _assumption_for_camera_pair(
                assumptions,
                "inter_camera_image_ransac_threshold_px",
                str(camera1),
                str(camera2),
                default_image_ransac_threshold_px,
            )
        )
        image_ransac_confidence = float(
            _assumption_for_camera_pair(
                assumptions,
                "inter_camera_image_ransac_confidence",
                str(camera1),
                str(camera2),
                default_image_ransac_confidence,
            )
        )
        evaluated = _annotate_match_quality_thresholds(
            evaluated,
            motion_pair_valid=bool(pair_baseline_m >= MIN_OBSERVABLE_BASELINE_M),
            threshold_px=threshold_px,
            max_triangulated_point_distance_m=max_triangulated_point_distance_m,
            min_triangulation_angle_deg=min_triangulation_angle_deg,
            max_ray_closest_distance_m=max_ray_closest_distance_m,
            ray_closest_distance_mode=ray_closest_distance_mode,
            ray_closest_distance_min_m=ray_closest_distance_min_m,
            ray_closest_distance_max_m=ray_closest_distance_max_m,
            ray_closest_distance_mad_k=ray_closest_distance_mad_k,
            ray_closest_distance_min_candidates=ray_closest_distance_min_candidates,
            ray_closest_distance_insufficient_candidates_threshold_m=(
                quality_threshold_insufficient_ray_closest_distance_m
            ),
            reprojection_threshold_mode=reprojection_threshold_mode,
            reprojection_threshold_min_px=(
                threshold_px
                if str(reprojection_threshold_mode) == "fixed"
                else reprojection_threshold_min_px
            ),
            reprojection_threshold_max_px=reprojection_threshold_max_px,
            reprojection_threshold_mad_k=reprojection_threshold_mad_k,
            reprojection_threshold_min_candidates=reprojection_threshold_min_candidates,
            reprojection_insufficient_candidates_threshold_px=(
                quality_threshold_insufficient_reprojection_px
            ),
        )
        evaluated = _apply_match_quality_filter(
            evaluated,
            image_ransac_threshold_px=image_ransac_threshold_px,
            image_ransac_confidence=image_ransac_confidence,
            pose_degraded_detection_enabled=pose_degraded_detection_enabled,
            pose_degraded_min_ransac_inliers=pose_degraded_min_ransac_inliers,
            pose_degraded_reprojection_median_px=pose_degraded_reprojection_median_px,
            pose_degraded_reprojection_p75_px=pose_degraded_reprojection_p75_px,
            pose_degraded_ray_gap_median_m=pose_degraded_ray_gap_median_m,
            pose_degraded_ray_gap_p75_m=pose_degraded_ray_gap_p75_m,
            pose_degraded_quality_mode=pose_degraded_quality_mode,
        )
        retained_mask = _retained_mask(evaluated)
        valid_values = evaluated.loc[
            retained_mask & np.isfinite(evaluated["reprojection_rms_px"]),
            "reprojection_rms_px",
        ].to_numpy(dtype=np.float64)
        metrics = summarize_errors(valid_values)
        pair_record = {
            "pair_type": "inter_camera",
            "camera1": camera1,
            "camera2": camera2,
            "frame1": int(frame1),
            "frame2": int(frame2),
            "timestamp1_ns": int(first_frame.timestamp_ns),
            "timestamp2_ns": int(second_frame.timestamp_ns),
            "temporal_delta_ns": int(second_frame.timestamp_ns - first_frame.timestamp_ns),
            "pair_baseline_m": pair_baseline_m,
            "reprojection_threshold_px": float(
                evaluated["reprojection_threshold_px"].iloc[0]
                if "reprojection_threshold_px" in evaluated.columns
                else threshold_px
            ),
            "reprojection_threshold_mode": str(reprojection_threshold_mode),
            "ray_closest_distance_max_m": ray_closest_distance_max_m,
            "ray_closest_distance_mad_k": ray_closest_distance_mad_k,
            "image_ransac_threshold_px": image_ransac_threshold_px,
            "image_ransac_confidence": image_ransac_confidence,
            "image_ransac_rejected": int(
                (~evaluated.get(
                    "image_ransac_inlier",
                    pd.Series([True] * len(evaluated), index=evaluated.index),
                ).astype(bool)).sum()
            ),
            "accepted_matches": int(len(evaluated)),
            "retained_matches": int(retained_mask.sum()),
            "valid_triangulations": int(metrics.get("count", 0)),
            "reprojection_median_px": metrics.get("median", np.nan),
            "reprojection_p95_px": metrics.get("p95", np.nan),
            "pose_degraded": bool(
                evaluated["pose_degraded"].iloc[0]
                if "pose_degraded" in evaluated.columns
                else False
            ),
            "pose_degraded_reprojection_median_px": float(
                evaluated["pose_degraded_reprojection_median_px"].iloc[0]
                if "pose_degraded_reprojection_median_px" in evaluated.columns
                else np.nan
            ),
            "pose_degraded_reprojection_p75_px": float(
                evaluated["pose_degraded_reprojection_p75_px"].iloc[0]
                if "pose_degraded_reprojection_p75_px" in evaluated.columns
                else np.nan
            ),
            "pose_degraded_ray_gap_median_m": float(
                evaluated["pose_degraded_ray_gap_median_m"].iloc[0]
                if "pose_degraded_ray_gap_median_m" in evaluated.columns
                else np.nan
            ),
            "pose_degraded_ray_gap_p75_m": float(
                evaluated["pose_degraded_ray_gap_p75_m"].iloc[0]
                if "pose_degraded_ray_gap_p75_m" in evaluated.columns
                else np.nan
            ),
        }
        if isinstance(pair_metrics, pd.DataFrame) and not pair_metrics.empty:
            base_rows = pair_metrics.loc[
                (pair_metrics["camera1"].astype(str) == str(camera1))
                & (pair_metrics["camera2"].astype(str) == str(camera2))
                & (pair_metrics["frame1"].astype(int) == int(frame1))
                & (pair_metrics["frame2"].astype(int) == int(frame2))
            ]
            if not base_rows.empty:
                for key, value in base_rows.iloc[0].to_dict().items():
                    pair_record.setdefault(key, value)
        pair_records.append(pair_record)
        evaluated.attrs.clear()
        evaluated_groups.append(evaluated)
    result = (
        pd.concat(evaluated_groups, ignore_index=True)
        if evaluated_groups
        else matches.iloc[0:0].copy()
    )
    result.attrs["pair_metrics"] = pd.DataFrame.from_records(pair_records)
    return result


def _filter_inter_cache_group_worker(
    config_path: Path,
    dataset_root: Path,
    cache_dirs: tuple[Path, ...],
    start_frame: int,
    frame_count: int | None,
    export_visualizations: bool,
) -> dict[str, int]:
    raw_parts: list[pd.DataFrame] = []
    pair_metric_parts: list[pd.DataFrame] = []
    raw_by_cache: list[tuple[Path, pd.DataFrame]] = []
    for cache_dir_value in cache_dirs:
        cache_dir = Path(cache_dir_value)
        raw = pd.read_parquet(cache_dir / "raw_matches.parquet")
        raw.attrs.clear()
        raw_parts.append(raw)
        raw_by_cache.append((cache_dir, raw))
        pair_metrics_path = cache_dir / "pair_metrics.csv"
        if pair_metrics_path.exists():
            pair_metric_parts.append(pd.read_csv(pair_metrics_path))

    all_matches = (
        pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    )
    if pair_metric_parts:
        all_matches.attrs["pair_metrics"] = pd.concat(
            pair_metric_parts, ignore_index=True
        )
    evaluated_all = (
        evaluate_inter_camera_match_quality(
            config_path=Path(config_path),
            dataset_root=Path(dataset_root),
            matches=all_matches,
            start_frame=int(start_frame),
            frame_count=frame_count,
        )
        if not all_matches.empty
        else all_matches.copy()
    )
    evaluated_pair_metrics = evaluated_all.attrs.get("pair_metrics")
    retained_count = 0
    for cache_dir, raw in raw_by_cache:
        if raw.empty:
            evaluated = raw.copy()
            cache_camera1 = cache_dir.parent.name.split("__", 1)[0]
            cache_camera2 = cache_dir.parent.name.split("__", 1)[-1]
            relation = cache_dir.name
        else:
            cache_camera1 = str(
                raw.get("cache_camera1", raw["camera1"]).astype(str).iloc[0]
            )
            cache_camera2 = str(
                raw.get("cache_camera2", raw["camera2"]).astype(str).iloc[0]
            )
            relation = str(
                raw.get(
                    "pair_relation",
                    pd.Series(["same_index"] * len(raw), index=raw.index),
                ).astype(str).iloc[0]
            )
            evaluated_cache_camera1 = evaluated_all.get(
                "cache_camera1", evaluated_all["camera1"]
            ).astype(str)
            evaluated_cache_camera2 = evaluated_all.get(
                "cache_camera2", evaluated_all["camera2"]
            ).astype(str)
            evaluated_relation = evaluated_all.get(
                "pair_relation",
                pd.Series(
                    ["same_index"] * len(evaluated_all),
                    index=evaluated_all.index,
                ),
            ).astype(str)
            evaluated = evaluated_all.loc[
                (evaluated_cache_camera1 == cache_camera1)
                & (evaluated_cache_camera2 == cache_camera2)
                & (evaluated_relation == relation)
            ].copy()

        retained = (
            evaluated.loc[_retained_mask(evaluated)].copy()
            if not evaluated.empty
            else evaluated.copy()
        )
        retained.attrs.clear()
        _atomic_write_parquet(retained, cache_dir / "retained_matches.parquet")
        retained_count += int(len(retained))

        if isinstance(evaluated_pair_metrics, pd.DataFrame):
            quality_metrics = evaluated_pair_metrics
            if not quality_metrics.empty:
                first_values = quality_metrics.get(
                    "cache_camera1", quality_metrics.get("camera1")
                )
                second_values = quality_metrics.get(
                    "cache_camera2", quality_metrics.get("camera2")
                )
                relation_values = quality_metrics.get("pair_relation")
                mask = pd.Series(True, index=quality_metrics.index)
                if first_values is not None:
                    mask &= first_values.astype(str) == cache_camera1
                if second_values is not None:
                    mask &= second_values.astype(str) == cache_camera2
                if relation_values is not None:
                    mask &= relation_values.astype(str) == relation
                quality_metrics = quality_metrics.loc[mask]
            _atomic_write_csv(
                quality_metrics.copy(), cache_dir / "pair_metrics_quality.csv"
            )

        if export_visualizations and not evaluated.empty:
            _export_inter_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=Path(dataset_root),
                config_path=Path(config_path),
                first_camera=cache_camera1,
                second_camera=cache_camera2,
                evaluated=evaluated,
            )
    return {
        "processed_dirs": len(raw_by_cache),
        "raw_matches": int(len(all_matches)),
        "retained_matches": int(retained_count),
    }


def match_multicamera_dataset_stage(
    *,
    config_path: Path,
    dataset_root: Path,
    cameras: tuple[str, ...],
    semantic_cache_root: Path = Path("data_cache/semantic_masks"),
    match_cache_root: Path = Path("data_cache/matches"),
    mask_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    mask_backend: str = "mask2former",
    mask_confidence: float = 0.25,
    mask_image_size: int = 2048,
    mask_dilation_pixels: int = 5,
    start_frame: int = 0,
    frame_count: int | None = None,
    relations: tuple[str, ...] = ("same_index",),
    image_cache_mode: str = "preload",
    preflight_cache_root: Path | None = None,
    filter_executor=None,
    pending_filter_jobs: list[object] | None = None,
) -> Path:
    config = _config_for_matching_scope(load_config(config_path), "inter")
    assumptions = dict(getattr(config, "assumptions", {}) or {})
    cache_config = (
        _config_for_inter_camera_pair(config, str(cameras[0]), str(cameras[1]))
        if len(cameras) == 2
        else config
    )
    export_cache_visualizations = bool(
        assumptions.get("export_match_cache_visualizations", True)
    )
    cache_key = match_cache_key(
        backend=str(cache_config.matching.backend),
        image_size=int(cache_config.matching.deep_image_size),
        max_keypoints=int(getattr(cache_config.matching, "deep_max_keypoints", 2048)),
        score_threshold=float(cache_config.matching.deep_score_threshold),
        semantic_backend="mask2former",
        semantic_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
        semantic_image_size=mask_image_size,
        semantic_dilation_pixels=mask_dilation_pixels,
    )
    matches = run_staged_inter_camera_matching(
        config_path=config_path,
        dataset_root=dataset_root,
        cameras=cameras,
        semantic_cache_root=semantic_cache_root,
        start_frame=start_frame,
        frame_count=frame_count,
        mask_model=mask_model,
        mask_backend=mask_backend,
        mask_confidence=mask_confidence,
        mask_image_size=mask_image_size,
        mask_dilation_pixels=mask_dilation_pixels,
        relations=relations,
        viz_cache_root=Path(match_cache_root),
        viz_cache_key=cache_key,
        matching_scope="inter",
        image_cache_mode=image_cache_mode,
        preflight_cache_root=preflight_cache_root,
    )
    matches_for_storage = matches.copy()
    matches_for_storage.attrs.clear()
    pair_metrics = matches.attrs.get("pair_metrics")
    if filter_executor is None:
        evaluated_matches = evaluate_inter_camera_match_quality(
            config_path=config_path,
            dataset_root=dataset_root,
            matches=matches,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        if evaluated_matches.empty or "pose_quality_valid" not in evaluated_matches.columns:
            retained_inter_camera = evaluated_matches.iloc[0:0].copy()
        else:
            retained_inter_camera = evaluated_matches.loc[
                _retained_mask(evaluated_matches)
            ].copy()
        retained_inter_camera.attrs.clear()
        evaluated_pair_metrics = evaluated_matches.attrs.get("pair_metrics")
    else:
        evaluated_matches = matches.iloc[0:0].copy()
        retained_inter_camera = matches.iloc[0:0].copy()
        evaluated_pair_metrics = None

    first_pair_dir: Path | None = None
    if not matches_for_storage.empty:
        cache_camera1_series = matches_for_storage.get(
            "cache_camera1", matches_for_storage.get("camera1", pd.Series(dtype=str))
        ).astype(str)
        cache_camera2_series = matches_for_storage.get(
            "cache_camera2", matches_for_storage.get("camera2", pd.Series(dtype=str))
        ).astype(str)
        relation_series = matches_for_storage.get(
            "pair_relation", pd.Series(["same_index"] * len(matches_for_storage))
        ).astype(str)
        pair_groups = matches_for_storage.groupby(
            [cache_camera1_series, cache_camera2_series, relation_series],
            sort=False,
        )
    else:
        pair_groups = []
    seen_cache_dir: Path | None = None
    deferred_cache_dirs: list[Path] = []
    for (first_camera, second_camera, relation), pair_raw in pair_groups:
        cache_dir = inter_camera_cache_dir(
            Path(match_cache_root),
            cache_key=cache_key,
            first_camera=str(first_camera),
            second_camera=str(second_camera),
            relation=str(relation),
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        if first_pair_dir is None:
            first_pair_dir = cache_dir
        if seen_cache_dir is None:
            seen_cache_dir = cache_dir
        deferred_cache_dirs.append(cache_dir)
        pair_raw.to_parquet(
            cache_dir / "raw_matches.parquet", index=False
        )

        if filter_executor is not None:
            pair_pm = (
                pair_metrics.loc[
                    (
                        pair_metrics.get("cache_camera1", pair_metrics["camera1"]).astype(str)
                        == str(first_camera)
                    )
                    & (
                        pair_metrics.get("cache_camera2", pair_metrics["camera2"]).astype(str)
                        == str(second_camera)
                    )
                    & (
                        pair_metrics.get(
                            "pair_relation",
                            pd.Series(["same_index"] * len(pair_metrics)),
                        ).astype(str)
                        == str(relation)
                    )
                ].copy()
                if isinstance(pair_metrics, pd.DataFrame) and not pair_metrics.empty
                else None
            )
            write_match_cache_artifacts(
                cache_dir,
                matches=pair_raw,
                pair_metrics=pair_pm,
                manifest={
                    "pair_type": "inter_camera",
                    "camera1": str(first_camera),
                    "camera2": str(second_camera),
                    "relation": str(relation),
                    "start_frame": int(start_frame),
                    "frame_count": None if frame_count is None else int(frame_count),
                },
            )
            continue

        retained_cache_camera1 = retained_inter_camera.get(
            "cache_camera1", retained_inter_camera.get("camera1", pd.Series(dtype=str))
        ).astype(str)
        retained_cache_camera2 = retained_inter_camera.get(
            "cache_camera2", retained_inter_camera.get("camera2", pd.Series(dtype=str))
        ).astype(str)
        pair_retained = retained_inter_camera.loc[
            (retained_cache_camera1 == str(first_camera))
            & (retained_cache_camera2 == str(second_camera))
            & (retained_inter_camera.get("pair_relation", pd.Series(["same_index"] * len(retained_inter_camera))).astype(str) == str(relation))
        ].copy()
        pair_retained.to_parquet(
            cache_dir / "retained_matches.parquet", index=False
        )

        pair_pm = (
            pair_metrics.loc[
                (
                    pair_metrics.get("cache_camera1", pair_metrics["camera1"]).astype(str)
                    == str(first_camera)
                )
                & (
                    pair_metrics.get("cache_camera2", pair_metrics["camera2"]).astype(str)
                    == str(second_camera)
                )
                & (
                    pair_metrics.get("pair_relation", pd.Series(["same_index"] * len(pair_metrics))).astype(str)
                    == str(relation)
                )
            ].copy()
            if isinstance(pair_metrics, pd.DataFrame) and not pair_metrics.empty
            else None
        )
        pair_epm = (
            evaluated_pair_metrics.loc[
                (
                    evaluated_pair_metrics.get(
                        "cache_camera1", evaluated_pair_metrics["camera1"]
                    ).astype(str)
                    == str(first_camera)
                )
                & (
                    evaluated_pair_metrics.get(
                        "cache_camera2", evaluated_pair_metrics["camera2"]
                    ).astype(str)
                    == str(second_camera)
                )
                & (
                    evaluated_pair_metrics.get("pair_relation", pd.Series(["same_index"] * len(evaluated_pair_metrics))).astype(str)
                    == str(relation)
                )
            ].copy()
            if isinstance(evaluated_pair_metrics, pd.DataFrame) and not evaluated_pair_metrics.empty
            else None
        )
        if isinstance(pair_epm, pd.DataFrame):
            pair_epm.to_csv(cache_dir / "pair_metrics_quality.csv", index=False)

        write_match_cache_artifacts(
            cache_dir,
            matches=pair_raw,
            pair_metrics=pair_pm,
            manifest={
                "pair_type": "inter_camera",
                "camera1": first_camera,
                "camera2": second_camera,
                "relation": relation,
                "start_frame": int(start_frame),
                "frame_count": None if frame_count is None else int(frame_count),
            },
        )

        if export_cache_visualizations:
            evaluated_cache_camera1 = evaluated_matches.get(
                "cache_camera1", evaluated_matches.get("camera1", pd.Series(dtype=str))
            ).astype(str)
            evaluated_cache_camera2 = evaluated_matches.get(
                "cache_camera2", evaluated_matches.get("camera2", pd.Series(dtype=str))
            ).astype(str)
            evaluated_relation = evaluated_matches.get(
                "pair_relation", pd.Series(["same_index"] * len(evaluated_matches))
            ).astype(str)
            pair_evaluated = evaluated_matches.loc[
                (evaluated_cache_camera1 == str(first_camera))
                & (evaluated_cache_camera2 == str(second_camera))
                & (evaluated_relation == str(relation))
            ].copy()
            _export_inter_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=dataset_root,
                config_path=config_path,
                first_camera=str(first_camera),
                second_camera=str(second_camera),
                evaluated=pair_evaluated,
            )

    if filter_executor is not None and deferred_cache_dirs:
        future = filter_executor.submit(
            _filter_inter_cache_group_worker,
            Path(config_path),
            Path(dataset_root),
            tuple(deferred_cache_dirs),
            int(start_frame),
            frame_count,
            bool(export_cache_visualizations),
        )
        if pending_filter_jobs is not None:
            pending_filter_jobs.append(future)

    return first_pair_dir or seen_cache_dir or Path(match_cache_root)


def run_persistent_matching_stage(
    *,
    config_path: Path,
    dataset_root: Path,
    cameras: tuple[str, ...],
    semantic_cache_root: Path,
    match_cache_root: Path,
    intra_baselines_m: tuple[float, ...] = (2.0, 4.0, 6.0),
    inter_pairs: tuple[str, ...] = (),
    inter_relations: tuple[str, ...] = ("same_index",),
    inter_relations_by_pair: dict[str, list[str]] | None = None,
    mask_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    mask_backend: str = "mask2former",
    mask_confidence: float = 0.25,
    mask_image_size: int = 2048,
    mask_dilation_pixels: int = 5,
    start_frame: int = 0,
    frame_count: int | None = None,
    image_cache_mode: str = "preload",
    preflight_cache_root: Path | None = None,
) -> Path:
    """Run every configured match relation in one process.

    The individual stage functions remain responsible for cache layout,
    quality evaluation, and visualization. Keeping their calls in this
    function means the LoMa backend and its module-level CPU/GPU feature
    caches survive across all intra and inter relations. The delivery CLI
    runs preflight and semantic generation in this same process before
    calling this function, so sampled preflight features are available here.
    """
    if Path(config_path).exists():
        config = load_config(config_path)
        filter_workers = int(getattr(config.matching, "filter_workers", 4))
        overlap_filtering = bool(getattr(config.matching, "overlap_filtering", True))
    else:
        # Lightweight callers and tests may provide stage doubles without a
        # materialized config. Keep the historical synchronous behavior.
        filter_workers = 1
        overlap_filtering = False
    pending_filter_jobs: list[object] = []

    def run_relations(filter_executor) -> None:
        if intra_baselines_m:
            for camera in cameras:
                match_dataset_stage(
                    config_path=config_path,
                    dataset_root=dataset_root,
                    camera=str(camera),
                    cache_root=semantic_cache_root,
                    match_cache_root=match_cache_root,
                    mask_model=mask_model,
                    mask_backend=mask_backend,
                    mask_confidence=mask_confidence,
                    mask_image_size=mask_image_size,
                    mask_dilation_pixels=mask_dilation_pixels,
                    start_frame=start_frame,
                    frame_count=frame_count,
                    target_baselines_m=tuple(float(value) for value in intra_baselines_m),
                    image_cache_mode=image_cache_mode,
                    preflight_cache_root=preflight_cache_root,
                    filter_executor=filter_executor,
                    pending_filter_jobs=pending_filter_jobs,
                )

        for pair in inter_pairs:
            pair_cameras = tuple(str(pair).split("__"))
            if len(pair_cameras) != 2 or not all(pair_cameras):
                raise ValueError(
                    f"inter pair must use '<first_camera>__<second_camera>': {pair}"
                )
            configured_relations = (
                inter_relations_by_pair.get(str(pair), list(inter_relations))
                if inter_relations_by_pair is not None
                else list(inter_relations)
            )
            relation_families = tuple(
                "temporal_overlap"
                if str(relation).startswith("temporal_overlap_")
                else str(relation)
                for relation in configured_relations
            )
            relation_families = tuple(dict.fromkeys(relation_families))
            match_multicamera_dataset_stage(
                config_path=config_path,
                dataset_root=dataset_root,
                cameras=(pair_cameras[0], pair_cameras[1]),
                semantic_cache_root=semantic_cache_root,
                match_cache_root=match_cache_root,
                mask_model=mask_model,
                mask_backend=mask_backend,
                mask_confidence=mask_confidence,
                mask_image_size=mask_image_size,
                mask_dilation_pixels=mask_dilation_pixels,
                start_frame=start_frame,
                frame_count=frame_count,
                relations=relation_families,
                image_cache_mode=image_cache_mode,
                preflight_cache_root=preflight_cache_root,
                filter_executor=filter_executor,
                pending_filter_jobs=pending_filter_jobs,
            )

    if overlap_filtering and filter_workers > 1:
        with ProcessPoolExecutor(
            max_workers=filter_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as filter_executor:
            run_relations(filter_executor)
            for future in pending_filter_jobs:
                future.result()
    else:
        run_relations(None)

    return Path(match_cache_root)


def _export_inter_camera_pair_visualizations(
    *,
    cache_dir: Path,
    dataset_root: Path,
    config_path: Path | None = None,
    first_camera: str,
    second_camera: str,
    evaluated: pd.DataFrame,
) -> None:
    from .visualization import draw_ransac_retained_reprojection_qc

    if evaluated.empty:
        return
    config = None
    if config_path is not None:
        config = load_config(config_path)
        config_dataset_root = Path(getattr(config, "dataset_root", dataset_root)).resolve()
        if Path(dataset_root).resolve() != config_dataset_root and hasattr(config, "model_copy"):
            config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    sample_ratio = float(
        getattr(
            getattr(config, "matching", None),
            "cache_visualization_pair_ratio",
            0.01,
        )
    )
    viz_root = cache_dir / "visualizations"
    retained_root = viz_root / "retained"
    qc_root = viz_root / "ransac_retained_qc"
    retained_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)

    cam_pair_label = f"{first_camera}__{second_camera}"
    frame_paths_by_camera: dict[str, dict[int, Path]] = {}
    for camera in sorted(
        set(evaluated.get("camera1", pd.Series(dtype=str)).astype(str))
        | set(evaluated.get("camera2", pd.Series(dtype=str)).astype(str))
    ):
        if not camera or camera == "nan":
            continue
        frames = (
            _frames_for_config(config, camera)
            if config is not None and hasattr(config, "dataset_root")
            else Dataset(dataset_root).frames(camera)
        )
        frame_paths_by_camera[camera] = {int(frame.index): frame.path for frame in frames}

    frame_pairs = (
        evaluated[["camera1", "camera2", "frame1", "frame2"]]
        .drop_duplicates()
        .assign(frame1=lambda df: df["frame1"].astype(int), frame2=lambda df: df["frame2"].astype(int))
        .itertuples(index=False, name=None)
    )
    for actual_first_camera, actual_second_camera, first_idx, second_idx in sorted(frame_pairs):
        if not _should_export_match_visualization(first_idx, sample_ratio):
            continue
        index_to_path1 = frame_paths_by_camera.get(str(actual_first_camera), {})
        index_to_path2 = frame_paths_by_camera.get(str(actual_second_camera), {})
        if int(first_idx) not in index_to_path1 or int(second_idx) not in index_to_path2:
            continue
        img1 = cv2.imread(str(index_to_path1[int(first_idx)]))
        img2 = cv2.imread(str(index_to_path2[int(second_idx)]))
        if img1 is None or img2 is None:
            continue

        frame_pair_eval = evaluated.loc[
            (evaluated["camera1"].astype(str) == str(actual_first_camera))
            & (evaluated["camera2"].astype(str) == str(actual_second_camera))
            & (evaluated["frame1"].astype(int) == int(first_idx))
            & (evaluated["frame2"].astype(int) == int(second_idx))
        ].copy()
        if frame_pair_eval.empty:
            continue

        stem = f"{cam_pair_label}_{int(first_idx):04d}_{int(second_idx):04d}"
        retained_mask = _retained_mask(frame_pair_eval).to_numpy(dtype=bool)
        ransac_inlier = (
            frame_pair_eval["image_ransac_inlier"].astype(bool).to_numpy()
            if "image_ransac_inlier" in frame_pair_eval.columns
            else np.ones((len(frame_pair_eval),), dtype=bool)
        )
        reprojection_rms = (
            frame_pair_eval["reprojection_rms_px"].astype(float).to_numpy()
            if "reprojection_rms_px" in frame_pair_eval.columns
            else np.full((len(frame_pair_eval),), np.nan)
        )
        ray_gap = (
            frame_pair_eval["ray_closest_distance_m"].astype(float).to_numpy()
            if "ray_closest_distance_m" in frame_pair_eval.columns
            else np.full((len(frame_pair_eval),), np.nan)
        )
        ransac_reprojection = reprojection_rms[ransac_inlier & np.isfinite(reprojection_rms)]
        ransac_ray = ray_gap[ransac_inlier & np.isfinite(ray_gap)]
        metadata = {
            "pose_degraded": bool(
                frame_pair_eval["pose_degraded"].iloc[0]
                if "pose_degraded" in frame_pair_eval.columns
                else False
            ),
            "reprojection_median_px": float(np.median(ransac_reprojection))
            if ransac_reprojection.size
            else np.nan,
            "reprojection_p75_px": float(np.percentile(ransac_reprojection, 75))
            if ransac_reprojection.size
            else np.nan,
            "ray_gap_median_m": float(np.median(ransac_ray))
            if ransac_ray.size
            else np.nan,
            "ray_gap_p75_m": float(np.percentile(ransac_ray, 75))
            if ransac_ray.size
            else np.nan,
            "reprojection_threshold_px": float(
                frame_pair_eval["reprojection_threshold_px"].iloc[0]
                if "reprojection_threshold_px" in frame_pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_median_px": float(
                frame_pair_eval["reprojection_threshold_median_px"].iloc[0]
                if "reprojection_threshold_median_px" in frame_pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_mad_px": float(
                frame_pair_eval["reprojection_threshold_mad_px"].iloc[0]
                if "reprojection_threshold_mad_px" in frame_pair_eval.columns
                else np.nan
            ),
            "reprojection_threshold_k": float(
                frame_pair_eval["reprojection_threshold_mad_k"].iloc[0]
                if "reprojection_threshold_mad_k" in frame_pair_eval.columns
                else 2.5
            ),
            "ray_gap_threshold_m": float(
                frame_pair_eval["ray_closest_distance_threshold_m"].iloc[0]
                if "ray_closest_distance_threshold_m" in frame_pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_median_m": float(
                frame_pair_eval["ray_closest_distance_median_m"].iloc[0]
                if "ray_closest_distance_median_m" in frame_pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_mad_m": float(
                frame_pair_eval["ray_closest_distance_mad_m"].iloc[0]
                if "ray_closest_distance_mad_m" in frame_pair_eval.columns
                else np.nan
            ),
            "ray_gap_threshold_k": float(
                frame_pair_eval["ray_closest_distance_mad_k"].iloc[0]
                if "ray_closest_distance_mad_k" in frame_pair_eval.columns
                else 2.5
            ),
        }
        metadata["reprojection_threshold_calculated_px"] = (
            float(
                metadata["reprojection_threshold_median_px"]
                + metadata["reprojection_threshold_k"]
                * 1.4826
                * metadata["reprojection_threshold_mad_px"]
            )
            if np.isfinite(metadata["reprojection_threshold_median_px"])
            and np.isfinite(metadata["reprojection_threshold_k"])
            and np.isfinite(metadata["reprojection_threshold_mad_px"])
            else np.nan
        )
        metadata["ray_gap_threshold_calculated_m"] = (
            float(
                metadata["ray_gap_threshold_median_m"]
                + metadata["ray_gap_threshold_k"]
                * 1.4826
                * metadata["ray_gap_threshold_mad_m"]
            )
            if np.isfinite(metadata["ray_gap_threshold_median_m"])
            and np.isfinite(metadata["ray_gap_threshold_k"])
            and np.isfinite(metadata["ray_gap_threshold_mad_m"])
            else np.nan
        )
        filter_steps: list[dict[str, int | str]] = []
        step_keep = np.ones((len(frame_pair_eval),), dtype=bool)
        for step_column in (
            "navigation_valid",
            "triangulation_valid_depth",
            "triangulation_distance_valid",
            "triangulation_angle_valid",
            "ray_closest_distance_valid",
            "reprojection_threshold_inlier",
            "image_ransac_inlier",
        ):
            step_condition = (
                frame_pair_eval[step_column].astype(bool).to_numpy()
                if step_column in frame_pair_eval.columns
                else np.ones((len(frame_pair_eval),), dtype=bool)
            )
            step_deleted = step_keep & ~step_condition
            step_keep = step_keep & step_condition
            filter_steps.append(
                {
                    "stage": step_column,
                    "deleted_this_step": int(step_deleted.sum()),
                    "remaining_after": int(step_keep.sum()),
                }
            )
        metadata["filter_steps"] = filter_steps
        if retained_mask.any():
            draw_ransac_retained_reprojection_qc(
                img1,
                img2,
                frame_pair_eval[["x1", "y1"]].to_numpy(),
                frame_pair_eval[["x2", "y2"]].to_numpy(),
                ransac_inlier=ransac_inlier,
                retained_mask=retained_mask,
                reprojected_first=frame_pair_eval[["reprojected1_x", "reprojected1_y"]].to_numpy(),
                reprojected_second=frame_pair_eval[["reprojected2_x", "reprojected2_y"]].to_numpy(),
                metadata=metadata,
                path=retained_root / f"{stem}_retained_matches.jpg",
                include_rejected_ransac=False,
                max_per_row=max(1, len(frame_pair_eval)),
                reprojection_layout="side_by_side",
                reprojection_points="retained",
                retained_match_color_mode="index",
                text_scale_multiplier=1.5,
            )
        draw_ransac_retained_reprojection_qc(
            img1,
            img2,
            frame_pair_eval[["x1", "y1"]].to_numpy(),
            frame_pair_eval[["x2", "y2"]].to_numpy(),
            ransac_inlier=ransac_inlier,
            retained_mask=retained_mask,
            reprojected_first=frame_pair_eval[["reprojected1_x", "reprojected1_y"]].to_numpy(),
            reprojected_second=frame_pair_eval[["reprojected2_x", "reprojected2_y"]].to_numpy(),
            metadata=metadata,
            path=qc_root / f"{stem}_ransac_retained_qc.jpg",
            include_rejected_ransac=True,
            reprojection_layout="side_by_side",
            reprojection_points="ransac_inliers",
            retained_match_color_mode="green",
            text_scale_multiplier=1.5,
        )


def _refresh_summary() -> dict[str, int]:
    return {
        "processed_dirs": 0,
        "raw_matches": 0,
        "retained_matches": 0,
        "skipped_dirs": 0,
    }


def _feature_cache_metrics(matches: pd.DataFrame) -> dict[str, int]:
    """Return stable per-pair LoMa feature-cache counters for pair metrics."""
    return {
        key: int(matches.attrs.get(key, 0) or 0)
        for key in (
            "feature_cache_hit_count",
            "feature_cache_miss_count",
            "feature_cache_gpu_hit_count",
            "feature_cache_cpu_hit_count",
        )
    }


def _remove_visualizations(cache_dir: Path) -> None:
    viz_dir = Path(cache_dir) / "visualizations"
    if viz_dir.exists():
        shutil.rmtree(viz_dir)


def _label_from_intra_cache_dir(cache_dir: Path) -> str:
    name = Path(cache_dir).name
    if name.startswith("baselines_"):
        return "baseline_" + name.removeprefix("baselines_")
    if name.startswith("offsets_"):
        return "offset_" + name.removeprefix("offsets_")
    return name


def refresh_intra_match_cache(
    *,
    config_path: Path,
    dataset_root: Path,
    match_cache_root: Path,
    regenerate_visualizations: bool = True,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> dict[str, int]:
    summary = _refresh_summary()
    intra_root = Path(match_cache_root) / "intra_camera"
    if not intra_root.exists():
        return summary
    for raw_path in sorted(intra_root.glob("*/*/raw_matches.parquet")):
        cache_dir = raw_path.parent
        raw = pd.read_parquet(raw_path)
        raw.attrs.clear()
        summary["processed_dirs"] += 1
        summary["raw_matches"] += int(len(raw))
        if raw.empty:
            raw.iloc[0:0].to_parquet(cache_dir / "retained_matches.parquet", index=False)
            summary["skipped_dirs"] += 1
            continue
        camera = (
            str(raw["camera"].dropna().iloc[0])
            if "camera" in raw.columns and raw["camera"].notna().any()
            else str(cache_dir.parent.name)
        )
        evaluated = evaluate_intra_camera_match_quality(
            config_path=config_path,
            dataset_root=dataset_root,
            matches=raw,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        retained = evaluated.loc[_retained_mask(evaluated)].copy()
        retained.attrs.clear()
        retained.to_parquet(cache_dir / "retained_matches.parquet", index=False)
        summary["retained_matches"] += int(len(retained))
        pair_metrics = evaluated.attrs.get("pair_metrics")
        if isinstance(pair_metrics, pd.DataFrame):
            pair_metrics.to_csv(cache_dir / "pair_metrics_quality.csv", index=False)
        if regenerate_visualizations:
            _remove_visualizations(cache_dir)
            _export_intra_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=dataset_root,
                config_path=config_path,
                camera=camera,
                evaluated=evaluated,
                label=_label_from_intra_cache_dir(cache_dir),
            )
    return summary


def refresh_inter_match_cache(
    *,
    config_path: Path,
    dataset_root: Path,
    match_cache_root: Path,
    regenerate_visualizations: bool = True,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> dict[str, int]:
    summary = _refresh_summary()
    inter_root = Path(match_cache_root) / "inter_camera"
    if not inter_root.exists():
        return summary
    for raw_path in sorted(inter_root.glob("*/*/raw_matches.parquet")):
        cache_dir = raw_path.parent
        raw = pd.read_parquet(raw_path)
        raw.attrs.clear()
        summary["processed_dirs"] += 1
        summary["raw_matches"] += int(len(raw))
        if raw.empty:
            raw.iloc[0:0].to_parquet(cache_dir / "retained_matches.parquet", index=False)
            summary["skipped_dirs"] += 1
            continue
        evaluated = evaluate_inter_camera_match_quality(
            config_path=config_path,
            dataset_root=dataset_root,
            matches=raw,
            start_frame=start_frame,
            frame_count=frame_count,
        )
        retained = evaluated.loc[_retained_mask(evaluated)].copy()
        retained.attrs.clear()
        retained.to_parquet(cache_dir / "retained_matches.parquet", index=False)
        summary["retained_matches"] += int(len(retained))
        pair_metrics = evaluated.attrs.get("pair_metrics")
        if isinstance(pair_metrics, pd.DataFrame):
            pair_metrics.to_csv(cache_dir / "pair_metrics_quality.csv", index=False)
        if regenerate_visualizations:
            pair_name = cache_dir.parent.name
            if "__" in pair_name:
                cache_camera1, cache_camera2 = pair_name.split("__", 1)
            else:
                cache_camera1 = str(raw.get("cache_camera1", raw["camera1"]).iloc[0])
                cache_camera2 = str(raw.get("cache_camera2", raw["camera2"]).iloc[0])
            _remove_visualizations(cache_dir)
            _export_inter_camera_pair_visualizations(
                cache_dir=cache_dir,
                dataset_root=dataset_root,
                config_path=config_path,
                first_camera=cache_camera1,
                second_camera=cache_camera2,
                evaluated=evaluated,
            )
    return summary


def refresh_match_cache(
    *,
    config_path: Path,
    dataset_root: Path,
    match_cache_root: Path,
    scope: str = "all",
    regenerate_visualizations: bool = True,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> dict[str, dict[str, int]]:
    if scope not in {"all", "intra", "inter"}:
        raise ValueError("scope must be one of: all, intra, inter")
    result: dict[str, dict[str, int]] = {}
    if scope in {"all", "intra"}:
        result["intra"] = refresh_intra_match_cache(
            config_path=config_path,
            dataset_root=dataset_root,
            match_cache_root=match_cache_root,
            regenerate_visualizations=regenerate_visualizations,
            start_frame=start_frame,
            frame_count=frame_count,
        )
    if scope in {"all", "inter"}:
        result["inter"] = refresh_inter_match_cache(
            config_path=config_path,
            dataset_root=dataset_root,
            match_cache_root=match_cache_root,
            regenerate_visualizations=regenerate_visualizations,
            start_frame=start_frame,
            frame_count=frame_count,
        )
    return result
