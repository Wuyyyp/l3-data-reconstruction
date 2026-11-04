from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2
import yaml

from .image_store import ImageStore, image_store_options


def make_preflight_pair_key(
    *,
    pair_type: str,
    camera1: str,
    camera2: str,
    relation: str,
    frame1: int,
    frame2: int,
) -> str:
    """Return a stable identity for one image pair before semantic filtering."""
    return "|".join(
        (
            str(pair_type),
            str(camera1),
            str(camera2),
            str(relation),
            str(int(frame1)),
            str(int(frame2)),
        )
    )


def _allocation_counts(
    group_sizes: dict[str, int],
    *,
    pairs_per_group: int,
    max_total_pairs: int | None,
) -> dict[str, int]:
    requested = {
        key: min(int(size), max(0, int(pairs_per_group)))
        for key, size in group_sizes.items()
    }
    if max_total_pairs is None or sum(requested.values()) <= int(max_total_pairs):
        return requested
    cap = max(0, int(max_total_pairs))
    if cap == 0 or not requested:
        return {key: 0 for key in requested}
    total_requested = float(sum(requested.values()))
    raw = {
        key: cap * value / total_requested for key, value in requested.items()
    }
    counts = {key: min(requested[key], int(np.floor(raw[key]))) for key in requested}
    remaining = cap - sum(counts.values())
    order = sorted(
        requested,
        key=lambda key: (raw[key] - np.floor(raw[key]), key),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        if counts[key] < requested[key]:
            counts[key] += 1
            remaining -= 1
    return counts


def _stratified_positions(size: int, count: int) -> np.ndarray:
    if count <= 0 or size <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= size:
        return np.arange(size, dtype=np.int64)
    positions = np.linspace(0, size - 1, count, dtype=np.float64)
    return np.unique(np.rint(positions).astype(np.int64))


def sample_preflight_pairs(
    records: pd.DataFrame,
    *,
    pairs_per_group: int = 20,
    max_total_pairs: int | None = 800,
    seed: int = 20260704,
) -> pd.DataFrame:
    """Select temporally distributed image pairs from a pair plan.

    ``records`` must contain ``group_key`` and ``frame1``. Groups are sampled
    independently, so a cap cannot cause one camera relation to consume all
    preflight budget. The seed is used for a stable tie-break order only; the
    temporal coverage itself is deterministic.
    """
    if records.empty:
        return records.copy()
    if "group_key" not in records.columns or "frame1" not in records.columns:
        raise KeyError("preflight pair records require group_key and frame1 columns")
    if int(pairs_per_group) < 1:
        raise ValueError("pairs_per_group must be positive")

    result = records.copy()
    result["frame1"] = result["frame1"].astype(int)
    sort_columns = [column for column in ("frame1", "frame2") if column in result]
    sort_columns.append("group_key")
    result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    grouped = {
        str(key): group.copy()
        for key, group in result.groupby("group_key", sort=False)
    }
    allocations = _allocation_counts(
        {key: len(group) for key, group in grouped.items()},
        pairs_per_group=int(pairs_per_group),
        max_total_pairs=max_total_pairs,
    )
    rng = np.random.default_rng(int(seed))
    selected: list[pd.DataFrame] = []
    for key, group in grouped.items():
        count = allocations.get(key, 0)
        positions = _stratified_positions(len(group), count)
        if len(positions) < count:
            remaining = np.setdiff1d(np.arange(len(group)), positions, assume_unique=True)
            extra = rng.choice(remaining, size=count - len(positions), replace=False)
            positions = np.sort(np.concatenate((positions, extra)))
        selected.append(group.iloc[positions])
    if not selected:
        return result.iloc[0:0].copy()
    sampled = pd.concat(selected, ignore_index=True)
    sampled["preflight_sampled"] = True
    return sampled.sort_values(
        [column for column in ("group_key", "frame1", "frame2") if column in sampled],
        kind="stable",
    ).reset_index(drop=True)


class PreflightMatchCache:
    """Small per-pair parquet cache shared by preflight and full matching."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if self.index_path.exists():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._index = dict(payload) if isinstance(payload, dict) else {}
        else:
            self._index: dict[str, dict[str, Any]] = {}

    @property
    def has_entries(self) -> bool:
        """Whether this cache contains at least one reusable pair."""
        return bool(self._index)

    @staticmethod
    def _file_name(key: str) -> str:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return f"pair_{digest}.parquet"

    def _flush(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def put(self, key: str, matches: pd.DataFrame, *, metadata: dict[str, Any] | None = None) -> None:
        filename = self._file_name(str(key))
        path = self.root / filename
        matches = matches.copy()
        matches.attrs.clear()
        matches.to_parquet(path, index=False)
        self._index[str(key)] = {
            "path": filename,
            "metadata": dict(metadata or {}),
            "row_count": int(len(matches)),
        }
        self._flush()

    def get(self, key: str) -> pd.DataFrame | None:
        item = self._index.get(str(key))
        if not isinstance(item, dict):
            return None
        path = self.root / str(item.get("path", ""))
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def metadata(self, key: str) -> dict[str, Any]:
        item = self._index.get(str(key), {})
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        return dict(metadata) if isinstance(metadata, dict) else {}

    def row_count(self, key: str) -> int:
        """Return the cached raw match count stored on the index entry."""
        item = self._index.get(str(key), {})
        if not isinstance(item, dict):
            return 0
        try:
            return int(item.get("row_count", 0))
        except (TypeError, ValueError):
            return 0

    def contains(self, key: str) -> bool:
        return self.get(key) is not None


def _metric_summary(values: pd.Series | None) -> dict[str, float | None]:
    if values is None:
        return {
            "mean": None,
            "median": None,
            "p75": None,
            "p95": None,
            "rmse": None,
        }
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"mean": None, "median": None, "p75": None, "p95": None, "rmse": None}
    array = numeric.to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "rmse": float(np.sqrt(np.mean(array * array))),
    }


def _summarize_subset(matches: pd.DataFrame) -> dict[str, Any]:
    count = int(len(matches))
    ransac = matches.get("image_ransac_inlier", pd.Series(False, index=matches.index)).astype(bool)
    valid_depth = matches.get(
        "triangulation_valid_depth", pd.Series(False, index=matches.index)
    ).astype(bool)
    reprojection = _metric_summary(matches.get("reprojection_rms_px"))
    ray_gap = _metric_summary(matches.get("ray_closest_distance_m"))
    if "quality_filter_valid" in matches.columns:
        retained = matches.loc[matches["quality_filter_valid"].astype(bool)]
    else:
        retained = matches
    retained_reprojection = _metric_summary(retained.get("reprojection_rms_px"))
    retained_ray_gap = _metric_summary(retained.get("ray_closest_distance_m"))
    summary = {
        "sample_count": count,
        "ransac_inlier_count": int(ransac.sum()),
        "ransac_inlier_ratio": float(ransac.mean()) if count else 0.0,
        "valid_depth_count": int(valid_depth.sum()),
        "valid_depth_ratio": float(valid_depth.mean()) if count else 0.0,
        "reprojection": reprojection,
        "ray_gap": ray_gap,
        "reprojection_mean_px": reprojection["mean"],
        "reprojection_median_px": reprojection["median"],
        "reprojection_p75_px": reprojection["p75"],
        "reprojection_p95_px": reprojection["p95"],
        "reprojection_rmse_px": reprojection["rmse"],
        "ray_gap_mean_m": ray_gap["mean"],
        "ray_gap_median_m": ray_gap["median"],
        "ray_gap_p75_m": ray_gap["p75"],
        "ray_gap_p95_m": ray_gap["p95"],
    }
    summary["retained_sample_count"] = int(len(retained))
    summary["retained_reprojection"] = retained_reprojection
    summary["retained_ray_gap"] = retained_ray_gap
    summary["retained_reprojection_mean_px"] = retained_reprojection["mean"]
    summary["retained_reprojection_median_px"] = retained_reprojection["median"]
    summary["retained_reprojection_p75_px"] = retained_reprojection["p75"]
    summary["retained_reprojection_p95_px"] = retained_reprojection["p95"]
    summary["retained_reprojection_rmse_px"] = retained_reprojection["rmse"]
    summary["retained_ray_gap_mean_m"] = retained_ray_gap["mean"]
    summary["retained_ray_gap_median_m"] = retained_ray_gap["median"]
    summary["retained_ray_gap_p75_m"] = retained_ray_gap["p75"]
    summary["retained_ray_gap_p95_m"] = retained_ray_gap["p95"]
    summary["retained_ray_gap_rmse_m"] = retained_ray_gap["rmse"]
    return summary


def summarize_preflight_matches(matches: pd.DataFrame) -> dict[str, Any]:
    """Summarize raw diagnostic rows while preserving intra/inter separation."""
    result: dict[str, Any] = {"overall": _summarize_subset(matches)}
    pair_types = matches["pair_type"].astype(str) if "pair_type" in matches else pd.Series(dtype=str)
    for pair_type, name in (("intra_camera", "intra_camera"), ("inter_camera", "inter_camera")):
        subset = matches.loc[pair_types == pair_type] if not pair_types.empty else matches.iloc[0:0]
        result[name] = _summarize_subset(subset)
    return result


def summarize_preflight_pair_metrics(matches: pd.DataFrame) -> pd.DataFrame:
    """Create one diagnostic row per sampled image pair."""
    if matches.empty or "pair_key" not in matches.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for pair_key, group in matches.groupby("pair_key", sort=False):
        item: dict[str, Any] = {
            "pair_key": str(pair_key),
            "evaluated_match_count": int(len(group)),
        }
        if "image_ransac_inlier" in group:
            ransac = group["image_ransac_inlier"].astype(bool)
            item["ransac_inlier_count"] = int(ransac.sum())
            item["ransac_inlier_ratio"] = float(ransac.mean())
        if "quality_filter_valid" in group.columns:
            retained = group.loc[group["quality_filter_valid"].astype(bool)]
        else:
            retained = group
        for column, prefix in (
            ("reprojection_rms_px", "reprojection"),
            ("ray_closest_distance_m", "ray_gap"),
        ):
            raw_values = _metric_summary(group.get(column))
            retained_values = _metric_summary(retained.get(column))
            item[f"raw_{prefix}_median"] = raw_values["median"]
            item[f"raw_{prefix}_p75"] = raw_values["p75"]
            item[f"raw_{prefix}_p95"] = raw_values["p95"]
            item[f"raw_{prefix}_rmse"] = raw_values["rmse"]
            item[f"{prefix}_median"] = retained_values["median"]
            item[f"{prefix}_p75"] = retained_values["p75"]
            item[f"{prefix}_p95"] = retained_values["p95"]
            item[f"{prefix}_rmse"] = retained_values["rmse"]
        if "triangulation_valid_depth" in group:
            depth = group["triangulation_valid_depth"].astype(bool)
            item["valid_depth_ratio"] = float(depth.mean())
        if "pose_degraded" in group:
            item["pose_degraded"] = bool(group["pose_degraded"].astype(bool).iloc[0])
        if "quality_filter_valid" in group:
            item["retained_matches"] = int(group["quality_filter_valid"].astype(bool).sum())
        rows.append(item)
    return pd.DataFrame.from_records(rows)


def _relation_target_baseline(relation: str, assumptions: dict[str, Any]) -> float:
    label = str(relation)
    if label.startswith("temporal_overlap_"):
        value = label.removeprefix("temporal_overlap_").removesuffix("m")
        return float(value.replace("p", "."))
    values = assumptions.get("inter_camera_temporal_overlap_target_baselines_m", [30.0])
    if isinstance(values, (list, tuple)) and values:
        return float(values[0])
    return float(values)


def _delivery_payload(path: Path) -> tuple[dict[str, Any], Path]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("delivery config must contain a mapping")
    delivery = payload.get("delivery", payload)
    if not isinstance(delivery, dict):
        raise ValueError("delivery config field must contain a mapping")
    base_config = delivery.get("base_config")
    if base_config is None:
        raise ValueError("delivery config requires base_config")
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        cwd_candidate = (Path.cwd() / base_path).resolve()
        config_candidate = (Path(path).resolve().parent / base_path).resolve()
        base_path = cwd_candidate if cwd_candidate.exists() else config_candidate
    return dict(delivery), base_path


def _resolve_preflight_pose_path(
    delivery: dict[str, Any], canonical_root: Path, root: Path
) -> Path:
    """Resolve the pose input used by the preflight stage.

    Preflight runs before dynamic compensation, so its default pose source is
    the canonical standardized pkl rather than the later compensated pkl.
    """
    configured = delivery.get("preflight_pose_pkl")
    if configured not in (None, ""):
        path = _resolve_delivery_path(root, configured)
        assert path is not None
        return path
    standard = canonical_root / "result_interpolation_standard.pkl"
    if standard.exists():
        return standard
    raise FileNotFoundError(
        "preflight requires result_interpolation_standard.pkl in canonical_root "
        "or an explicit delivery.preflight_pose_pkl"
    )


def _resolve_preflight_pose_field(delivery: dict[str, Any]) -> str:
    """Return the ego field used before exposure-offset compensation exists."""
    value = delivery.get(
        "preflight_pose_field", "ego2global_transformation_matrix"
    )
    return str(value)


def _run_preflight_match_backend(
    first_image,
    second_image,
    *,
    calibration,
    matching_config,
    first_mask,
    second_mask,
    first_frame,
    second_frame,
    first_original_shape=None,
    second_original_shape=None,
):
    """Run one sampled pair with the shared LoMa feature-cache context."""
    from .match_backends import run_match_backend

    return run_match_backend(
        first_image,
        second_image,
        calibration,
        matching_config,
        first_semantic_mask=first_mask,
        second_semantic_mask=second_mask,
        first_image_key=str(first_frame.path),
        second_image_key=str(second_frame.path),
        first_original_shape=first_original_shape,
        second_original_shape=second_original_shape,
    )


def _sampled_semantic_items(
    sampled: pd.DataFrame,
    frames_by_camera: dict[str, dict[int, Any]],
) -> list[tuple[Path, str, int]]:
    """Return unique semantic-cache items needed by sampled image pairs."""
    items: list[tuple[Path, str, int]] = []
    seen: set[tuple[str, int]] = set()
    for row in sampled.to_dict(orient="records"):
        for camera_key, frame_key in (
            ("match_camera1", "frame1"),
            ("match_camera2", "frame2"),
        ):
            camera = str(row[camera_key])
            frame_index = int(row[frame_key])
            key = (camera, frame_index)
            if key in seen:
                continue
            frame = frames_by_camera[camera][frame_index]
            items.append((Path(frame.path), camera, int(frame.timestamp_ns)))
            seen.add(key)
    return items


def _prepare_preflight_semantic_masks(
    *,
    sampled: pd.DataFrame,
    frames_by_camera: dict[str, dict[int, Any]],
    base_config: Any,
    cache_root: Path,
    image_store: ImageStore | None = None,
) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    """Generate/reuse semantic masks for sampled images and load them for LoMa."""
    from .semantic_masks import (
        DEFAULT_DYNAMIC_CLASSES,
        DynamicMaskConfig,
        MaskCache,
        create_predictor,
        generate_cached_masks_batch,
        load_dynamic_mask,
        write_cached_preview,
    )

    assumptions = dict(getattr(base_config, "assumptions", {}) or {})
    semantic = base_config.semantic_segmentation
    dynamic_classes = frozenset(
        str(item)
        for item in assumptions.get("semantic_mask_classes", DEFAULT_DYNAMIC_CLASSES)
    )
    semantic_config = DynamicMaskConfig(
        cache_root=Path(cache_root),
        backend=str(assumptions.get("semantic_backend_default", "mask2former")),
        model_name=str(
            assumptions.get(
                "semantic_mask2former_model",
                "facebook/mask2former-swin-large-mapillary-vistas-semantic",
            )
        ),
        confidence_threshold=float(semantic.confidence),
        processor_image_size=int(semantic.processor_image_size),
        inference_dtype=str(getattr(semantic, "inference_dtype", "fp16")),
        use_fast_processor=bool(getattr(semantic, "use_fast_processor", True)),
        dilation_pixels=int(semantic.dilation_pixels),
        batch_size=int(semantic.batch_size),
        write_workers=int(semantic.write_workers),
        preview_ratio=float(getattr(semantic, "preview_ratio", 0.1)),
        dynamic_classes=dynamic_classes,
        device=str(getattr(base_config.matching, "model_device", "cpu")),
        source_validation="none",
        image_cache_mode="preload",
        image_cache_enabled=bool(
            getattr(getattr(base_config, "image_cache", None), "enabled", True)
        ),
        image_cache_max_long_side=int(
            getattr(getattr(base_config, "image_cache", None), "max_long_side", 1600)
        ),
    )
    cache = MaskCache(semantic_config)
    items = _sampled_semantic_items(sampled, frames_by_camera)
    from .semantic_pipeline import preview_indices_for_count

    preview_keys: set[tuple[str, int]] = set()
    for camera, frames_by_index in frames_by_camera.items():
        ordered_frames = list(frames_by_index.values())
        for index in preview_indices_for_count(
            len(ordered_frames),
            semantic_config.preview_ratio,
        ):
            preview_keys.add((str(camera), int(ordered_frames[index].timestamp_ns)))
    cached_results: dict[tuple[str, int], Any] = {}
    pending: list[tuple[Path, str, int]] = []
    manifest_rows: list[dict[str, Any]] = []
    for image_path, camera, timestamp_ns in items:
        write_preview = (camera, timestamp_ns) in preview_keys
        cached = cache.load_if_valid(
            image_path,
            camera,
            timestamp_ns,
            semantic_config.model_name,
            require_preview=write_preview,
        )
        if cached is None:
            pending.append((image_path, camera, timestamp_ns))
        else:
            if write_preview and not cached.preview_path.exists():
                image = (
                    image_store.get(image_path)
                    if image_store is not None
                    else cv2.imread(str(image_path))
                )
                if image is None:
                    raise ValueError(f"could not read semantic source image: {image_path}")
                write_cached_preview(cached, image)
            cached_results[(camera, timestamp_ns)] = cached
            manifest_rows.append(
                {
                    "camera": camera,
                    "timestamp_ns": timestamp_ns,
                    "image_path": str(image_path),
                    "cache_hit": True,
                    "class_mask_path": str(cached.class_mask_path),
                }
            )
    if pending:
        predictor = create_predictor(semantic_config)
        generated: list[Any] = []
        batch_size = max(1, int(semantic_config.batch_size))
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            generated.extend(
                generate_cached_masks_batch(
                    batch,
                    predictor,
                    cache,
                    force=True,
                    write_preview_flags=[
                        (camera, timestamp_ns) in preview_keys
                        for _image_path, camera, timestamp_ns in batch
                    ],
                    write_workers=semantic_config.write_workers,
                    image_store=image_store,
                )
            )
        for (image_path, camera, timestamp_ns), result in zip(
            pending, generated, strict=True
        ):
            cached_results[(camera, timestamp_ns)] = result
            manifest_rows.append(
                {
                    "camera": camera,
                    "timestamp_ns": timestamp_ns,
                    "image_path": str(image_path),
                    "cache_hit": False,
                    "class_mask_path": str(result.class_mask_path),
                }
            )
    masks: dict[tuple[str, int], np.ndarray] = {}
    for key, result in cached_results.items():
        masks[key] = load_dynamic_mask(
            result,
            dynamic_classes=semantic_config.dynamic_classes,
            dilation_pixels=semantic_config.dilation_pixels,
        )
    return masks, pd.DataFrame.from_records(manifest_rows)


def build_preflight_pair_plan(
    *,
    base_config_path: Path,
    dataset_root: Path,
    cameras: list[str],
    intra_baselines_m: list[float],
    inter_pairs: list[str],
    inter_relations_by_pair: dict[str, list[str]],
    start_frame: int = 0,
    frame_count: int | None = None,
    preflight_pose_path: Path | None = None,
    preflight_pose_field: str | None = None,
) -> pd.DataFrame:
    """Build pair identities using the same geometry rules as staged matching."""
    from .config import load_config
    from . import staged_pipeline as staged

    config = load_config(Path(base_config_path))
    if Path(dataset_root).resolve() != Path(config.dataset_root).resolve():
        config = config.model_copy(update={"dataset_root": Path(dataset_root).resolve()})
    assumptions = dict(config.assumptions or {})
    if preflight_pose_path is not None:
        assumptions["result_interpolation_pkl_path"] = str(
            Path(preflight_pose_path).resolve()
        )
    if preflight_pose_field is not None:
        assumptions["result_interpolation_ego_field"] = str(preflight_pose_field)
    if preflight_pose_path is not None or preflight_pose_field is not None:
        config = config.model_copy(update={"assumptions": assumptions})
    navigation = staged._navigation_for_config(config)
    navigation_offset_ns = staged.navigation_time_offset_ns(assumptions)
    selected_by_camera: dict[str, list[Any]] = {}
    centers_by_camera: dict[str, dict[int, np.ndarray]] = {}
    view_dirs_by_camera: dict[str, dict[int, np.ndarray]] = {}
    calibrations: dict[str, Any] = {}
    for camera in cameras:
        frames = staged._frames_for_config(config, camera)
        begin, end = staged._resolve_window(
            len(frames), start_frame=start_frame, frame_count=frame_count
        )
        selected = frames[begin:end]
        selected_by_camera[str(camera)] = selected
        calibration = staged._calibration_for_config(config, str(camera))
        calibrations[str(camera)] = calibration
        centers_by_camera[str(camera)] = staged._camera_centers_by_frame_index(
            frames=selected,
            calibration=calibration,
            navigation=navigation,
            navigation_offset_ns=navigation_offset_ns,
        )
        view_dirs_by_camera[str(camera)] = staged._camera_view_dirs_by_frame_index(
            frames=selected,
            calibration=calibration,
            navigation=navigation,
            navigation_offset_ns=navigation_offset_ns,
        )

    rows: list[dict[str, Any]] = []
    for camera in cameras:
        frame_indices = [int(frame.index) for frame in selected_by_camera[camera]]
        for baseline in intra_baselines_m:
            pairs = staged._build_baseline_frame_pairs(
                frame_indices=frame_indices,
                camera_centers_by_index=centers_by_camera[camera],
                target_baseline_m=float(baseline),
            )
            relation = f"baseline_{float(baseline):g}m"
            for first, second, pair_baseline in pairs:
                key = make_preflight_pair_key(
                    pair_type="intra_camera",
                    camera1=camera,
                    camera2=camera,
                    relation=relation,
                    frame1=first,
                    frame2=second,
                )
                rows.append(
                    {
                        "pair_key": key,
                        "group_key": f"intra_camera|{camera}|{relation}",
                        "pair_type": "intra_camera",
                        "camera": camera,
                        "camera1": camera,
                        "camera2": camera,
                        "match_camera1": camera,
                        "match_camera2": camera,
                        "cache_camera1": camera,
                        "cache_camera2": camera,
                        "relation": relation,
                        "frame1": int(first),
                        "frame2": int(second),
                        "baseline_target_m": float(baseline),
                        "baseline_pair_m": float(pair_baseline),
                    }
                )

    for pair in inter_pairs:
        parts = str(pair).split("__", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid inter pair: {pair}")
        declared_first, declared_second = parts
        if declared_first not in selected_by_camera or declared_second not in selected_by_camera:
            continue
        relation_names = inter_relations_by_pair.get(str(pair), ["same_index"])
        first_frames = selected_by_camera[declared_first]
        second_by_index = {int(frame.index): frame for frame in selected_by_camera[declared_second]}
        if "same_index" in relation_names and staged._is_same_index_pair_allowed(
            assumptions, declared_first, declared_second
        ):
            match_first, match_second = staged._declared_same_index_pair(
                assumptions, declared_first, declared_second
            )
            for first in selected_by_camera[match_first]:
                second = {
                    int(frame.index): frame for frame in selected_by_camera[match_second]
                }.get(int(first.index))
                if second is None:
                    continue
                key = make_preflight_pair_key(
                    pair_type="inter_camera",
                    camera1=match_first,
                    camera2=match_second,
                    relation="same_index",
                    frame1=int(first.index),
                    frame2=int(second.index),
                )
                rows.append(
                    {
                        "pair_key": key,
                        "group_key": f"inter_camera|{declared_first}__{declared_second}|same_index",
                        "pair_type": "inter_camera",
                        "camera1": match_first,
                        "camera2": match_second,
                        "match_camera1": match_first,
                        "match_camera2": match_second,
                        "cache_camera1": declared_first,
                        "cache_camera2": declared_second,
                        "relation": "same_index",
                        "frame1": int(first.index),
                        "frame2": int(second.index),
                    }
                )
        for relation in relation_names:
            if not str(relation).startswith("temporal_overlap"):
                continue
            if not staged._is_temporal_overlap_direction_allowed(
                assumptions, declared_first, declared_second
            ):
                continue
            cache_first, cache_second = staged._declared_temporal_overlap_pair(
                assumptions, declared_first, declared_second
            )
            search_first, search_second = staged._temporal_overlap_search_direction_for_pair(
                assumptions, cache_first, cache_second
            )
            if search_first not in selected_by_camera or search_second not in selected_by_camera:
                continue
            target = _relation_target_baseline(str(relation), assumptions)
            pairs = staged._build_temporal_overlap_pairs(
                first_frames=selected_by_camera[search_first],
                second_frames=selected_by_camera[search_second],
                first_centers_by_index=centers_by_camera[search_first],
                second_centers_by_index=centers_by_camera[search_second],
                first_view_dirs_by_index=view_dirs_by_camera[search_first],
                second_view_dirs_by_index=view_dirs_by_camera[search_second],
                target_baseline_m=target,
                max_frame_gap=int(assumptions.get("inter_camera_temporal_overlap_max_frame_gap", 300)),
                min_opposite_view_dot=staged._temporal_overlap_view_dot_threshold_for_pair(
                    assumptions, cache_first, cache_second
                ),
            )
            relation_label = str(relation)
            if relation_label == "temporal_overlap":
                relation_label = staged._temporal_overlap_relation_label(target)
            for first, second in pairs:
                key = make_preflight_pair_key(
                    pair_type="inter_camera",
                    camera1=search_first,
                    camera2=search_second,
                    relation=relation_label,
                    frame1=first,
                    frame2=second,
                )
                rows.append(
                    {
                        "pair_key": key,
                        "group_key": f"inter_camera|{cache_first}__{cache_second}|{relation_label}",
                        "pair_type": "inter_camera",
                        "camera1": search_first,
                        "camera2": search_second,
                        "match_camera1": search_first,
                        "match_camera2": search_second,
                        "cache_camera1": cache_first,
                        "cache_camera2": cache_second,
                        "relation": relation_label,
                        "frame1": int(first),
                        "frame2": int(second),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows).drop_duplicates("pair_key").reset_index(drop=True)


def classify_preflight_quality(
    *,
    pair_metrics: pd.DataFrame,
    summary: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    settings = getattr(config, "preflight_diagnostic", config)
    fail_action = str(
        getattr(settings, "fail_action", "stop_before_dynamic_compensation")
    )
    failure_reasons: list[str] = []
    warnings: list[str] = []
    if pair_metrics.empty:
        failure_reasons.append("no_preflight_pairs")
    else:
        raw_counts = pd.to_numeric(pair_metrics.get("raw_match_count"), errors="coerce").fillna(0)
        if int((raw_counts >= int(settings.min_raw_matches_per_pair)).sum()) == 0:
            failure_reasons.append("no_pair_has_enough_raw_matches")
        valid_rows = pair_metrics.loc[
            raw_counts >= int(settings.min_raw_matches_per_pair)
        ]
        if int(settings.min_camera_graph_degree) > 0 and not valid_rows.empty:
            degrees: dict[str, int] = {}
            for row in valid_rows.to_dict(orient="records"):
                first = str(row.get("camera1", ""))
                second = str(row.get("camera2", ""))
                if first:
                    degrees[first] = degrees.get(first, 0) + 1
                if second and second != first:
                    degrees[second] = degrees.get(second, 0) + 1
            expected_cameras = [str(item) for item in getattr(config, "cameras", [])]
            missing_graph = [
                camera
                for camera in expected_cameras
                if degrees.get(camera, 0) < int(settings.min_camera_graph_degree)
            ]
            if missing_graph:
                failure_reasons.append(
                    "camera_graph_disconnected:" + ",".join(sorted(missing_graph))
                )
        if "reprojection_median" in pair_metrics.columns:
            max_bad_pair_fraction = float(
                getattr(settings, "max_bad_pair_fraction", 0.25)
            )
            min_bad_pairs_for_failure = int(
                getattr(settings, "min_bad_pairs_for_failure", 3)
            )
            min_failed_relation_fraction_for_failure = float(
                getattr(settings, "min_failed_relation_fraction_for_failure", 0.20)
            )
            group_columns = [
                column
                for column in ("group_key", "pair_type", "camera1", "camera2", "relation")
                if column in valid_rows.columns
            ]
            if not group_columns:
                valid_rows = valid_rows.assign(_preflight_group="all")
                group_columns = ["_preflight_group"]

            relation_group_count = int(
                valid_rows.groupby(group_columns, dropna=False).ngroups
            )

            def _classify_relation_outliers(
                column: str,
                threshold: float,
                failure_reason: str,
                warning_reason: str,
            ) -> None:
                if column not in valid_rows.columns:
                    return
                values = pd.to_numeric(valid_rows[column], errors="coerce")
                bad = values > float(threshold)
                if not bool(bad.any()):
                    return
                grouped = valid_rows.assign(_bad_pair=bad).groupby(
                    group_columns, dropna=False
                )
                failed_relation_count = 0
                for _, group in grouped:
                    bad_count = int(group["_bad_pair"].sum())
                    bad_fraction = bad_count / max(len(group), 1)
                    if (
                        bad_count >= min_bad_pairs_for_failure
                        and bad_fraction >= max_bad_pair_fraction
                    ):
                        failed_relation_count += 1
                failed_relation_fraction = failed_relation_count / max(
                    relation_group_count, 1
                )
                if failed_relation_fraction >= min_failed_relation_fraction_for_failure:
                    failure_reasons.append(failure_reason)
                else:
                    warnings.append(warning_reason)

            _classify_relation_outliers(
                "reprojection_median",
                float(settings.reprojection_median_fail_px),
                "relation_reprojection_median_exceeded",
                "relation_reprojection_median_outliers_within_tolerance",
            )
        for scope in ("intra_camera", "inter_camera"):
            scope_rows = pair_metrics.loc[pair_metrics["pair_type"] == scope]
            if scope_rows.empty:
                warnings.append(f"no_{scope}_preflight_pairs")
                continue
            valid = scope_rows.loc[
                pd.to_numeric(scope_rows["raw_match_count"], errors="coerce")
                >= int(settings.min_raw_matches_per_pair)
            ]
            minimum = int(
                settings.min_intra_valid_pairs
                if scope == "intra_camera"
                else settings.min_inter_valid_pairs
            )
            if len(valid) < minimum:
                failure_reasons.append(f"insufficient_{scope}_valid_pairs")
        overall = summary.get("overall", {})
        if float(overall.get("ransac_inlier_ratio", 0.0)) < float(settings.min_ransac_inlier_ratio):
            failure_reasons.append("low_overall_ransac_inlier_ratio")
        if float(overall.get("valid_depth_ratio", 0.0)) < float(settings.min_valid_depth_ratio):
            warnings.append("low_overall_valid_depth_ratio")
        median = overall.get("retained_reprojection_median_px")
        if median is not None and float(median) > float(settings.reprojection_median_fail_px):
            failure_reasons.append("high_overall_reprojection_median")
        elif median is not None and float(median) > float(settings.reprojection_median_warn_px):
            warnings.append("elevated_overall_reprojection_median")
    status = "FAIL" if failure_reasons else ("WARN" if warnings else "PASS")
    return {
        "overall_status": status,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "fail_action": fail_action,
        "recommended_action": (
            "stop_before_dynamic_compensation"
            if status == "FAIL" and fail_action != "report_only"
            else "continue_delivery"
        ),
    }


def _resolve_delivery_path(root: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _pair_metric_plan(plan: pd.DataFrame, cache: PreflightMatchCache) -> pd.DataFrame:
    rows = plan.copy()
    rows["raw_match_count"] = [
        cache.row_count(str(key)) for key in rows["pair_key"]
    ]
    rows["matching_source"] = [
        str(cache.metadata(str(key)).get("matching_source", "unknown"))
        for key in rows["pair_key"]
    ]
    return rows


def _split_intra_preflight_matches(
    matches: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    """Split intra-camera diagnostics because their evaluator is single-camera."""
    if matches.empty:
        return []
    return [
        (str(camera), subset.copy())
        for camera, subset in matches.groupby("match_camera1", sort=False)
    ]


def run_preflight_from_delivery_config(delivery_config_path: Path) -> dict[str, Any]:
    """Run the sampled preflight stage from a delivery configuration."""
    from .config import load_config
    from . import staged_pipeline as staged

    delivery, base_config_path = _delivery_payload(Path(delivery_config_path))
    root = Path.cwd().resolve()
    base_config_path = (
        base_config_path
        if base_config_path.exists()
        else (root / base_config_path).resolve()
    )
    base_config = load_config(base_config_path)
    canonical_root = _resolve_delivery_path(root, delivery.get("canonical_root"))
    if canonical_root is None:
        canonical_root = Path(base_config.dataset_root).resolve()
    preflight_pose_path = _resolve_preflight_pose_path(delivery, canonical_root, root)
    preflight_pose_field = _resolve_preflight_pose_field(delivery)
    base_assumptions = dict(base_config.assumptions or {})
    base_assumptions["result_interpolation_pkl_path"] = str(
        preflight_pose_path.resolve()
    )
    base_assumptions["result_interpolation_ego_field"] = preflight_pose_field
    base_config = base_config.model_copy(
        update={"dataset_root": canonical_root, "assumptions": base_assumptions}
    )

    preflight = base_config.preflight_diagnostic
    if not preflight.enabled:
        return {
            "overall_status": "DISABLED",
            "failure_reasons": [],
            "warnings": [],
            "fail_action": str(preflight.fail_action),
            "recommended_action": "continue_delivery",
        }

    cameras = [str(item) for item in delivery.get("cameras", base_config.cameras)]
    intra_baselines = [
        float(item)
        for item in delivery.get(
            "intra_baselines_m",
            base_config.assumptions.get("intra_camera_target_baselines_m", [2, 4, 6]),
        )
    ]
    inter_pairs = [str(item) for item in delivery.get("inter_pairs", [])]
    global_relations = [str(item) for item in delivery.get("inter_relations", ["same_index"])]
    raw_relation_mapping = delivery.get("inter_relations_by_pair", {})
    if not isinstance(raw_relation_mapping, dict):
        raise ValueError("inter_relations_by_pair must contain a mapping")
    relation_mapping = {
        pair: [str(item) for item in raw_relation_mapping.get(pair, global_relations)]
        for pair in inter_pairs
    }
    start_frame = int(delivery.get("start_frame", 0))
    frame_count = delivery.get("frame_count")
    frame_count = None if frame_count is None else int(frame_count)

    plan = build_preflight_pair_plan(
        base_config_path=base_config_path,
        dataset_root=canonical_root,
        cameras=cameras,
        intra_baselines_m=intra_baselines,
        inter_pairs=inter_pairs,
        inter_relations_by_pair=relation_mapping,
        start_frame=start_frame,
        frame_count=frame_count,
        preflight_pose_path=preflight_pose_path,
        preflight_pose_field=preflight_pose_field,
    )
    sampled = sample_preflight_pairs(
        plan,
        pairs_per_group=preflight.sample_pairs_per_relation,
        max_total_pairs=preflight.max_total_pairs,
        seed=preflight.random_seed,
    )
    run_root = _resolve_delivery_path(root, delivery.get("run_output_root"))
    output_root = _resolve_delivery_path(
        root,
        delivery.get("preflight_output_root")
        or (run_root / "preflight_quality" if run_root else "outputs/preflight_quality"),
    )
    cache_root = _resolve_delivery_path(
        root,
        delivery.get("preflight_match_cache_root")
        or (run_root / "preflight_match_cache" if run_root else "data_cache/preflight_matches"),
    )
    assert output_root is not None
    assert cache_root is not None
    output_root.mkdir(parents=True, exist_ok=True)
    cache = PreflightMatchCache(cache_root)

    frames_by_camera: dict[str, dict[int, Any]] = {}
    calibration_by_camera: dict[str, Any] = {}
    for camera in cameras:
        frames = staged._frames_for_config(base_config, camera)
        begin, end = staged._resolve_window(
            len(frames), start_frame=start_frame, frame_count=frame_count
        )
        frames_by_camera[camera] = {
            int(frame.index): frame for frame in frames[begin:end]
        }
        calibration_by_camera[camera] = staged._calibration_for_config(base_config, camera)

    semantic_masks: dict[tuple[str, int], np.ndarray] = {}
    semantic_manifest = pd.DataFrame()
    run_root = _resolve_delivery_path(root, delivery.get("run_output_root"))
    semantic_cache_root = _resolve_delivery_path(
        root,
        delivery.get("semantic_cache_root")
        or (run_root / "semantic_masks" if run_root else "data_cache/semantic_masks"),
    )
    assert semantic_cache_root is not None
    sampled_image_items = _sampled_semantic_items(sampled, frames_by_camera)
    preflight_image_store = ImageStore.from_paths(
        [item[0] for item in sampled_image_items],
        **image_store_options(base_config),
    )
    semantic_masks, semantic_manifest = _prepare_preflight_semantic_masks(
        sampled=sampled,
        frames_by_camera=frames_by_camera,
        base_config=base_config,
        cache_root=semantic_cache_root,
        image_store=preflight_image_store,
    )
    semantic_manifest.to_csv(output_root / "semantic_manifest.csv", index=False)

    match_records: list[pd.DataFrame] = []
    current_matching_sources: dict[str, str] = {}
    for row in sampled.to_dict(orient="records"):
        pair_key = str(row["pair_key"])
        matched = cache.get(pair_key)
        matching_source = "preflight_reused"
        if matched is None:
            camera1 = str(row["match_camera1"])
            camera2 = str(row["match_camera2"])
            first = frames_by_camera[camera1][int(row["frame1"])]
            second = frames_by_camera[camera2][int(row["frame2"])]
            image1 = preflight_image_store.get(first.path)
            image2 = preflight_image_store.get(second.path)
            if row["pair_type"] == "inter_camera":
                pair_config = staged._config_for_inter_camera_pair(
                    staged._config_for_matching_scope(base_config, "inter"),
                    camera1,
                    camera2,
                )
            else:
                pair_config = staged._config_for_matching_scope(base_config, "intra")
            first_mask = semantic_masks.get((camera1, int(first.timestamp_ns)))
            second_mask = semantic_masks.get((camera2, int(second.timestamp_ns)))
            matched = _run_preflight_match_backend(
                image1,
                image2,
                calibration=calibration_by_camera[camera1],
                matching_config=pair_config.matching,
                first_mask=first_mask,
                second_mask=second_mask,
                first_frame=first,
                second_frame=second,
                first_original_shape=preflight_image_store.original_shape(first.path),
                second_original_shape=preflight_image_store.original_shape(second.path),
            )
            matching_source = "preflight_loma"
            cache.put(
                pair_key,
                matched,
                metadata={
                    "matching_source": matching_source,
                    "pair_type": row["pair_type"],
                    "camera1": row["match_camera1"],
                    "camera2": row["match_camera2"],
                    "relation": row["relation"],
                    "frame1": int(row["frame1"]),
                    "frame2": int(row["frame2"]),
                },
            )
        current_matching_sources[pair_key] = matching_source
        matched = matched.copy()
        matched.attrs.clear()
        for key, value in row.items():
            matched[key] = value
        matched["matching_source"] = matching_source
        match_records.append(matched)

    sampled.to_csv(output_root / "sampled_pairs.csv", index=False)
    pair_metrics = _pair_metric_plan(sampled, cache)
    if not pair_metrics.empty:
        pair_metrics["matching_source"] = pair_metrics["pair_key"].map(
            current_matching_sources
        ).fillna(pair_metrics["matching_source"])
    if match_records:
        all_matches = pd.concat(match_records, ignore_index=True)
    else:
        all_matches = pd.DataFrame()

    evaluated_parts: list[pd.DataFrame] = []
    for pair_type in ("intra_camera", "inter_camera"):
        subset = all_matches.loc[
            all_matches.get("pair_type", pd.Series(dtype=str)).astype(str) == pair_type
        ].copy()
        if subset.empty:
            continue
        if pair_type == "intra_camera":
            intra_pair_metrics = pair_metrics.loc[
                pair_metrics["pair_type"] == pair_type
            ].copy()
            if "camera" not in intra_pair_metrics.columns and "match_camera1" in intra_pair_metrics.columns:
                intra_pair_metrics["camera"] = intra_pair_metrics["match_camera1"]
            intra_pair_metrics = intra_pair_metrics.drop(
                columns=["match_camera1"], errors="ignore"
            )
            for _camera, camera_subset in _split_intra_preflight_matches(subset):
                camera_subset["camera"] = camera_subset["match_camera1"]
                camera_subset["frame_offset"] = (
                    camera_subset["frame2"].astype(int)
                    - camera_subset["frame1"].astype(int)
                )
                camera_subset.attrs["pair_metrics"] = intra_pair_metrics
                evaluated = staged.evaluate_intra_camera_match_quality(
                    config_path=base_config_path,
                    dataset_root=canonical_root,
                    matches=camera_subset,
                    start_frame=start_frame,
                    frame_count=frame_count,
                    pose_pkl_path=preflight_pose_path,
                    pose_field=preflight_pose_field,
                )
                evaluated.attrs.clear()
                evaluated_parts.append(evaluated)
            continue
        else:
            subset["camera1"] = subset["match_camera1"]
            subset["camera2"] = subset["match_camera2"]
            subset["pair_relation"] = subset["relation"]
            subset.attrs["pair_metrics"] = pair_metrics.loc[
                pair_metrics["pair_type"] == pair_type
            ]
            evaluated = staged.evaluate_inter_camera_match_quality(
                config_path=base_config_path,
                dataset_root=canonical_root,
                matches=subset,
                start_frame=start_frame,
                frame_count=frame_count,
                pose_pkl_path=preflight_pose_path,
                pose_field=preflight_pose_field,
            )
        evaluated.attrs.clear()
        evaluated_parts.append(evaluated)
    evaluated_matches = (
        pd.concat(evaluated_parts, ignore_index=True)
        if evaluated_parts
        else pd.DataFrame()
    )
    evaluated_matches.attrs.clear()
    if not evaluated_matches.empty:
        evaluated_matches.to_parquet(
            output_root / "raw_match_diagnostics.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_root / "raw_match_diagnostics.parquet", index=False)

    summary = summarize_preflight_matches(evaluated_matches)
    summary["sampled_pair_count"] = int(len(sampled))
    summary["sampled_group_count"] = int(sampled["group_key"].nunique()) if not sampled.empty else 0
    pair_quality = summarize_preflight_pair_metrics(evaluated_matches)
    if not pair_quality.empty:
        pair_metrics = pair_metrics.merge(pair_quality, on="pair_key", how="left")
        if int(preflight.visualize_top_bad_pairs) > 0 and not evaluated_matches.empty:
            from .visualization import draw_tiled_match_overlay

            viz_root = output_root / "visualizations"
            viz_root.mkdir(parents=True, exist_ok=True)
            ranking = pair_quality.sort_values(
                ["reprojection_p95", "evaluated_match_count"],
                ascending=[False, False],
                na_position="last",
            ).head(int(preflight.visualize_top_bad_pairs))
            for item in ranking.to_dict(orient="records"):
                pair_key = str(item["pair_key"])
                pair_row = sampled.loc[sampled["pair_key"] == pair_key]
                if pair_row.empty:
                    continue
                pair = pair_row.iloc[0].to_dict()
                camera1 = str(pair["match_camera1"])
                camera2 = str(pair["match_camera2"])
                first = frames_by_camera[camera1].get(int(pair["frame1"]))
                second = frames_by_camera[camera2].get(int(pair["frame2"]))
                if first is None or second is None:
                    continue
                image1 = cv2.imread(str(first.path), cv2.IMREAD_COLOR)
                image2 = cv2.imread(str(second.path), cv2.IMREAD_COLOR)
                points = evaluated_matches.loc[
                    evaluated_matches["pair_key"].astype(str) == pair_key
                ]
                if image1 is None or image2 is None or points.empty:
                    continue
                stem = (
                    f"{pair['pair_type']}_{camera1}__{camera2}_"
                    f"{pair['relation']}_{int(pair['frame1']):05d}_{int(pair['frame2']):05d}"
                )
                draw_tiled_match_overlay(
                    image1,
                    image2,
                    points[["x1", "y1"]].to_numpy(dtype=np.float64),
                    points[["x2", "y2"]].to_numpy(dtype=np.float64),
                    viz_root / f"{stem}_preflight.jpg",
                    title_prefix=(
                        f"preflight reproj_p95={item.get('reprojection_p95')} "
                        f"ransac={item.get('ransac_inlier_ratio')}"
                    ),
                )
    pair_metrics.to_csv(output_root / "pair_metrics.csv", index=False)
    decision = classify_preflight_quality(
        pair_metrics=pair_metrics,
        summary=summary,
        config=base_config,
    )
    report = {
        "clip_id": str(delivery.get("vehicle_model", "unknown")),
        "vehicle": str(delivery.get("vehicle_model", "unknown")),
        "pose_field": str(base_config.assumptions.get("result_interpolation_ego_field", "")),
        "preflight": summary,
        **decision,
        "sampled_pair_count": int(len(sampled)),
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "fail_action": str(preflight.fail_action),
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    report_lines = [
        f"# Clip Preflight Quality",
        "",
        f"- Status: **{report['overall_status']}**",
        f"- Vehicle: `{report['vehicle']}`",
        f"- Sampled pairs: `{report['sampled_pair_count']}`",
        f"- Reused/new cache: `{sum(1 for item in pair_metrics['matching_source'] if item == 'preflight_reused')}` / `{len(pair_metrics)}`",
        "",
        "## Failure reasons",
        *(f"- {item}" for item in report["failure_reasons"]),
        "",
        "## Warnings",
        *(f"- {item}" for item in report["warnings"]),
    ]
    (output_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report
