from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple

import numpy as np
import yaml

from dongfeng_registration.dynamic_object_compensation import (
    DEFAULT_REFERENCE_CAMERAS_6,
    IcpMotionConfig,
    ObjectIcpMotionCandidate,
    ObjectTrackSample,
    PcdCloud,
    _interpolate_ego_transform,
    _interpolate_track_transform,
    box_corners_ego,
    box_corners_from_transform,
    box_transform_world,
    compensate_points_between_box_poses,
    dynamic_objects_for_info,
    ego_world_transform,
    estimate_icp_corrected_motion,
    interpolate_relative_transform,
    object_id,
    pcd_path_is_ego_frame,
    points_in_box_mask,
    read_pcd_cloud,
    reference_time_seconds_from_cameras,
    select_object_motion_velocity,
    split_indices_by_timestamp_clusters,
    transform_points,
    write_pcd_cloud,
)


RAW_LIDAR_NAMES = (
    "front_lidar",
    "rear_lidar",
    "left_lidar",
    "right_lidar",
    "top_center_lidar",
    "front_left_lidar",
    "front_right_lidar",
)

RAW_LIDAR_FOLDER_ALIASES = {
    "front_left_lidar": "top_left_lidar",
    "front_right_lidar": "top_right_lidar",
}


class ObjectPointSelection(NamedTuple):
    total_point_count: int
    clusters: tuple[np.ndarray, ...]


class RawLidarTrackBuildResult(NamedTuple):
    tracks: dict[tuple[str, str], list[ObjectTrackSample]]
    selections: dict[tuple[int, str, str], ObjectPointSelection]


class FrameLidarContext(NamedTuple):
    path: Path
    cloud: PcdCloud
    points_ego: np.ndarray
    timestamps: np.ndarray | None


class PcdCloudCache:
    """Bounded all-or-nothing cache for decompressed immutable PCD clouds."""

    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._usable = self.max_bytes > 0
        self._stored_bytes = 0
        self._clouds: dict[tuple[int, str], PcdCloud] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @property
    def usable(self) -> bool:
        with self._lock:
            return self._usable

    @property
    def stored_bytes(self) -> int:
        with self._lock:
            return self._stored_bytes

    def put(self, key: tuple[int, str], cloud: PcdCloud) -> bool:
        with self._lock:
            if not self._usable:
                return False
            size = int(cloud.data.nbytes)
            previous = self._clouds.get(key)
            previous_size = 0 if previous is None else int(previous.data.nbytes)
            next_size = self._stored_bytes - previous_size + size
            if next_size > self.max_bytes:
                self._clouds.clear()
                self._stored_bytes = 0
                self._usable = False
                return False
            self._clouds[key] = cloud
            self._stored_bytes = next_size
            return True

    def get(self, key: tuple[int, str]) -> PcdCloud | None:
        with self._lock:
            if not self._usable:
                self.misses += 1
                return None
            cloud = self._clouds.get(key)
            if cloud is None:
                self.misses += 1
            else:
                self.hits += 1
            return cloud


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = getattr(args, "config", None)
    config = _load_yaml_config(config_path)
    dynamic_config = config.get(
        "dynamic_object_compensation",
        config.get("dynamic_compensation", config),
    )
    if not isinstance(dynamic_config, dict):
        raise ValueError("dynamic_object_compensation config must be a mapping")
    for key, value in dynamic_config.items():
        if key in {"icp", "object_motion_voting"}:
            continue
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)
    icp_payload = dynamic_config.get("icp", {})
    if icp_payload is None:
        icp_payload = {}
    if not isinstance(icp_payload, dict):
        raise ValueError("dynamic_object_compensation.icp must be a mapping")
    args.icp_payload = dict(icp_payload)
    voting_payload = dynamic_config.get("object_motion_voting", {})
    if voting_payload is None:
        voting_payload = {}
    if not isinstance(voting_payload, dict):
        raise ValueError("dynamic_object_compensation.object_motion_voting must be a mapping")
    args.object_motion_voting_payload = dict(voting_payload)
    return args


def _icp_config_from_args(args: argparse.Namespace) -> IcpMotionConfig:
    payload = dict(getattr(args, "icp_payload", {}) or {})
    if bool(getattr(args, "use_icp", False)):
        payload["enabled"] = True
    if hasattr(args, "icp_max_correspondence_distance_m") and args.icp_max_correspondence_distance_m:
        payload["max_correspondence_distance_m"] = float(args.icp_max_correspondence_distance_m)
    return IcpMotionConfig(**payload)


def _object_motion_voting_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(getattr(args, "object_motion_voting_payload", {}) or {})
    if bool(getattr(args, "object_motion_voting_enabled", False)):
        payload["enabled"] = True
    return payload


def _reference_time_seconds_with_offset(
    info: dict[str, Any], reference_cameras: tuple[str, ...] | list[str], offset_s: float
) -> float:
    return reference_time_seconds_from_cameras(info, reference_cameras) + float(offset_s)


def output_lidar_data(
    lidar_payloads: dict[str, Any], *, lidar_names: tuple[str, ...]
) -> tuple[str, dict[str, Any]]:
    """Return the lidar metadata that defines the fused output filename/template.

    Multi-lidar datasets expose the fused ego-frame cloud as ``perception``.
    A single-lidar canonical dataset, such as T3, may only expose the requested
    raw lidar. In that case its compensated ego-frame cloud is the output.
    """
    perception = lidar_payloads.get("perception")
    if isinstance(perception, dict):
        return "perception", perception
    for lidar_name in lidar_names:
        lidar_data = lidar_payloads.get(lidar_name)
        if isinstance(lidar_data, dict):
            return lidar_name, lidar_data
    raise ValueError("frame has neither perception nor a requested lidar payload")


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def load_canonical_infos(path: Path) -> list[dict[str, Any]]:
    """Load infos from a standardized canonical result_interpolation pkl.

    Downstream dynamic compensation should never read vendor/raw dataset pkl files
    directly. Raw inputs must first go through dataset normalization, which adds
    canonical metadata and rewrites sensor paths relative to the canonical dataset.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("canonical_dataset") is not True:
        raise ValueError(
            f"{path} is not a canonical standardized pkl. "
            "Run normalize-dataset first and pass result_interpolation_standard.pkl."
        )
    if not isinstance(metadata.get("standardized_source"), dict):
        raise ValueError(
            f"{path} is missing canonical standardized_source metadata. "
            "Run normalize-dataset first and pass result_interpolation_standard.pkl."
        )
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_lidar_file(dataset_root: Path, lidar_name: str, lidar_data: dict[str, Any]) -> Path:
    raw_path = Path(str(lidar_data.get("data_path") or lidar_data.get("aws_path") or ""))
    name = raw_path.name
    raw_folder = raw_path.parent.name
    folders = [RAW_LIDAR_FOLDER_ALIASES.get(lidar_name, lidar_name), lidar_name]
    if lidar_name == "perception":
        folders.insert(0, "perception")
    candidates = []
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    if str(raw_path):
        candidates.append(dataset_root / raw_path)
    if raw_folder:
        candidates.extend(
            [
                dataset_root / raw_folder / name,
                dataset_root / "lidar" / raw_folder / name,
            ]
        )
    for folder in folders:
        candidates.extend(
            [
                dataset_root / folder / name,
                dataset_root / "lidar" / folder / name,
                dataset_root / "lidar" / f"{folder}_compensation" / name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(f"cannot resolve {lidar_name}: {name}")


def _materialize_single_lidar_output(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _build_raw_lidar_object_tracks(
    *,
    infos: list[dict[str, Any]],
    dataset_root: Path,
    frame_start: int,
    frame_end: int,
    lidar_names: tuple[str, ...],
    min_object_points: int,
    box_margin_m: float,
    timestamp_cluster_max_gap_s: float,
    timestamp_cluster_max_gap_fraction: float,
    track_workers: int = 1,
    cloud_cache: PcdCloudCache | None = None,
) -> RawLidarTrackBuildResult:
    def extract_frame(
        frame_index: int,
    ) -> tuple[
        dict[tuple[str, str], list[ObjectTrackSample]],
        dict[tuple[int, str, str], ObjectPointSelection],
    ]:
        frame_tracks: dict[tuple[str, str], list[ObjectTrackSample]] = {}
        frame_selections: dict[tuple[int, str, str], ObjectPointSelection] = {}
        info = infos[frame_index]
        lidar_payloads = info.get("sensors", {}).get("lidar", {})
        objects = dynamic_objects_for_info(info)
        if not objects:
            return frame_tracks, frame_selections
        source_ego_to_world = ego_world_transform(info)
        for lidar_name in lidar_names:
            lidar_data = lidar_payloads.get(lidar_name)
            if not isinstance(lidar_data, dict) or "extrinsic" not in lidar_data:
                continue
            lidar_file = resolve_lidar_file(dataset_root, lidar_name, lidar_data)
            cloud = read_pcd_cloud(lidar_file)
            if cloud_cache is not None:
                cloud_cache.put((frame_index, lidar_name), cloud)
            if cloud.timestamps is None:
                continue
            if pcd_path_is_ego_frame(lidar_file):
                points_ego = cloud.xyz
            else:
                points_ego = transform_points(
                    cloud.xyz,
                    np.asarray(lidar_data["extrinsic"], dtype=np.float64).reshape(4, 4),
                )
            timestamps = np.asarray(cloud.timestamps, dtype=np.float64)
            claimed = np.zeros(len(points_ego), dtype=bool)
            for obj in objects:
                oid = object_id(obj)
                if not oid:
                    continue
                mask = points_in_box_mask(points_ego, obj, margin_m=box_margin_m) & ~claimed
                total_point_count = int(np.count_nonzero(mask))
                clusters = split_indices_by_timestamp_clusters(
                    np.flatnonzero(mask),
                    timestamps,
                    min_points=int(min_object_points),
                    max_gap_s=float(timestamp_cluster_max_gap_s),
                    max_gap_fraction=float(timestamp_cluster_max_gap_fraction),
                )
                frame_selections[(frame_index, lidar_name, oid)] = ObjectPointSelection(
                    total_point_count=total_point_count,
                    clusters=tuple(cluster.copy() for cluster in clusters),
                )
                for cluster in clusters:
                    frame_tracks.setdefault((oid, lidar_name), []).append(
                        ObjectTrackSample(
                            frame_index=frame_index,
                            time_s=float(np.mean(timestamps[cluster])),
                            transform_world_from_box=box_transform_world(info, obj),
                            obj=obj,
                            points_world=transform_points(points_ego[cluster], source_ego_to_world),
                        )
                    )
                    claimed[cluster] = True
        return frame_tracks, frame_selections

    workers = max(1, int(track_workers))
    frame_indices = range(frame_start, frame_end)
    if workers == 1:
        frame_results = map(extract_frame, frame_indices)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dynamic-track")
        frame_results = executor.map(extract_frame, frame_indices)

    tracks: dict[tuple[str, str], list[ObjectTrackSample]] = {}
    selections: dict[tuple[int, str, str], ObjectPointSelection] = {}
    try:
        for frame_tracks, frame_selections in frame_results:
            for key, samples in frame_tracks.items():
                tracks.setdefault(key, []).extend(samples)
            selections.update(frame_selections)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    for samples in tracks.values():
        samples.sort(key=lambda item: item.time_s)
    return RawLidarTrackBuildResult(tracks=tracks, selections=selections)


def _neighbor_sample_for_reference_time(
    samples: list[ObjectTrackSample],
    *,
    frame_index: int,
    source_time_s: float,
    reference_time_s: float,
) -> ObjectTrackSample | None:
    if len(samples) < 2:
        return None
    current_candidates = [
        sample
        for sample in samples
        if sample.frame_index == int(frame_index) and abs(sample.time_s - float(source_time_s)) < 1.0e-6
    ]
    current = current_candidates[0] if current_candidates else None
    if current is None:
        return None
    if float(reference_time_s) >= float(source_time_s):
        forward = [sample for sample in samples if sample.time_s > float(source_time_s)]
        if forward:
            return forward[0]
        backward = [sample for sample in samples if sample.time_s < float(source_time_s)]
        return backward[-1] if backward else None
    backward = [sample for sample in samples if sample.time_s < float(source_time_s)]
    if backward:
        return backward[-1]
    forward = [sample for sample in samples if sample.time_s > float(source_time_s)]
    return forward[0] if forward else None


def _collect_object_icp_motion_candidates(
    *,
    infos: list[dict[str, Any]],
    frame_index: int,
    frame_lidar_contexts: dict[str, FrameLidarContext],
    object_tracks: dict[tuple[str, str], list[ObjectTrackSample]],
    min_object_points: int,
    box_margin_m: float,
    icp_config: IcpMotionConfig,
    timestamp_cluster_max_gap_s: float,
    timestamp_cluster_max_gap_fraction: float,
    object_selections: dict[tuple[int, str, str], ObjectPointSelection] | None = None,
) -> dict[str, list[ObjectIcpMotionCandidate]]:
    if not bool(icp_config.enabled):
        return {}
    info = infos[frame_index]
    source_ego_to_world = ego_world_transform(info)
    candidates: dict[str, list[ObjectIcpMotionCandidate]] = {}
    objects = dynamic_objects_for_info(info)
    if not objects:
        return candidates
    selections = object_selections or {}
    for lidar_name, context in frame_lidar_contexts.items():
        if context.timestamps is None:
            continue
        points_ego = context.points_ego
        timestamps = context.timestamps
        claimed = np.zeros(len(points_ego), dtype=bool)
        for obj in objects:
            oid = object_id(obj)
            cached_selection = selections.get((frame_index, lidar_name, oid))
            if cached_selection is None:
                mask = points_in_box_mask(points_ego, obj, margin_m=box_margin_m) & ~claimed
                clusters = split_indices_by_timestamp_clusters(
                    np.flatnonzero(mask),
                    timestamps,
                    min_points=int(min_object_points),
                    max_gap_s=float(timestamp_cluster_max_gap_s),
                    max_gap_fraction=float(timestamp_cluster_max_gap_fraction),
                )
            else:
                clusters = [cluster.copy() for cluster in cached_selection.clusters]
            if not clusters:
                continue
            track_samples = object_tracks.get((oid, lidar_name), [])
            for cluster in clusters:
                source_time_s = float(np.mean(timestamps[cluster]))
                source_box_world = _interpolate_track_transform(track_samples, source_time_s)
                if source_box_world is None:
                    continue
                neighbor = _neighbor_sample_for_reference_time(
                    track_samples,
                    frame_index=frame_index,
                    source_time_s=source_time_s,
                    reference_time_s=source_time_s,
                )
                if neighbor is None or neighbor.points_world is None:
                    continue
                box_pair_motion = (
                    np.asarray(neighbor.transform_world_from_box, dtype=np.float64)
                    @ np.linalg.inv(np.asarray(source_box_world, dtype=np.float64))
                )
                current_points_world = transform_points(
                    points_ego[cluster], source_ego_to_world
                ).astype(np.float64)
                icp_estimate = estimate_icp_corrected_motion(
                    source_points_world=current_points_world,
                    target_points_world=np.asarray(neighbor.points_world, dtype=np.float64),
                    box_motion_target_from_source=box_pair_motion,
                    config=icp_config,
                )
                if not icp_estimate.used_icp:
                    continue
                candidates.setdefault(oid, []).append(
                    ObjectIcpMotionCandidate(
                        object_id=oid,
                        lidar_name=lidar_name,
                        transform_target_from_source=icp_estimate.transform_target_from_source,
                        source_time_s=source_time_s,
                        target_time_s=float(neighbor.time_s),
                        fitness=icp_estimate.fitness,
                        rmse_m=icp_estimate.rmse_m,
                        inlier_count=icp_estimate.inlier_count,
                    )
                )
                claimed[cluster] = True
    return candidates


def _feature(
    *,
    frame_index: int,
    obj: dict[str, Any],
    timestamp_s: float,
    corners: np.ndarray,
    status: str,
    point_count: int,
    lidar_name: str | None = None,
    source_time_s: float | None = None,
) -> dict[str, Any]:
    corners = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    return {
        "frame": int(frame_index),
        "object_id": object_id(obj),
        "object_type": str(obj.get("type", "")),
        "timestamp_s": float(timestamp_s),
        "corners": corners.tolist(),
        "length": float(np.asarray(obj.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)[0]),
        "width": float(np.asarray(obj.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)[1]),
        "height": float(np.asarray(obj.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)[2]),
        "yaw": float(np.asarray(obj.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[2]),
        "status": status,
        "point_count": int(point_count),
        "lidar_name": lidar_name or "",
        "source_time_s": None if source_time_s is None else float(source_time_s),
    }


def _box_payload_for_lidar(payload: dict[str, Any], lidar_name: str) -> dict[str, Any]:
    reports = [
        dict(row)
        for row in payload.get("lidar_reports", [])
        if str(row.get("lidar_name", "")) == str(lidar_name)
    ]
    object_ids = {str(row.get("object_id", "")) for row in reports if row.get("object_id")}
    before = [
        dict(row)
        for row in payload.get("before", [])
        if str(row.get("object_id", "")) in object_ids
    ]
    after_by_lidar = payload.get("after_by_lidar", {})
    if isinstance(after_by_lidar, dict) and isinstance(after_by_lidar.get(str(lidar_name)), dict):
        best_after = {
            str(oid): dict(row)
            for oid, row in after_by_lidar[str(lidar_name)].items()
            if str(oid) in object_ids
        }
    else:
        best_after: dict[str, dict[str, Any]] = {}
        for row in payload.get("after", []):
            if str(row.get("lidar_name", "")) != str(lidar_name):
                continue
            oid = str(row.get("object_id", ""))
            if oid not in object_ids:
                continue
            existing = best_after.get(oid)
            if existing is None or int(row.get("point_count") or 0) > int(
                existing.get("point_count") or 0
            ):
                best_after[oid] = dict(row)
    after = list(best_after.values())
    out = {
        **payload,
        "lidar_name": str(lidar_name),
        "lidar_names": [str(lidar_name)],
        "before": before,
        "after": after,
        "after_by_object_id": best_after,
        "lidar_reports": reports,
    }
    return out


def compensate_frame_from_raw_lidars(
    *,
    infos: list[dict[str, Any]],
    frame_index: int,
    dataset_root: Path,
    lidar_names: tuple[str, ...],
    object_tracks: dict[tuple[str, str], list[ObjectTrackSample]],
    reference_cameras: tuple[str, ...],
    min_object_points: int,
    box_margin_m: float,
    icp_config: IcpMotionConfig,
    timestamp_cluster_max_gap_s: float,
    timestamp_cluster_max_gap_fraction: float,
    max_motion_translation_m: float | None,
    reference_time_offset_s: float = 0.0,
    object_motion_voting: dict[str, Any] | None = None,
    object_selections: dict[tuple[int, str, str], ObjectPointSelection] | None = None,
    cloud_cache: PcdCloudCache | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    info = infos[frame_index]
    lidar_payloads = info.get("sensors", {}).get("lidar", {})
    output_lidar_name, output_lidar = output_lidar_data(lidar_payloads, lidar_names=lidar_names)

    frame_time_s = float(info["timestamp"]) / 1.0e9
    reference_time_s = _reference_time_seconds_with_offset(
        info, reference_cameras, reference_time_offset_s
    )
    source_ego_to_world = ego_world_transform(info)
    target_ego_to_world = _interpolate_ego_transform(infos, reference_time_s)
    if target_ego_to_world is None:
        target_ego_to_world = source_ego_to_world
    world_to_source_ego = np.linalg.inv(source_ego_to_world)

    raw_arrays: list[np.ndarray] = []
    raw_lidar_names: list[str] = []
    per_lidar_clouds: dict[str, Any] = {}
    before_features: dict[str, dict[str, Any]] = {}
    after_features: dict[str, dict[str, Any]] = {}
    after_features_by_lidar: dict[str, dict[str, dict[str, Any]]] = {}
    lidar_reports: list[dict[str, Any]] = []
    object_selections = object_selections or {}
    object_motion_voting = dict(object_motion_voting or {})
    frame_lidar_contexts: dict[str, FrameLidarContext] = {}
    for lidar_name in lidar_names:
        lidar_data = lidar_payloads.get(lidar_name)
        if not isinstance(lidar_data, dict) or "extrinsic" not in lidar_data:
            continue
        lidar_file = resolve_lidar_file(dataset_root, lidar_name, lidar_data)
        cloud = cloud_cache.get((frame_index, lidar_name)) if cloud_cache is not None else None
        if cloud is None:
            cloud = read_pcd_cloud(lidar_file)
        if pcd_path_is_ego_frame(lidar_file):
            points_ego = cloud.xyz.astype(np.float64)
        else:
            points_ego = transform_points(
                cloud.xyz,
                np.asarray(lidar_data["extrinsic"], dtype=np.float64).reshape(4, 4),
            ).astype(np.float64)
        timestamps = (
            None
            if cloud.timestamps is None
            else np.asarray(cloud.timestamps, dtype=np.float64)
        )
        frame_lidar_contexts[lidar_name] = FrameLidarContext(
            path=lidar_file,
            cloud=cloud,
            points_ego=points_ego,
            timestamps=timestamps,
        )
    shared_object_velocities = {}
    if bool(object_motion_voting.get("enabled", False)):
        object_candidates = _collect_object_icp_motion_candidates(
            infos=infos,
            frame_index=frame_index,
            frame_lidar_contexts=frame_lidar_contexts,
            object_tracks=object_tracks,
            min_object_points=min_object_points,
            box_margin_m=box_margin_m,
            icp_config=icp_config,
            timestamp_cluster_max_gap_s=timestamp_cluster_max_gap_s,
            timestamp_cluster_max_gap_fraction=timestamp_cluster_max_gap_fraction,
            object_selections=object_selections,
        )
        for oid, candidates in object_candidates.items():
            velocity = select_object_motion_velocity(
                candidates,
                min_candidates=int(object_motion_voting.get("min_candidates", 2)),
                max_velocity_deviation_mps=float(
                    object_motion_voting.get("max_velocity_deviation_mps", 2.0)
                ),
                mode=str(object_motion_voting.get("mode", "weighted_vote")),
            )
            if velocity is not None:
                shared_object_velocities[oid] = velocity

    for obj in dynamic_objects_for_info(info):
        before_features[object_id(obj)] = _feature(
            frame_index=frame_index,
            obj=obj,
            timestamp_s=frame_time_s,
            corners=box_corners_ego(obj),
            status="before",
            point_count=0,
        )

    for lidar_name in lidar_names:
        context = frame_lidar_contexts.get(lidar_name)
        if context is None:
            continue
        cloud = context.cloud
        data = cloud.data.copy()
        points_ego = context.points_ego.copy()
        timestamps = context.timestamps
        moved = np.zeros(len(points_ego), dtype=bool)
        for obj in dynamic_objects_for_info(info):
            oid = object_id(obj)
            if timestamps is None:
                mask = points_in_box_mask(points_ego, obj, margin_m=box_margin_m) & ~moved
                total_point_count = int(np.count_nonzero(mask))
                lidar_reports.append(
                    {
                        "frame_index": frame_index,
                        "lidar_name": lidar_name,
                        "object_id": oid,
                        "object_type": str(obj.get("type", "")),
                        "point_count": total_point_count,
                        "total_object_point_count": total_point_count,
                        "timestamp_cluster_index": -1,
                        "timestamp_cluster_count": 0,
                        "source_time_s": float("nan"),
                        "reference_time_s": reference_time_s,
                        "motion_translation_m": float("nan"),
                        "status": "missing_point_timestamps",
                        "motion_source": "none",
                        "icp_status": "",
                        "icp_fitness": float("nan"),
                        "icp_rmse_m": float("nan"),
                        "icp_inlier_count": 0,
                        "icp_translation_delta_from_box_m": float("nan"),
                        "icp_rotation_delta_from_box_deg": float("nan"),
                    }
                )
                continue
            cached_selection = object_selections.get((frame_index, lidar_name, oid))
            if cached_selection is None:
                mask = points_in_box_mask(points_ego, obj, margin_m=box_margin_m) & ~moved
                total_point_count = int(np.count_nonzero(mask))
                clusters = split_indices_by_timestamp_clusters(
                    np.flatnonzero(mask),
                    np.asarray(timestamps, dtype=np.float64),
                    min_points=int(min_object_points),
                    max_gap_s=float(timestamp_cluster_max_gap_s),
                    max_gap_fraction=float(timestamp_cluster_max_gap_fraction),
                )
            else:
                total_point_count = int(cached_selection.total_point_count)
                clusters = [cluster.copy() for cluster in cached_selection.clusters]
            if not clusters:
                lidar_reports.append(
                    {
                        "frame_index": frame_index,
                        "lidar_name": lidar_name,
                        "object_id": oid,
                        "object_type": str(obj.get("type", "")),
                        "point_count": total_point_count,
                        "total_object_point_count": total_point_count,
                        "timestamp_cluster_index": -1,
                        "timestamp_cluster_count": 0,
                        "source_time_s": float("nan"),
                        "reference_time_s": reference_time_s,
                        "motion_translation_m": float("nan"),
                        "status": "too_few_timestamp_cluster_points",
                        "motion_source": "none",
                        "icp_status": "",
                        "icp_fitness": float("nan"),
                        "icp_rmse_m": float("nan"),
                        "icp_inlier_count": 0,
                        "icp_translation_delta_from_box_m": float("nan"),
                        "icp_rotation_delta_from_box_deg": float("nan"),
                    }
                )
                continue
            track_samples = object_tracks.get((oid, lidar_name), [])
            target_box_world = _interpolate_track_transform(track_samples, reference_time_s)
            for cluster_index, cluster in enumerate(clusters):
                point_count = int(len(cluster))
                status = "too_few_points"
                source_time_s = float(np.mean(np.asarray(timestamps, dtype=np.float64)[cluster]))
                motion_m = float("nan")
                source_box_world = _interpolate_track_transform(
                    track_samples, source_time_s
                )
                if source_box_world is not None and target_box_world is not None:
                    icp_status = "box_only"
                    icp_used = False
                    icp_fitness = float("nan")
                    icp_rmse_m = float("nan")
                    icp_inlier_count = 0
                    icp_translation_delta_from_box_m = float("nan")
                    icp_rotation_delta_from_box_deg = float("nan")
                    shared_velocity = shared_object_velocities.get(oid)
                    if shared_velocity is not None:
                        current_points_world = transform_points(
                            points_ego[cluster], source_ego_to_world
                        ).astype(np.float64)
                        dt = float(reference_time_s) - float(source_time_s)
                        moved_points_world = current_points_world + (
                            np.asarray(shared_velocity.velocity_world_mps, dtype=np.float64).reshape(1, 3)
                            * dt
                        )
                        moved_points = transform_points(moved_points_world, world_to_source_ego).astype(
                            np.float64
                        )
                        partial_motion = np.eye(4, dtype=np.float64)
                        partial_motion[:3, 3] = (
                            np.asarray(shared_velocity.velocity_world_mps, dtype=np.float64).reshape(3)
                            * dt
                        )
                        after_corners_world = transform_points(
                            box_corners_from_transform(obj["size"], source_box_world),
                            partial_motion,
                        )
                        icp_status = str(shared_velocity.source)
                        icp_used = False
                        icp_fitness = float("nan")
                        icp_rmse_m = float("nan")
                        icp_inlier_count = int(shared_velocity.candidate_count)
                        icp_translation_delta_from_box_m = float("nan")
                        icp_rotation_delta_from_box_deg = 0.0
                    elif bool(icp_config.enabled):
                        neighbor = _neighbor_sample_for_reference_time(
                            track_samples,
                            frame_index=frame_index,
                            source_time_s=source_time_s,
                            reference_time_s=reference_time_s,
                        )
                        current_points_world = transform_points(
                            points_ego[cluster], source_ego_to_world
                        ).astype(np.float64)
                        if neighbor is not None and neighbor.points_world is not None:
                            time_span = float(neighbor.time_s - source_time_s)
                            alpha = 0.0 if abs(time_span) < 1.0e-9 else (
                                float(reference_time_s - source_time_s) / time_span
                            )
                            box_pair_motion = (
                                np.asarray(neighbor.transform_world_from_box, dtype=np.float64)
                                @ np.linalg.inv(np.asarray(source_box_world, dtype=np.float64))
                            )
                            icp_estimate = estimate_icp_corrected_motion(
                                source_points_world=current_points_world,
                                target_points_world=np.asarray(neighbor.points_world, dtype=np.float64),
                                box_motion_target_from_source=box_pair_motion,
                                config=icp_config,
                            )
                            partial_motion = interpolate_relative_transform(
                                icp_estimate.transform_target_from_source,
                                alpha,
                            )
                            moved_points = transform_points(
                                transform_points(points_ego[cluster], source_ego_to_world),
                                partial_motion,
                            )
                            moved_points = transform_points(moved_points, world_to_source_ego).astype(
                                np.float64
                            )
                            icp_status = icp_estimate.status
                            icp_used = bool(icp_estimate.used_icp)
                            icp_fitness = icp_estimate.fitness
                            icp_rmse_m = icp_estimate.rmse_m
                            icp_inlier_count = icp_estimate.inlier_count
                            icp_translation_delta_from_box_m = (
                                icp_estimate.translation_delta_from_box_m
                            )
                            icp_rotation_delta_from_box_deg = icp_estimate.rotation_delta_from_box_deg
                            after_corners_world = transform_points(
                                box_corners_from_transform(obj["size"], source_box_world),
                                partial_motion,
                            )
                        else:
                            moved_points = compensate_points_between_box_poses(
                                points_ego[cluster],
                                source_box_to_world=source_box_world,
                                target_box_to_world=target_box_world,
                                world_to_output_ego=world_to_source_ego,
                                source_ego_to_world=source_ego_to_world,
                            ).astype(np.float64)
                            icp_status = "icp_missing_neighbor"
                            after_corners_world = box_corners_from_transform(
                                obj["size"], target_box_world
                            )
                    else:
                        moved_points = compensate_points_between_box_poses(
                            points_ego[cluster],
                            source_box_to_world=source_box_world,
                            target_box_to_world=target_box_world,
                            world_to_output_ego=world_to_source_ego,
                            source_ego_to_world=source_ego_to_world,
                        ).astype(np.float64)
                        after_corners_world = box_corners_from_transform(
                            obj["size"], target_box_world
                        )
                    motion = np.linalg.norm(moved_points - points_ego[cluster], axis=1)
                    motion_m = float(np.median(motion)) if motion.size else 0.0
                    motion_limit = (
                        float(max_motion_translation_m)
                        if max_motion_translation_m is not None
                        else float("inf")
                    )
                    if np.isfinite(motion_limit) and motion_m > motion_limit:
                        status = "motion_translation_too_large"
                    else:
                        points_ego[cluster] = moved_points
                        moved[cluster] = True
                        status = "compensated"
                        if oid not in after_features:
                            after_corners = transform_points(
                                after_corners_world,
                                world_to_source_ego,
                            )
                            after_features[oid] = _feature(
                                frame_index=frame_index,
                                obj=obj,
                                timestamp_s=reference_time_s,
                                corners=after_corners,
                                status="after_comp",
                                point_count=point_count,
                                lidar_name=lidar_name,
                                source_time_s=source_time_s,
                            )
                        lidar_after = after_features_by_lidar.setdefault(lidar_name, {})
                        existing_lidar_after = lidar_after.get(oid)
                        if existing_lidar_after is None or point_count > int(
                            existing_lidar_after.get("point_count") or 0
                        ):
                            after_corners = transform_points(
                                after_corners_world,
                                world_to_source_ego,
                            )
                            lidar_after[oid] = _feature(
                                frame_index=frame_index,
                                obj=obj,
                                timestamp_s=reference_time_s,
                                corners=after_corners,
                                status="after_comp",
                                point_count=point_count,
                                lidar_name=lidar_name,
                                source_time_s=source_time_s,
                            )
                else:
                    status = "no_bracketing_object_track"
                    icp_status = ""
                    icp_used = False
                    icp_fitness = float("nan")
                    icp_rmse_m = float("nan")
                    icp_inlier_count = 0
                    icp_translation_delta_from_box_m = float("nan")
                    icp_rotation_delta_from_box_deg = float("nan")
                lidar_reports.append(
                    {
                        "frame_index": frame_index,
                        "lidar_name": lidar_name,
                        "object_id": oid,
                        "object_type": str(obj.get("type", "")),
                        "point_count": point_count,
                        "total_object_point_count": total_point_count,
                        "timestamp_cluster_index": int(cluster_index),
                        "timestamp_cluster_count": int(len(clusters)),
                        "source_time_s": source_time_s,
                        "reference_time_s": reference_time_s,
                        "motion_translation_m": motion_m,
                        "status": status,
                        "motion_source": (
                            "object_icp"
                            if status == "compensated"
                            and oid in shared_object_velocities
                            else ("icp" if icp_used else ("box" if status == "compensated" else "none"))
                        ),
                        "icp_status": icp_status,
                        "icp_fitness": icp_fitness,
                        "icp_rmse_m": icp_rmse_m,
                        "icp_inlier_count": icp_inlier_count,
                        "icp_translation_delta_from_box_m": icp_translation_delta_from_box_m,
                        "icp_rotation_delta_from_box_deg": icp_rotation_delta_from_box_deg,
                        "object_motion_candidate_count": int(
                            shared_object_velocities[oid].candidate_count
                            if oid in shared_object_velocities
                            else 0
                        ),
                        "object_motion_candidate_lidars": ",".join(
                            shared_object_velocities[oid].candidate_lidars
                            if oid in shared_object_velocities
                            else ()
                        ),
                    }
                )
        data["x"] = points_ego[:, 0].astype(data["x"].dtype, copy=False)
        data["y"] = points_ego[:, 1].astype(data["y"].dtype, copy=False)
        data["z"] = points_ego[:, 2].astype(data["z"].dtype, copy=False)
        per_lidar_clouds[lidar_name] = cloud.__class__(
            header_lines=cloud.header_lines,
            header=cloud.header,
            data=data,
            dtype=data.dtype,
        )
        raw_arrays.append(data)
        raw_lidar_names.append(lidar_name)

    if not raw_arrays:
        raise ValueError(f"frame {frame_index} has no raw lidar arrays")
    fused_data = raw_arrays[0] if len(raw_arrays) == 1 else np.concatenate(raw_arrays)
    raw_template_cloud = next(iter(per_lidar_clouds.values()))
    output_cloud = raw_template_cloud.__class__(
        header_lines=raw_template_cloud.header_lines,
        header=raw_template_cloud.header,
        data=fused_data,
        dtype=fused_data.dtype,
    )
    for oid, before in before_features.items():
        before["point_count"] = int(
            sum(row["point_count"] for row in lidar_reports if row["object_id"] == oid)
        )
        if oid not in after_features:
            after_features[oid] = {
                **before,
                "timestamp_s": reference_time_s,
                "status": "not_compensated",
                "source_time_s": None,
            }
    box_payload = {
        "frame_index": frame_index,
        "frame_timestamp_s": frame_time_s,
        "reference_time_s": reference_time_s,
        "reference_time_offset_s": float(reference_time_offset_s),
        "lidar_names": raw_lidar_names,
        "output_lidar_name": output_lidar_name,
        "before": list(before_features.values()),
        "after": list(after_features.values()),
        "after_by_lidar": after_features_by_lidar,
        "lidar_reports": lidar_reports,
    }
    return output_cloud, box_payload, per_lidar_clouds


def run(args: argparse.Namespace) -> None:
    stage_started = time.perf_counter()
    args = _apply_config(args)
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_canonical_infos(pkl_path)
    start = int(args.start)
    requested_end = int(args.end)
    end = len(infos) if requested_end < 0 else min(len(infos), requested_end)
    track_start = max(0, start - 1)
    track_end = min(len(infos), end + 1)
    lidar_names = tuple(item.strip() for item in str(args.lidars).split(",") if item.strip())
    reference_cameras = tuple(
        item.strip() for item in str(args.reference_cameras).split(",") if item.strip()
    )
    output_pcd_dir = Path(args.output_pcd_dir)
    output_box_dir = Path(args.output_box_dir)
    output_raw_lidar_dir = Path(args.output_raw_lidar_dir) if args.output_raw_lidar_dir else None
    output_box_by_lidar_dir = (
        Path(args.output_box_by_lidar_dir) if args.output_box_by_lidar_dir else None
    )
    summary_path = Path(args.summary_csv)
    icp_config = _icp_config_from_args(args)
    object_motion_voting = _object_motion_voting_config_from_args(args)
    output_pcd_dir.mkdir(parents=True, exist_ok=True)
    output_box_dir.mkdir(parents=True, exist_ok=True)
    if output_raw_lidar_dir is not None:
        output_raw_lidar_dir.mkdir(parents=True, exist_ok=True)
    if output_box_by_lidar_dir is not None:
        output_box_by_lidar_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    track_workers = int(args.track_workers)
    if track_workers < 1:
        raise ValueError("track_workers must be at least 1")
    cache_bytes = int(max(0.0, float(args.cloud_cache_max_gb)) * (1024**3))
    cloud_cache = PcdCloudCache(max_bytes=cache_bytes)
    track_started = time.perf_counter()
    track_build = _build_raw_lidar_object_tracks(
        infos=infos,
        dataset_root=dataset_root,
        frame_start=track_start,
        frame_end=track_end,
        lidar_names=lidar_names,
        min_object_points=int(args.min_object_points),
        box_margin_m=float(args.box_margin_m),
        timestamp_cluster_max_gap_s=float(args.timestamp_cluster_max_gap_s),
        timestamp_cluster_max_gap_fraction=float(args.timestamp_cluster_max_gap_fraction),
        track_workers=track_workers,
        cloud_cache=cloud_cache,
    )
    track_elapsed_s = time.perf_counter() - track_started
    if output_raw_lidar_dir is not None:
        for lidar_name in lidar_names:
            (output_raw_lidar_dir / lidar_name).mkdir(parents=True, exist_ok=True)
    if output_box_by_lidar_dir is not None:
        for lidar_name in lidar_names:
            (output_box_by_lidar_dir / lidar_name).mkdir(parents=True, exist_ok=True)

    single_lidar_output_mode = str(args.single_lidar_output_mode).strip().lower()
    if single_lidar_output_mode not in {"hardlink", "write"}:
        raise ValueError("single_lidar_output_mode must be 'hardlink' or 'write'")

    def compensate_and_write_frame(frame_index: int) -> dict[str, Any]:
        cloud, boxes, per_lidar_clouds = compensate_frame_from_raw_lidars(
            infos=infos,
            frame_index=frame_index,
            dataset_root=dataset_root,
            lidar_names=lidar_names,
            object_tracks=track_build.tracks,
            reference_cameras=reference_cameras,
            min_object_points=int(args.min_object_points),
            box_margin_m=float(args.box_margin_m),
            icp_config=icp_config,
            timestamp_cluster_max_gap_s=float(args.timestamp_cluster_max_gap_s),
            timestamp_cluster_max_gap_fraction=float(args.timestamp_cluster_max_gap_fraction),
            max_motion_translation_m=float(args.max_motion_translation_m),
            reference_time_offset_s=float(args.reference_time_offset_s),
            object_motion_voting=object_motion_voting,
            object_selections=track_build.selections,
            cloud_cache=cloud_cache,
        )
        _, output_lidar = output_lidar_data(
            infos[frame_index]["sensors"]["lidar"], lidar_names=lidar_names
        )
        pcd_name = Path(str(output_lidar.get("data_path") or output_lidar.get("aws_path") or "")).name
        if not pcd_name:
            pcd_name = f"frame_{frame_index:06d}.pcd"
        out_pcd = output_pcd_dir / pcd_name
        write_pcd_cloud(out_pcd, cloud)
        out_json = output_box_dir / f"frame_{frame_index:06d}_{Path(pcd_name).stem}_boxes.json"
        out_json.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
        per_lidar_written = 0
        if output_raw_lidar_dir is not None:
            if len(per_lidar_clouds) == 1 and single_lidar_output_mode == "hardlink":
                lidar_name = next(iter(per_lidar_clouds))
                _materialize_single_lidar_output(
                    out_pcd,
                    output_raw_lidar_dir / lidar_name / pcd_name,
                )
                per_lidar_written = 1
            else:
                for lidar_name, lidar_cloud in per_lidar_clouds.items():
                    lidar_dir = output_raw_lidar_dir / lidar_name
                    write_pcd_cloud(lidar_dir / pcd_name, lidar_cloud)
                    per_lidar_written += 1
        if output_box_by_lidar_dir is not None:
            for lidar_name in per_lidar_clouds:
                lidar_box_dir = output_box_by_lidar_dir / lidar_name
                lidar_payload = _box_payload_for_lidar(boxes, lidar_name)
                (lidar_box_dir / out_json.name).write_text(
                    json.dumps(lidar_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        compensated = sum(1 for row in boxes["lidar_reports"] if row["status"] == "compensated")
        icp_used = sum(1 for row in boxes["lidar_reports"] if row.get("motion_source") == "icp")
        object_icp_used = sum(
            1 for row in boxes["lidar_reports"] if row.get("motion_source") == "object_icp"
        )
        box_used = sum(1 for row in boxes["lidar_reports"] if row.get("motion_source") == "box")
        split_reports = sum(
            1 for row in boxes["lidar_reports"] if int(row.get("timestamp_cluster_count") or 0) > 1
        )
        return {
            "frame_index": frame_index,
            "pcd": pcd_name,
            "output_pcd": str(out_pcd),
            "box_json": str(out_json),
            "per_lidar_pcd_count": per_lidar_written,
            "reference_time_s": f"{boxes['reference_time_s']:.9f}",
            "dynamic_object_count": len(boxes["before"]),
            "lidar_object_compensated_count": compensated,
            "lidar_object_icp_count": icp_used,
            "lidar_object_object_icp_count": object_icp_used,
            "lidar_object_box_count": box_used,
            "timestamp_split_report_count": split_reports,
            "lidar_report_count": len(boxes["lidar_reports"]),
        }

    compensation_started = time.perf_counter()
    frame_indices = range(start, end)
    if workers == 1:
        frame_results = map(compensate_and_write_frame, frame_indices)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dynamic-pcd")
        frame_results = executor.map(compensate_and_write_frame, frame_indices)
    rows: list[dict[str, Any]] = []
    try:
        for row in frame_results:
            rows.append(row)
            print(
                f"[{int(row['frame_index']):04d}] pcd={row['pcd']} "
                f"objects={int(row['dynamic_object_count'])} "
                f"compensated_lidar_objects={int(row['lidar_object_compensated_count'])} "
                f"object_icp={int(row['lidar_object_object_icp_count'])} "
                f"icp={int(row['lidar_object_icp_count'])} "
                f"box={int(row['lidar_object_box_count'])} "
                f"split_reports={int(row['timestamp_split_report_count'])} -> {row['output_pcd']}"
            )
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    compensation_elapsed_s = time.perf_counter() - compensation_started
    total_elapsed_s = time.perf_counter() - stage_started
    timing_payload = {
        "frame_count": int(max(0, end - start)),
        "lidar_count": int(len(lidar_names)),
        "track_workers": track_workers,
        "compensation_workers": workers,
        "cloud_cache_max_bytes": cache_bytes,
        "cloud_cache_usable": cloud_cache.usable,
        "cloud_cache_stored_bytes": cloud_cache.stored_bytes,
        "cloud_cache_hits": cloud_cache.hits,
        "cloud_cache_misses": cloud_cache.misses,
        "track_build_seconds": track_elapsed_s,
        "compensation_and_write_seconds": compensation_elapsed_s,
        "total_seconds": total_elapsed_s,
    }
    if args.timing_json is not None:
        timing_path = Path(args.timing_json)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(timing_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] pcd_dir={output_pcd_dir}")
    if output_raw_lidar_dir is not None:
        print(f"[done] raw_lidar_pcd_dir={output_raw_lidar_dir}")
    print(f"[done] box_dir={output_box_dir}")
    if output_box_by_lidar_dir is not None:
        print(f"[done] box_by_lidar_dir={output_box_by_lidar_dir}")
    print(f"[done] summary={summary_path}")
    if args.timing_json is not None:
        print(f"[done] timing={args.timing_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compensate dynamic objects per raw lidar, then fuse to ego-frame PCD."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10, help="exclusive end frame; use -1 for all frames")
    parser.add_argument("--workers", type=int, default=1, help="parallel frame workers")
    parser.add_argument("--track-workers", type=int, default=1, help="parallel track-prepass workers")
    parser.add_argument(
        "--cloud-cache-max-gb",
        type=float,
        default=0.0,
        help="all-or-nothing decompressed PCD cache budget; 0 disables it",
    )
    parser.add_argument(
        "--single-lidar-output-mode",
        choices=("hardlink", "write"),
        default="hardlink",
    )
    parser.add_argument("--output-pcd-dir", type=Path, required=True)
    parser.add_argument("--output-box-dir", type=Path, required=True)
    parser.add_argument("--output-raw-lidar-dir", type=Path)
    parser.add_argument("--output-box-by-lidar-dir", type=Path)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timing-json", type=Path)
    parser.add_argument("--lidars", default=",".join(RAW_LIDAR_NAMES))
    parser.add_argument("--reference-cameras", default=",".join(DEFAULT_REFERENCE_CAMERAS_6))
    parser.add_argument("--min-object-points", type=int, default=5)
    parser.add_argument("--box-margin-m", type=float, default=0.0)
    parser.add_argument("--timestamp-cluster-max-gap-s", type=float, default=0.030)
    parser.add_argument("--timestamp-cluster-max-gap-fraction", type=float, default=0.30)
    parser.add_argument("--reference-time-offset-s", type=float, default=0.0)
    parser.add_argument("--max-motion-translation-m", type=float, default=float("inf"))
    parser.add_argument("--object-motion-voting-enabled", action="store_true")
    parser.add_argument("--use-icp", action="store_true")
    parser.add_argument("--icp-max-correspondence-distance-m", type=float)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

