"""COLMAP external-pose BA and dense reconstruction helpers."""

from __future__ import annotations

import json
import csv
import pickle
import shutil
import sqlite3
import struct
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image
import yaml

from .deliverables import update_colmap_delivery_index
from .semantic_labels import DEFAULT_DYNAMIC_CLASSES, canonicalize_label
from .openmvs_backend import (
    OpenMVSPaths,
    OpenMVSSettings,
    count_ply_vertices,
    run_openmvs_dense,
)


DEFAULT_COLMAP_DENSE_MASK_CLASSES = frozenset(DEFAULT_DYNAMIC_CLASSES - {"vegetation"})


@dataclass(frozen=True)
class ColmapReconstructionConfig:
    config_path: Path
    canonical_root: Path
    image_root: Path
    semantic_cache_root: Path
    delivery_root: Path
    pose_variant: str
    pose_csv: Path
    calibration_pkl: Path
    output_root: Path
    colmap_executable: Path
    feature_max_image_size: int
    matching_pair_offsets: tuple[int, ...]
    matching_inter_camera_pairs: tuple[str, ...]
    use_gpu: bool
    gpu_index: str
    run_dense: bool
    mvs_backend: str
    openmvs_interface_colmap_executable: Path
    openmvs_densify_point_cloud_executable: Path
    openmvs_resolution_level: int
    openmvs_max_resolution: int
    openmvs_min_resolution: int
    openmvs_number_views: int
    openmvs_number_views_fuse: int
    openmvs_gpu_device: str | None
    run_bundle_adjustment: bool
    ba_mode: str
    absolute_pose_prior_colmap_executable: Path
    ba_position_prior_sigma_m: float
    ba_rotation_prior_sigma_deg: float
    ba_refine_points3d: bool
    ba_min_track_len: int
    ba_max_reproj_error: float
    ba_min_tri_angle_deg: float
    dense_max_image_size: int
    patch_match_max_num_neighbors: int
    patch_match_num_iterations: int
    patch_match_num_samples: int
    patch_match_depth_min_m: float
    patch_match_depth_max_m: float
    patch_match_filter: bool
    patch_match_write_consistency_graph: bool
    stereo_fusion_use_cache: bool
    stereo_fusion_cache_size_gb: float
    pose_mask_classes: frozenset[str]
    dense_mask_classes: frozenset[str]
    start_frame: int
    frame_count: int | None


def _config_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _executable_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        return _config_path(root, path)
    found = shutil.which(str(path))
    return Path(found) if found else _config_path(root, path)


def _optional_gpu_device(value: object) -> str | None:
    if value is None:
        return None
    device = str(value).strip()
    return None if device.lower() in {"", "cpu", "none"} else device


def load_colmap_config(
    config_path: Path,
    *,
    run_output_root: Path | None = None,
    pose_variant: str | None = None,
    output_root: Path | None = None,
) -> ColmapReconstructionConfig:
    """Load standalone or vehicle-embedded COLMAP reconstruction settings."""
    config_path = Path(config_path).resolve()
    source_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("COLMAP configuration must contain a mapping")
    root = config_path.parent
    embedded = source_payload.get("colmap")
    if isinstance(embedded, dict):
        delivery = source_payload.get("delivery", {})
        if not isinstance(delivery, dict):
            raise ValueError("vehicle configuration delivery field must contain a mapping")
        payload = dict(embedded)
        for key in ("canonical_root", "semantic_cache_root", "run_output_root"):
            if key not in payload and delivery.get(key):
                payload[key] = delivery[key]
    else:
        payload = source_payload
    configured_run_output_root = payload.get("run_output_root")
    if run_output_root is None and configured_run_output_root:
        run_output_root = _config_path(root, configured_run_output_root)
    elif run_output_root is not None:
        run_output_root = Path(run_output_root).resolve()

    required = (
        "canonical_root",
        "semantic_cache_root",
        "colmap_executable",
    )
    if run_output_root is None:
        required += ("delivery_root",)
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"COLMAP configuration is missing: {', '.join(missing)}")
    selected_pose_variant = str(pose_variant or payload.get("pose_variant", "after")).lower()
    if selected_pose_variant not in {"before", "after"}:
        raise ValueError("pose_variant must be 'before' or 'after'")
    ba_mode = str(payload.get("ba_mode", "absolute_pose_prior")).lower()
    if ba_mode not in {"standard", "absolute_pose_prior"}:
        raise ValueError("colmap ba_mode must be 'standard' or 'absolute_pose_prior'")
    mvs_backend = str(payload.get("mvs_backend", "colmap")).lower()
    if mvs_backend not in {"colmap", "openmvs"}:
        raise ValueError("colmap mvs_backend must be 'colmap' or 'openmvs'")
    canonical_root = _config_path(root, payload["canonical_root"])
    delivery_root = (
        run_output_root / "deliverables"
        if run_output_root is not None
        else _config_path(root, payload["delivery_root"])
    )
    return ColmapReconstructionConfig(
        config_path=config_path,
        canonical_root=canonical_root,
        image_root=_config_path(root, payload.get("image_root", canonical_root / "images")),
        semantic_cache_root=_config_path(root, payload["semantic_cache_root"]),
        delivery_root=delivery_root,
        pose_variant=selected_pose_variant,
        pose_csv=_config_path(
            root,
            payload.get(
                "pose_csv",
                delivery_root
                / "image_extrinsics"
                / f"image_camera_to_world_{selected_pose_variant}.csv",
            ),
        ),
        calibration_pkl=_config_path(
            root,
            payload.get("calibration_pkl", delivery_root / "result_interpolation_optimized.pkl"),
        ),
        output_root=(
            Path(output_root).resolve()
            if output_root is not None
            else (
                delivery_root / "colmap-ba"
                if run_output_root is not None
                else _config_path(root, payload.get("output_root", delivery_root / "colmap-ba"))
            )
        ),
        colmap_executable=_executable_path(root, payload["colmap_executable"]),
        feature_max_image_size=int(payload.get("feature_max_image_size", 2400)),
        matching_pair_offsets=tuple(int(value) for value in payload.get("matching_pair_offsets", [1, 2, 5])),
        matching_inter_camera_pairs=tuple(str(value) for value in payload.get("matching_inter_camera_pairs", [])),
        use_gpu=bool(payload.get("use_gpu", True)),
        gpu_index=str(payload.get("gpu_index", "0")),
        run_dense=bool(payload.get("run_dense", False)),
        mvs_backend=mvs_backend,
        openmvs_interface_colmap_executable=_executable_path(
            root,
            payload.get(
                "openmvs_interface_colmap_executable",
                "../tools/openmvs/bin/InterfaceCOLMAP",
            ),
        ),
        openmvs_densify_point_cloud_executable=_executable_path(
            root,
            payload.get(
                "openmvs_densify_point_cloud_executable",
                "../tools/openmvs/bin/DensifyPointCloud",
            ),
        ),
        openmvs_resolution_level=int(payload.get("openmvs_resolution_level", 1)),
        openmvs_max_resolution=int(payload.get("openmvs_max_resolution", 1280)),
        openmvs_min_resolution=int(payload.get("openmvs_min_resolution", 640)),
        openmvs_number_views=int(payload.get("openmvs_number_views", 8)),
        openmvs_number_views_fuse=int(payload.get("openmvs_number_views_fuse", 3)),
        openmvs_gpu_device=_optional_gpu_device(payload.get("openmvs_gpu_device")),
        run_bundle_adjustment=bool(payload.get("run_bundle_adjustment", True)),
        ba_mode=ba_mode,
        absolute_pose_prior_colmap_executable=_executable_path(
            root,
            payload.get("absolute_pose_prior_colmap_executable", payload["colmap_executable"]),
        ),
        ba_position_prior_sigma_m=float(payload.get("ba_position_prior_sigma_m", 0.1)),
        ba_rotation_prior_sigma_deg=float(payload.get("ba_rotation_prior_sigma_deg", 0.5)),
        ba_refine_points3d=bool(payload.get("ba_refine_points3d", True)),
        ba_min_track_len=int(payload.get("ba_min_track_len", 3)),
        ba_max_reproj_error=float(payload.get("ba_max_reproj_error", 4.0)),
        ba_min_tri_angle_deg=float(payload.get("ba_min_tri_angle_deg", 1.5)),
        dense_max_image_size=int(payload.get("dense_max_image_size", 1280)),
        patch_match_max_num_neighbors=int(payload.get("patch_match_max_num_neighbors", 12)),
        patch_match_num_iterations=int(payload.get("patch_match_num_iterations", 4)),
        patch_match_num_samples=int(payload.get("patch_match_num_samples", 10)),
        patch_match_depth_min_m=float(payload.get("patch_match_depth_min_m", 0.1)),
        patch_match_depth_max_m=float(payload.get("patch_match_depth_max_m", 200.0)),
        patch_match_filter=bool(payload.get("patch_match_filter", True)),
        patch_match_write_consistency_graph=bool(
            payload.get("patch_match_write_consistency_graph", True)
        ),
        stereo_fusion_use_cache=bool(payload.get("stereo_fusion_use_cache", True)),
        stereo_fusion_cache_size_gb=float(payload.get("stereo_fusion_cache_size_gb", 16)),
        pose_mask_classes=frozenset(
            canonicalize_label(value)
            for value in payload.get("colmap_pose_mask_classes", DEFAULT_DYNAMIC_CLASSES)
        ),
        dense_mask_classes=frozenset(
            canonicalize_label(value)
            for value in payload.get("colmap_dense_mask_classes", DEFAULT_COLMAP_DENSE_MASK_CLASSES)
        ),
        start_frame=int(payload.get("start_frame", 0)),
        frame_count=(int(payload["frame_count"]) if payload.get("frame_count") is not None else None),
    )


def build_feature_extractor_command(
    *,
    colmap_executable: Path,
    database_path: Path,
    image_root: Path,
    image_list_path: Path,
    camera_spec: dict[str, Any],
    static_valid_mask_root: Path,
    gpu_index: str,
    max_image_size: int,
    use_gpu: bool = True,
) -> list[str]:
    """Build a masked COLMAP feature extraction command.

    ``ImageReader.mask_path`` is deliberately mandatory: a matching database
    without this option is not valid for this reconstruction workflow.
    """
    return [
        str(colmap_executable),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_root),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.camera_model",
        str(camera_spec["model"]),
        "--ImageReader.camera_params",
        ",".join(f"{float(value):.15g}" for value in camera_spec["params"]),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.mask_path",
        str(static_valid_mask_root),
        "--FeatureExtraction.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureExtraction.gpu_index",
        str(gpu_index),
        "--FeatureExtraction.max_image_size",
        str(max_image_size),
    ]


def build_static_valid_masks(
    *,
    semantic_root: Path,
    output_root: Path,
    dynamic_classes: set[str] | frozenset[str] = DEFAULT_DYNAMIC_CLASSES,
    required_relative_names: set[str] | None = None,
    image_root: Path | None = None,
) -> dict[str, int]:
    """Build COLMAP valid-pixel masks from semantic class-index cache files.

    COLMAP uses a zero-valued pixel to suppress feature extraction.  The output
    mirrors the image root: ``<camera>/<image-stem>.png``.
    """
    semantic_root = Path(semantic_root)
    output_root = Path(output_root)
    normalized_dynamic = {canonicalize_label(value) for value in dynamic_classes}
    from .semantic_masks import build_dynamic_mask_from_class_labels
    required = set(required_relative_names) if required_relative_names is not None else None
    # Semantic caching resolves symlinks; keep the delivery image names for COLMAP.
    source_names = {
        (Path(name).parent.as_posix(), (image_root / name).resolve()): name
        for name in (required or ())
    } if image_root is not None else {}
    created = 0
    excluded = 0

    for camera_root in sorted(path for path in semantic_root.iterdir() if path.is_dir()):
        metadata_root = camera_root / "metadata"
        if not metadata_root.is_dir():
            continue
        destination_root = output_root / camera_root.name
        destination_root.mkdir(parents=True, exist_ok=True)
        for metadata_path in sorted(metadata_root.glob("*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata.get("source")
            if not isinstance(source, dict) or not source.get("path"):
                raise ValueError(f"semantic metadata has no source path: {metadata_path}")
            source_filename = Path(str(source["path"]).replace("\\", "/")).name
            image_relative_name = f"{camera_root.name}/{source_filename}"
            if required is not None and image_relative_name not in required:
                image_relative_name = source_names.get(
                    (camera_root.name, Path(str(source["path"])).resolve())
                )
                if image_relative_name is None:
                    continue
                source_filename = Path(image_relative_name).name
            class_mask_value = metadata.get("class_mask_path")
            if not class_mask_value:
                raise ValueError(
                    f"semantic cache entry {metadata_path} is schema v1 and has no class_mask_path; "
                    "regenerate semantic masks with schema v2 before COLMAP reconstruction"
                )
            class_mask_path = Path(str(class_mask_value))
            if not class_mask_path.is_file():
                class_mask_filename = Path(str(class_mask_value).replace("\\", "/")).name
                relocated_class_mask = camera_root / "class_masks" / class_mask_filename
                if relocated_class_mask.is_file():
                    class_mask_path = relocated_class_mask
                else:
                    raise FileNotFoundError(
                        f"semantic class mask listed by {metadata_path} is missing: "
                        f"{class_mask_path} (also checked {relocated_class_mask})"
                    )
            labels = np.asarray(Image.open(class_mask_path))
            if labels.ndim != 2:
                raise ValueError(
                    f"semantic class mask must be single-channel: {class_mask_path}"
                )
            dynamic_mask = build_dynamic_mask_from_class_labels(
                labels,
                {
                    int(class_id): str(class_name)
                    for class_id, class_name in metadata.get(
                        "class_id_to_name", {}
                    ).items()
                },
                dynamic_classes=normalized_dynamic,
                dilation_pixels=0,
            )
            valid_mask = np.where(dynamic_mask > 0, 0, 255).astype(np.uint8)
            destination = destination_root / Path(source_filename).with_suffix(".png").name
            mask_image = Image.fromarray(valid_mask, mode="L")
            if image_root is not None:
                with Image.open(image_root / image_relative_name) as source_image:
                    mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)
            mask_image.save(destination)
            created += 1
            excluded += int(np.count_nonzero(np.asarray(mask_image) == 0))

    if not created:
        raise ValueError(f"no semantic metadata files found below {semantic_root}")
    return {"masks": created, "dynamic_pixels": excluded}


def resize_masks_for_dense_workspace(
    *,
    source_mask_root: Path,
    dense_image_root: Path,
    output_mask_root: Path,
) -> dict[str, int]:
    """Resize static masks to the images written by COLMAP image_undistorter."""
    if output_mask_root.exists():
        shutil.rmtree(output_mask_root)
    output_mask_root.mkdir(parents=True, exist_ok=True)
    masks = 0
    resized = 0
    for image_path in dense_image_root.rglob("*"):
        if not image_path.is_file():
            continue
        relative = image_path.relative_to(dense_image_root)
        source_path = source_mask_root / relative.with_suffix(".png")
        if not source_path.is_file():
            raise FileNotFoundError(f"dense mask missing for image: {source_path}")
        destination = output_mask_root / relative.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as dense_image, Image.open(source_path) as source_mask:
            target_size = dense_image.size
            mask = source_mask.convert("L")
            if mask.size != target_size:
                mask = mask.resize(target_size, Image.Resampling.NEAREST)
                resized += 1
            mask.save(destination)
        masks += 1
    if not masks:
        raise ValueError(f"no dense images found below {dense_image_root}")
    return {"masks": masks, "resized": resized}


def build_stereo_fusion_command(
    *,
    colmap_executable: Path,
    dense_root: Path,
    fused_ply: Path,
    dense_max_image_size: int,
    stereo_fusion_use_cache: bool,
    stereo_fusion_cache_size_gb: float,
    mask_root: Path,
) -> list[str]:
    return [
        str(colmap_executable), "stereo_fusion", "--workspace_path", str(dense_root),
        "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_type", "PLY",
        "--output_path", str(fused_ply), "--StereoFusion.max_image_size", str(dense_max_image_size),
        "--StereoFusion.use_cache", "1" if stereo_fusion_use_cache else "0",
        "--StereoFusion.cache_size", str(stereo_fusion_cache_size_gb),
        "--StereoFusion.mask_path", str(mask_root),
    ]


def _resolve_semantic_model_root(cache_root: Path) -> Path:
    """Accept either a semantic model cache root or the parent cache directory."""
    cache_root = Path(cache_root)
    if (cache_root / "backends").is_dir():
        candidates = [
            child
            for backend in (cache_root / "backends").iterdir()
            if backend.is_dir()
            for child in backend.iterdir()
            if child.is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError(
                "semantic_cache_root contains multiple model caches; pass the selected "
                "backends/<backend>/<cache-key> directory explicitly"
            )
        return candidates[0]
    return cache_root


def _read_pose_csv(csv_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            camera_to_world = np.asarray(
                [[float(row[f"T_cw_{r}{c}"]) for c in range(4)] for r in range(4)],
                dtype=np.float64,
            )
            records.append(
                {
                    "camera": str(row["camera"]),
                    "frame": int(row["frame"]),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "image_filename": str(row["image_filename"]),
                    "image_path": Path(row["image_path"]),
                    "camera_to_world": camera_to_world,
                }
            )
    if not records:
        raise ValueError(f"no pose records found in {csv_path}")
    return records


def _read_delivery_pose_records(config: ColmapReconstructionConfig) -> list[dict[str, Any]]:
    """Read current aggregate CSVs or the historical per-camera layout."""
    if config.pose_csv.is_file():
        return _read_pose_csv(config.pose_csv)
    legacy_root = config.delivery_root / "image_extrinsics" / config.pose_variant
    legacy_paths = sorted(
        legacy_root.glob(f"*_image_camera_to_world_{config.pose_variant}.csv")
    )
    if not legacy_paths:
        raise FileNotFoundError(
            f"delivery pose CSV does not exist: {config.pose_csv}; "
            f"no legacy per-camera CSVs found below {legacy_root}"
        )
    records = [record for path in legacy_paths for record in _read_pose_csv(path)]
    keys = {(str(record["camera"]), int(record["frame"])) for record in records}
    if len(keys) != len(records):
        raise ValueError("delivery pose CSVs contain duplicate camera-frame records")
    return records


def _select_records(
    records: list[dict[str, Any]], *, start_frame: int, frame_count: int | None
) -> list[dict[str, Any]]:
    stop_frame = start_frame + frame_count if frame_count is not None else None
    selected = [
        record
        for record in records
        if int(record["frame"]) >= start_frame
        and (stop_frame is None or int(record["frame"]) < stop_frame)
    ]
    if not selected:
        raise ValueError("the requested frame range contains no delivery pose records")
    return selected


def _relative_image_name(record: dict[str, Any], image_root: Path) -> str:
    try:
        image_path = Path(record["image_path"]).resolve()
        relative = image_path.relative_to(image_root.resolve()).as_posix()
        record["image_path"] = image_path
        return relative
    except ValueError as error:
        camera = str(record.get("camera", "")).strip()
        filename = str(record.get("image_filename", "")).strip()
        relocated = image_root / camera / filename
        if camera and filename and relocated.is_file():
            record["image_path"] = relocated
            return relocated.relative_to(image_root).as_posix()
        raise ValueError(
            f"delivery image path is not below canonical image root: {record['image_path']}"
        ) from error


def _camera_specs(
    calibration_pkl: Path, records: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    with Path(calibration_pkl).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list) or not infos:
        raise ValueError(f"invalid result_interpolation pkl: {calibration_pkl}")
    source_cameras = infos[0]["sensors"]["cams"]
    camera_names = sorted({str(record["camera"]) for record in records})
    camera_ids = {name: index + 1 for index, name in enumerate(camera_names)}
    specs: dict[int, dict[str, Any]] = {}
    for camera_name, camera_id in camera_ids.items():
        sample = next(record for record in records if record["camera"] == camera_name)
        image_path = Path(sample["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"image from delivery pose CSV is missing: {image_path}")
        if camera_name not in source_cameras:
            raise KeyError(f"camera {camera_name} is absent from {calibration_pkl}")
        intrinsic = np.asarray(source_cameras[camera_name]["cam_intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"camera {camera_name} has invalid 3x3 intrinsic matrix")
        with Image.open(image_path) as image:
            width, height = image.size
        specs[camera_id] = {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [
                float(intrinsic[0, 0]), float(intrinsic[1, 1]),
                float(intrinsic[0, 2]), float(intrinsic[1, 2]),
            ],
        }
    return camera_ids, specs


def _build_pairs(
    records: list[dict[str, Any]],
    offsets: tuple[int, ...],
    inter_pairs: tuple[str, ...],
) -> list[tuple[str, str]]:
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_camera.setdefault(str(record["camera"]), []).append(record)
    pairs: set[tuple[str, str]] = set()
    for camera_records in by_camera.values():
        ordered = sorted(camera_records, key=lambda item: int(item["frame"]))
        for index, first in enumerate(ordered):
            for offset in offsets:
                second_index = index + offset
                if offset > 0 and second_index < len(ordered):
                    pairs.add((str(first["name"]), str(ordered[second_index]["name"])))
    for configured_pair in inter_pairs:
        first_camera, separator, second_camera = configured_pair.partition("__")
        if not separator:
            raise ValueError(f"invalid matching_inter_camera_pairs value: {configured_pair}")
        first_by_frame = {int(item["frame"]): item for item in by_camera.get(first_camera, [])}
        second_by_frame = {int(item["frame"]): item for item in by_camera.get(second_camera, [])}
        for frame in sorted(set(first_by_frame) & set(second_by_frame)):
            pairs.add((str(first_by_frame[frame]["name"]), str(second_by_frame[frame]["name"])))
    if not pairs:
        raise ValueError("COLMAP matching configuration generated no image pairs")
    return sorted(pairs)


def _write_pairs_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{first} {second}" for first, second in pairs) + "\n", encoding="utf-8")


def _run_logged_command(command: list[str], *, output_root: Path, stage: str) -> None:
    log_path = output_root / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
        if returncode:
            raise RuntimeError(f"COLMAP stage {stage} exited {returncode}; see {log_path}")
    finally:
        timing_path = output_root / "timing_summary.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timings.setdefault("stages", {})[stage] = {
            "seconds": time.perf_counter() - started,
            "returncode": returncode,
            "command": [str(value) for value in command],
            "log": f"logs/{stage}.log",
        }
        timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


def _export_ba_pose_csv(
    path: Path, records: list[dict[str, Any]], optimized: dict[str, np.ndarray]
) -> None:
    fields = ["camera", "frame", "image_filename"] + [
        f"T_cw_{r}{c}" for r in range(4) for c in range(4)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (str(item["camera"]), int(item["frame"]))):
            pose = optimized[str(record["name"])]
            row = {"camera": record["camera"], "frame": record["frame"],
                   "image_filename": Path(str(record["name"])).name}
            row.update({f"T_cw_{r}{c}": float(pose[r, c]) for r in range(4) for c in range(4)})
            writer.writerow(row)


def _prepare_matching_database(
    *,
    config: ColmapReconstructionConfig,
    records: list[dict[str, Any]],
    camera_ids: dict[str, int],
    camera_specs: dict[int, dict[str, Any]],
    pose_mask_root: Path,
    pairs_path: Path,
    database_path: Path,
) -> dict[str, int]:
    for candidate in (database_path, Path(f"{database_path}-shm"), Path(f"{database_path}-wal")):
        if candidate.exists():
            candidate.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    image_lists_root = database_path.parent / "image_lists"
    image_lists_root.mkdir(parents=True, exist_ok=True)
    for camera_name, camera_id in camera_ids.items():
        names = sorted(str(record["name"]) for record in records if record["camera"] == camera_name)
        image_list_path = image_lists_root / f"{camera_name}.txt"
        image_list_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        _run_logged_command(
            build_feature_extractor_command(
                colmap_executable=config.colmap_executable,
                database_path=database_path,
                image_root=config.image_root,
                image_list_path=image_list_path,
                camera_spec=camera_specs[camera_id],
                static_valid_mask_root=pose_mask_root,
                use_gpu=config.use_gpu,
                gpu_index=config.gpu_index,
                max_image_size=config.feature_max_image_size,
            ),
            output_root=config.output_root, stage=f"features_{camera_name}",
        )
    _run_logged_command(
        [
            str(config.colmap_executable), "matches_importer", "--database_path", str(database_path),
            "--match_list_path", str(pairs_path), "--match_type", "pairs",
            "--FeatureMatching.use_gpu", "1" if config.use_gpu else "0",
            "--FeatureMatching.gpu_index", config.gpu_index,
        ],
        output_root=config.output_root, stage="matching",
    )
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    image_ids = {str(name): int(image_id) for image_id, name, _ in rows}
    expected_names = {str(record["name"]) for record in records}
    if set(image_ids) != expected_names:
        raise RuntimeError("COLMAP database image records differ from the delivery pose CSV")
    configured_camera_ids = dict(camera_ids)
    camera_ids.clear()
    camera_specs_by_database_id: dict[int, dict[str, Any]] = {}
    record_by_name = {str(record["name"]): record for record in records}
    for _, name, database_camera_id in rows:
        camera_name = str(record_by_name[str(name)]["camera"])
        previous = camera_ids.setdefault(camera_name, int(database_camera_id))
        if previous != int(database_camera_id):
            raise RuntimeError(f"COLMAP assigned multiple camera IDs to {camera_name}")
    for camera_name, database_camera_id in camera_ids.items():
        camera_specs_by_database_id[database_camera_id] = camera_specs[
            configured_camera_ids[camera_name]
        ]
    camera_specs.clear()
    camera_specs.update(camera_specs_by_database_id)
    return image_ids


def _rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qvec = np.asarray([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            qvec = np.asarray([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            qvec = np.asarray([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            qvec = np.asarray([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    return qvec / np.linalg.norm(qvec)


def _write_text_model(
    output_root: Path,
    camera_specs: dict[int, dict[str, Any]],
    records: list[dict[str, Any]],
    camera_ids: dict[str, int],
    image_ids: dict[str, int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    camera_lines = []
    for camera_id, spec in sorted(camera_specs.items()):
        params = " ".join(f"{float(value):.15g}" for value in spec["params"])
        camera_lines.append(
            f"{camera_id} {spec['model']} {spec['width']} {spec['height']} {params}"
        )
    (output_root / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    image_lines: list[str] = []
    for record in sorted(records, key=lambda item: image_ids[str(item["name"])]):
        camera_to_world = np.asarray(record["camera_to_world"], dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        qvec = _rotation_to_qvec(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]
        image_lines.extend((
            " ".join([
                str(image_ids[str(record["name"])]),
                *(f"{float(value):.15g}" for value in qvec),
                *(f"{float(value):.15g}" for value in tvec),
                str(camera_ids[str(record["camera"])]),
                str(record["name"]),
            ]),
            "",
        ))
    (output_root / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (output_root / "points3D.txt").write_text("", encoding="utf-8")


def _read_camera_to_world_poses(sparse_root: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    images_bin = Path(sparse_root) / "images.bin"
    if not images_bin.is_file():
        raise FileNotFoundError(f"COLMAP BA did not create {images_bin}")
    with images_bin.open("rb") as handle:
        image_count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(image_count):
            handle.read(4)  # image_id
            qvec = np.asarray(struct.unpack("<4d", handle.read(32)), dtype=np.float64)
            translation = np.asarray(struct.unpack("<3d", handle.read(24)), dtype=np.float64)
            handle.read(4)  # camera_id
            name_bytes = bytearray()
            while True:
                byte = handle.read(1)
                if byte == b"\x00":
                    break
                if not byte:
                    raise RuntimeError(f"truncated COLMAP image name in {images_bin}")
                name_bytes.extend(byte)
            point_count = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(point_count * 24, 1)  # x, y, point3D_id
            qw, qx, qy, qz = qvec
            rotation = np.asarray(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float64,
            )
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = rotation
            world_to_camera[:3, 3] = translation
            poses[name_bytes.decode("utf-8")] = np.linalg.inv(world_to_camera)
    return poses


def _count_colmap_points3d(sparse_root: Path) -> int:
    points_path = Path(sparse_root) / "points3D.bin"
    if not points_path.is_file():
        raise FileNotFoundError(f"COLMAP model has no {points_path}")
    with points_path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise RuntimeError(f"truncated COLMAP points3D header: {points_path}")
    return int(struct.unpack("<Q", header)[0])


def _export_ba_corrections(
    output_path: Path,
    records: list[dict[str, Any]],
    optimized: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["name"])):
        initial = np.asarray(record["camera_to_world"], dtype=np.float64)
        result = optimized[str(record["name"])]
        delta = result @ np.linalg.inv(initial)
        center_delta = result[:3, 3] - initial[:3, 3]
        angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        row: dict[str, Any] = {
            "image": str(record["name"]),
            "camera": str(record["camera"]),
            "frame": int(record["frame"]),
            "camera_center_delta_world_x_m": float(center_delta[0]),
            "camera_center_delta_world_y_m": float(center_delta[1]),
            "camera_center_delta_world_z_m": float(center_delta[2]),
            "camera_center_delta_norm_m": float(np.linalg.norm(center_delta)),
            "relative_rotation_angle_deg": float(angle),
        }
        for row_index in range(4):
            for column_index in range(4):
                row[f"delta_cw_{row_index}{column_index}"] = float(delta[row_index, column_index])
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)


def _build_patch_match_command(
    config: ColmapReconstructionConfig, dense_root: Path, *, geometric: bool
) -> list[str]:
    return [
        str(config.colmap_executable), "patch_match_stereo",
        "--workspace_path", str(dense_root), "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(config.dense_max_image_size),
        "--PatchMatchStereo.gpu_index", config.gpu_index,
        # COLMAP 4.x parses these boolean options as numeric values. Use the
        # attached 0/1 form; geometric mode itself performs an internal
        # photometric seed pass before writing geometric maps.
        f"--PatchMatchStereo.geom_consistency={'1' if geometric else '0'}",
        "--PatchMatchStereo.num_iterations", str(config.patch_match_num_iterations),
        "--PatchMatchStereo.num_samples", str(config.patch_match_num_samples),
        "--PatchMatchStereo.depth_min", str(config.patch_match_depth_min_m),
        "--PatchMatchStereo.depth_max", str(config.patch_match_depth_max_m),
        f"--PatchMatchStereo.filter={'1' if config.patch_match_filter else '0'}",
        "--PatchMatchStereo.write_consistency_graph="
        f"{'1' if geometric and config.patch_match_write_consistency_graph else '0'}",
    ]


def _limit_patch_match_neighbors(config_path: Path, maximum: int) -> None:
    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    if len(lines) % 2:
        raise RuntimeError(f"invalid PatchMatch source config: {config_path}")
    for index in range(1, len(lines), 2):
        if lines[index].strip().startswith("__auto__,"):
            lines[index] = f"__auto__, {maximum}"
    Path(config_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patch_match_sources(
    config_path: Path, pairs_path: Path, *, maximum: int
) -> None:
    """Write explicit source images when no sparse 3D seed exists.

    COLMAP's ``__auto__`` neighbor selection is based on sparse 3D points.
    Fixed-pose reconstruction intentionally has none before MVS, so use the
    verified image-pair graph instead.
    """
    adjacency: dict[str, list[str]] = {}
    for line in Path(pairs_path).read_text(encoding="utf-8").splitlines():
        first, separator, second = line.partition(" ")
        if not separator or not first or not second:
            continue
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    if len(lines) % 2:
        raise RuntimeError(f"invalid PatchMatch source config: {config_path}")
    output: list[str] = []
    for index in range(0, len(lines), 2):
        reference = lines[index].strip()
        sources = adjacency.get(reference, [])[:maximum]
        output.append(reference)
        # PatchMatch parses the second line as a comma-separated CSV list.
        output.append(", ".join(sources))
        output.append("")
    Path(config_path).write_text("\n".join(output), encoding="utf-8")


def _reset_output_dirs(*paths: Path) -> None:
    """Create empty directories required by COLMAP output commands."""
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _build_bundle_adjustment_command(
    config: ColmapReconstructionConfig,
    *,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    shared_options = [
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--BundleAdjustment.refine_focal_length", "0",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "0",
        "--BundleAdjustment.refine_rig_from_world", "1",
        "--BundleAdjustment.refine_sensor_from_rig", "0",
        "--BundleAdjustment.refine_points3D", "1" if config.ba_refine_points3d else "0",
    ]
    if config.ba_mode == "standard":
        return [str(config.colmap_executable), "bundle_adjuster", *shared_options]
    return [
        str(config.absolute_pose_prior_colmap_executable),
        "absolute_pose_prior_bundle_adjuster",
        *shared_options,
        "--pose_csv", str(config.pose_csv),
        "--position_prior_sigma_m", str(config.ba_position_prior_sigma_m),
        "--rotation_prior_sigma_deg", str(config.ba_rotation_prior_sigma_deg),
    ]


def _run_colmap_dense(
    *,
    config: ColmapReconstructionConfig,
    pose_model: Path,
    dense_mask_root: Path,
    database_path: Path,
) -> dict[str, Any]:
    dense_root = config.output_root / "dense"
    if dense_root.exists():
        shutil.rmtree(dense_root)
    undistort_started = time.perf_counter()
    subprocess.run(
        [
            str(config.colmap_executable), "image_undistorter", "--image_path", str(config.image_root),
            "--input_path", str(pose_model), "--output_path", str(dense_root),
            "--output_type", "COLMAP", "--max_image_size", str(config.dense_max_image_size),
        ],
        check=True,
    )
    undistort_seconds = time.perf_counter() - undistort_started
    _limit_patch_match_neighbors(
        dense_root / "stereo" / "patch-match.cfg", config.patch_match_max_num_neighbors
    )
    if not config.run_bundle_adjustment:
        _write_patch_match_sources(
            dense_root / "stereo" / "patch-match.cfg",
            database_path.parent / "pairs.txt",
            maximum=config.patch_match_max_num_neighbors,
        )
    patch_match_started = time.perf_counter()
    for geometric in (False, True):
        subprocess.run(_build_patch_match_command(config, dense_root, geometric=geometric), check=True)
    patch_match_seconds = time.perf_counter() - patch_match_started
    fused_ply = dense_root / "fused.ply"
    dense_workspace_mask_root = config.output_root / "matching" / "dense_valid_masks_undistorted"
    resize_masks_for_dense_workspace(
        source_mask_root=dense_mask_root,
        dense_image_root=dense_root / "images",
        output_mask_root=dense_workspace_mask_root,
    )
    fusion_started = time.perf_counter()
    subprocess.run(
        build_stereo_fusion_command(
            colmap_executable=config.colmap_executable,
            dense_root=dense_root,
            fused_ply=fused_ply,
            dense_max_image_size=config.dense_max_image_size,
            stereo_fusion_use_cache=config.stereo_fusion_use_cache,
            stereo_fusion_cache_size_gb=config.stereo_fusion_cache_size_gb,
            mask_root=dense_workspace_mask_root,
        ),
        check=True,
    )
    fusion_seconds = time.perf_counter() - fusion_started
    if not fused_ply.is_file() or fused_ply.stat().st_size == 0:
        raise RuntimeError(f"COLMAP dense PLY was not created or is empty: {fused_ply}")
    point_count = count_ply_vertices(fused_ply)
    return {
        "mvs_backend": "colmap",
        "fused_ply": str(fused_ply),
        "dense_point_count": point_count,
        "dense_file_size_bytes": fused_ply.stat().st_size,
        "dense_stage_seconds": {
            "undistort": undistort_seconds,
            "patch_match": patch_match_seconds,
            "fusion": fusion_seconds,
        },
    }


def _run_dense_backend(
    *,
    config: ColmapReconstructionConfig,
    pose_model: Path,
    dense_mask_root: Path,
    database_path: Path,
) -> dict[str, Any]:
    if config.mvs_backend == "colmap":
        return _run_colmap_dense(
            config=config,
            pose_model=pose_model,
            dense_mask_root=dense_mask_root,
            database_path=database_path,
        )
    dense_root = config.output_root / "dense"
    if dense_root.exists():
        shutil.rmtree(dense_root)
    return run_openmvs_dense(
        settings=OpenMVSSettings(
            interface_colmap_executable=config.openmvs_interface_colmap_executable,
            densify_point_cloud_executable=config.openmvs_densify_point_cloud_executable,
            resolution_level=config.openmvs_resolution_level,
            max_resolution=config.openmvs_max_resolution,
            min_resolution=config.openmvs_min_resolution,
            number_views=config.openmvs_number_views,
            number_views_fuse=config.openmvs_number_views_fuse,
            gpu_device=config.openmvs_gpu_device,
        ),
        paths=OpenMVSPaths.for_dense_root(dense_root),
        colmap_executable=config.colmap_executable,
        image_root=config.image_root,
        sparse_model=pose_model,
        source_mask_root=dense_mask_root,
        max_image_size=config.dense_max_image_size,
    )
def _run_ba_and_dense(
    *,
    config: ColmapReconstructionConfig,
    records: list[dict[str, Any]],
    camera_ids: dict[str, int],
    camera_specs: dict[int, dict[str, Any]],
    image_ids: dict[str, int],
    dense_mask_root: Path,
    database_path: Path,
) -> dict[str, Any]:
    sparse_input = config.output_root / "sparse_before" / "input"
    sparse_triangulated = config.output_root / "sparse_before" / "triangulated_binary"
    sparse_ba = config.output_root / "sparse_after"
    sparse_filtered = config.output_root / "sparse_before" / "filtered_binary"
    _reset_output_dirs(sparse_input, sparse_triangulated, sparse_filtered, sparse_ba)
    _write_text_model(sparse_input, camera_specs, records, camera_ids, image_ids)
    pose_model = sparse_ba
    if config.run_bundle_adjustment:
        _run_logged_command(
            [
                str(config.colmap_executable), "point_triangulator", "--database_path", str(database_path),
                "--image_path", str(config.image_root), "--input_path", str(sparse_input),
                "--output_path", str(sparse_triangulated), "--clear_points", "1",
                "--refine_intrinsics", "0", "--Mapper.fix_existing_frames", "1",
                "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
            ],
            output_root=config.output_root, stage="triangulation",
        )
        _run_logged_command(
            [
                str(config.colmap_executable), "point_filtering",
                "--input_path", str(sparse_triangulated),
                "--output_path", str(sparse_filtered),
                "--min_track_len", str(config.ba_min_track_len),
                "--max_reproj_error", str(config.ba_max_reproj_error),
                "--min_tri_angle", str(config.ba_min_tri_angle_deg),
            ],
            output_root=config.output_root, stage="point_filtering",
        )
        _run_logged_command(
            _build_bundle_adjustment_command(
                config,
                input_path=sparse_filtered,
                output_path=sparse_ba,
            ),
            output_root=config.output_root, stage="colmap_ba",
        )
    else:
        # Convert the fixed external-pose text model directly. This avoids
        # point_triangulator, whose mapper stage performs an implicit BA.
        _run_logged_command(
            [
                str(config.colmap_executable), "model_converter",
                "--input_path", str(sparse_input),
                "--output_path", str(sparse_ba),
                "--output_type", "BIN",
            ],
            output_root=config.output_root, stage="model_conversion",
        )
    optimized = _read_camera_to_world_poses(pose_model)
    expected = {str(record["name"]) for record in records}
    if set(optimized) != expected:
        raise RuntimeError("COLMAP BA did not produce poses for every selected delivery image")
    correction_csv = config.output_root / "pose_delta.csv"
    _export_ba_corrections(correction_csv, records, optimized)
    pose_csv = config.output_root / "image_camera_to_world_after_ba.csv"
    _export_ba_pose_csv(pose_csv, records, optimized)
    report: dict[str, Any] = {
        "sparse_ba": str(sparse_ba),
        "image_camera_to_world_after_ba": str(pose_csv),
        "ba_pose_corrections": str(correction_csv),
        "bundle_adjustment_enabled": config.run_bundle_adjustment,
        "ba_mode": config.ba_mode if config.run_bundle_adjustment else "disabled",
        "pose_prior": (
            {
                "pose_csv": str(config.pose_csv),
                "position_sigma_m": config.ba_position_prior_sigma_m,
                "rotation_sigma_deg": config.ba_rotation_prior_sigma_deg,
            }
            if config.run_bundle_adjustment and config.ba_mode == "absolute_pose_prior"
            else None
        ),
        "selected_images": len(records),
        "registered_images": len(optimized),
        "sparse_point_count": _count_colmap_points3d(pose_model),
        "points3d_refined": config.ba_refine_points3d,
        "point_filter": {
            "min_track_len": config.ba_min_track_len,
            "max_reproj_error_px": config.ba_max_reproj_error,
            "min_tri_angle_deg": config.ba_min_tri_angle_deg,
        },
    }
    if not config.run_dense:
        return report
    report.update(
        _run_dense_backend(
            config=config,
            pose_model=pose_model,
            dense_mask_root=dense_mask_root,
            database_path=database_path,
        )
    )
    return report


def _validate_reconstruction_executables(config: ColmapReconstructionConfig) -> None:
    if not config.colmap_executable.is_file():
        raise FileNotFoundError(f"COLMAP executable does not exist: {config.colmap_executable}")
    if (
        config.run_bundle_adjustment
        and config.ba_mode == "absolute_pose_prior"
        and not config.absolute_pose_prior_colmap_executable.is_file()
    ):
        raise FileNotFoundError(
            "COLMAP absolute pose-prior executable does not exist: "
            f"{config.absolute_pose_prior_colmap_executable}; build it with "
            "scripts/build_colmap_absolute_pose_prior.sh"
        )
    if config.run_dense and config.mvs_backend == "openmvs":
        if not config.openmvs_interface_colmap_executable.is_file():
            raise FileNotFoundError(
                "OpenMVS InterfaceCOLMAP executable does not exist: "
                f"{config.openmvs_interface_colmap_executable}; build it with "
                "scripts/build_openmvs.sh"
            )
        if not config.openmvs_densify_point_cloud_executable.is_file():
            raise FileNotFoundError(
                "OpenMVS DensifyPointCloud executable does not exist: "
                f"{config.openmvs_densify_point_cloud_executable}; build it with "
                "scripts/build_openmvs.sh"
            )


def run_colmap_reconstruction(
    config_path: Path,
    *,
    run_output_root: Path | None = None,
    pose_variant: str | None = None,
    output_root: Path | None = None,
    run_bundle_adjustment: bool | None = None,
) -> Path:
    """Run static-feature COLMAP matching, external-pose BA, and optional MVS."""
    config = load_colmap_config(
        config_path,
        run_output_root=run_output_root,
        pose_variant=pose_variant,
        output_root=output_root,
    )
    if run_bundle_adjustment is not None:
        config = replace(config, run_bundle_adjustment=run_bundle_adjustment)
    config.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "RUNNING", "bundle_adjustment_enabled": config.run_bundle_adjustment}
    status_path = config.output_root / "STATUS.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    for name in ("report.json", "colmap_reconstruction_report.json", "timing_summary.json"):
        (config.output_root / name).unlink(missing_ok=True)
    try:
        report_path = _run_colmap_reconstruction(config)
        status["status"] = "SUCCESS"
        return report_path
    except Exception as error:
        status.update(status="FAILED", error=str(error))
        raise
    finally:
        timing_path = config.output_root / "timing_summary.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timings.update(total_seconds=time.perf_counter() - started, status=status["status"])
        timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if config.output_root.resolve() == (config.delivery_root / "colmap-ba").resolve():
            update_colmap_delivery_index(config.delivery_root)


def _run_colmap_reconstruction(config: ColmapReconstructionConfig) -> Path:
    _validate_reconstruction_executables(config)
    if not config.calibration_pkl.is_file():
        raise FileNotFoundError(f"calibration pkl does not exist: {config.calibration_pkl}")
    available_records = _read_delivery_pose_records(config)
    records = _select_records(
        available_records,
        start_frame=config.start_frame,
        frame_count=config.frame_count,
    )
    for record in records:
        record["name"] = _relative_image_name(record, config.image_root)
    model_root = _resolve_semantic_model_root(config.semantic_cache_root)
    matching_root = config.output_root / "matching"
    if matching_root.exists():
        shutil.rmtree(matching_root)
    pose_mask_root = matching_root / "pose_valid_masks"
    dense_mask_root = matching_root / "dense_valid_masks"
    pose_mask_summary = build_static_valid_masks(
        semantic_root=model_root,
        output_root=pose_mask_root,
        dynamic_classes=config.pose_mask_classes,
        required_relative_names={str(record["name"]) for record in records},
        image_root=config.image_root,
    )
    dense_mask_summary = build_static_valid_masks(
        semantic_root=model_root,
        output_root=dense_mask_root,
        dynamic_classes=config.dense_mask_classes,
        required_relative_names={str(record["name"]) for record in records},
        image_root=config.image_root,
    )
    expected_masks = [pose_mask_root / Path(str(record["name"])).with_suffix(".png") for record in records]
    expected_dense_masks = [
        dense_mask_root / Path(str(record["name"])).with_suffix(".png") for record in records
    ]
    missing_masks = [path for path in expected_masks + expected_dense_masks if not path.is_file()]
    if missing_masks:
        raise FileNotFoundError(f"static masks missing for selected images: {missing_masks[:3]}")
    camera_ids, camera_specs = _camera_specs(config.calibration_pkl, records)
    pairs = _build_pairs(records, config.matching_pair_offsets, config.matching_inter_camera_pairs)
    pairs_path = matching_root / "pairs.txt"
    _write_pairs_file(pairs_path, pairs)
    database_path = matching_root / "database.db"
    image_ids = _prepare_matching_database(
        config=config,
        records=records,
        camera_ids=camera_ids,
        camera_specs=camera_specs,
        pose_mask_root=pose_mask_root,
        pairs_path=pairs_path,
        database_path=database_path,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    report = _run_ba_and_dense(
        config=config,
        records=records,
        camera_ids=camera_ids,
        camera_specs=camera_specs,
        image_ids=image_ids,
        dense_mask_root=dense_mask_root,
        database_path=database_path,
    )
    report_path = config.output_root / "colmap_reconstruction_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "frames": len({int(record["frame"]) for record in records}),
                "frame_range": [min(int(r["frame"]) for r in records), max(int(r["frame"]) for r in records)],
                "selected_frames": sorted({int(r["frame"]) for r in records}),
                "images": len(records),
                "available_images": len(available_records),
                "coverage": "complete" if len(records) == len(available_records) else "partial",
                "source_delivery_unchanged": True,
                "pose_variant": config.pose_variant,
                "pose_csv": str(config.pose_csv),
                "legacy_per_camera_pose_csvs": not config.pose_csv.is_file(),
                "dynamic_features_masked": True,
                "feature_mask_root": str(pose_mask_root),
                "dense_mask_root": str(dense_mask_root),
                "pose_mask_classes": sorted(config.pose_mask_classes),
                "dense_mask_classes": sorted(config.dense_mask_classes),
                "pose_mask_summary": pose_mask_summary,
                "dense_mask_summary": dense_mask_summary,
                "pair_count": len(pairs),
                "dense_enabled": config.run_dense,
                "bundle_adjustment_enabled": config.run_bundle_adjustment,
                "ba_scope": (
                    "full_se3_pose_prior_and_points"
                    if config.run_bundle_adjustment and config.ba_mode == "absolute_pose_prior"
                    else "image_extrinsics_only"
                    if config.run_bundle_adjustment
                    else "disabled_input_poses_fixed"
                ),
                "intrinsics_optimized": False,
                "sensor_from_rig_optimized": False,
                "points3d_optimized": (
                    config.run_bundle_adjustment and config.ba_refine_points3d
                ),
                "result": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    standard_report = config.output_root / "report.json"
    standard_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (config.output_root / "README.md").write_text(
        "# COLMAP BA 交付结果\n\n"
        "- `sparse_before/filtered_binary/`：BA 前模型；`sparse_after/`：BA 后模型。\n"
        "- `image_camera_to_world_after_ba.csv`：BA 后逐图像 camera-to-world 位姿。\n"
        "- `pose_delta.csv`：相对于输入位姿的改变量。\n"
        "- `report.json`：实际帧范围、图像数量、约束和结果；coverage=partial 表示部分帧验证。\n"
        "- `STATUS.json`、`timing_summary.json`、`logs/`：状态、耗时和命令日志。\n\n"
        "基础 result_interpolation_optimized.pkl 和 image_extrinsics/* 不回写 BA。\n"
        "使用 BA 位姿时读取本目录的 after_ba CSV；未覆盖的帧没有 BA 结果。\n"
        "位置/旋转先验是软约束，sigma 不是最大改动上限。\n",
        encoding="utf-8",
    )
    return standard_report
