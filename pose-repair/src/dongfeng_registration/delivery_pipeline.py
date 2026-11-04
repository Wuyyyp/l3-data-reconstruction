from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


STAGE_ORDER = (
    "normalize",
    "dynamic-compensation",
    "export-dynamic-pkl",
    "persistent-match",
    "optimize",
    "projection-qc",
    "export-deliverables",
    "colmap-reconstruction",
)


@dataclass(frozen=True)
class DeliveryStage:
    name: str
    command: tuple[str, ...]


def _as_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _append_repeated(args: list[str], option: str, values: list[Any]) -> None:
    for value in values:
        args.extend([option, str(value)])


def _inter_relations_by_pair(
    config: dict[str, Any],
    inter_pairs: list[str],
) -> dict[str, list[str]] | None:
    raw_mapping = config.get("inter_relations_by_pair")
    if raw_mapping is None:
        return None
    if not isinstance(raw_mapping, dict):
        raise ValueError("inter_relations_by_pair must contain a mapping")
    mapping: dict[str, list[str]] = {}
    for pair in inter_pairs:
        if pair not in raw_mapping:
            raise ValueError(
                f"inter_relations_by_pair is missing configured pair: {pair}"
            )
        relations = [str(item) for item in _as_list(raw_mapping[pair])]
        if not relations:
            raise ValueError(f"inter_relations_by_pair has no relations for: {pair}")
        mapping[pair] = relations
    return mapping


def _matching_relation_name(relation: str) -> str:
    """Match generation accepts the temporal relation family, not its cache suffix."""
    if relation.startswith("temporal_overlap_"):
        return "temporal_overlap"
    return relation


def load_delivery_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("delivery config must contain a mapping")
    delivery = payload.get("delivery", payload)
    if not isinstance(delivery, dict):
        raise ValueError("delivery config field must contain a mapping")
    config = dict(delivery)
    dynamic = payload.get("dynamic_compensation", {})
    semantic = payload.get("semantic_segmentation", {})
    if not isinstance(dynamic, dict) or not isinstance(semantic, dict):
        raise ValueError("stage settings must contain mappings")

    # Keep flat delivery keys as a fallback for older configuration files.
    if "workers" in dynamic:
        config["dynamic_workers"] = dynamic["workers"]
    if "batch_size" in semantic:
        config["semantic_segmentation_batch_size"] = semantic["batch_size"]
    if "write_workers" in semantic:
        config["semantic_write_workers"] = semantic["write_workers"]
    for semantic_key, delivery_key in (
        ("processor_image_size", "semantic_processor_image_size"),
        ("inference_dtype", "semantic_inference_dtype"),
        ("use_fast_processor", "semantic_use_fast_processor"),
        ("confidence", "semantic_confidence"),
        ("dilation_pixels", "semantic_dilation_pixels"),
        ("preview_ratio", "semantic_preview_ratio"),
    ):
        if semantic_key in semantic:
            config[delivery_key] = semantic[semantic_key]
    return config


def build_delivery_stages(config_path: Path) -> list[DeliveryStage]:
    config_path = Path(config_path).resolve()
    root = Path.cwd().resolve()
    cfg = load_delivery_config(config_path)

    python = str(cfg.get("python", sys.executable))
    vehicle_model = str(cfg.get("vehicle_model", "unknown"))
    cameras = [str(item) for item in _as_list(cfg.get("cameras"))]
    normalize_cameras = [
        str(item) for item in _as_list(cfg.get("normalize_cameras"))
    ]
    reference_cameras = [str(item) for item in _as_list(cfg.get("reference_cameras"))]
    raw_lidars = [str(item) for item in _as_list(cfg.get("raw_lidars"))]
    intra_baselines = _as_list(cfg.get("intra_baselines_m", [2, 4, 6]))
    inter_pairs = [str(item) for item in _as_list(cfg.get("inter_pairs"))]
    inter_relations = [str(item) for item in _as_list(cfg.get("inter_relations", ["same_index"]))]
    inter_relations_by_pair = _inter_relations_by_pair(
        cfg,
        inter_pairs,
    )

    raw_dataset_root = _as_path(root, cfg.get("raw_dataset_root"))
    normalize_image_root = _as_path(root, cfg.get("normalize_image_root"))
    canonical_root = _as_path(root, cfg.get("canonical_root"))
    base_config = _as_path(root, cfg.get("base_config"))
    run_output_root = _as_path(root, cfg.get("run_output_root"))
    dynamic_output_root = _as_path(root, cfg.get("dynamic_output_root"))
    dynamic_pkl = _as_path(root, cfg.get("dynamic_pkl"))
    semantic_cache_root = _as_path(root, cfg.get("semantic_cache_root"))
    match_cache_root = _as_path(root, cfg.get("match_cache_root"))
    preflight_match_cache_root = _as_path(
        root, cfg.get("preflight_match_cache_root")
    )
    if preflight_match_cache_root is None and run_output_root is not None:
        preflight_match_cache_root = run_output_root / "preflight_match_cache"
    match_cache_key_root = _as_path(root, cfg.get("match_cache_key_root"))
    optimization_output_root = _as_path(root, cfg.get("optimization_output_root"))
    projection_qc_root = _as_path(root, cfg.get("projection_qc_root"))
    deliverable_root = _as_path(root, cfg.get("deliverable_root"))
    colmap_config = _as_path(root, cfg.get("colmap_config"))

    if run_output_root is not None:
        dynamic_output_root = run_output_root / "dynamic_compensation"
        dynamic_pkl = (
            dynamic_output_root / "result_interpolation_dynamic_compensated.pkl"
        )
        optimization_output_root = run_output_root / "joint_optimization"
        projection_qc_root = run_output_root / "quality_control"
        deliverable_root = run_output_root / "deliverables"

    if base_config is None:
        raise ValueError("delivery config requires base_config")
    if canonical_root is None:
        raise ValueError("delivery config requires canonical_root")

    if colmap_config is None:
        base_payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        embedded_colmap = base_payload.get("colmap") if isinstance(base_payload, dict) else None
        if isinstance(embedded_colmap, dict) and bool(embedded_colmap.get("enabled", True)):
            colmap_config = base_config

    raw_pkl = str(cfg.get("raw_pkl", "result_interpolation.pkl"))
    reference_time_offset_s = str(float(cfg.get("reference_time_offset_s", 0.0)))
    frame_count = cfg.get("frame_count")
    start_frame_int = int(cfg.get("start_frame", 0))
    start_frame = str(start_frame_int)
    end_frame = cfg.get("end_frame")
    if end_frame is None:
        end_frame = (
            start_frame_int + int(frame_count)
            if frame_count is not None
            else -1
        )

    stages: list[DeliveryStage] = []

    if raw_dataset_root is not None:
        normalize_args = [
            python,
            "-m",
            "dongfeng_registration.cli",
            "normalize-dataset",
            "--pose-backend",
            str(cfg.get("pose_backend", "result_interpolation_pkl")),
            "--dataset-root",
            str(raw_dataset_root),
            "--output-root",
            str(canonical_root),
            "--vehicle-model",
            vehicle_model,
            "--pkl",
            raw_pkl,
        ]
        _append_repeated(normalize_args, "--camera", normalize_cameras)
        if normalize_image_root is not None:
            normalize_args.extend(["--image-root", str(normalize_image_root)])
        stages.append(
            DeliveryStage(
                "normalize",
                tuple(normalize_args),
            )
        )

    if dynamic_output_root is not None:
        dynamic_pcd_dir = dynamic_output_root / "compensated_perception"
        dynamic_box_dir = dynamic_output_root / "dynamic_3d_boxes_json"
        dynamic_args = [
            python,
            str(Path("scripts") / "compensate_dynamic_raw_lidars.py"),
            "--config",
            str(config_path),
            "--dataset-root",
            str(canonical_root),
            "--pkl",
            str(canonical_root / "result_interpolation_standard.pkl"),
            "--start",
            start_frame,
            "--end",
            str(end_frame),
            "--workers",
            str(int(cfg.get("dynamic_workers", 1))),
            "--output-pcd-dir",
            str(dynamic_pcd_dir),
            "--output-box-dir",
            str(dynamic_box_dir),
            "--output-raw-lidar-dir",
            str(dynamic_output_root / "compensated_raw_lidars"),
            "--output-box-by-lidar-dir",
            str(dynamic_output_root / "dynamic_3d_boxes_by_lidar"),
            "--summary-csv",
            str(dynamic_output_root / "summary.csv"),
            "--timing-json",
            str(dynamic_output_root / "timing.json"),
            "--reference-time-offset-s",
            reference_time_offset_s,
        ]
        if raw_lidars:
            dynamic_args.extend(["--lidars", ",".join(raw_lidars)])
        if reference_cameras:
            dynamic_args.extend(["--reference-cameras", ",".join(reference_cameras)])
        stages.append(DeliveryStage("dynamic-compensation", tuple(dynamic_args)))

        if dynamic_pkl is not None:
            export_dynamic_args = [
                python,
                str(
                    Path("scripts")
                    / "export_dynamic_compensated_result_interpolation_pkl.py"
                ),
                "--input-pkl",
                str(canonical_root / "result_interpolation_standard.pkl"),
                "--box-dir",
                str(dynamic_box_dir),
                "--compensated-pcd-dir",
                str(dynamic_pcd_dir),
                "--output-pkl",
                str(dynamic_pkl),
                "--reference-time-offset-s",
                reference_time_offset_s,
            ]
            if reference_cameras:
                export_dynamic_args.extend(["--reference-cameras", ",".join(reference_cameras)])
            stages.append(DeliveryStage("export-dynamic-pkl", tuple(export_dynamic_args)))

        if projection_qc_root is not None:
            stages.append(
                DeliveryStage(
                    "dynamic-pointcloud-qc",
                    (
                        python,
                        str(Path("scripts") / "visualize_saved_dynamic_compensation_qc.py"),
                        "--dataset-root",
                        str(canonical_root),
                        "--pkl",
                        str(canonical_root / "result_interpolation_standard.pkl"),
                        "--box-dir",
                        str(dynamic_box_dir),
                        "--compensated-pcd-dir",
                        str(dynamic_pcd_dir),
                        "--output-dir",
                        str(projection_qc_root / "dynamic_pointcloud"),
                        "--start",
                        start_frame,
                        "--end",
                        str(end_frame),
                        "--cameras",
                        ",".join(cameras),
                        "--sample-ratio",
                        str(float(cfg.get("dynamic_qc_sample_ratio", 1.0))),
                        "--panel-width",
                        str(int(cfg.get("dynamic_qc_panel_width", 1280))),
                        "--write-workers",
                        str(int(cfg.get("qc_write_workers", 4))),
                        "--jpeg-quality",
                        str(int(cfg.get("qc_jpeg_quality", 90))),
                    ),
                )
            )

    if semantic_cache_root is not None and match_cache_root is not None:
        stages.append(
            DeliveryStage(
                "persistent-match",
                (
                    python,
                    "-m",
                    "dongfeng_registration.cli",
                    "match-delivery",
                    "--config",
                    str(config_path),
                ),
            )
        )

    if optimization_output_root is not None:
        optimize_args = [
            python,
            "-m",
            "dongfeng_registration.cli",
            "optimize-multicamera-extrinsics",
            "--config",
            str(base_config),
            "--match-cache-root",
            str(match_cache_key_root or match_cache_root),
            "--output",
            str(optimization_output_root),
            "--torch-device",
            str(cfg.get("torch_device", "cuda:0")),
            "--projection-qc-sample-ratio",
            str(float(cfg.get("projection_qc_sample_ratio", 0.01))),
        ]
        _append_repeated(optimize_args, "--camera", cameras)
        _append_repeated(optimize_args, "--intra-baseline-m", intra_baselines)
        _append_repeated(optimize_args, "--inter-pair", inter_pairs)
        if inter_relations_by_pair is None:
            _append_repeated(optimize_args, "--inter-relation", inter_relations)
        else:
            relation_bindings = [
                f"{pair}={relation}"
                for pair in inter_pairs
                for relation in inter_relations_by_pair[pair]
            ]
            _append_repeated(
                optimize_args,
                "--inter-pair-relation",
                relation_bindings,
            )
        stages.append(DeliveryStage("optimize", tuple(optimize_args)))

    if projection_qc_root is not None:
        if optimization_output_root is None:
            raise ValueError("projection_qc requires optimization_output_root")
        stages.append(
            DeliveryStage(
                "projection-qc",
                (
                    python,
                    str(
                        Path("scripts")
                        / "generate_reprojection_qc.py"
                    ),
                    "--optimization-output",
                    str(optimization_output_root),
                    "--dataset-root",
                    str(canonical_root),
                    "--output-dir",
                    str(projection_qc_root / "reprojection"),
                    "--sample-ratio",
                    "1.0",
                    "--tile-width",
                    str(int(cfg.get("reprojection_qc_tile_width", 1280))),
                    "--tile-height",
                    str(int(cfg.get("reprojection_qc_tile_height", 720))),
                    "--workers",
                    str(int(cfg.get("qc_write_workers", 4))),
                    "--jpeg-quality",
                    str(int(cfg.get("qc_jpeg_quality", 90))),
                ),
            )
        )

    if deliverable_root is not None:
        optimized_pkl = (
            optimization_output_root / "deliverables" / "result_interpolation_optimized.pkl"
            if optimization_output_root is not None
            else None
        )
        export_args = [
            python,
            "-m",
            "dongfeng_registration.cli",
            "export-deliverables",
            "--output-root",
            str(deliverable_root),
            "--vehicle-model",
            vehicle_model,
        ]
        if optimized_pkl is not None:
            export_args.extend(["--optimized-pkl", str(optimized_pkl)])
        elif dynamic_pkl is not None:
            export_args.extend(["--standard-pkl", str(dynamic_pkl)])
        if dynamic_output_root is not None:
            export_args.extend(
                [
                    "--compensated-perception-dir",
                    str(dynamic_output_root / "compensated_perception"),
                    "--remove-compensated-perception-source",
                ]
            )
        if optimization_output_root is not None:
            export_args.extend(["--optimization-output-dir", str(optimization_output_root)])
        if projection_qc_root is not None:
            export_args.extend(["--projection-qc-dir", str(projection_qc_root)])
        stages.append(DeliveryStage("export-deliverables", tuple(export_args)))

    if colmap_config is not None:
        colmap_args = [
            python,
            "-m",
            "dongfeng_registration.cli",
            "reconstruct-colmap",
            "--config",
            str(colmap_config),
        ]
        if run_output_root is not None:
            colmap_args.extend(["--run-output-root", str(run_output_root)])
        stages.append(
            DeliveryStage(
                "colmap-reconstruction",
                tuple(colmap_args),
            )
        )

    return stages


def select_delivery_stages(
    stages: list[DeliveryStage],
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[DeliveryStage]:
    names = [stage.name for stage in stages]
    if from_stage is not None and from_stage not in STAGE_ORDER:
        raise ValueError(f"unknown from_stage: {from_stage}")
    if to_stage is not None and to_stage not in STAGE_ORDER:
        raise ValueError(f"unknown to_stage: {to_stage}")
    start = 0
    end = len(stages)
    if from_stage is not None:
        start = next((idx for idx, name in enumerate(names) if name == from_stage), end)
    if to_stage is not None:
        end = next((idx for idx in range(len(names) - 1, -1, -1) if names[idx] == to_stage), -1) + 1
    return stages[start:end]


def _run_delivery_stage(index: int, stage: DeliveryStage, *, dry_run: bool) -> None:
    command_text = " ".join(stage.command)
    prefix = f"[{index:02d}] {stage.name}"
    print(f"{prefix}: {command_text}", flush=True)
    if dry_run:
        return
    started = time.perf_counter()
    print(f"{prefix}: started", flush=True)
    try:
        subprocess.run(stage.command, check=True)
    except subprocess.CalledProcessError:
        elapsed = time.perf_counter() - started
        print(f"{prefix}: failed after {elapsed:.1f}s", flush=True)
        raise
    elapsed = time.perf_counter() - started
    print(f"{prefix}: completed in {elapsed:.1f}s", flush=True)


def run_delivery_stages(stages: list[DeliveryStage], *, dry_run: bool = False) -> None:
    for index, stage in enumerate(stages, start=1):
        _run_delivery_stage(index, stage, dry_run=dry_run)
