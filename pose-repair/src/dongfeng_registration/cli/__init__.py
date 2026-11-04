from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ..config import load_config
from ..colmap_reconstruction import run_colmap_reconstruction
from ..clip_preflight import run_preflight_from_delivery_config
from ..dataset_normalization import (
    normalize_result_interpolation_dataset,
    normalize_ved_localization_dataset,
)
from ..deliverables import (
    DEFAULT_OPTIMIZED_EGO_FIELD,
    export_project_deliverables,
)
from ..delivery_pipeline import (
    _as_path,
    build_delivery_stages,
    load_delivery_config,
    run_delivery_stages,
    select_delivery_stages,
)
from ..extrinsic_optimization import optimize_multi_camera_alternating_from_match_cache
from ..extrinsic_optimization import optimize_multi_camera_extrinsics_from_match_cache
from ..inspection import inspect_dataset
from ..pipeline import run_camera
from ..semantic_masks import (
    DEFAULT_DYNAMIC_CLASSES,
    DEFAULT_MASK2FORMER_MODEL,
    DynamicMaskConfig,
    resolve_model_name,
)
from ..semantic_cache_migration import migrate_semantic_cache_layout
from ..staged_pipeline import (
    match_dataset_stage,
    match_multicamera_dataset_stage,
    refresh_match_cache,
    run_persistent_matching_stage,
    segment_dataset_stage,
)


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("reconstruct-colmap")
def reconstruct_colmap(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_output_root: Path | None = typer.Option(None),
    pose_variant: str | None = typer.Option(None, help="Override configured pose variant for one reconstruction."),
    output_root: Path | None = typer.Option(None, help="Override COLMAP output directory for one reconstruction."),
    skip_ba: bool = typer.Option(False, help="Skip COLMAP bundle adjustment and keep input image poses fixed."),
) -> None:
    """Run masked COLMAP external-pose BA and optional dense reconstruction."""
    report = run_colmap_reconstruction(
        config,
        run_output_root=run_output_root,
        pose_variant=pose_variant,
        output_root=output_root,
        run_bundle_adjustment=False if skip_ba else None,
    )
    console.print(f"COLMAP reconstruction report: {report}")


def _parse_camera_mapping(values: list[str]) -> tuple[list[str], dict[str, str]]:
    cameras: list[str] = []
    camera_name_map: dict[str, str] = {}
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "=" in item:
            target, source = item.split("=", maxsplit=1)
            target = target.strip()
            source = source.strip()
            if not target or not source:
                raise typer.BadParameter(
                    "camera mapping must be formatted as target=source"
                )
            cameras.append(target)
            camera_name_map[target] = source
        else:
            cameras.append(item)
    return cameras, camera_name_map


def _parse_inter_pair_relations(values: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "=" not in item:
            raise typer.BadParameter(
                "inter pair relation must be formatted as pair=relation"
            )
        pair, relation = item.split("=", maxsplit=1)
        pair = pair.strip()
        relation = relation.strip()
        if not pair or not relation:
            raise typer.BadParameter(
                "inter pair relation must be formatted as pair=relation"
            )
        mapping.setdefault(pair, []).append(relation)
    return mapping


def _show_frame(frame: pd.DataFrame) -> None:
    table = Table(show_header=True)
    for column in frame.columns:
        table.add_column(str(column))
    for row in frame.itertuples(index=False):
        table.add_row(*(str(value) for value in row))
    console.print(table)


@app.command()
def inspect(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    settings = load_config(config)
    report = inspect_dataset(settings.dataset_root, settings.cameras)
    console.print("Cameras: " + ", ".join(report["camera"].tolist()))
    _show_frame(report)
    failures = report[
        ~(
            report["frame_count"].eq(600)
            & report["timestamps_monotonic"]
            & report["all_images_exist"]
            & report["image_size_matches"]
            & report["calibration_valid"]
        )
    ]
    console.print(f"{len(report)} camera(s) checked; {len(failures)} failure(s)")
    if len(failures):
        raise typer.Exit(1)


@app.command("normalize-dataset")
def normalize_dataset(
    pose_backend: str = typer.Option("result_interpolation_pkl"),
    dataset_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path = typer.Option(...),
    pkl: Path = typer.Option(Path("result_interpolation.pkl")),
    vehicle_model: str = typer.Option("unknown"),
    image_root: Path | None = typer.Option(None),
    camera: list[str] = typer.Option([], "--camera"),
    extrinsic_direction: str = typer.Option("car_to_camera"),
    navigation_csv: Path | None = typer.Option(None),
) -> None:
    normalized_backend = str(pose_backend).lower()
    if normalized_backend not in {"result_interpolation_pkl", "ved_localization"}:
        raise typer.BadParameter(
            "normalize-dataset supports pose_backend=result_interpolation_pkl "
            "or ved_localization"
        )
    cameras, camera_name_map = _parse_camera_mapping(camera)
    if normalized_backend == "result_interpolation_pkl":
        result = normalize_result_interpolation_dataset(
            dataset_root=dataset_root,
            pkl_path=pkl,
            output_root=output_root,
            vehicle_model=vehicle_model,
            image_root=image_root,
            cameras=tuple(cameras) if cameras else None,
            camera_name_map=camera_name_map,
            extrinsic_direction=extrinsic_direction,
        )
    else:
        result = normalize_ved_localization_dataset(
            dataset_root=dataset_root,
            output_root=output_root,
            vehicle_model=vehicle_model,
            cameras=tuple(cameras) if cameras else None,
            camera_name_map=camera_name_map,
            navigation_csv=navigation_csv,
        )
    console.print(f"Canonical dataset: {result.output_root}")
    console.print(f"Standard pkl: {result.standard_pkl_path}")
    console.print(f"Normalization report: {result.report_path}")


@app.command("export-deliverables")
def export_deliverables(
    output_root: Path = typer.Option(...),
    vehicle_model: str = typer.Option("unknown"),
    optimized_pkl: Path | None = typer.Option(None, exists=True, dir_okay=False),
    standard_pkl: Path | None = typer.Option(None, exists=True, dir_okay=False),
    compensated_perception_dir: Path | None = typer.Option(
        None,
        file_okay=False,
    ),
    optimization_output_dir: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
    ),
    image_extrinsics_dir: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
    ),
    metrics_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    projection_qc_dir: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
    ),
    ego_field: str = typer.Option(DEFAULT_OPTIMIZED_EGO_FIELD),
    remove_compensated_perception_source: bool = typer.Option(
        False,
        help="Remove the source point-cloud directory after a successful export.",
    ),
) -> None:
    try:
        result = export_project_deliverables(
            output_root=output_root,
            vehicle_model=vehicle_model,
            optimized_pkl=optimized_pkl,
            standard_pkl=standard_pkl,
            compensated_perception_dir=compensated_perception_dir,
            optimization_output_dir=optimization_output_dir,
            image_extrinsics_dir=image_extrinsics_dir,
            metrics_dir=metrics_dir,
            projection_qc_dir=projection_qc_dir,
            ego_field=ego_field,
            remove_compensated_perception_source=remove_compensated_perception_source,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Deliverables: {result.output_root}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"README: {result.readme_path}")


@app.command("run-delivery")
def run_delivery(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    from_stage: str | None = typer.Option(None),
    to_stage: str | None = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    try:
        stages = build_delivery_stages(config)
        selected = select_delivery_stages(
            stages,
            from_stage=from_stage,
            to_stage=to_stage,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_delivery_stages(selected, dry_run=dry_run)


@app.command("diagnose-clip-quality")
def diagnose_clip_quality(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Run the sampled LoMA pose-quality preflight before expensive stages."""
    report = run_preflight_from_delivery_config(config)
    console.print(f"Preflight quality: {report.get('overall_status', 'UNKNOWN')}")
    console.print(f"Report: {report.get('output_root', '')}")
    if (
        report.get("overall_status") == "FAIL"
        and report.get("fail_action") != "report_only"
    ):
        raise typer.Exit(code=2)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    camera: str = typer.Option(...),
    start_frame: int = typer.Option(0, min=0),
    frame_count: int | None = typer.Option(None, min=2),
    frame_offset: int = typer.Option(1, min=1),
    run_name: str | None = typer.Option(None),
) -> None:
    settings = load_config(config)
    if camera not in settings.cameras:
        raise typer.BadParameter(f"unknown camera: {camera}")
    output = run_camera(settings, camera, start_frame, frame_count, frame_offset, run_name)
    console.print(f"Output: {output}")


@app.command("validate-real-data")
def validate_real_data(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    skip_full_sequences: bool = typer.Option(False),
) -> None:
    settings = load_config(config)
    inspection = inspect_dataset(settings.dataset_root, settings.cameras)
    inspection_output = settings.output_root / "real-data-validation"
    inspection_output.mkdir(parents=True, exist_ok=True)
    inspection.to_csv(inspection_output / "inspection.csv", index=False)
    for camera in settings.cameras:
        run_camera(settings, camera, 285, 30, 1, f"real-data-validation/window-{camera}")
    if not skip_full_sequences:
        for camera in ("center_camera_fov30", "front_camera_fov195"):
            run_camera(settings, camera, 0, 600, 1, f"real-data-validation/full-{camera}")
    console.print("Real-data validation completed")


@app.command("segment-dataset")
def segment_dataset(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    camera: str = typer.Option(...),
    start_frame: int = typer.Option(0, min=0),
    frame_count: int | None = typer.Option(None, min=1),
    cache_root: Path = typer.Option(Path("data_cache/semantic_masks")),
    backend: str = typer.Option("mask2former"),
    model: str = typer.Option(DEFAULT_MASK2FORMER_MODEL),
    confidence: float | None = typer.Option(None, min=0.0, max=1.0),
    processor_image_size: int | None = typer.Option(None, min=1),
    dilation_pixels: int | None = typer.Option(None, min=0),
    batch_size: int | None = typer.Option(None, min=1),
    write_workers: int | None = typer.Option(None, min=1),
    preview_ratio: float | None = typer.Option(None, min=0.0, max=1.0),
    device: str = typer.Option("cuda:0"),
    source_validation: str = typer.Option("none"),
    image_cache_mode: str = typer.Option("preload"),
    force: bool = typer.Option(False),
) -> None:
    settings = load_config(config)
    semantic = getattr(settings, "semantic_segmentation", None)
    if semantic is None:
        from types import SimpleNamespace

        semantic = SimpleNamespace(
            processor_image_size=1024,
            inference_dtype="fp16",
            use_fast_processor=True,
            confidence=0.25,
            dilation_pixels=5,
            batch_size=4,
            write_workers=1,
            preview_ratio=0.1,
        )
    mask_config = DynamicMaskConfig(
        cache_root=cache_root.resolve(),
        backend=backend,
        model_name=resolve_model_name(backend, model),
        confidence_threshold=(confidence if confidence is not None else semantic.confidence),
        processor_image_size=(processor_image_size if processor_image_size is not None else semantic.processor_image_size),
        inference_dtype=str(getattr(semantic, "inference_dtype", "fp16")),
        use_fast_processor=bool(getattr(semantic, "use_fast_processor", True)),
        dilation_pixels=dilation_pixels if dilation_pixels is not None else semantic.dilation_pixels,
        batch_size=batch_size if batch_size is not None else semantic.batch_size,
        write_workers=write_workers if write_workers is not None else semantic.write_workers,
        preview_ratio=preview_ratio if preview_ratio is not None else semantic.preview_ratio,
        dynamic_classes=DEFAULT_DYNAMIC_CLASSES,
        device=device,
        source_validation=source_validation,
        image_cache_mode=image_cache_mode,
        image_cache_enabled=bool(
            getattr(getattr(settings, "image_cache", None), "enabled", True)
        ),
        image_cache_max_long_side=int(
            getattr(getattr(settings, "image_cache", None), "max_long_side", 1600)
        ),
        image_cache_max_items=getattr(
            getattr(settings, "image_cache", None), "max_items", 1024
        ),
    )
    stage_output = segment_dataset_stage(
        config_path=config,
        dataset_root=settings.dataset_root,
        cache_root=cache_root,
        camera=camera,
        mask_config=mask_config,
        start_frame=start_frame,
        frame_count=frame_count,
        force=force,
    )
    console.print(f"Semantic stage output: {stage_output}")


@app.command("match-dataset")
def match_dataset(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    camera: str = typer.Option(...),
    cache_root: Path = typer.Option(Path("data_cache/semantic_masks")),
    match_cache_root: Path = typer.Option(Path("data_cache/matches")),
    mask_model: str = typer.Option(
        "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    ),
    mask_backend: str = typer.Option("mask2former"),
    mask_confidence: float = typer.Option(0.25),
    mask_image_size: int = typer.Option(2048),
    mask_dilation_pixels: int = typer.Option(5),
    start_frame: int = typer.Option(0, min=0),
    frame_count: int | None = typer.Option(None, min=1),
    frame_offsets: list[int] = typer.Option([], "--frame-offset"),
    target_baselines_m: list[float] = typer.Option([2.0, 4.0, 6.0], "--baseline-m"),
    image_cache_mode: str = typer.Option("preload"),
    preflight_cache_root: Path | None = typer.Option(None),
) -> None:
    settings = load_config(config)
    stage_output = match_dataset_stage(
        config_path=config,
        dataset_root=settings.dataset_root,
        camera=camera,
        cache_root=cache_root,
        match_cache_root=match_cache_root,
        mask_model=mask_model,
        mask_backend=mask_backend,
        mask_confidence=mask_confidence,
        mask_image_size=mask_image_size,
        mask_dilation_pixels=mask_dilation_pixels,
        start_frame=start_frame,
        frame_count=frame_count,
        frame_offsets=tuple(frame_offsets),
        target_baselines_m=tuple(target_baselines_m) if target_baselines_m else None,
        image_cache_mode=image_cache_mode,
        preflight_cache_root=preflight_cache_root,
    )
    console.print(f"Raw matching stage output: {stage_output}")


@app.command("match-multicamera")
def match_multicamera(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    semantic_cache_root: Path = typer.Option(Path("data_cache/semantic_masks")),
    match_cache_root: Path = typer.Option(Path("data_cache/matches")),
    cameras: list[str] = typer.Option(
        [
            "left_front_camera",
            "left_rear_camera",
            "rear_camera",
            "right_front_camera",
            "right_rear_camera",
            "center_camera_fov30",
        ]
    ),
    mask_model: str = typer.Option(
        "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    ),
    mask_backend: str = typer.Option("mask2former"),
    mask_confidence: float = typer.Option(0.25),
    mask_image_size: int = typer.Option(2048),
    mask_dilation_pixels: int = typer.Option(5),
    start_frame: int = typer.Option(0, min=0),
    frame_count: int | None = typer.Option(None, min=1),
    relations: list[str] = typer.Option(["same_index"], "--relation"),
    image_cache_mode: str = typer.Option("preload"),
    preflight_cache_root: Path | None = typer.Option(None),
) -> None:
    supported_relations = {"same_index", "temporal_overlap"}
    unsupported_relations = sorted(set(relations) - supported_relations)
    if unsupported_relations:
        raise typer.BadParameter(
            f"unsupported inter-camera relation(s): {', '.join(unsupported_relations)}; "
            f"supported relations: {', '.join(sorted(supported_relations))}",
            param_hint="--relation",
        )
    settings = load_config(config)
    stage_output = match_multicamera_dataset_stage(
        config_path=config,
        dataset_root=settings.dataset_root,
        cameras=tuple(cameras),
        semantic_cache_root=semantic_cache_root,
        match_cache_root=match_cache_root,
        mask_model=mask_model,
        mask_backend=mask_backend,
        mask_confidence=mask_confidence,
        mask_image_size=mask_image_size,
        mask_dilation_pixels=mask_dilation_pixels,
        start_frame=start_frame,
        frame_count=frame_count,
        relations=tuple(relations),
        image_cache_mode=image_cache_mode,
        preflight_cache_root=preflight_cache_root,
    )
    console.print(f"Inter-camera matching stage output: {stage_output}")


@app.command("match-delivery")
def match_delivery(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Run preflight, semantic masks, and all configured LoMa matching in one process."""
    delivery = load_delivery_config(config.resolve())
    root = Path.cwd().resolve()
    base_config = _as_path(root, delivery.get("base_config"))
    dataset_root = _as_path(root, delivery.get("canonical_root"))
    semantic_cache_root = _as_path(root, delivery.get("semantic_cache_root"))
    match_cache_root = _as_path(root, delivery.get("match_cache_root"))
    if base_config is None or dataset_root is None:
        raise typer.BadParameter("delivery config requires base_config and canonical_root")
    if semantic_cache_root is None or match_cache_root is None:
        raise typer.BadParameter(
            "delivery config requires semantic_cache_root and match_cache_root"
        )
    preflight_cache_root = _as_path(root, delivery.get("preflight_match_cache_root"))

    cameras = tuple(str(item) for item in (delivery.get("cameras") or []))
    inter_pairs = tuple(str(item) for item in (delivery.get("inter_pairs") or []))
    inter_relations = tuple(
        str(item)
        for item in (delivery.get("inter_relations") or ["same_index"])
    )
    relation_mapping = delivery.get("inter_relations_by_pair")
    if relation_mapping is not None and not isinstance(relation_mapping, dict):
        raise typer.BadParameter("inter_relations_by_pair must contain a mapping")

    preflight_report = run_preflight_from_delivery_config(config.resolve())
    console.print(
        f"Preflight quality: {preflight_report.get('overall_status', 'UNKNOWN')}"
    )
    console.print(f"Preflight report: {preflight_report.get('output_root', '')}")
    if (
        preflight_report.get("overall_status") == "FAIL"
        and preflight_report.get("fail_action") != "report_only"
    ):
        raise typer.Exit(code=2)

    settings = load_config(base_config)
    semantic = getattr(settings, "semantic_segmentation", None)
    if semantic is None:
        from types import SimpleNamespace

        semantic = SimpleNamespace(
            processor_image_size=1024,
            inference_dtype="fp16",
            use_fast_processor=True,
            confidence=0.25,
            dilation_pixels=5,
            batch_size=4,
            write_workers=1,
            preview_ratio=0.1,
        )
    mask_backend = str(delivery.get("mask_backend", "mask2former"))
    mask_config = DynamicMaskConfig(
        cache_root=semantic_cache_root.resolve(),
        backend=mask_backend,
        model_name=resolve_model_name(
            mask_backend,
            str(
                delivery.get(
                    "mask_model",
                    "facebook/mask2former-swin-large-mapillary-vistas-semantic",
                )
            ),
        ),
        confidence_threshold=float(
            delivery.get("semantic_confidence", semantic.confidence)
        ),
        processor_image_size=int(
            delivery.get("semantic_processor_image_size", semantic.processor_image_size)
        ),
        inference_dtype=str(
            delivery.get(
                "semantic_inference_dtype",
                getattr(semantic, "inference_dtype", "fp16"),
            )
        ),
        use_fast_processor=bool(
            delivery.get(
                "semantic_use_fast_processor",
                getattr(semantic, "use_fast_processor", True),
            )
        ),
        dilation_pixels=int(
            delivery.get("semantic_dilation_pixels", semantic.dilation_pixels)
        ),
        batch_size=int(
            delivery.get("semantic_segmentation_batch_size", semantic.batch_size)
        ),
        write_workers=int(delivery.get("semantic_write_workers", semantic.write_workers)),
        preview_ratio=float(delivery.get("semantic_preview_ratio", semantic.preview_ratio)),
        dynamic_classes=DEFAULT_DYNAMIC_CLASSES,
        device=str(delivery.get("torch_device", "cuda:0")),
        source_validation="none",
        image_cache_mode=str(delivery.get("image_cache_mode", "preload")),
        image_cache_enabled=bool(
            getattr(getattr(settings, "image_cache", None), "enabled", True)
        ),
        image_cache_max_long_side=int(
            getattr(getattr(settings, "image_cache", None), "max_long_side", 1600)
        ),
        image_cache_max_items=getattr(
            getattr(settings, "image_cache", None), "max_items", 1024
        ),
    )
    for camera in cameras:
        segment_dataset_stage(
            config_path=base_config,
            dataset_root=dataset_root,
            cache_root=semantic_cache_root,
            camera=str(camera),
            mask_config=mask_config,
            start_frame=int(delivery.get("start_frame", 0)),
            frame_count=(
                None
                if delivery.get("frame_count") is None
                else int(delivery["frame_count"])
            ),
        )

    result = run_persistent_matching_stage(
        config_path=base_config,
        dataset_root=dataset_root,
        cameras=cameras,
        semantic_cache_root=semantic_cache_root,
        match_cache_root=match_cache_root,
        intra_baselines_m=tuple(
            float(item) for item in (delivery.get("intra_baselines_m") or [2, 4, 6])
        ),
        inter_pairs=inter_pairs,
        inter_relations=inter_relations,
        inter_relations_by_pair=relation_mapping,
        mask_model=str(
            delivery.get(
                "mask_model",
                "facebook/mask2former-swin-large-mapillary-vistas-semantic",
            )
        ),
        mask_backend=str(delivery.get("mask_backend", "mask2former")),
        mask_confidence=float(delivery.get("semantic_confidence", 0.25)),
        mask_image_size=2048,
        mask_dilation_pixels=int(delivery.get("semantic_dilation_pixels", 5)),
        start_frame=int(delivery.get("start_frame", 0)),
        frame_count=(
            None
            if delivery.get("frame_count") is None
            else int(delivery["frame_count"])
        ),
        image_cache_mode=str(delivery.get("image_cache_mode", "preload")),
        preflight_cache_root=preflight_cache_root,
    )
    console.print(f"Persistent matching stage output: {result}")


@app.command("refresh-match-cache")
def refresh_match_cache_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    match_cache_root: Path = typer.Option(...),
    scope: str = typer.Option("all"),
    start_frame: int = typer.Option(0, min=0),
    frame_count: int | None = typer.Option(None, min=1),
    regenerate_visualizations: bool = typer.Option(
        True, "--regenerate-visualizations/--no-regenerate-visualizations"
    ),
) -> None:
    if scope not in {"all", "intra", "inter"}:
        raise typer.BadParameter("scope must be one of: all, intra, inter")
    settings = load_config(config)
    summary = refresh_match_cache(
        config_path=config,
        dataset_root=settings.dataset_root,
        match_cache_root=match_cache_root,
        scope=scope,
        regenerate_visualizations=regenerate_visualizations,
        start_frame=start_frame,
        frame_count=frame_count,
    )
    console.print(summary)


@app.command("optimize-multicamera-extrinsics")
def optimize_multicamera_extrinsics(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    match_cache_root: list[Path] = typer.Option(..., "--match-cache-root"),
    output: Path = typer.Option(...),
    camera: list[str] = typer.Option([], "--camera"),
    intra_offset: list[int] = typer.Option([], "--intra-offset"),
    intra_baseline_m: list[float] = typer.Option([], "--intra-baseline-m"),
    inter_pair: list[str] = typer.Option([], "--inter-pair"),
    inter_relation: list[str] = typer.Option([], "--inter-relation"),
    inter_pair_relation: list[str] = typer.Option(
        [],
        "--inter-pair-relation",
        help="Pair-specific relation in pair=relation format; repeat for multiple relations.",
    ),
    max_observations: int | None = typer.Option(None, min=1),
    pair_sample_max_matches: int | None = typer.Option(None, min=1),
    pair_sample_seed: int | None = typer.Option(None),
    pair_sample_strategy: str | None = typer.Option(None),
    pair_sample_grid_cols: int | None = typer.Option(None, min=1),
    pair_sample_grid_rows: int | None = typer.Option(None, min=1),
    translation_bound_m: float | None = typer.Option(None, min=0.0),
    rotation_bound_deg: float | None = typer.Option(None, min=0.0),
    fixed_translation_l2_weight: float | None = typer.Option(None, min=0.0),
    fixed_rotation_l2_weight: float | None = typer.Option(None, min=0.0),
    max_nfev: int | None = typer.Option(None, min=1),
    evaluation_max_observations: int | None = typer.Option(None, min=1),
    projection_qc_sample_ratio: float = typer.Option(0.01, min=0.0, max=1.0),
    optimizer_method: str = typer.Option("torch_sparse_graph"),
    torch_device: str = typer.Option("cuda:0"),
    torch_learning_rate: float | None = typer.Option(None, min=0.0),
    torch_lr_schedule: str | None = typer.Option(None),
    torch_lr_step_size: int | None = typer.Option(None, min=1),
    torch_lr_decay: float | None = typer.Option(None, min=0.0),
    torch_lr_min_factor: float | None = typer.Option(None, min=0.0),
    torch_dtype: str = typer.Option("float32"),
    torch_early_stopping_patience: int | None = typer.Option(100, min=1),
    torch_early_stopping_min_delta: float | None = typer.Option(None, min=0.0),
    torch_early_stopping_min_steps: int | None = typer.Option(None, min=0),
    adaptive_weight_mode: str = typer.Option("none"),
    adaptive_weight_alpha: float = typer.Option(0.5, min=0.0),
    adaptive_weight_min: float = typer.Option(0.5, min=0.0),
    adaptive_weight_max: float = typer.Option(2.0, min=0.0),
    adaptive_weight_long_tail_ratio_guard: float = typer.Option(8.0, min=0.0),
    adaptive_weight_min_group_count: int = typer.Option(500, min=1),
    intra_weight: float = typer.Option(1.0, min=0.0),
    inter_weight: float | None = typer.Option(None, min=0.0),
    optimize_time_varying_disturbance: bool | None = typer.Option(None),
    time_varying_disturbance_dof: int | None = typer.Option(None, min=3, max=6),
    optimize_body_time_varying_disturbance: bool | None = typer.Option(None),
    body_time_varying_disturbance_dof: int | None = typer.Option(None, min=3, max=6),
    time_varying_rotation_bound_deg: float | None = typer.Option(None, min=0.0),
    time_varying_translation_bound_m: float | None = typer.Option(None, min=0.0),
    time_varying_translation_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_rotation_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_translation_smooth_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_rotation_smooth_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_translation_mean_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_rotation_mean_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_frame_common_translation_l2_weight: float | None = typer.Option(None, min=0.0),
    time_varying_frame_common_rotation_l2_weight: float | None = typer.Option(None, min=0.0),
    body_time_varying_rotation_bound_deg: float | None = typer.Option(None, min=0.0),
    body_time_varying_translation_bound_m: float | None = typer.Option(None, min=0.0),
    body_time_varying_translation_l2_weight: float | None = typer.Option(None, min=0.0),
    body_time_varying_rotation_l2_weight: float | None = typer.Option(None, min=0.0),
    body_time_varying_translation_smooth_l2_weight: float | None = typer.Option(None, min=0.0),
    body_time_varying_rotation_smooth_l2_weight: float | None = typer.Option(None, min=0.0),
    mode_projection_enabled: bool | None = typer.Option(None),
    mode_projection_interval_steps: int | None = typer.Option(None, min=1),
    mode_projection_absorb_frame_common: bool | None = typer.Option(None),
    mode_projection_absorb_camera_long_term: bool | None = typer.Option(None),
    mode_projection_reset_optimizer_state: bool | None = typer.Option(None),
    absorb_camera_common_mode: bool | None = typer.Option(None),
    camera_common_mode_min_cameras: int | None = typer.Option(None, min=1),
    camera_common_mode_min_median_cos: float | None = typer.Option(None),
    camera_common_mode_min_positive_projection_fraction: float | None = typer.Option(None),
    absorb_camera_long_term_mode: bool | None = typer.Option(None),
    camera_long_term_mode_min_frames: int | None = typer.Option(None, min=1),
    camera_long_term_mode_min_median_cos: float | None = typer.Option(None),
    camera_long_term_mode_min_positive_projection_fraction: float | None = typer.Option(None),
    camera_long_term_mode_min_mean_to_rms: float | None = typer.Option(None, min=0.0),
) -> None:
    settings = load_config(config)
    optimization = settings.optimization
    inter_relations_by_pair = _parse_inter_pair_relations(inter_pair_relation)

    def _optimization_value(name: str, explicit: float | None) -> float:
        return float(explicit if explicit is not None else getattr(optimization, name))

    def _optimization_int(name: str, explicit: int | None) -> int:
        return int(explicit if explicit is not None else getattr(optimization, name))

    def _optimization_bool(name: str, explicit: bool | None) -> bool:
        return bool(explicit if explicit is not None else getattr(optimization, name))

    result = optimize_multi_camera_extrinsics_from_match_cache(
        config_path=config,
        match_cache_root=match_cache_root,
        output=output,
        cameras=list(camera) if camera else list(settings.cameras),
        intra_offsets=list(intra_offset) if intra_offset else None,
        intra_baselines_m=list(intra_baseline_m) if intra_baseline_m else None,
        inter_pairs=list(inter_pair) if inter_pair else None,
        inter_relations=list(inter_relation) if inter_relation else None,
        inter_relations_by_pair=inter_relations_by_pair or None,
        max_observations=max_observations,
        pair_sample_max_matches=(
            pair_sample_max_matches
            if pair_sample_max_matches is not None
            else optimization.pair_sample_max_matches
        ),
        pair_sample_seed=(
            int(pair_sample_seed)
            if pair_sample_seed is not None
            else int(optimization.pair_sample_seed)
        ),
        pair_sample_strategy=(
            str(pair_sample_strategy)
            if pair_sample_strategy is not None
            else str(optimization.pair_sample_strategy)
        ),
        pair_sample_grid_cols=(
            int(pair_sample_grid_cols)
            if pair_sample_grid_cols is not None
            else int(optimization.pair_sample_grid_cols)
        ),
        pair_sample_grid_rows=(
            int(pair_sample_grid_rows)
            if pair_sample_grid_rows is not None
            else int(optimization.pair_sample_grid_rows)
        ),
        translation_bound_m=_optimization_value("translation_bound_m", translation_bound_m),
        rotation_bound_deg=_optimization_value("rotation_bound_deg", rotation_bound_deg),
        fixed_translation_l2_weight=_optimization_value(
            "fixed_translation_l2_weight",
            fixed_translation_l2_weight,
        ),
        fixed_rotation_l2_weight=_optimization_value(
            "fixed_rotation_l2_weight",
            fixed_rotation_l2_weight,
        ),
        max_nfev=_optimization_int("max_nfev", max_nfev),
        evaluation_max_observations=evaluation_max_observations,
        projection_qc_sample_ratio=projection_qc_sample_ratio,
        optimizer_method=optimizer_method,
        torch_device=torch_device,
        torch_learning_rate=_optimization_value("torch_learning_rate", torch_learning_rate),
        torch_lr_schedule=(
            str(torch_lr_schedule)
            if torch_lr_schedule is not None
            else str(optimization.torch_lr_schedule)
        ),
        torch_lr_step_size=_optimization_int("torch_lr_step_size", torch_lr_step_size),
        torch_lr_decay=_optimization_value("torch_lr_decay", torch_lr_decay),
        torch_lr_min_factor=_optimization_value("torch_lr_min_factor", torch_lr_min_factor),
        torch_dtype=torch_dtype,
        torch_early_stopping_patience=torch_early_stopping_patience,
        torch_early_stopping_min_delta=_optimization_value(
            "torch_early_stopping_min_delta",
            torch_early_stopping_min_delta,
        ),
        torch_early_stopping_min_steps=_optimization_int(
            "torch_early_stopping_min_steps",
            torch_early_stopping_min_steps,
        ),
        adaptive_weight_mode=adaptive_weight_mode,
        adaptive_weight_alpha=adaptive_weight_alpha,
        adaptive_weight_min=adaptive_weight_min,
        adaptive_weight_max=adaptive_weight_max,
        adaptive_weight_long_tail_ratio_guard=adaptive_weight_long_tail_ratio_guard,
        adaptive_weight_min_group_count=adaptive_weight_min_group_count,
        intra_weight=intra_weight,
        inter_weight=_optimization_value("inter_weight", inter_weight),
        optimize_time_varying_disturbance=_optimization_bool(
            "optimize_time_varying_disturbance",
            optimize_time_varying_disturbance,
        ),
        time_varying_disturbance_dof=_optimization_int(
            "time_varying_disturbance_dof",
            time_varying_disturbance_dof,
        ),
        optimize_body_time_varying_disturbance=_optimization_bool(
            "optimize_body_time_varying_disturbance",
            optimize_body_time_varying_disturbance,
        ),
        body_time_varying_disturbance_dof=_optimization_int(
            "body_time_varying_disturbance_dof",
            body_time_varying_disturbance_dof,
        ),
        time_varying_rotation_bound_deg=_optimization_value(
            "time_varying_rotation_bound_deg",
            time_varying_rotation_bound_deg,
        ),
        time_varying_translation_bound_m=_optimization_value(
            "time_varying_translation_bound_m",
            time_varying_translation_bound_m,
        ),
        time_varying_translation_l2_weight=_optimization_value(
            "time_varying_translation_l2_weight",
            time_varying_translation_l2_weight,
        ),
        time_varying_rotation_l2_weight=_optimization_value(
            "time_varying_rotation_l2_weight",
            time_varying_rotation_l2_weight,
        ),
        time_varying_translation_smooth_l2_weight=_optimization_value(
            "time_varying_translation_smooth_l2_weight",
            time_varying_translation_smooth_l2_weight,
        ),
        time_varying_rotation_smooth_l2_weight=_optimization_value(
            "time_varying_rotation_smooth_l2_weight",
            time_varying_rotation_smooth_l2_weight,
        ),
        time_varying_translation_mean_l2_weight=_optimization_value(
            "time_varying_translation_mean_l2_weight",
            time_varying_translation_mean_l2_weight,
        ),
        time_varying_rotation_mean_l2_weight=_optimization_value(
            "time_varying_rotation_mean_l2_weight",
            time_varying_rotation_mean_l2_weight,
        ),
        time_varying_frame_common_translation_l2_weight=_optimization_value(
            "time_varying_frame_common_translation_l2_weight",
            time_varying_frame_common_translation_l2_weight,
        ),
        time_varying_frame_common_rotation_l2_weight=_optimization_value(
            "time_varying_frame_common_rotation_l2_weight",
            time_varying_frame_common_rotation_l2_weight,
        ),
        body_time_varying_rotation_bound_deg=_optimization_value(
            "body_time_varying_rotation_bound_deg",
            body_time_varying_rotation_bound_deg,
        ),
        body_time_varying_translation_bound_m=_optimization_value(
            "body_time_varying_translation_bound_m",
            body_time_varying_translation_bound_m,
        ),
        body_time_varying_translation_l2_weight=_optimization_value(
            "body_time_varying_translation_l2_weight",
            body_time_varying_translation_l2_weight,
        ),
        body_time_varying_rotation_l2_weight=_optimization_value(
            "body_time_varying_rotation_l2_weight",
            body_time_varying_rotation_l2_weight,
        ),
        body_time_varying_translation_smooth_l2_weight=_optimization_value(
            "body_time_varying_translation_smooth_l2_weight",
            body_time_varying_translation_smooth_l2_weight,
        ),
        body_time_varying_rotation_smooth_l2_weight=_optimization_value(
            "body_time_varying_rotation_smooth_l2_weight",
            body_time_varying_rotation_smooth_l2_weight,
        ),
        mode_projection_enabled=_optimization_bool(
            "mode_projection_enabled",
            mode_projection_enabled,
        ),
        mode_projection_interval_steps=_optimization_int(
            "mode_projection_interval_steps",
            mode_projection_interval_steps,
        ),
        mode_projection_absorb_frame_common=_optimization_bool(
            "mode_projection_absorb_frame_common",
            mode_projection_absorb_frame_common,
        ),
        mode_projection_absorb_camera_long_term=_optimization_bool(
            "mode_projection_absorb_camera_long_term",
            mode_projection_absorb_camera_long_term,
        ),
        mode_projection_reset_optimizer_state=_optimization_bool(
            "mode_projection_reset_optimizer_state",
            mode_projection_reset_optimizer_state,
        ),
        absorb_camera_common_mode=_optimization_bool(
            "absorb_camera_common_mode",
            absorb_camera_common_mode,
        ),
        camera_common_mode_min_cameras=_optimization_int(
            "camera_common_mode_min_cameras",
            camera_common_mode_min_cameras,
        ),
        camera_common_mode_min_median_cos=_optimization_value(
            "camera_common_mode_min_median_cos",
            camera_common_mode_min_median_cos,
        ),
        camera_common_mode_min_positive_projection_fraction=_optimization_value(
            "camera_common_mode_min_positive_projection_fraction",
            camera_common_mode_min_positive_projection_fraction,
        ),
        absorb_camera_long_term_mode=_optimization_bool(
            "absorb_camera_long_term_mode",
            absorb_camera_long_term_mode,
        ),
        camera_long_term_mode_min_frames=_optimization_int(
            "camera_long_term_mode_min_frames",
            camera_long_term_mode_min_frames,
        ),
        camera_long_term_mode_min_median_cos=_optimization_value(
            "camera_long_term_mode_min_median_cos",
            camera_long_term_mode_min_median_cos,
        ),
        camera_long_term_mode_min_positive_projection_fraction=_optimization_value(
            "camera_long_term_mode_min_positive_projection_fraction",
            camera_long_term_mode_min_positive_projection_fraction,
        ),
        camera_long_term_mode_min_mean_to_rms=_optimization_value(
            "camera_long_term_mode_min_mean_to_rms",
            camera_long_term_mode_min_mean_to_rms,
        ),
    )
    console.print(f"Multi-camera extrinsic optimization output: {result}")


@app.command("optimize-multicamera-alternating")
def optimize_multicamera_alternating(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    match_cache_root: list[Path] = typer.Option(..., "--match-cache-root"),
    output: Path = typer.Option(...),
    camera: list[str] = typer.Option(..., "--camera"),
    intra_offset: list[int] = typer.Option([], "--intra-offset"),
    intra_baseline_m: list[float] = typer.Option([], "--intra-baseline-m"),
    inter_pair: list[str] = typer.Option([], "--inter-pair"),
    inter_relation: list[str] = typer.Option(["same_index"], "--inter-relation"),
    block_rounds: int = typer.Option(1, min=1),
    block_max_nfev: int = typer.Option(500, min=1),
    global_max_nfev: int = typer.Option(1500, min=1),
    max_observations: int | None = typer.Option(None, min=1),
    evaluation_max_observations: int | None = typer.Option(None, min=1),
    torch_device: str = typer.Option("cuda:0"),
    torch_learning_rate: float = typer.Option(0.003, min=0.0),
    torch_lr_schedule: str = typer.Option("step"),
    torch_lr_step_size: int = typer.Option(100, min=1),
    torch_lr_decay: float = typer.Option(0.5, min=0.0),
    torch_lr_min_factor: float = typer.Option(0.1, min=0.0),
    torch_dtype: str = typer.Option("float32"),
    torch_early_stopping_patience: int | None = typer.Option(100, min=1),
    torch_early_stopping_min_delta: float = typer.Option(1e-3, min=0.0),
    torch_early_stopping_min_steps: int = typer.Option(500, min=0),
    time_varying_rotation_bound_deg: float = typer.Option(3.0, min=0.0),
    time_varying_prior_sigma_deg: float = typer.Option(0.3, min=0.0),
    time_varying_prior_weight: float = typer.Option(10.0, min=0.0),
    time_varying_smooth_sigma_deg: float = typer.Option(0.3, min=0.0),
    time_varying_smooth_weight: float = typer.Option(1.0, min=0.0),
    block_intra_weight: float = typer.Option(1.0, min=0.0),
    block_inter_weight: float = typer.Option(0.5, min=0.0),
    run_global_refine: bool = typer.Option(True, "--global-refine/--no-global-refine"),
) -> None:
    result = optimize_multi_camera_alternating_from_match_cache(
        config_path=config,
        match_cache_root=match_cache_root,
        output=output,
        cameras=camera,
        intra_offsets=list(intra_offset) if intra_offset else None,
        intra_baselines_m=list(intra_baseline_m) if intra_baseline_m else None,
        inter_pairs=list(inter_pair) if inter_pair else None,
        inter_relations=list(inter_relation) if inter_relation else None,
        block_rounds=block_rounds,
        block_max_nfev=block_max_nfev,
        global_max_nfev=global_max_nfev,
        max_observations=max_observations,
        evaluation_max_observations=evaluation_max_observations,
        torch_device=torch_device,
        torch_learning_rate=torch_learning_rate,
        torch_lr_schedule=torch_lr_schedule,
        torch_lr_step_size=torch_lr_step_size,
        torch_lr_decay=torch_lr_decay,
        torch_lr_min_factor=torch_lr_min_factor,
        torch_dtype=torch_dtype,
        torch_early_stopping_patience=torch_early_stopping_patience,
        torch_early_stopping_min_delta=torch_early_stopping_min_delta,
        torch_early_stopping_min_steps=torch_early_stopping_min_steps,
        time_varying_rotation_bound_deg=time_varying_rotation_bound_deg,
        time_varying_prior_sigma_deg=time_varying_prior_sigma_deg,
        time_varying_prior_weight=time_varying_prior_weight,
        time_varying_smooth_sigma_deg=time_varying_smooth_sigma_deg,
        time_varying_smooth_weight=time_varying_smooth_weight,
        block_intra_weight=block_intra_weight,
        block_inter_weight=block_inter_weight,
        run_global_refine=run_global_refine,
    )
    console.print(f"Multi-camera alternating optimization output: {result}")


@app.command("migrate-semantic-cache")
def migrate_semantic_cache(
    cache_root: Path = typer.Option(Path("data_cache/semantic_masks")),
    remove_legacy: bool = typer.Option(False, "--remove-legacy/--keep-legacy"),
) -> None:
    summary = migrate_semantic_cache_layout(
        cache_root,
        remove_legacy=remove_legacy,
    )
    console.print(summary)
