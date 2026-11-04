from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .semantic_masks import (
    DEFAULT_DYNAMIC_CLASSES,
    DEFAULT_MASK2FORMER_MODEL,
    DynamicMaskConfig,
    resolve_model_name,
)
from .semantic_pipeline import segment_camera


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("segment-dynamic")
def segment_dynamic(
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
    pipeline_config = load_config(config)
    if camera not in pipeline_config.cameras:
        raise typer.BadParameter(f"unknown camera: {camera}")
    mask_config = DynamicMaskConfig(
        cache_root=cache_root.resolve(),
        backend=backend,
        model_name=resolve_model_name(backend, model),
        confidence_threshold=confidence if confidence is not None else pipeline_config.semantic_segmentation.confidence,
        processor_image_size=(processor_image_size if processor_image_size is not None else pipeline_config.semantic_segmentation.processor_image_size),
        inference_dtype=str(pipeline_config.semantic_segmentation.inference_dtype),
        use_fast_processor=bool(pipeline_config.semantic_segmentation.use_fast_processor),
        dilation_pixels=dilation_pixels if dilation_pixels is not None else pipeline_config.semantic_segmentation.dilation_pixels,
        batch_size=batch_size if batch_size is not None else pipeline_config.semantic_segmentation.batch_size,
        write_workers=write_workers if write_workers is not None else pipeline_config.semantic_segmentation.write_workers,
        preview_ratio=preview_ratio if preview_ratio is not None else pipeline_config.semantic_segmentation.preview_ratio,
        dynamic_classes=DEFAULT_DYNAMIC_CLASSES,
        device=device,
        source_validation=source_validation,
        image_cache_mode=image_cache_mode,
        image_cache_enabled=bool(pipeline_config.image_cache.enabled),
        image_cache_max_long_side=int(pipeline_config.image_cache.max_long_side),
        image_cache_max_items=pipeline_config.image_cache.max_items,
    )
    report = segment_camera(
        pipeline_config.dataset_root,
        camera,
        mask_config,
        start_frame,
        frame_count,
        force,
    )
    console.print(
        f"{camera}: {len(report)} frame(s), "
        f"{int(report['cache_hit'].sum())} cache hit(s), "
        f"{int(report['instance_count'].sum())} dynamic instance(s)"
    )


if __name__ == "__main__":
    app()
