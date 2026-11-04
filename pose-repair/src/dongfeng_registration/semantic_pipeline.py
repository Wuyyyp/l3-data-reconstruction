from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .dataset import Dataset
from .image_store import ImageStore, image_store_options
from .semantic_masks import (
    DynamicMaskConfig,
    MaskCache,
    create_predictor,
    generate_cached_masks_batch,
    write_cached_preview,
)


def preview_indices_for_count(frame_count: int, preview_ratio: float) -> set[int]:
    if frame_count <= 0 or preview_ratio <= 0.0:
        return set()
    if preview_ratio >= 1.0:
        return set(range(frame_count))
    preview_count = max(1, int(np.ceil(frame_count * preview_ratio)))
    return {
        int(round(value))
        for value in np.linspace(0, frame_count - 1, num=preview_count)
    }


def segment_camera(
    dataset_root: Path,
    camera: str,
    config: DynamicMaskConfig,
    start_frame: int = 0,
    frame_count: int | None = None,
    force: bool = False,
    frames: list | None = None,
) -> pd.DataFrame:
    frames = list(frames) if frames is not None else Dataset(dataset_root).frames(camera)
    stop = len(frames) if frame_count is None else min(len(frames), start_frame + frame_count)
    selected = frames[start_frame:stop]
    cache = MaskCache(config)
    image_store: ImageStore | None = None
    predictor = None
    records_by_index: dict[int, dict[str, object]] = {}
    pending: list[tuple[int, object]] = []
    missing_previews: list[tuple[object, object]] = []
    preview_indices = preview_indices_for_count(
        len(selected),
        float(config.preview_ratio),
    )

    def process_batch(batch: list[tuple[int, object]]) -> None:
        nonlocal predictor
        if image_store is None:
            raise RuntimeError("semantic image store was not initialized")
        if predictor is None:
            predictor = create_predictor(config)
        batch_started = time.perf_counter()
        items = [
            (frame.path, camera, frame.timestamp_ns)
            for _relative_index, frame in batch
        ]
        write_preview_flags = [
            bool(relative_index in preview_indices)
            for relative_index, _frame in batch
        ]
        results = generate_cached_masks_batch(
            items,
            predictor,
            cache,
            force=True,
            write_preview_flags=write_preview_flags,
            image_store=image_store,
            write_workers=getattr(config, "write_workers", 1),
        )
        elapsed = time.perf_counter() - batch_started
        per_frame_elapsed = elapsed / max(1, len(batch))
        for (relative_index, frame), result in zip(batch, results, strict=True):
            if relative_index not in preview_indices:
                Path(result.preview_path).unlink(missing_ok=True)
            records_by_index[relative_index] = {
                "camera": camera,
                "frame": frame.index,
                "timestamp_ns": frame.timestamp_ns,
                "image_path": str(frame.path),
                "class_mask_path": str(result.class_mask_path),
                "metadata_path": str(result.metadata_path),
                "preview_path": str(result.preview_path),
                "cache_hit": result.cache_hit,
                "instance_count": result.instance_count,
                "dynamic_pixel_ratio": result.dynamic_pixel_ratio,
                "elapsed_seconds": per_frame_elapsed,
            }

    batch_size = max(1, int(config.batch_size))
    for relative_index, frame in enumerate(selected):
        started = time.perf_counter()
        write_preview = bool(relative_index in preview_indices)
        cached = None if force else cache.load_if_valid(
            frame.path,
            camera,
            frame.timestamp_ns,
            config.model_name,
            require_preview=write_preview,
        )
        if cached is not None:
            result = cached
            if write_preview and not result.preview_path.exists():
                missing_previews.append((frame, result))
            if relative_index not in preview_indices:
                Path(result.preview_path).unlink(missing_ok=True)
            records_by_index[relative_index] = {
                "camera": camera,
                "frame": frame.index,
                "timestamp_ns": frame.timestamp_ns,
                "image_path": str(frame.path),
                "class_mask_path": str(result.class_mask_path),
                "metadata_path": str(result.metadata_path),
                "preview_path": str(result.preview_path),
                "cache_hit": result.cache_hit,
                "instance_count": result.instance_count,
                "dynamic_pixel_ratio": result.dynamic_pixel_ratio,
                "elapsed_seconds": time.perf_counter() - started,
            }
            continue
        pending.append((relative_index, frame))
    required_image_paths = [frame.path for _index, frame in pending]
    required_image_paths.extend(frame.path for frame, _result in missing_previews)
    if required_image_paths:
        image_store = ImageStore.from_paths(
            list(dict.fromkeys(required_image_paths)),
            **image_store_options(config),
        )
    for frame, result in missing_previews:
        if image_store is None:
            raise RuntimeError("semantic image store was not initialized")
        write_cached_preview(result, image_store.get(frame.path))
    for start in range(0, len(pending), batch_size):
        process_batch(pending[start : start + batch_size])
    report = pd.DataFrame.from_records(
        [records_by_index[index] for index in range(len(selected))]
    )
    report_root = config.cache_root / "reports" / camera
    report_root.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_root / "segment_report.csv", index=False)
    (report_root / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "cache_root": str(config.cache_root),
                "dynamic_classes": sorted(config.dynamic_classes),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report
