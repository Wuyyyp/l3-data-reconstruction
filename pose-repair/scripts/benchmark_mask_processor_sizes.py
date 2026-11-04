from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from dongfeng_registration.semantic_masks import DynamicMaskConfig, create_predictor


ROOT = Path(r"D:\DongFeng\data\M18_canonical\images\front_camera_fov120")
OUTPUT = Path(r"D:\DongFeng\mask_processor_benchmark")
IMAGES = sorted(ROOT.glob("*.jpg"))[:3]


def class_color(name: str) -> tuple[int, int, int]:
    seed = zlib.crc32(name.encode("utf-8"))
    return (
        64 + seed % 160,
        64 + (seed >> 8) % 160,
        64 + (seed >> 16) % 160,
    )


def render_overlay(image: np.ndarray, items) -> np.ndarray:
    overlay = image.copy()
    for item in items:
        mask = item.mask.astype(bool)
        color = np.asarray(class_color(item.class_name), dtype=np.uint8)
        overlay[mask] = color
    return cv2.addWeighted(image, 0.55, overlay, 0.45, 0)


def main() -> None:
    if len(IMAGES) != 3:
        raise RuntimeError(f"expected 3 images, found {len(IMAGES)}")
    results = []
    overlays_by_image: dict[str, list[np.ndarray]] = {path.name: [] for path in IMAGES}
    for processor_size in (384, 512, 768, 1024, 1280):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        config = DynamicMaskConfig(
            cache_root=OUTPUT / f"cache_{processor_size}",
            image_size=2048,
            processor_image_size=processor_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            preview_ratio=1.0,
        )
        predictor = create_predictor(config)
        load_seconds = time.perf_counter() - load_start
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in IMAGES]
        if any(image is None for image in images):
            raise RuntimeError("failed to load benchmark image")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        predictions = predictor.predict_batch(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        output_dir = OUTPUT / f"processor_{processor_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        class_counts = []
        for source_path, image, items in zip(IMAGES, images, predictions):
            labels = []
            for item in items:
                labels.append(item.class_name)
            blended = render_overlay(image, items)
            cv2.imwrite(str(output_dir / source_path.name), blended)
            overlays_by_image[source_path.name].append(blended)
            class_counts.append({"image": source_path.name, "classes": sorted(set(labels))})
        results.append({
            "processor_image_size": processor_size,
            "model_load_seconds": load_seconds,
            "three_image_inference_seconds": inference_seconds,
            "average_inference_seconds_per_image": inference_seconds / len(images),
            "peak_allocated_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
            "peak_reserved_gb": (torch.cuda.max_memory_reserved() / 1024**3) if torch.cuda.is_available() else None,
            "classes": class_counts,
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison_root = OUTPUT / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    for name, overlays in overlays_by_image.items():
        labeled = []
        for size, overlay in zip((384, 512, 768, 1024, 1280), overlays):
            preview = cv2.resize(overlay, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.putText(preview, f"processor {size} x {size}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(preview, f"processor {size} x {size}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
            labeled.append(preview)
        cv2.imwrite(str(comparison_root / name), np.hstack(labeled))
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
