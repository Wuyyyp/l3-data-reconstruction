from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from dongfeng_registration.config import load_config
from dongfeng_registration.deep_matchers import (
    _as_loma_descriptor_tensor_from_cv2_image,
    _as_loma_detector_tensor_from_cv2_image,
    _get_loma_backend_bundle,
)
from dongfeng_registration.image_store import ImageStore, image_store_options
from dongfeng_registration.staged_pipeline import (
    _config_for_matching_scope,
    _frames_for_config,
)


def _prepare(image) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _as_loma_detector_tensor_from_cv2_image(image),
        _as_loma_descriptor_tensor_from_cv2_image(image),
    )


def _extract(model, prepared, num_keypoints: int):
    detector_tensor, descriptor_tensor = prepared
    keypoints = model._detector.detect(  # noqa: SLF001
        {"image": detector_tensor}, num_keypoints=num_keypoints
    )["keypoints"]
    descriptions = model._descriptor.describe_keypoints(  # noqa: SLF001
        descriptor_tensor, keypoints
    )["descriptions"]
    return keypoints, descriptions


def _run_inline(model, images, num_keypoints: int):
    return [_extract(model, _prepare(image), num_keypoints) for image in images]


def _run_prefetched(
    model,
    images,
    *,
    num_keypoints: int,
    workers: int,
    prefetch: int,
):
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future] = deque()
        next_index = 0
        while next_index < min(prefetch, len(images)):
            pending.append(executor.submit(_prepare, images[next_index]))
            next_index += 1
        while pending:
            prepared = pending.popleft().result()
            if next_index < len(images):
                pending.append(executor.submit(_prepare, images[next_index]))
                next_index += 1
            outputs.append(_extract(model, prepared, num_keypoints))
    return outputs


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark overlapping LoMa CPU preprocessing with GPU inference."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", default="center_camera_fov30")
    parser.add_argument("--num-images", type=int, default=96)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = _config_for_matching_scope(load_config(args.config), "intra")
    frames = _frames_for_config(config, args.camera)[: int(args.num_images)]
    store = ImageStore.from_paths(
        [frame.path for frame in frames], **image_store_options(config)
    )
    images = [store.get(frame.path) for frame in frames]
    bundle = _get_loma_backend_bundle(config.matching.model_device)
    model = bundle["model"]
    device = str(bundle["resolved_device"])
    num_keypoints = int(config.matching.deep_max_keypoints)

    _extract(model, _prepare(images[0]), num_keypoints)
    _sync(device)

    preprocess_trials = []
    for _ in range(int(args.repeats)):
        started = time.perf_counter()
        prepared = [_prepare(image) for image in images]
        preprocess_trials.append(time.perf_counter() - started)
        del prepared

    configurations = [("inline", 0)] + [
        (f"workers_{int(workers)}", int(workers)) for workers in args.workers
    ]
    results = {}
    reference = None
    for name, workers in configurations:
        timings = []
        final_outputs = None
        for _ in range(int(args.repeats)):
            _sync(device)
            started = time.perf_counter()
            if workers == 0:
                outputs = _run_inline(model, images, num_keypoints)
            else:
                outputs = _run_prefetched(
                    model,
                    images,
                    num_keypoints=num_keypoints,
                    workers=workers,
                    prefetch=int(args.prefetch),
                )
            _sync(device)
            timings.append(time.perf_counter() - started)
            final_outputs = outputs
        if reference is None:
            reference = final_outputs
        exact = all(
            torch.equal(base_keypoints, keypoints)
            and torch.equal(base_descriptions, descriptions)
            for (base_keypoints, base_descriptions), (keypoints, descriptions) in zip(
                reference, final_outputs, strict=True
            )
        )
        results[name] = {
            "workers": workers,
            "timings_seconds": timings,
            "median_seconds": statistics.median(timings),
            "outputs_exact": exact,
        }

    baseline = float(results["inline"]["median_seconds"])
    for result in results.values():
        elapsed = float(result["median_seconds"])
        result["speedup_vs_inline"] = baseline / elapsed
        result["time_reduction_fraction"] = 1.0 - elapsed / baseline

    report = {
        "config": str(args.config.resolve()),
        "camera": args.camera,
        "num_images": len(images),
        "input_shape": list(images[0].shape),
        "device": device,
        "prefetch": int(args.prefetch),
        "preprocess_only_median_seconds": statistics.median(preprocess_trials),
        "preprocess_only_trials_seconds": preprocess_trials,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
