from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from dongfeng_registration.config import load_config
from dongfeng_registration.staged_pipeline import (
    _config_for_matching_scope,
    _frames_for_config,
)
from dongfeng_registration.deep_matchers import (
    _LOMA_FEATURE_CACHE,
    _LOMA_FEATURE_GPU_CACHE,
    _get_loma_backend_bundle,
    _resize_for_deep_with_scale,
    _run_loma_matching,
)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _run_once(
    *,
    bundle: dict[str, object],
    pairs: list[tuple[Path, Path]],
    config,
    feature_cache,
    cache_device: str = "cpu",
    cache_max_items: int | None = None,
    gpu_cache=None,
    cache_cpu_max_items: int | None = None,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    outputs = []
    _sync_cuda()
    started = time.perf_counter()
    for first_path, second_path in pairs:
        first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
        second = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
        if first is None or second is None:
            raise FileNotFoundError(f"failed to read pair: {first_path}, {second_path}")
        first_resized, first_scale = _resize_for_deep_with_scale(
            first, int(config.deep_image_size)
        )
        second_resized, second_scale = _resize_for_deep_with_scale(
            second, int(config.deep_image_size)
        )
        outputs.append(
            _run_loma_matching(
                bundle,
                first_resized,
                second_resized,
                first_scale,
                second_scale,
                float(config.deep_score_threshold),
                int(config.deep_max_keypoints),
                first_image_key=str(first_path),
                second_image_key=str(second_path),
                feature_cache=feature_cache,
                cache_device=cache_device,
                cache_max_items=cache_max_items,
                gpu_cache=gpu_cache,
                cache_cpu_max_items=cache_cpu_max_items,
            )
        )
    _sync_cuda()
    return (time.perf_counter() - started) * 1000.0, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LoMa image feature reuse.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 5])
    args = parser.parse_args()

    config = _config_for_matching_scope(load_config(args.config), "intra")
    frames = _frames_for_config(config, args.camera)
    candidates = [
        [
            (frames[index].path, frames[index + int(offset)].path)
            for index in range(max(0, len(frames) - int(offset)))
        ]
        for offset in args.offsets
    ]
    pairs: list[tuple[Path, Path]] = []
    for index in range(max((len(items) for items in candidates), default=0)):
        for items in candidates:
            if index < len(items):
                pairs.append(items[index])
                if len(pairs) >= int(args.max_pairs):
                    break
        if len(pairs) >= int(args.max_pairs):
            break
    if not pairs:
        raise SystemExit("no image pairs selected")

    bundle = _get_loma_backend_bundle(config.matching.model_device)
    _LOMA_FEATURE_CACHE.clear()
    baseline_ms, baseline_outputs = _run_once(
        bundle=bundle,
        pairs=pairs,
        config=config.matching,
        feature_cache=None,
    )
    _LOMA_FEATURE_CACHE.clear()
    optimized_ms, optimized_outputs = _run_once(
        bundle=bundle,
        pairs=pairs,
        config=config.matching,
        feature_cache=_LOMA_FEATURE_CACHE,
        cache_device="two_tier",
        cache_max_items=512,
        gpu_cache=_LOMA_FEATURE_GPU_CACHE,
        cache_cpu_max_items=None,
    )
    cpu_cache_items = len(_LOMA_FEATURE_CACHE)
    gpu_cache_items = len(_LOMA_FEATURE_GPU_CACHE)

    _LOMA_FEATURE_CACHE.clear()
    _LOMA_FEATURE_GPU_CACHE.clear()
    model_cache_ms, model_cache_outputs = _run_once(
        bundle=bundle,
        pairs=pairs,
        config=config.matching,
        feature_cache=_LOMA_FEATURE_CACHE,
        cache_device="model",
        cache_max_items=512,
    )

    equal = len(baseline_outputs) == len(optimized_outputs)
    max_point_delta = 0.0
    if equal:
        for (base_first, base_second), (new_first, new_second) in zip(
            baseline_outputs, optimized_outputs, strict=True
        ):
            equal = equal and base_first.shape == new_first.shape
            equal = equal and base_second.shape == new_second.shape
            if base_first.size and new_first.size:
                max_point_delta = max(
                    max_point_delta,
                    float(np.max(np.abs(base_first - new_first))),
                    float(np.max(np.abs(base_second - new_second))),
                )

    model_equal = len(baseline_outputs) == len(model_cache_outputs)
    model_max_point_delta = 0.0
    if model_equal:
        for (base_first, base_second), (new_first, new_second) in zip(
            baseline_outputs, model_cache_outputs, strict=True
        ):
            model_equal = model_equal and base_first.shape == new_first.shape
            model_equal = model_equal and base_second.shape == new_second.shape
            if base_first.size and new_first.size:
                model_max_point_delta = max(
                    model_max_point_delta,
                    float(np.max(np.abs(base_first - new_first))),
                    float(np.max(np.abs(base_second - new_second))),
                )

    speedup = baseline_ms / optimized_ms if optimized_ms > 0 else float("inf")
    print(
        json.dumps(
            {
                "camera": args.camera,
                "pairs": len(pairs),
                "baseline_no_feature_cache_ms": baseline_ms,
                "optimized_feature_cache_ms": optimized_ms,
                "baseline_ms_per_pair": baseline_ms / len(pairs),
                "optimized_ms_per_pair": optimized_ms / len(pairs),
                "speedup": speedup,
                "time_reduction_fraction": 1.0 - optimized_ms / baseline_ms,
                "model_cache_ms": model_cache_ms,
                "model_cache_ms_per_pair": model_cache_ms / len(pairs),
                "model_cache_speedup": baseline_ms / model_cache_ms
                if model_cache_ms > 0
                else float("inf"),
                "cpu_cache_items": cpu_cache_items,
                "gpu_cache_items": gpu_cache_items,
                "model_cache_items": len(_LOMA_FEATURE_CACHE),
                "match_outputs_equal": bool(equal and max_point_delta == 0.0),
                "max_match_point_delta": max_point_delta,
                "model_cache_outputs_equal": bool(
                    model_equal and model_max_point_delta == 0.0
                ),
                "model_cache_max_match_point_delta": model_max_point_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
