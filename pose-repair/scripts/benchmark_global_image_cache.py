from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dongfeng_registration.config import load_config
from dongfeng_registration.deep_matchers import (
    _get_loma_backend_bundle,
    _map_points_to_shape,
    _resize_for_deep_with_scale,
    _run_loma_matching,
)
from dongfeng_registration.image_store import ImageStore
from dongfeng_registration.staged_pipeline import (
    _config_for_matching_scope,
    _frames_for_config,
)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _select_pairs(frames, offsets: tuple[int, ...], max_pairs: int) -> list[tuple[object, object]]:
    groups = [
        [
            (frames[index], frames[index + int(offset)])
            for index in range(max(0, len(frames) - int(offset)))
        ]
        for offset in offsets
    ]
    pairs: list[tuple[object, object]] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                pairs.append(group[index])
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _run_once(
    *,
    bundle: dict[str, object],
    pairs: list[tuple[object, object]],
    config,
    image_store: ImageStore,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    _sync_cuda()
    started = time.perf_counter()
    for first_frame, second_frame in pairs:
        first = image_store.get(first_frame.path)
        second = image_store.get(second_frame.path)
        first_resized, first_scale = _resize_for_deep_with_scale(
            first, int(config.deep_image_size)
        )
        second_resized, second_scale = _resize_for_deep_with_scale(
            second, int(config.deep_image_size)
        )
        first_points, second_points = _run_loma_matching(
            bundle,
            first_resized,
            second_resized,
            first_scale,
            second_scale,
            float(config.deep_score_threshold),
            int(config.deep_max_keypoints),
            first_image_key=None,
            second_image_key=None,
            feature_cache=None,
            cache_device="model",
        )
        outputs.append(
            (
                _map_points_to_shape(
                    first_points,
                    first.shape[:2],
                    image_store.original_shape(first_frame.path),
                ),
                _map_points_to_shape(
                    second_points,
                    second.shape[:2],
                    image_store.original_shape(second_frame.path),
                ),
            )
        )
    _sync_cuda()
    return (time.perf_counter() - started) * 1000.0, outputs


def _compare_outputs(
    first: list[tuple[np.ndarray, np.ndarray]],
    second: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[bool, float]:
    if len(first) != len(second):
        return False, float("inf")
    max_delta = 0.0
    for (first_a, first_b), (second_a, second_b) in zip(first, second, strict=True):
        if first_a.shape != second_a.shape or first_b.shape != second_b.shape:
            return False, float("inf")
        if first_a.size:
            max_delta = max(max_delta, float(np.max(np.abs(first_a - second_a))))
        if first_b.size:
            max_delta = max(max_delta, float(np.max(np.abs(first_b - second_b))))
    return max_delta <= 1e-6, max_delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare uncached source-image loading against the 1600-long-side "
            "global CPU image cache during LoMa matching."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = _config_for_matching_scope(load_config(args.config), "intra")
    frames = _frames_for_config(config, args.camera)
    pairs = _select_pairs(frames, tuple(args.offsets), int(args.max_pairs))
    if not pairs:
        raise SystemExit("no image pairs selected")

    bundle = _get_loma_backend_bundle(config.matching.model_device)
    warmup_store = ImageStore.from_paths(
        [pairs[0][0].path, pairs[0][1].path],
        mode="stream",
        target_long_side=None,
        global_cache=False,
    )
    _run_once(
        bundle=bundle,
        pairs=[pairs[0]],
        config=config.matching,
        image_store=warmup_store,
    )

    ImageStore.clear_global_cache()
    uncached_store = ImageStore.from_paths(
        [frame.path for pair in pairs for frame in pair],
        mode="stream",
        target_long_side=None,
        global_cache=False,
    )
    uncached_ms, uncached_outputs = _run_once(
        bundle=bundle,
        pairs=pairs,
        config=config.matching,
        image_store=uncached_store,
    )
    uncached_stats = ImageStore.global_cache_stats()

    ImageStore.clear_global_cache()
    cached_store = ImageStore.from_paths(
        [frame.path for pair in pairs for frame in pair],
        mode="preload",
        target_long_side=int(config.image_cache.max_long_side),
        global_cache=True,
    )
    cached_load_stats = ImageStore.global_cache_stats()
    cached_matching_store = ImageStore.from_paths(
        [frame.path for pair in pairs for frame in pair],
        mode="lazy",
        target_long_side=int(config.image_cache.max_long_side),
        global_cache=True,
    )
    cached_ms, cached_outputs = _run_once(
        bundle=bundle,
        pairs=pairs,
        config=config.matching,
        image_store=cached_matching_store,
    )
    cached_stats = ImageStore.global_cache_stats()
    outputs_equal, max_delta = _compare_outputs(uncached_outputs, cached_outputs)

    result = {
        "config": str(args.config.resolve()),
        "camera": args.camera,
        "pairs": len(pairs),
        "offsets": list(args.offsets),
        "cache_max_long_side": int(config.image_cache.max_long_side),
        "uncached_matching_wall_ms": uncached_ms,
        "cached_matching_wall_ms": cached_ms,
        "matching_wall_time_reduction_fraction": (
            1.0 - cached_ms / uncached_ms if uncached_ms > 0 else 0.0
        ),
        "matching_speedup": uncached_ms / cached_ms if cached_ms > 0 else float("inf"),
        "uncached_image_cache_stats": uncached_stats,
        "cached_preload_stats": cached_load_stats,
        "cached_total_stats": cached_stats,
        "outputs_equal": outputs_equal,
        "max_match_point_delta": max_delta,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
