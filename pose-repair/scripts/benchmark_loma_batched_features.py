from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
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


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _extract_features(
    model: object,
    detector_tensors: list[torch.Tensor],
    descriptor_tensors: list[torch.Tensor],
    *,
    num_keypoints: int,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(detector_tensors), batch_size):
        detector_batch = torch.cat(detector_tensors[start : start + batch_size], dim=0)
        descriptor_batch = torch.cat(
            descriptor_tensors[start : start + batch_size], dim=0
        )
        keypoints = model._detector.detect(  # noqa: SLF001
            {"image": detector_batch},
            num_keypoints=num_keypoints,
        )["keypoints"]
        descriptions = model._descriptor.describe_keypoints(  # noqa: SLF001
            descriptor_batch,
            keypoints,
        )["descriptions"]
        outputs.extend(
            (keypoints[index : index + 1], descriptions[index : index + 1])
            for index in range(keypoints.shape[0])
        )
    return outputs


def _match_indices(
    model: object,
    features: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    threshold: float,
) -> list[np.ndarray]:
    from loma.loma import filter_matches

    pair_matches: list[np.ndarray] = []
    for index in range(0, len(features) - 1, 2):
        keypoints_first, descriptors_first = features[index]
        keypoints_second, descriptors_second = features[index + 1]
        scores = model(
            keypoints_first,
            keypoints_second,
            descriptors_first,
            descriptors_second,
        )["scores"]
        matched_first, _, _, _ = filter_matches(scores, threshold)
        valid = matched_first[0] > -1
        first_indices = torch.arange(
            matched_first.shape[1], device=matched_first.device
        )[valid]
        pair_matches.append(
            torch.stack([first_indices, matched_first[0][valid]], dim=1)
            .cpu()
            .numpy()
        )
    return pair_matches


def _feature_consistency(
    baseline: list[tuple[torch.Tensor, torch.Tensor]],
    candidate: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    if len(baseline) != len(candidate):
        return {"same_shapes": False, "max_keypoint_delta": None, "max_descriptor_delta": None}
    same_shapes = True
    max_keypoint_delta = 0.0
    max_descriptor_delta = 0.0
    for (base_keypoints, base_descriptors), (
        candidate_keypoints,
        candidate_descriptors,
    ) in zip(baseline, candidate, strict=True):
        if (
            base_keypoints.shape != candidate_keypoints.shape
            or base_descriptors.shape != candidate_descriptors.shape
        ):
            same_shapes = False
            continue
        max_keypoint_delta = max(
            max_keypoint_delta,
            float((base_keypoints - candidate_keypoints).abs().max().item()),
        )
        max_descriptor_delta = max(
            max_descriptor_delta,
            float((base_descriptors - candidate_descriptors).abs().max().item()),
        )
    return {
        "same_shapes": same_shapes,
        "max_keypoint_delta": max_keypoint_delta,
        "max_descriptor_delta": max_descriptor_delta,
    }


def _match_consistency(
    baseline: list[np.ndarray],
    candidate: list[np.ndarray],
) -> dict[str, object]:
    if len(baseline) != len(candidate):
        return {"same_pair_count": False, "exact_pair_fraction": 0.0}
    exact = [
        first.shape == second.shape and np.array_equal(first, second)
        for first, second in zip(baseline, candidate, strict=True)
    ]
    return {
        "same_pair_count": True,
        "exact_pair_fraction": float(np.mean(exact)) if exact else 1.0,
        "match_counts": [int(len(matches)) for matches in candidate],
    }


def _matched_pixel_coordinates(
    features: list[tuple[torch.Tensor, torch.Tensor]],
    matches: list[np.ndarray],
    *,
    height: int,
    width: int,
) -> list[np.ndarray]:
    from loma.loma import to_pixel_coords

    coordinates: list[np.ndarray] = []
    for pair_index, pair_matches in enumerate(matches):
        first_keypoints = features[pair_index * 2][0][0]
        second_keypoints = features[pair_index * 2 + 1][0][0]
        if len(pair_matches) == 0:
            coordinates.append(np.empty((0, 4), dtype=np.float32))
            continue
        first = to_pixel_coords(
            first_keypoints[pair_matches[:, 0]], height, width
        ).cpu()
        second = to_pixel_coords(
            second_keypoints[pair_matches[:, 1]], height, width
        ).cpu()
        coordinates.append(torch.cat([first, second], dim=1).numpy())
    return coordinates


def _nearest_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.full((len(first),), np.inf, dtype=np.float64)
    distances: list[np.ndarray] = []
    for start in range(0, len(first), 256):
        chunk = first[start : start + 256].astype(np.float64, copy=False)
        squared = np.sum((chunk[:, None, :] - second[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def _geometric_match_consistency(
    baseline: list[np.ndarray],
    candidate: list[np.ndarray],
) -> dict[str, object]:
    bidirectional_distances: list[np.ndarray] = []
    for base_points, candidate_points in zip(baseline, candidate, strict=True):
        bidirectional_distances.extend(
            [
                _nearest_distances(base_points, candidate_points),
                _nearest_distances(candidate_points, base_points),
            ]
        )
    finite = np.concatenate(bidirectional_distances)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "bidirectional_median_distance_px": None,
            "bidirectional_p95_distance_px": None,
            "fraction_within_1px": 0.0,
            "fraction_within_2px": 0.0,
        }
    return {
        "bidirectional_median_distance_px": float(np.median(finite)),
        "bidirectional_p95_distance_px": float(np.percentile(finite, 95)),
        "fraction_within_1px": float(np.mean(finite <= 1.0)),
        "fraction_within_2px": float(np.mean(finite <= 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark real batched LoMa detector/descriptor inference."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", default="center_camera_fov30")
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = _config_for_matching_scope(load_config(args.config), "intra")
    frames = _frames_for_config(config, args.camera)[: int(args.num_images)]
    if len(frames) < 2:
        raise SystemExit("at least two camera frames are required")

    store = ImageStore.from_paths(
        [frame.path for frame in frames],
        **image_store_options(config),
    )
    images = [store.get(frame.path) for frame in frames]

    preprocess_started = time.perf_counter()
    detector_tensors = [
        _as_loma_detector_tensor_from_cv2_image(image) for image in images
    ]
    descriptor_tensors = [
        _as_loma_descriptor_tensor_from_cv2_image(image) for image in images
    ]
    preprocess_seconds = time.perf_counter() - preprocess_started
    detector_shapes = sorted({tuple(tensor.shape) for tensor in detector_tensors})
    if len(detector_shapes) != 1:
        raise SystemExit(
            f"camera images produced multiple detector shapes: {detector_shapes}"
        )

    model_load_started = time.perf_counter()
    bundle = _get_loma_backend_bundle(config.matching.model_device)
    model_load_seconds = time.perf_counter() - model_load_started
    model = bundle["model"]
    resolved_device = str(bundle["resolved_device"])
    num_keypoints = int(config.matching.deep_max_keypoints)

    results: dict[int, dict[str, object]] = {}
    feature_outputs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for batch_size in [int(value) for value in args.batch_sizes]:
        try:
            warm_count = min(batch_size, len(detector_tensors))
            warm = _extract_features(
                model,
                detector_tensors[:warm_count],
                descriptor_tensors[:warm_count],
                num_keypoints=num_keypoints,
                batch_size=batch_size,
            )
            _sync(resolved_device)
            del warm
            gc.collect()

            timings: list[float] = []
            peak_allocated_bytes = 0
            final_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(int(args.repeats)):
                if resolved_device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(resolved_device)
                    baseline_allocated = torch.cuda.memory_allocated(resolved_device)
                else:
                    baseline_allocated = 0
                _sync(resolved_device)
                started = time.perf_counter()
                outputs = _extract_features(
                    model,
                    detector_tensors,
                    descriptor_tensors,
                    num_keypoints=num_keypoints,
                    batch_size=batch_size,
                )
                _sync(resolved_device)
                timings.append(time.perf_counter() - started)
                if resolved_device.startswith("cuda"):
                    peak_allocated_bytes = max(
                        peak_allocated_bytes,
                        torch.cuda.max_memory_allocated(resolved_device)
                        - baseline_allocated,
                    )
                final_outputs = outputs
            median_seconds = statistics.median(timings)
            feature_outputs[batch_size] = final_outputs
            results[batch_size] = {
                "status": "ok",
                "timings_seconds": timings,
                "median_seconds": median_seconds,
                "milliseconds_per_image": median_seconds * 1000.0 / len(images),
                "peak_incremental_allocated_gb": peak_allocated_bytes / 1024**3,
            }
        except torch.OutOfMemoryError as error:
            results[batch_size] = {"status": "oom", "error": str(error)}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    baseline_batch_size = 1
    if baseline_batch_size not in feature_outputs:
        raise SystemExit("batch size 1 must complete to provide the baseline")
    baseline_features = feature_outputs[baseline_batch_size]
    baseline_matches = _match_indices(
        model,
        baseline_features,
        threshold=float(config.matching.deep_score_threshold),
    )
    image_height, image_width = images[0].shape[:2]
    baseline_match_coordinates = _matched_pixel_coordinates(
        baseline_features,
        baseline_matches,
        height=int(image_height),
        width=int(image_width),
    )
    baseline_seconds = float(results[baseline_batch_size]["median_seconds"])
    for batch_size, outputs in feature_outputs.items():
        candidate_matches = _match_indices(
            model,
            outputs,
            threshold=float(config.matching.deep_score_threshold),
        )
        candidate_match_coordinates = _matched_pixel_coordinates(
            outputs,
            candidate_matches,
            height=int(image_height),
            width=int(image_width),
        )
        candidate_seconds = float(results[batch_size]["median_seconds"])
        results[batch_size].update(
            {
                "speedup_vs_batch1": baseline_seconds / candidate_seconds,
                "time_reduction_fraction": 1.0
                - candidate_seconds / baseline_seconds,
                "feature_consistency": _feature_consistency(
                    baseline_features, outputs
                ),
                "match_consistency": _match_consistency(
                    baseline_matches, candidate_matches
                ),
                "geometric_match_consistency": _geometric_match_consistency(
                    baseline_match_coordinates,
                    candidate_match_coordinates,
                ),
            }
        )

    report = {
        "config": str(args.config.resolve()),
        "camera": args.camera,
        "num_images": len(images),
        "input_shape": list(images[0].shape),
        "detector_tensor_shape": list(detector_tensors[0].shape),
        "descriptor_tensor_shape": list(descriptor_tensors[0].shape),
        "num_keypoints": num_keypoints,
        "requested_device": str(config.matching.model_device),
        "resolved_device": resolved_device,
        "preprocess_seconds": preprocess_seconds,
        "model_load_seconds": model_load_seconds,
        "results": {str(key): value for key, value in results.items()},
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
