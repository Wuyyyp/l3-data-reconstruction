from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .camera_models import CameraCalibration, undistort_normalized
from .config import MatchingConfig


MATCH_COLUMNS = [
    "kp1", "kp2", "x1", "y1", "x2", "y2", "nx1", "ny1", "nx2", "ny2",
    "descriptor_distance", "ratio", "lk_fb_error", "lk_valid", "ransac_inlier",
    "backend", "backend_score",
]


def _prepare(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    if image is None:
        raise ValueError("image could not be read")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, max_dimension / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _static_feature_mask(
    dynamic_mask: np.ndarray | None,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    if dynamic_mask is None:
        return None
    mask = np.asarray(dynamic_mask)
    if mask.shape != target_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def _empty_matches(**diagnostics) -> pd.DataFrame:
    result = pd.DataFrame(columns=MATCH_COLUMNS)
    result.attrs.update(diagnostics)
    return result


def _gradient_energy(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def _corner_response(gray: np.ndarray) -> np.ndarray:
    return cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)


def _candidate_signal_mask(
    keypoints: list[cv2.KeyPoint],
    gray: np.ndarray,
    min_texture_strength: float,
    min_corner_response: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not keypoints:
        empty = np.zeros((0,), dtype=np.float32)
        return np.zeros((0,), dtype=bool), empty, empty
    gradient = _gradient_energy(gray)
    corner = _corner_response(gray)
    points = np.float32([kp.pt for kp in keypoints])
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, gray.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, gray.shape[0] - 1)
    texture_values = gradient[y, x]
    corner_values = corner[y, x]
    valid = (texture_values >= min_texture_strength) & (
        corner_values >= min_corner_response
    )
    return valid, texture_values, corner_values


def _mutual_ratio_matches(first: np.ndarray, second: np.ndarray, threshold: float):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first, second, k=2)
    reverse = matcher.knnMatch(second, first, k=2)
    reverse_good = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < threshold * pair[1].distance
    }
    return [
        (pair[0], pair[0].distance / max(pair[1].distance, 1e-12))
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < threshold * pair[1].distance
        and reverse_good.get(pair[0].trainIdx) == pair[0].queryIdx
    ]


def match_frame_pair(
    first_image,
    second_image,
    calibration: CameraCalibration,
    config: MatchingConfig,
    first_dynamic_mask: np.ndarray | None = None,
    second_dynamic_mask: np.ndarray | None = None,
):
    max_dimension = getattr(config, "max_image_dimension", 1600)
    first_gray, first_scale = _prepare(first_image, max_dimension)
    second_gray, second_scale = _prepare(second_image, max_dimension)
    first_static = _static_feature_mask(first_dynamic_mask, first_gray.shape)
    second_static = _static_feature_mask(second_dynamic_mask, second_gray.shape)
    sift = cv2.SIFT_create(nfeatures=config.max_features)
    first_keypoints, first_descriptors = sift.detectAndCompute(first_gray, first_static)
    second_keypoints, second_descriptors = sift.detectAndCompute(second_gray, second_static)
    diagnostics = {
        "first_keypoints": len(first_keypoints),
        "second_keypoints": len(second_keypoints),
        "mutual_ratio_candidates": 0,
        "pre_lk_matches": 0,
        "lk_valid_before_semantic": 0,
        "lk_failure_count": 0,
        "semantic_destination_rejected": 0,
        "texture_corner_rejected": 0,
        "lk_pre_points1": np.empty((0, 2), dtype=np.float32),
        "lk_pre_points2": np.empty((0, 2), dtype=np.float32),
        "lk_post_points1": np.empty((0, 2), dtype=np.float32),
        "lk_post_points2": np.empty((0, 2), dtype=np.float32),
    }
    if first_descriptors is None or second_descriptors is None:
        return _empty_matches(**diagnostics)
    candidates = _mutual_ratio_matches(first_descriptors, second_descriptors, config.ratio_threshold)
    diagnostics["mutual_ratio_candidates"] = len(candidates)
    if not candidates:
        return _empty_matches(**diagnostics)
    first_candidate_points = np.float32(
        [first_keypoints[m.queryIdx].pt for m, _ in candidates]
    )
    second_candidate_points = np.float32(
        [second_keypoints[m.trainIdx].pt for m, _ in candidates]
    )
    signal_valid, _, _ = _candidate_signal_mask(
        [first_keypoints[m.queryIdx] for m, _ in candidates],
        first_gray,
        config.min_texture_strength,
        config.min_corner_response,
    )
    diagnostics["pre_lk_matches"] = int(signal_valid.sum())
    diagnostics["texture_corner_rejected"] = int(len(candidates) - signal_valid.sum())
    diagnostics["lk_pre_points1"] = first_candidate_points[signal_valid] / first_scale
    diagnostics["lk_pre_points2"] = second_candidate_points[signal_valid] / second_scale
    if not np.any(signal_valid):
        return _empty_matches(**diagnostics)
    filtered_candidates = [item for item, keep in zip(candidates, signal_valid) if keep]
    first_scaled = np.float32(
        [first_keypoints[m.queryIdx].pt for m, _ in filtered_candidates]
    ).reshape(-1, 1, 2)
    predicted, forward_status, _ = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, first_scaled, None)
    if predicted is None or forward_status is None:
        diagnostics["lk_failure_count"] += 1
        return _empty_matches(**diagnostics)
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(second_gray, first_gray, predicted, None)
    if returned is None or backward_status is None:
        diagnostics["lk_failure_count"] += 1
        return _empty_matches(**diagnostics)
    errors = np.linalg.norm(returned.reshape(-1, 2) - first_scaled.reshape(-1, 2), axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (errors <= config.lk_forward_backward_threshold)
    )
    diagnostics["lk_valid_before_semantic"] = int(valid.sum())
    if second_dynamic_mask is not None:
        second_full = np.asarray(second_dynamic_mask)
        predicted_full = predicted.reshape(-1, 2) / second_scale
        x = np.clip(np.rint(predicted_full[:, 0]).astype(int), 0, second_full.shape[1] - 1)
        y = np.clip(np.rint(predicted_full[:, 1]).astype(int), 0, second_full.shape[0] - 1)
        destination_static = second_full[y, x] == 0
        diagnostics["semantic_destination_rejected"] = int(
            np.count_nonzero(valid & ~destination_static)
        )
        valid &= destination_static
    diagnostics["lk_post_points1"] = (
        first_scaled.reshape(-1, 2)[valid] / first_scale if np.any(valid) else np.empty((0, 2), dtype=np.float32)
    )
    diagnostics["lk_post_points2"] = (
        predicted.reshape(-1, 2)[valid] / second_scale if np.any(valid) else np.empty((0, 2), dtype=np.float32)
    )
    records = []
    for index, ((match, ratio), is_valid) in enumerate(zip(filtered_candidates, valid)):
        if not is_valid:
            continue
        p1 = np.asarray(first_keypoints[match.queryIdx].pt) / first_scale
        p2 = predicted[index, 0].astype(np.float64) / second_scale
        records.append({
            "kp1": int(match.queryIdx), "kp2": int(match.trainIdx),
            "x1": float(p1[0]), "y1": float(p1[1]),
            "x2": float(p2[0]), "y2": float(p2[1]),
            "descriptor_distance": float(match.distance), "ratio": float(ratio),
            "lk_fb_error": float(errors[index] / first_scale), "lk_valid": True,
            "backend": "sift_lk", "backend_score": np.nan,
        })
    if not records:
        return _empty_matches(**diagnostics)
    result = pd.DataFrame(records)
    result[["nx1", "ny1"]] = undistort_normalized(result[["x1", "y1"]].to_numpy(), calibration)
    result[["nx2", "ny2"]] = undistort_normalized(result[["x2", "y2"]].to_numpy(), calibration)
    if len(result) >= 8:
        _, mask = cv2.findFundamentalMat(
            result[["x1", "y1"]].to_numpy(), result[["x2", "y2"]].to_numpy(),
            cv2.FM_RANSAC, config.ransac_threshold_pixels, 0.999,
        )
        result["ransac_inlier"] = mask.reshape(-1).astype(bool) if mask is not None else False
    else:
        result["ransac_inlier"] = False
    result = result.reindex(columns=MATCH_COLUMNS)
    result.attrs.update(diagnostics)
    return result
