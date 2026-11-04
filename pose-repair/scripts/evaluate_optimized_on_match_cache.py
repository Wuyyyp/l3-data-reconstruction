from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dongfeng_registration.camera_models import CameraCalibration
from dongfeng_registration.config import load_config
from dongfeng_registration.extrinsic_optimization import (
    _build_reprojection_qc_matches,
    _collect_multicamera_match_cache,
    _evaluate_multicamera_matches,
    _navigation_coverage_mask,
    _select_optimization_matches,
    _summarize_multicamera_edges,
    _summarize_match_quality,
    _write_reprojection_outlier_diagnostics,
)
from dongfeng_registration.navigation import NavigationTrajectory, navigation_time_offset_ns
from dongfeng_registration.pkl_pose import (
    calibrations_from_result_interpolation,
    navigation_from_result_interpolation,
    resolve_result_interpolation_path,
)


def _parse_csv_values(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_optimized_yaml(path: Path) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, int], np.ndarray],
    dict[int, np.ndarray],
]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    optimized_sensor_to_car = {
        str(camera): np.asarray(matrix, dtype=np.float64)
        for camera, matrix in payload["optimized_sensor_to_car"].items()
    }
    frame_rotvecs: dict[tuple[str, int], np.ndarray] = {}
    for key, value in payload.get("optimized_frame_rotvecs", {}).items():
        camera, frame_text = str(key).rsplit(":", 1)
        frame_rotvecs[(camera, int(frame_text))] = np.asarray(value, dtype=np.float64)
    body_frame_disturbances = {
        int(frame): np.asarray(value, dtype=np.float64)
        for frame, value in payload.get("optimized_body_frame_disturbances", {}).items()
    }
    return optimized_sensor_to_car, frame_rotvecs, body_frame_disturbances


def evaluate_optimized_on_match_cache(
    *,
    config_path: Path,
    match_cache_root: Path,
    optimized_yaml: Path,
    output: Path,
    cameras: list[str] | None = None,
    inter_pairs: list[str] | None = None,
    inter_relations: list[str] | None = None,
    evaluation_backend: str = "torch",
    evaluation_device: str = "cuda:0",
    evaluation_batch_size: int = 50_000,
) -> Path:
    config = load_config(config_path)
    cameras = cameras or list(config.cameras)
    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)

    result_interpolation_path = resolve_result_interpolation_path(
        config.assumptions,
        config.dataset_root,
    )
    if result_interpolation_path is not None:
        extrinsic_direction = str(
            config.assumptions.get("result_interpolation_extrinsic_direction", "camera_to_car")
        )
        calibrations = calibrations_from_result_interpolation(
            result_interpolation_path,
            tuple(cameras),
            extrinsic_direction=extrinsic_direction,
            images_undistorted=bool(
                config.assumptions.get("result_interpolation_images_undistorted", False)
            ),
        )
        navigation = navigation_from_result_interpolation(
            result_interpolation_path,
            ego_field=str(
                config.assumptions.get(
                    "result_interpolation_ego_field",
                    "ego2global_transformation_matrix",
                )
            ),
            allow_end_extrapolation=bool(
                config.assumptions.get("extrapolate_navigation", False)
            ),
        )
    else:
        calibrations = {
            camera: CameraCalibration.from_dataset(config.dataset_root, camera)
            for camera in cameras
        }
        navigation = NavigationTrajectory.from_csv(
            config.dataset_root / config.assumptions["navigation_source"]
        )
    navigation_offset_ns = navigation_time_offset_ns(config.assumptions)

    matches = _collect_multicamera_match_cache(
        match_cache_root,
        tuple(cameras),
        intra_offsets=None,
        intra_baselines_m=None,
        inter_pairs=None if inter_pairs is None else tuple(inter_pairs),
        inter_relations=None if inter_relations is None else tuple(inter_relations),
    )
    retained = _select_optimization_matches(matches)
    retained = retained.loc[
        _navigation_coverage_mask(retained, navigation, navigation_offset_ns)
    ].reset_index(drop=True)
    if retained.empty:
        raise ValueError("no current-cache matches remain after quality/navigation filtering")

    initial_sensor_to_car = {
        camera: calibration.sensor_to_car for camera, calibration in calibrations.items()
    }
    optimized_sensor_to_car, frame_rotvecs, body_frame_disturbances = _load_optimized_yaml(
        optimized_yaml
    )
    optimized_sensor_to_car = {
        camera: optimized_sensor_to_car[camera] for camera in cameras
    }

    before = _evaluate_multicamera_matches(
        retained,
        navigation,
        calibrations,
        initial_sensor_to_car,
        navigation_offset_ns=navigation_offset_ns,
        evaluation_backend=evaluation_backend,
        evaluation_device=evaluation_device,
        evaluation_batch_size=evaluation_batch_size,
    )
    after = _evaluate_multicamera_matches(
        retained,
        navigation,
        calibrations,
        optimized_sensor_to_car,
        navigation_offset_ns=navigation_offset_ns,
        frame_rotvecs=frame_rotvecs,
        body_frame_disturbances=body_frame_disturbances,
        evaluation_backend=evaluation_backend,
        evaluation_device=evaluation_device,
        evaluation_batch_size=evaluation_batch_size,
    )

    before_summary = _summarize_match_quality(before)
    after_summary = _summarize_match_quality(after)
    outlier_diagnostics = _write_reprojection_outlier_diagnostics(
        output=output,
        before_matches=before,
        after_matches=after,
        hard_threshold_px=100.0,
    )
    pair_summary = _summarize_multicamera_edges(before, after)
    pair_summary.to_csv(output / "metrics" / "camera_pair_summary.csv", index=False)
    qc_matches = _build_reprojection_qc_matches(
        before,
        after,
        sample_ratio=0.01,
    )
    qc_matches.to_parquet(
        output / "observations" / "qc_matches.parquet",
        index=False,
    )
    (output / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "config": str(config_path),
                "match_cache_root": str(match_cache_root),
                "optimized_yaml": str(optimized_yaml),
                "retained_observations": int(len(retained)),
                "before": before_summary,
                "after": after_summary,
                "outlier_diagnostics": outlier_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / "observations" / "qc_matches.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--match-cache-root", type=Path, required=True)
    parser.add_argument("--optimized-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=str, default=None)
    parser.add_argument("--inter-pairs", type=str, default=None)
    parser.add_argument("--inter-relations", type=str, default=None)
    parser.add_argument("--evaluation-backend", type=str, default="torch")
    parser.add_argument("--evaluation-device", type=str, default="cuda:0")
    parser.add_argument("--evaluation-batch-size", type=int, default=50_000)
    args = parser.parse_args()

    qc_path = evaluate_optimized_on_match_cache(
        config_path=args.config,
        match_cache_root=args.match_cache_root,
        optimized_yaml=args.optimized_yaml,
        output=args.output,
        cameras=_parse_csv_values(args.cameras),
        inter_pairs=_parse_csv_values(args.inter_pairs),
        inter_relations=_parse_csv_values(args.inter_relations),
        evaluation_backend=args.evaluation_backend,
        evaluation_device=args.evaluation_device,
        evaluation_batch_size=args.evaluation_batch_size,
    )
    print(f"qc_matches={qc_path}")


if __name__ == "__main__":
    main()
