from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import yaml

from ..camera_models import CameraCalibration, project_points
from ..config import PipelineConfig
from ..dataset import Dataset
from ..evaluation import summarize_errors
from ..features import match_frame_pair
from ..geometry import (
    epipolar_errors,
    essential_from_relative_pose,
    fundamental_from_essential,
    ray_geometry_metrics,
    triangulate_and_reproject,
)
from ..navigation import NavigationTrajectory
from ..tracks import build_tracks
from ..visualization import (
    draw_epipolar_overlay,
    draw_match_overlay,
    draw_reprojection_overlay,
    plot_error_over_time,
)


def _write_run_metadata(config: PipelineConfig, output: Path) -> None:
    resolved = config.model_dump(mode="json")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output / "assumptions.json").write_text(
        json.dumps(config.assumptions, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _scaled_visual(image: np.ndarray, points: np.ndarray, maximum: int = 1600):
    scale = min(1.0, maximum / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, np.asarray(points) * scale


def _evaluate_navigation(
    matches,
    first_timestamp,
    second_timestamp,
    calibration,
    navigation,
    navigation_time_offset_ns: int = 0,
):
    result = matches.copy()
    result["navigation_valid"] = False
    numeric_columns = (
        "nav_point_to_epiline_px",
        "nav_symmetric_epipolar_px",
        "nav_sampson_px2",
        "normalized_algebraic_residual",
        "reprojection1_px",
        "reprojection2_px",
        "reprojection_rms_px",
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "triangulated_point_distance_m",
        "triangulation_angle_deg",
        "ray_closest_distance_m",
    )
    for column in numeric_columns:
        result[column] = np.nan
    result["triangulation_valid_depth"] = False
    try:
        first_body = navigation.pose_at(int(first_timestamp) + int(navigation_time_offset_ns))
        second_body = navigation.pose_at(int(second_timestamp) + int(navigation_time_offset_ns))
    except ValueError:
        return result, None
    first_camera_to_world = first_body @ calibration.sensor_to_car
    second_camera_to_world = second_body @ calibration.sensor_to_car
    first_to_second = np.linalg.inv(second_camera_to_world) @ first_camera_to_world
    essential = essential_from_relative_pose(first_to_second)
    fundamental = fundamental_from_essential(
        essential, calibration.intrinsic, calibration.intrinsic
    )
    pixel_errors = epipolar_errors(
        result[["x1", "y1"]].to_numpy(),
        result[["x2", "y2"]].to_numpy(),
        fundamental,
    )
    normalized_errors = epipolar_errors(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        essential,
    )
    first_world_to_camera = np.linalg.inv(first_camera_to_world)
    second_world_to_camera = np.linalg.inv(second_camera_to_world)
    triangulation = triangulate_and_reproject(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        first_world_to_camera,
        second_world_to_camera,
    )
    ray_metrics = ray_geometry_metrics(
        result[["nx1", "ny1"]].to_numpy(),
        result[["nx2", "ny2"]].to_numpy(),
        first_camera_to_world,
        second_camera_to_world,
    )
    first_camera_points = triangulation["first_camera_points"]
    second_camera_points = triangulation["second_camera_points"]
    first_pixels = project_points(first_camera_points, calibration)
    second_pixels = project_points(second_camera_points, calibration)
    first_reprojection = np.linalg.norm(
        first_pixels - result[["x1", "y1"]].to_numpy(), axis=1
    )
    second_reprojection = np.linalg.norm(
        second_pixels - result[["x2", "y2"]].to_numpy(), axis=1
    )
    first_camera_center_world = first_camera_to_world[:3, 3]
    result["triangulated_point_distance_m"] = np.linalg.norm(
        triangulation["world_points"] - first_camera_center_world,
        axis=1,
    )
    result["triangulation_angle_deg"] = ray_metrics["triangulation_angle_deg"]
    result["ray_closest_distance_m"] = ray_metrics["ray_closest_distance_m"]
    result["navigation_valid"] = True
    result["nav_point_to_epiline_px"] = pixel_errors["point_to_epiline"]
    result["nav_symmetric_epipolar_px"] = pixel_errors["symmetric_epipolar"]
    result["nav_sampson_px2"] = pixel_errors["sampson"]
    result["normalized_algebraic_residual"] = normalized_errors["algebraic"]
    result["triangulation_valid_depth"] = triangulation["valid_depth"]
    result["reprojection1_px"] = first_reprojection
    result["reprojection2_px"] = second_reprojection
    result["reprojection_rms_px"] = np.sqrt(
        (first_reprojection**2 + second_reprojection**2) / 2
    )
    result[["reprojected1_x", "reprojected1_y"]] = first_pixels
    result[["reprojected2_x", "reprojected2_y"]] = second_pixels
    return result, fundamental


def _render_pair_visuals(
    output, camera, pair_index, first_image, second_image, matches, fundamental
):
    first_points = matches[["x1", "y1"]].to_numpy()
    second_points = matches[["x2", "y2"]].to_numpy()
    first_small, first_scaled = _scaled_visual(first_image, first_points)
    second_small, second_scaled = _scaled_visual(second_image, second_points)
    identifier = f"{int(matches['frame1'].iloc[0]):04d}_{int(matches['frame2'].iloc[0]):04d}"
    draw_match_overlay(
        first_small,
        second_small,
        first_scaled,
        second_scaled,
        output / f"visualizations/{camera}/matches/{identifier}.jpg",
    )
    scale = second_small.shape[1] / second_image.shape[1]
    scale_matrix = np.diag([scale, scale, 1.0])
    scaled_fundamental = np.linalg.inv(scale_matrix).T @ fundamental @ np.linalg.inv(scale_matrix)
    draw_epipolar_overlay(
        first_small,
        second_small,
        first_scaled,
        second_scaled,
        scaled_fundamental,
        output / f"visualizations/{camera}/epipolar/{identifier}.jpg",
    )
    reprojected = matches[["reprojected2_x", "reprojected2_y"]].to_numpy() * scale
    draw_reprojection_overlay(
        second_small,
        second_scaled,
        reprojected,
        output / f"visualizations/{camera}/reprojection/{identifier}.jpg",
    )


def run_camera(
    config: PipelineConfig,
    camera: str,
    start_frame: int = 0,
    frame_count: int | None = None,
    frame_offset: int = 1,
    run_name: str | None = None,
) -> Path:
    started = time.perf_counter()
    output = config.output_root / (run_name or config.run_name)
    for relative in ("observations", "metrics", "visualizations", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    _write_run_metadata(config, output)
    logging.basicConfig(
        filename=output / "logs/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    dataset = Dataset(config.dataset_root)
    calibration = CameraCalibration.from_dataset(config.dataset_root, camera)
    navigation = NavigationTrajectory.from_csv(
        config.dataset_root / config.assumptions["navigation_source"]
    )
    all_frames = dataset.frames(camera)
    stop = len(all_frames) if frame_count is None else min(len(all_frames), start_frame + frame_count)
    selected = all_frames[start_frame:stop]
    pair_records, metrics_records = [], []
    rendered = 0
    for local_index in range(max(0, len(selected) - frame_offset)):
        first, second = selected[local_index], selected[local_index + frame_offset]
        first_image, second_image = cv2.imread(str(first.path)), cv2.imread(str(second.path))
        matches = match_frame_pair(first_image, second_image, calibration, config.matching)
        if matches.empty:
            logging.warning("%s frames %d-%d produced no matches", camera, first.index, second.index)
            continue
        matches.insert(0, "camera", camera)
        matches["frame1"], matches["frame2"] = first.index, second.index
        matches["timestamp1_ns"], matches["timestamp2_ns"] = first.timestamp_ns, second.timestamp_ns
        matches, fundamental = _evaluate_navigation(
            matches, first.timestamp_ns, second.timestamp_ns, calibration, navigation
        )
        pair_records.append(matches)
        epipolar_summary = summarize_errors(
            matches.loc[matches["navigation_valid"], "nav_symmetric_epipolar_px"].to_numpy()
        )
        reprojection_summary = summarize_errors(
            matches.loc[
                matches["triangulation_valid_depth"], "reprojection_rms_px"
            ].to_numpy()
        )
        metrics_records.append({
            "camera": camera,
            "frame1": first.index,
            "frame2": second.index,
            "timestamp1_ns": first.timestamp_ns,
            "candidate_matches": len(matches),
            "ransac_inliers": int(matches["ransac_inlier"].sum()),
            "navigation_matches": int(matches["navigation_valid"].sum()),
            "valid_triangulations": int(matches["triangulation_valid_depth"].sum()),
            "nav_symmetric_median_px": epipolar_summary.get("median", np.nan),
            "nav_symmetric_p95_px": epipolar_summary.get("p95", np.nan),
            "reprojection_median_px": reprojection_summary.get("median", np.nan),
            "reprojection_p95_px": reprojection_summary.get("p95", np.nan),
        })
        if (
            fundamental is not None
            and rendered < config.matching.visualization_pairs
            and matches["navigation_valid"].any()
        ):
            _render_pair_visuals(
                output, camera, rendered, first_image, second_image, matches, fundamental
            )
            rendered += 1
    matches_all = pd.concat(pair_records, ignore_index=True) if pair_records else pd.DataFrame()
    if matches_all.empty:
        raise RuntimeError(f"{camera} produced no valid frame-pair matches")
    tracks = build_tracks(matches_all, camera)
    lookup = {
        (int(row.frame), int(row.keypoint_id)): int(row.track_id)
        for row in tracks.itertuples(index=False)
    }
    matches_all["track_id"] = [
        lookup[(int(frame), int(keypoint))]
        for frame, keypoint in zip(matches_all["frame1"], matches_all["kp1"])
    ]
    matches_all.to_parquet(output / f"observations/{camera}_matches.parquet", index=False)
    tracks.to_parquet(output / f"observations/{camera}_tracks.parquet", index=False)
    pair_metrics = pd.DataFrame(metrics_records)
    pair_metrics.to_csv(output / "metrics/pair_metrics.csv", index=False)
    epipolar = summarize_errors(
        matches_all.loc[matches_all["navigation_valid"], "nav_symmetric_epipolar_px"].to_numpy()
    )
    reprojection = summarize_errors(
        matches_all.loc[
            matches_all["triangulation_valid_depth"], "reprojection_rms_px"
        ].to_numpy()
    )
    camera_summary = {
        "camera": camera,
        "pair_count": int(len(pair_metrics)),
        "match_count": int(len(matches_all)),
        "track_count": int(tracks["track_id"].nunique()),
        "elapsed_seconds": time.perf_counter() - started,
        **{f"epipolar_{key}": value for key, value in epipolar.items()},
        **{f"reprojection_{key}": value for key, value in reprojection.items()},
    }
    pd.DataFrame([camera_summary]).to_csv(output / "metrics/camera_summary.csv", index=False)
    (output / "metrics/global_summary.json").write_text(
        json.dumps(camera_summary, indent=2), encoding="utf-8"
    )
    plot_error_over_time(
        pair_metrics, output / f"visualizations/{camera}_error_over_time.png", camera
    )
    logging.info("completed %s in %.3f seconds", camera, time.perf_counter() - started)
    return output
