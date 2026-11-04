from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    _interpolate_ego_transform,
    ego_world_transform,
    read_pcd_cloud,
    timestamp_to_seconds,
    transform_points,
)
from compensate_dynamic_perception_pcd import (
    image_for_camera,
    render_projection,
    resolve_sensor_file,
)


DEFAULT_CAMERAS_6 = (
    "center_camera_fov30",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_front_camera",
    "right_rear_camera",
)


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def camera_time_with_offset_seconds(info: dict[str, Any], camera: str, offset_s: float) -> float:
    cam = info.get("sensors", {}).get("cams", {}).get(camera)
    if not isinstance(cam, dict) or "timestamp" not in cam:
        raise KeyError(f"missing camera timestamp: {camera}")
    return timestamp_to_seconds(cam["timestamp"]) + float(offset_s)


def points_in_camera_time_ego(
    points_source_ego: np.ndarray,
    *,
    source_ego_to_world: np.ndarray,
    target_ego_to_world: np.ndarray,
) -> np.ndarray:
    points_world = transform_points(points_source_ego, source_ego_to_world)
    return transform_points(points_world, np.linalg.inv(target_ego_to_world))


def _put_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(output.shape[1] / 1300.0, 0.85))
    thickness = max(1, int(round(font_scale * 2)))
    line_height = int(round(30 * font_scale))
    y = 14 + line_height
    for line in lines:
        (width, height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        x = 12
        cv2.rectangle(
            output,
            (x - 5, y - height - 5),
            (x + width + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            line,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height + 8
    return output


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    target_height = max(1, int(height))
    if image.shape[0] == target_height:
        return image
    scale = target_height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, target_height), interpolation=interpolation)


def _parse_offsets_ms(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("--offsets-ms must contain at least one value")
    return values


def sample_points_for_visualization(
    points: np.ndarray, *, sample_ratio: float, seed: int
) -> np.ndarray:
    values = np.asarray(points)
    ratio = float(sample_ratio)
    if ratio >= 1.0 or len(values) == 0:
        return values
    if ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    keep_count = max(1, int(round(len(values) * ratio)))
    if keep_count >= len(values):
        return values
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(values), size=keep_count, replace=False)
    indices.sort()
    return values[indices]


def select_frame_indices(
    *, start: int, end: int, stride: int, sample_ratio: float, seed: int
) -> list[int]:
    indices = list(range(int(start), int(end), max(1, int(stride))))
    ratio = float(sample_ratio)
    if ratio >= 1.0 or len(indices) == 0:
        return indices
    if ratio <= 0.0:
        raise ValueError("frame_sample_ratio must be in (0, 1]")
    keep_count = max(1, int(round(len(indices) * ratio)))
    if keep_count >= len(indices):
        return indices
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(indices, dtype=np.int64), size=keep_count, replace=False)
    return sorted(int(value) for value in selected)


def render_offset_sweep_for_camera(
    *,
    infos: list[dict[str, Any]],
    frame_index: int,
    camera: str,
    image: np.ndarray,
    cam_data: dict[str, Any],
    points_source_ego: np.ndarray,
    source_ego_to_world: np.ndarray,
    offsets_ms: list[float],
    max_depth_m: float,
    max_points: int | None,
    point_stride: int,
    point_radius: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
    camera_timestamp_s = timestamp_to_seconds(cam_data["timestamp"])
    panels = [
        _put_label(
            render_projection(
                image,
                points_source_ego,
                ego_to_camera=ego_to_camera,
                intrinsic=intrinsic,
                max_depth_m=max_depth_m,
                max_points=max_points,
                point_stride=point_stride,
                point_radius=point_radius,
            ),
            ["source ego / no interp", "raw perception PCD"],
        )
    ]
    rows: list[dict[str, Any]] = []
    for offset_ms in offsets_ms:
        target_time_s = camera_timestamp_s + float(offset_ms) * 1.0e-3
        target_ego_to_world = _interpolate_ego_transform(infos, target_time_s)
        fallback = target_ego_to_world is None
        if fallback:
            target_ego_to_world = source_ego_to_world
        points_target_ego = points_in_camera_time_ego(
            points_source_ego,
            source_ego_to_world=source_ego_to_world,
            target_ego_to_world=target_ego_to_world,
        )
        label = [
            f"camera ego +{offset_ms:g} ms",
            f"cam+offset={target_time_s:.6f}s",
        ]
        if fallback:
            label.append("interp unavailable -> source ego")
        panels.append(
            _put_label(
                render_projection(
                    image,
                    points_target_ego,
                    ego_to_camera=ego_to_camera,
                    intrinsic=intrinsic,
                    max_depth_m=max_depth_m,
                    max_points=max_points,
                    point_stride=point_stride,
                    point_radius=point_radius,
                ),
                label,
            )
        )
        rows.append(
            {
                "offset_ms": float(offset_ms),
                "camera_timestamp_s": f"{camera_timestamp_s:.9f}",
                "target_time_s": f"{target_time_s:.9f}",
                "interpolation_fallback": bool(fallback),
            }
        )
    height = max(panel.shape[0] for panel in panels)
    resized = [_resize_to_height(panel, height) for panel in panels]
    return np.concatenate(resized, axis=1), rows


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    output_dir = Path(args.output_dir)
    cameras = [item.strip() for item in str(args.cameras).split(",") if item.strip()]
    offsets_ms = _parse_offsets_ms(str(args.offsets_ms))
    frame_end = min(len(infos), int(args.end))
    summary_rows: list[dict[str, Any]] = []

    frame_indices = select_frame_indices(
        start=int(args.start),
        end=frame_end,
        stride=int(args.frame_stride),
        sample_ratio=float(args.frame_sample_ratio),
        seed=int(args.frame_sample_seed),
    )
    print(
        f"[info] selected_frames={len(frame_indices)}/{len(range(int(args.start), frame_end, max(1, int(args.frame_stride))))} "
        f"frame_sample_ratio={float(args.frame_sample_ratio):.3f}"
    )
    for frame_index in frame_indices:
        info = infos[frame_index]
        lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception")
        if not isinstance(lidar_data, dict):
            print(f"[skip] frame={frame_index}: missing perception lidar")
            continue
        pcd_path = resolve_sensor_file(dataset_root, str(lidar_data["data_path"]), "perception")
        cloud = read_pcd_cloud(pcd_path)
        points_source_ego = sample_points_for_visualization(
            cloud.xyz,
            sample_ratio=float(args.point_sample_ratio),
            seed=int(args.point_sample_seed) + frame_index,
        )
        source_ego_to_world = ego_world_transform(info)

        for camera in cameras:
            cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(cam_data, dict):
                print(f"[skip] frame={frame_index} camera={camera}: missing camera data")
                continue
            image_path, image = image_for_camera(dataset_root, cam_data, camera)
            rendered, offset_rows = render_offset_sweep_for_camera(
                infos=infos,
                frame_index=frame_index,
                camera=camera,
                image=image,
                cam_data=cam_data,
                points_source_ego=points_source_ego,
                source_ego_to_world=source_ego_to_world,
                offsets_ms=offsets_ms,
                max_depth_m=float(args.max_depth_m),
                max_points=None if args.max_points is None else int(args.max_points),
                point_stride=int(args.point_stride),
                point_radius=int(args.point_radius),
            )
            output_path = (
                output_dir
                / camera
                / f"{camera}_{frame_index:06d}_{Path(image_path).stem}_perception_offset_sweep.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), rendered)
            for row in offset_rows:
                summary_rows.append(
                    {
                        "frame_index": frame_index,
                        "camera": camera,
                        "pcd": pcd_path.name,
                        "image": Path(image_path).name,
                        "source_point_count": int(len(cloud.xyz)),
                        "visualized_point_count": int(len(points_source_ego)),
                        "point_sample_ratio": float(args.point_sample_ratio),
                        "output": str(output_path),
                        **row,
                    }
                )
        print(f"[done] frame={frame_index:04d} pcd={pcd_path.name}")

    summary_path = output_dir / "perception_projection_offset_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"[done] visualizations={len(set(row['output'] for row in summary_rows)) if summary_rows else 0}")
    print(f"[done] summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw perception PCD projection with camera-time ego interpolation "
            "under several image timestamp offsets."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-sample-ratio", type=float, default=1.0)
    parser.add_argument("--frame-sample-seed", type=int, default=0)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS_6))
    parser.add_argument("--offsets-ms", default="0,10,15,20")
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--point-sample-ratio", type=float, default=1.0)
    parser.add_argument("--point-sample-seed", type=int, default=0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

