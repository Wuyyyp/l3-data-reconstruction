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
    CAMERA_TO_IMAGE_FOLDER,
    image_for_camera,
    render_projection,
    resolve_sensor_file,
)


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def pcd_align_time_seconds(point_timestamps: np.ndarray, ratio: float) -> float:
    values = np.asarray(point_timestamps, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("PCD timestamp field has no finite values")
    t_min = float(np.min(finite))
    t_max = float(np.max(finite))
    return t_min + float(ratio) * (t_max - t_min)


def camera_time_seconds(info: dict[str, Any], camera: str) -> float:
    cam = info.get("sensors", {}).get("cams", {}).get(camera)
    if not isinstance(cam, dict) or "timestamp" not in cam:
        raise KeyError(f"missing camera timestamp: {camera}")
    return timestamp_to_seconds(cam["timestamp"])


def points_in_target_ego(
    points_source_ego: np.ndarray,
    *,
    source_ego_to_world: np.ndarray,
    target_ego_to_world: np.ndarray,
) -> np.ndarray:
    points_world = transform_points(points_source_ego, source_ego_to_world)
    return transform_points(points_world, np.linalg.inv(target_ego_to_world))


def put_label(image: np.ndarray, lines: list[str]) -> None:
    y = 34
    for line in lines:
        cv2.putText(
            image,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30


def write_pose_time_projection_compare(
    *,
    image: np.ndarray,
    pkl_points_target_ego: np.ndarray,
    align_points_target_ego: np.ndarray,
    cam_data: dict[str, Any],
    output_path: Path,
    max_depth_m: float,
    max_points: int | None,
    point_stride: int,
    point_radius: int,
    label_lines: list[str],
) -> None:
    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
    pkl_panel = render_projection(
        image,
        pkl_points_target_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        max_points=max_points,
        point_stride=point_stride,
        point_radius=point_radius,
    )
    align_panel = render_projection(
        image,
        align_points_target_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        max_points=max_points,
        point_stride=point_stride,
        point_radius=point_radius,
    )
    put_label(pkl_panel, ["left: source ego = pkl timestamp (~0%)", *label_lines])
    put_label(align_panel, ["right: source ego = PCD t_align 20%", *label_lines])
    joined = np.concatenate([pkl_panel, align_panel], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), joined)


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    output_dir = Path(args.output_dir)
    cameras = [item.strip() for item in str(args.cameras).split(",") if item.strip()]
    frame_end = min(len(infos), int(args.end))
    summary_rows: list[dict[str, object]] = []

    for frame_index in range(int(args.start), frame_end, max(1, int(args.frame_stride))):
        info = infos[frame_index]
        lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception")
        if not isinstance(lidar_data, dict):
            print(f"[skip] frame={frame_index}: missing perception lidar")
            continue
        pcd_path = resolve_sensor_file(dataset_root, str(lidar_data["data_path"]), "perception")
        cloud = read_pcd_cloud(pcd_path)
        timestamps = cloud.timestamps
        if timestamps is None:
            print(f"[skip] frame={frame_index}: PCD has no timestamp field: {pcd_path.name}")
            continue
        pkl_time_s = timestamp_to_seconds(info["timestamp"])
        t_min = float(np.nanmin(timestamps))
        t_max = float(np.nanmax(timestamps))
        align_time_s = pcd_align_time_seconds(timestamps, float(args.align_ratio))
        pkl_ego_to_world = ego_world_transform(info)
        align_ego_to_world = _interpolate_ego_transform(infos, align_time_s)
        if align_ego_to_world is None:
            print(f"[skip] frame={frame_index}: cannot interpolate ego at t_align={align_time_s:.9f}")
            continue

        for camera in cameras:
            cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(cam_data, dict):
                print(f"[skip] frame={frame_index} camera={camera}: missing camera data")
                continue
            image_path, image = image_for_camera(dataset_root, cam_data, camera)
            cam_time_s = camera_time_seconds(info, camera)
            cam_ego_to_world = _interpolate_ego_transform(infos, cam_time_s)
            if cam_ego_to_world is None:
                cam_ego_to_world = pkl_ego_to_world

            pkl_points_target = points_in_target_ego(
                cloud.xyz,
                source_ego_to_world=pkl_ego_to_world,
                target_ego_to_world=cam_ego_to_world,
            )
            align_points_target = points_in_target_ego(
                cloud.xyz,
                source_ego_to_world=align_ego_to_world,
                target_ego_to_world=cam_ego_to_world,
            )
            stem = f"{camera}_{frame_index:04d}_{Path(image_path).stem}_pkl_vs_talign"
            out_path = output_dir / camera / f"{stem}.jpg"
            label = [
                f"frame={frame_index} image={Path(image_path).name}",
                f"pkl-min={1000*(pkl_time_s-t_min):.1f}ms | t_align-pkl={1000*(align_time_s-pkl_time_s):.1f}ms",
                f"cam-pkl={1000*(cam_time_s-pkl_time_s):.3f}ms | span={1000*(t_max-t_min):.1f}ms",
            ]
            write_pose_time_projection_compare(
                image=image,
                pkl_points_target_ego=pkl_points_target,
                align_points_target_ego=align_points_target,
                cam_data=cam_data,
                output_path=out_path,
                max_depth_m=float(args.max_depth_m),
                max_points=None if args.max_points is None else int(args.max_points),
                point_stride=int(args.point_stride),
                point_radius=int(args.point_radius),
                label_lines=label,
            )
            summary_rows.append(
                {
                    "frame_index": frame_index,
                    "camera": camera,
                    "pcd": pcd_path.name,
                    "image": Path(image_path).name,
                    "pkl_time_s": f"{pkl_time_s:.9f}",
                    "pcd_min_time_s": f"{t_min:.9f}",
                    "pcd_max_time_s": f"{t_max:.9f}",
                    "pcd_align_time_s": f"{align_time_s:.9f}",
                    "camera_time_s": f"{cam_time_s:.9f}",
                    "pkl_minus_pcd_min_ms": f"{1000*(pkl_time_s-t_min):.6f}",
                    "align_minus_pkl_ms": f"{1000*(align_time_s-pkl_time_s):.6f}",
                    "camera_minus_pkl_ms": f"{1000*(cam_time_s-pkl_time_s):.6f}",
                    "output": str(out_path),
                }
            )
        print(f"[done] frame={frame_index:04d} pcd={pcd_path.name}")

    summary_path = output_dir / "pose_time_projection_compare_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw perception PCD projection using pkl ego pose versus PCD "
            "20%-span t_align ego pose."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--cameras",
        default="center_camera_fov120,left_front_camera,right_front_camera",
        help="Comma-separated result_interpolation camera names.",
    )
    parser.add_argument("--align-ratio", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--max-points", type=int, default=70000)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

