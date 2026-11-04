from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    ego_world_transform,
    read_pcd_cloud,
    transform_points,
)
from compensate_dynamic_perception_pcd import (
    CAMERA_TO_IMAGE_FOLDER,
    image_for_camera,
    render_projection,
    resolve_sensor_file,
)


DEFAULT_CAMERAS_7 = (
    "center_camera_fov120",
    "center_camera_fov30",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_front_camera",
    "right_rear_camera",
)
DEFAULT_OFFSET_EGO_FIELD = "ego2global_transformation_matrix_camera_reference_offset"


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def offset_ego_to_world_for_info(
    info: dict[str, Any], field_name: str = DEFAULT_OFFSET_EGO_FIELD
) -> np.ndarray:
    if field_name not in info:
        raise KeyError(f"missing offset ego field: {field_name}")
    return np.asarray(info[field_name], dtype=np.float64).reshape(4, 4)


def points_in_offset_ego(
    points_source_ego: np.ndarray,
    *,
    source_ego_to_world: np.ndarray,
    offset_ego_to_world: np.ndarray,
) -> np.ndarray:
    points_world = transform_points(points_source_ego, source_ego_to_world)
    return transform_points(points_world, np.linalg.inv(offset_ego_to_world))


def resolve_pkl_relative_sensor_file(
    *,
    pkl_path: Path,
    data_path: str,
    fallback_root: Path,
    fallback_folder: str,
    point_cloud_root: Path | None = None,
) -> Path:
    raw = Path(str(data_path))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path(pkl_path).parent / raw)
    if point_cloud_root is not None:
        candidates.append(Path(point_cloud_root) / raw.name)
    candidates.extend(
        [
            Path(fallback_root) / fallback_folder / raw.name,
            Path(fallback_root) / raw.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return resolve_sensor_file(Path(fallback_root), str(data_path), fallback_folder)


def _put_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(output.shape[1] / 1300.0, 0.85))
    thickness = max(1, int(round(font_scale * 2)))
    y = int(18 + 28 * font_scale)
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
        y += int(34 * font_scale)
    return output


def run(args: argparse.Namespace) -> None:
    pkl_path = Path(args.pkl)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    infos = load_infos(pkl_path)
    cameras = tuple(item.strip() for item in str(args.cameras).split(",") if item.strip())
    start = int(args.start)
    end = min(len(infos), int(args.end))
    rows: list[dict[str, Any]] = []

    for frame_index in range(start, end, max(1, int(args.frame_stride))):
        info = infos[frame_index]
        perception = info.get("sensors", {}).get("lidar", {}).get("perception")
        if not isinstance(perception, dict):
            print(f"[skip] frame={frame_index}: missing perception lidar")
            continue
        pcd_path = resolve_pkl_relative_sensor_file(
            pkl_path=pkl_path,
            data_path=str(perception.get("data_path") or perception.get("aws_path") or ""),
            fallback_root=image_root,
            fallback_folder="perception",
            point_cloud_root=args.point_cloud_root,
        )
        cloud = read_pcd_cloud(pcd_path)
        source_ego_to_world = ego_world_transform(info)
        offset_ego_to_world = offset_ego_to_world_for_info(info, str(args.offset_ego_field))
        points_offset_ego = points_in_offset_ego(
            cloud.xyz,
            source_ego_to_world=source_ego_to_world,
            offset_ego_to_world=offset_ego_to_world,
        )
        ego_delta_m = float(
            np.linalg.norm(offset_ego_to_world[:3, 3] - source_ego_to_world[:3, 3])
        )

        for camera in cameras:
            cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(cam_data, dict):
                print(f"[skip] frame={frame_index} camera={camera}: missing camera data")
                continue
            image_path, image = image_for_camera(image_root, cam_data, camera)
            rendered = render_projection(
                image,
                points_offset_ego,
                ego_to_camera=np.asarray(cam_data["extrinsic"], dtype=np.float64),
                intrinsic=np.asarray(cam_data["cam_intrinsic"], dtype=np.float64),
                max_depth_m=float(args.max_depth_m),
                max_points=None if args.max_points is None else int(args.max_points),
                point_stride=int(args.point_stride),
                point_radius=int(args.point_radius),
            )
            rendered = _put_label(
                rendered,
                [
                    "dynamic compensated perception PCD",
                    f"offset ego field: {args.offset_ego_field}",
                    f"ego delta={ego_delta_m:.3f}m | points={len(cloud.xyz)}",
                ],
            )
            output_path = (
                output_dir
                / camera
                / f"{camera}_{frame_index:06d}_{Path(image_path).stem}_dynamic_compensated_pkl_projection.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), rendered)
            rows.append(
                {
                    "frame_index": frame_index,
                    "camera": camera,
                    "pcd": str(pcd_path),
                    "image": str(image_path),
                    "point_count": int(len(cloud.xyz)),
                    "ego_delta_m": f"{ego_delta_m:.9f}",
                    "output": str(output_path),
                }
            )
        print(f"[done] frame={frame_index:04d} pcd={pcd_path.name}")

    summary_path = output_dir / "dynamic_compensated_pkl_projection_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[done] visualizations={len(rows)} -> {output_dir}")
    print(f"[done] summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project dynamic-compensated perception PCD from a deliverable pkl, using the "
            "offset ego2global field for camera vehicle pose."
        )
    )
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--point-cloud-root",
        type=Path,
        help="Directory containing compensated perception PCD files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset-ego-field", default=DEFAULT_OFFSET_EGO_FIELD)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS_7))
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
