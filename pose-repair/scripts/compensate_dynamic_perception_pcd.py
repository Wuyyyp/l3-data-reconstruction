from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    DEFAULT_REFERENCE_CAMERAS_6,
    build_object_tracks,
    compensate_dynamic_objects_for_frame,
    load_frame_point_clouds,
    read_pcd_cloud,
    write_pcd_cloud,
)


CAMERA_TO_IMAGE_FOLDER = {
    "center_camera_fov120": "front_camera_fov120",
    "center_camera_fov30": "front_camera_fov30",
    "left_front_camera": "left_front_camera",
    "right_front_camera": "right_front_camera",
    "left_rear_camera": "left_rear_camera",
    "right_rear_camera": "right_rear_camera",
    "rear_camera": "rear_camera",
}


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_sensor_file(dataset_root: Path, data_path: str, fallback_folder: str) -> Path:
    raw = Path(str(data_path))
    if raw.is_absolute() and raw.exists():
        return raw
    candidate = dataset_root / raw
    if candidate.exists():
        return candidate
    candidate = dataset_root / "images" / fallback_folder / raw.name
    if candidate.exists():
        return candidate
    candidate = dataset_root / "lidar" / fallback_folder / raw.name
    if candidate.exists():
        return candidate
    candidate = dataset_root / fallback_folder / raw.name
    if candidate.exists():
        return candidate
    if raw.exists():
        return raw
    candidate = dataset_root / raw.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot resolve sensor file for {data_path}")


def image_for_camera(dataset_root: Path, cam_data: dict[str, Any], camera: str) -> tuple[Path, np.ndarray]:
    folder = CAMERA_TO_IMAGE_FOLDER.get(camera, camera)
    image_path = resolve_sensor_file(dataset_root, str(cam_data["data_path"]), folder)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    return image_path, image


def project_to_pixels(
    points_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int, int],
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    if len(points_ego) == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_camera = (np.asarray(ego_to_camera, dtype=np.float64).reshape(4, 4) @ homogeneous.T)[
        :3
    ]
    depth = points_camera[2]
    uvw = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points_camera
    uv = uvw[:2] / np.clip(uvw[2:3], 1e-8, None)
    valid = (
        np.isfinite(depth)
        & (depth > 0.05)
        & (depth <= float(max_depth_m))
        & np.isfinite(uv[0])
        & np.isfinite(uv[1])
        & (uv[0] >= 0)
        & (uv[0] < width)
        & (uv[1] >= 0)
        & (uv[1] < height)
    )
    xs = np.rint(uv[0, valid]).astype(np.int32)
    ys = np.rint(uv[1, valid]).astype(np.int32)
    depths = depth[valid].astype(np.float32)
    return xs, ys, depths


def render_projection(
    image: np.ndarray,
    points_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    max_depth_m: float,
    max_points: int | None,
    point_stride: int,
    point_radius: int,
) -> np.ndarray:
    xs, ys, depths = project_to_pixels(
        points_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        image_shape=image.shape,
        max_depth_m=max_depth_m,
    )
    if len(xs) == 0:
        return image.copy()
    order = np.argsort(depths)
    xs, ys, depths = xs[order], ys[order], depths[order]
    _, unique_indices = np.unique(np.stack([xs, ys], axis=1), axis=0, return_index=True)
    xs, ys, depths = xs[unique_indices], ys[unique_indices], depths[unique_indices]

    stride = max(1, int(point_stride))
    if stride > 1:
        xs, ys, depths = xs[::stride], ys[::stride], depths[::stride]
    if max_points is not None and len(xs) > int(max_points):
        indices = np.linspace(0, len(xs) - 1, int(max_points), dtype=np.int32)
        xs, ys, depths = xs[indices], ys[indices], depths[indices]

    normalized = np.clip(depths / max(float(max_depth_m), 1.0), 0.0, 1.0)
    colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA).reshape(
        -1, 3
    )
    output = image.copy()
    radius = max(1, int(point_radius))
    for x, y, color in zip(xs, ys, colors, strict=False):
        cv2.circle(output, (int(x), int(y)), radius, color.tolist(), -1)
    return output


def put_label(image: np.ndarray, text: str, *, y: int = 42) -> None:
    cv2.putText(
        image,
        text,
        (24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_projection_compare(
    *,
    image: np.ndarray,
    before_points: np.ndarray,
    after_points: np.ndarray,
    cam_data: dict[str, Any],
    output_path: Path,
    moved_point_count: int,
    compensated_object_count: int,
    max_depth_m: float,
    max_points: int | None,
    point_stride: int,
    point_radius: int,
) -> None:
    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
    before = render_projection(
        image,
        before_points,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        max_points=max_points,
        point_stride=point_stride,
        point_radius=point_radius,
    )
    after = render_projection(
        image,
        after_points,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        max_points=max_points,
        point_stride=point_stride,
        point_radius=point_radius,
    )
    put_label(before, "before dynamic compensation")
    put_label(
        after,
        f"after | moved_points={moved_point_count} | compensated_objects={compensated_object_count}",
    )
    joined = np.concatenate([before, after], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), joined)


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    frame_end = min(len(infos), int(args.end))
    reference_cameras = tuple(
        item.strip() for item in str(args.reference_cameras).split(",") if item.strip()
    )
    frame_point_clouds = load_frame_point_clouds(
        infos,
        dataset_root=dataset_root,
        resolve_sensor_file=resolve_sensor_file,
    )
    object_tracks = build_object_tracks(
        infos,
        frame_point_clouds,
        min_object_points=int(args.min_object_points),
        box_margin_m=float(args.box_margin_m),
    )
    output_pcd_dir = Path(args.output_pcd_dir)
    visualization_dir = Path(args.visualization_dir)
    summary_path = Path(args.summary_csv)
    output_pcd_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for frame_index in range(int(args.start), frame_end):
        info = infos[frame_index]
        lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception")
        if not isinstance(lidar_data, dict):
            print(f"[skip] frame={frame_index}: missing perception lidar")
            continue
        pcd_path = resolve_sensor_file(dataset_root, str(lidar_data["data_path"]), "perception")
        cloud = read_pcd_cloud(pcd_path)
        result = compensate_dynamic_objects_for_frame(
            infos=infos,
            frame_index=frame_index,
            cloud=cloud,
            object_tracks=object_tracks,
            reference_cameras=reference_cameras,
            min_object_points=int(args.min_object_points),
            box_margin_m=float(args.box_margin_m),
        )
        out_pcd = output_pcd_dir / pcd_path.name
        write_pcd_cloud(out_pcd, result.cloud)

        compensated_objects = sum(1 for report in result.reports if report.status == "compensated")
        rows.append(
            {
                "frame_index": frame_index,
                "pcd": pcd_path.name,
                "output_pcd": str(out_pcd),
                "reference_time_s": f"{result.reference_time_s:.9f}",
                "moved_point_count": result.moved_point_count,
                "compensated_object_count": compensated_objects,
                "dynamic_object_report_count": len(result.reports),
            }
        )
        for report in result.reports:
            rows.append(
                {
                    "frame_index": frame_index,
                    "pcd": pcd_path.name,
                    "object_id": report.object_id,
                    "object_type": report.object_type,
                    "object_point_count": report.point_count,
                    "source_time_s": f"{report.source_time_s:.9f}"
                    if np.isfinite(report.source_time_s)
                    else "",
                    "reference_time_s": f"{report.reference_time_s:.9f}",
                    "motion_translation_m": f"{report.motion_translation_m:.6f}"
                    if np.isfinite(report.motion_translation_m)
                    else "",
                    "status": report.status,
                }
            )

        if not args.no_visualization:
            cams = info.get("sensors", {}).get("cams", {})
            cam_data = cams.get(args.projection_camera)
            if isinstance(cam_data, dict):
                image_path, image = image_for_camera(dataset_root, cam_data, args.projection_camera)
                output_path = (
                    visualization_dir
                    / f"{args.projection_camera}_{frame_index:06d}_{image_path.stem}_compare.jpg"
                )
                write_projection_compare(
                    image=image,
                    before_points=cloud.xyz,
                    after_points=result.cloud.xyz,
                    cam_data=cam_data,
                    output_path=output_path,
                    moved_point_count=result.moved_point_count,
                    compensated_object_count=compensated_objects,
                    max_depth_m=float(args.max_depth_m),
                    max_points=args.max_visualization_points,
                    point_stride=int(args.point_stride),
                    point_radius=int(args.point_radius),
                )
        print(
            f"[{frame_index:04d}] {pcd_path.name}: moved_points={result.moved_point_count}, "
            f"objects={compensated_objects}, out={out_pcd}"
        )

    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[done] pcd_dir={output_pcd_dir}")
    print(f"[done] visualizations={visualization_dir}")
    print(f"[done] summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compensate dynamic objects in ego-frame perception PCDs and visualize "
            "before/after projection to a camera."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--output-pcd-dir", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument(
        "--reference-cameras",
        default=",".join(DEFAULT_REFERENCE_CAMERAS_6),
        help="Comma-separated pkl camera names used to compute the image reference time.",
    )
    parser.add_argument("--projection-camera", default="center_camera_fov120")
    parser.add_argument("--min-object-points", type=int, default=5)
    parser.add_argument("--box-margin-m", type=float, default=0.0)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--max-visualization-points", type=int, default=250000)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--no-visualization", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

