from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


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
        infos = pickle.load(handle).get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain an infos list")
    return infos


def transform_points_world_to_camera(points_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4) @ homogeneous.T).T[:, :3]


def world_to_camera_from_info(
    info: dict[str, Any],
    *,
    camera: str,
    ego_field: str,
) -> np.ndarray:
    camera_data = info["sensors"]["cams"][camera]
    world_from_ego = np.asarray(info[ego_field], dtype=np.float64).reshape(4, 4)
    ego_to_camera = np.asarray(camera_data["extrinsic"], dtype=np.float64).reshape(4, 4)
    world_from_camera = world_from_ego @ np.linalg.inv(ego_to_camera)
    return np.linalg.inv(world_from_camera)


def image_path_for_camera(dataset_root: Path, camera_data: dict[str, Any], camera: str) -> Path:
    source = Path(str(camera_data["data_path"]))
    candidates = [
        source,
        dataset_root / source,
        dataset_root / "images" / CAMERA_TO_IMAGE_FOLDER[camera] / source.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for {camera}: {source}")


def ply_header_line_count(path: Path) -> int:
    with Path(path).open("rb") as handle:
        for line_index, line in enumerate(handle, start=1):
            if line.strip() == b"end_header":
                return line_index
    raise ValueError(f"PLY header has no end_header line: {path}")


def read_ascii_ply_xyz(path: Path) -> np.ndarray:
    header_lines = ply_header_line_count(path)
    points = np.loadtxt(
        path,
        dtype=np.float32,
        comments=None,
        skiprows=header_lines,
        usecols=(0, 1, 2),
    )
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def project_world_points(
    points_world: np.ndarray,
    *,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int, int],
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    points_camera = transform_points_world_to_camera(points_world, world_to_camera)
    depths = points_camera[:, 2]
    projected = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points_camera.T
    u = projected[0] / np.maximum(projected[2], 1e-8)
    v = projected[1] / np.maximum(projected[2], 1e-8)
    valid = (
        np.isfinite(depths)
        & (depths > 0.05)
        & (depths <= float(max_depth_m))
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    return (
        np.rint(u[valid]).astype(np.int32),
        np.rint(v[valid]).astype(np.int32),
        depths[valid].astype(np.float32),
    )


def draw_depth_overlay(
    image: np.ndarray,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    point_radius: int,
) -> np.ndarray:
    output = image.copy()
    if len(xs) == 0:
        return output
    height, width = image.shape[:2]
    linear = ys.astype(np.int64) * width + xs.astype(np.int64)
    order = np.lexsort((depths, linear))
    ordered_linear = linear[order]
    _, first = np.unique(ordered_linear, return_index=True)
    chosen = order[first]
    xs, ys, depths = xs[chosen], ys[chosen], depths[chosen]
    near, far = np.percentile(depths, [1.0, 99.0])
    far = max(float(far), float(near) + 1.0)
    colors = cv2.applyColorMap(
        (255.0 * np.clip((depths - near) / (far - near), 0.0, 1.0)).astype(np.uint8),
        cv2.COLORMAP_PLASMA,
    ).reshape(-1, 3)
    for x, y, color in zip(xs, ys, colors, strict=True):
        cv2.circle(output, (int(x), int(y)), max(1, int(point_radius)), color.tolist(), -1)
    return output


def label_image(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    y = 38
    for line in lines:
        cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return output


def selected_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    count = min(max(1, int(sample_count)), frame_count)
    return np.unique(np.linspace(0, frame_count - 1, count, dtype=np.int32)).tolist()


def run(args: argparse.Namespace) -> None:
    infos = load_infos(args.pkl)
    points_world = read_ascii_ply_xyz(args.fused_ply)
    xy_tree = cKDTree(points_world[:, :2])
    output_dir = Path(args.output_dir)
    cameras = [value.strip() for value in str(args.cameras).split(",") if value.strip()]
    frame_indices = selected_frame_indices(len(infos), args.sample_count)
    rows: list[dict[str, object]] = []

    for frame_index in frame_indices:
        info = infos[frame_index]
        ego_translation = np.asarray(info[args.ego_field], dtype=np.float64).reshape(4, 4)[:2, 3]
        candidate_indices = xy_tree.query_ball_point(ego_translation, r=float(args.world_radius_m))
        local_world = points_world[np.asarray(candidate_indices, dtype=np.int64)]
        for camera in cameras:
            camera_data = info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(camera_data, dict):
                continue
            image_path = image_path_for_camera(args.dataset_root, camera_data, camera)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot read image: {image_path}")
            world_to_camera = world_to_camera_from_info(info, camera=camera, ego_field=args.ego_field)
            xs, ys, depths = project_world_points(
                local_world,
                world_to_camera=world_to_camera,
                intrinsic=np.asarray(camera_data["cam_intrinsic"], dtype=np.float64),
                image_shape=image.shape,
                max_depth_m=float(args.max_depth_m),
            )
            rendered = draw_depth_overlay(
                image,
                xs=xs,
                ys=ys,
                depths=depths,
                point_radius=args.point_radius,
            )
            rendered = label_image(
                rendered,
                [
                    f"fused world points | frame={frame_index} | camera={camera}",
                    f"projected={len(xs)} | local candidates={len(local_world)} | depth<= {args.max_depth_m:.0f}m",
                ],
            )
            path = output_dir / camera / f"{camera}_{frame_index:04d}_{image_path.stem}_fused.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(f"failed to write {path}")
            rows.append(
                {
                    "frame_index": frame_index,
                    "camera": camera,
                    "image": str(image_path),
                    "output": str(path),
                    "local_candidate_points": len(local_world),
                    "projected_points_before_zbuffer": len(xs),
                }
            )
        print(f"[done] frame={frame_index} local_candidates={len(local_world)}")

    summary = output_dir / "projection_summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[done] outputs -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a world-coordinate fused LiDAR PLY to sampled camera images.")
    parser.add_argument("--fused-ply", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ego-field", default="ego2global_transformation_matrix")
    parser.add_argument("--cameras", default=",".join(CAMERA_TO_IMAGE_FOLDER))
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--world-radius-m", type=float, default=110.0)
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--point-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
