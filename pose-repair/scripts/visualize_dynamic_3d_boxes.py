from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    DEFAULT_REFERENCE_CAMERAS_6,
    _interpolate_ego_transform,
    _interpolate_track_transform,
    box_corners_ego,
    build_object_tracks,
    is_dynamic_object,
    load_frame_point_clouds,
    object_id,
    reference_time_seconds_from_cameras,
    transform_points,
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

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_image_path(dataset_root: Path, cam_data: dict[str, Any], camera: str) -> Path:
    folder = CAMERA_TO_IMAGE_FOLDER.get(camera, camera)
    name = Path(str(cam_data["data_path"])).name
    candidate = dataset_root / folder / name
    if candidate.exists():
        return candidate
    raw = Path(str(cam_data["data_path"]))
    if raw.exists():
        return raw
    raise FileNotFoundError(candidate)


def resolve_sensor_file(dataset_root: Path, data_path: str, fallback_folder: str) -> Path:
    name = Path(str(data_path)).name
    candidate = dataset_root / fallback_folder / name
    if candidate.exists():
        return candidate
    raw = Path(str(data_path))
    if raw.exists():
        return raw
    raise FileNotFoundError(candidate)


def project_corners(
    corners_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = transform_points(corners_ego, ego_to_camera).astype(np.float64)
    depth = points_camera[:, 2]
    uvw = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points_camera.T
    uv = (uvw[:2] / np.clip(uvw[2:3], 1e-8, None)).T
    return uv, depth


def draw_box(
    image: np.ndarray,
    corners_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 3,
) -> bool:
    height, width = image.shape[:2]
    uv, depth = project_corners(
        corners_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
    )
    drawn = False
    for a, b in BOX_EDGES:
        if depth[a] <= 0.05 or depth[b] <= 0.05:
            continue
        pa = tuple(np.rint(uv[a]).astype(int).tolist())
        pb = tuple(np.rint(uv[b]).astype(int).tolist())
        if (
            max(pa[0], pb[0]) < -width
            or min(pa[0], pb[0]) > 2 * width
            or max(pa[1], pb[1]) < -height
            or min(pa[1], pb[1]) > 2 * height
        ):
            continue
        cv2.line(image, pa, pb, color, thickness, cv2.LINE_AA)
        drawn = True
    if drawn and label:
        valid = (depth > 0.05) & np.isfinite(uv).all(axis=1)
        if valid.any():
            anchor = np.rint(uv[valid].mean(axis=0)).astype(int)
            if -width <= anchor[0] <= 2 * width and -height <= anchor[1] <= 2 * height:
                cv2.putText(
                    image,
                    label,
                    (int(anchor[0]), int(anchor[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return drawn


def put_label(image: np.ndarray, text: str, *, color: tuple[int, int, int], y: int = 42) -> None:
    cv2.putText(image, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def visualize_frame(
    *,
    infos: list[dict[str, Any]],
    tracks,
    frame_index: int,
    camera: str,
    dataset_root: Path,
    output_dir: Path,
    reference_cameras: tuple[str, ...],
) -> bool:
    info = infos[int(frame_index)]
    cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
    if not isinstance(cam_data, dict):
        return False
    image_path = resolve_image_path(dataset_root, cam_data, camera)
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    reference_time_s = reference_time_seconds_from_cameras(info, reference_cameras)
    target_ego_to_world = _interpolate_ego_transform(infos, reference_time_s)
    if target_ego_to_world is None:
        target_ego_to_world = np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64)
    world_to_target_ego = np.linalg.inv(target_ego_to_world)

    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
    before = image.copy()
    after = image.copy()
    before_count = 0
    after_count = 0
    for obj in info.get("objects", []):
        if not is_dynamic_object(obj):
            continue
        oid = object_id(obj)
        obj_type = str(obj.get("type", ""))
        before_corners = box_corners_ego(obj)
        if draw_box(
            before,
            before_corners,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 0, 255),
            label=f"{oid}:{obj_type}",
        ):
            before_count += 1
        target_box_world = _interpolate_track_transform(tracks.get(oid, []), reference_time_s)
        if target_box_world is None:
            continue
        size = np.asarray(obj["size"], dtype=np.float64).reshape(3)
        half = size / 2.0
        local = np.asarray(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=np.float64,
        )
        target_corners_ego = transform_points(
            transform_points(local, target_box_world),
            world_to_target_ego,
        )
        if draw_box(
            after,
            target_corners_ego,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 255, 0),
            label=f"{oid}:{obj_type}",
        ):
            after_count += 1

    put_label(before, f"before boxes | red | drawn={before_count}", color=(0, 0, 255))
    put_label(after, f"after boxes | green | drawn={after_count}", color=(0, 255, 0))
    canvas = np.concatenate([before, after], axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera}_{frame_index:06d}_{image_path.stem}_boxes_compare.jpg"
    return bool(cv2.imwrite(str(output_path), canvas))


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    frame_point_clouds = load_frame_point_clouds(
        infos,
        dataset_root=dataset_root,
        resolve_sensor_file=resolve_sensor_file,
    )
    tracks = build_object_tracks(infos, frame_point_clouds)
    cameras = tuple(item.strip() for item in str(args.cameras).split(",") if item.strip())
    reference_cameras = tuple(
        item.strip() for item in str(args.reference_cameras).split(",") if item.strip()
    )
    end = min(len(infos), int(args.end))
    total = 0
    for camera in cameras:
        out_dir = Path(args.output_dir) / camera
        rendered = 0
        for frame_index in range(int(args.start), end):
            if visualize_frame(
                infos=infos,
                tracks=tracks,
                frame_index=frame_index,
                camera=camera,
                dataset_root=dataset_root,
                output_dir=out_dir,
                reference_cameras=reference_cameras,
            ):
                rendered += 1
                total += 1
        print(f"[{camera}] rendered={rendered} -> {out_dir}")
    print(f"[done] total={total} -> {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize before/after dynamic 3D boxes.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cameras",
        default="center_camera_fov120,left_front_camera,right_front_camera",
    )
    parser.add_argument(
        "--reference-cameras",
        default=",".join(DEFAULT_REFERENCE_CAMERAS_6),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

