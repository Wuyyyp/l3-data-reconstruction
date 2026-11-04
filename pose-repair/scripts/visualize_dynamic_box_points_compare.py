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
    box_corners_from_transform,
    box_correction_delta_ego,
    box_edges_polylinez,
    build_object_tracks,
    compensate_points_between_box_poses,
    dynamic_objects_for_info,
    ego_world_transform,
    load_frame_point_clouds,
    object_id,
    points_in_box_mask,
    read_pcd_cloud,
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


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_sensor_file(dataset_root: Path, data_path: str, fallback_folder: str) -> Path:
    raw_path = Path(str(data_path))
    name = raw_path.name
    raw_folder = raw_path.parent.name
    candidates = [
        dataset_root / raw_path,
        dataset_root / "images" / raw_folder / name,
        dataset_root / "images" / fallback_folder / name,
        dataset_root / raw_folder / name,
        dataset_root / "lidar" / raw_folder / name,
        dataset_root / fallback_folder / name,
        dataset_root / "lidar" / fallback_folder / name,
        dataset_root / "lidar" / f"{fallback_folder}_compensation" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(candidates[0])


def project_points(
    points_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int, int],
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    points_camera = (np.asarray(ego_to_camera, dtype=np.float64).reshape(4, 4) @ homogeneous.T)[
        :3
    ]
    depth = points_camera[2]
    uvw = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points_camera
    uv = uvw[:2] / np.clip(uvw[2:3], 1e-8, None)
    valid = (
        np.isfinite(depth)
        & np.isfinite(uv[0])
        & np.isfinite(uv[1])
        & (depth > 0.05)
        & (depth <= float(max_depth_m))
        & (uv[0] >= 0)
        & (uv[0] < width)
        & (uv[1] >= 0)
        & (uv[1] < height)
    )
    return (
        np.rint(uv[0, valid]).astype(np.int32),
        np.rint(uv[1, valid]).astype(np.int32),
        depth[valid].astype(np.float32),
    )


def draw_points(
    image: np.ndarray,
    points_ego: np.ndarray,
    *,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    max_depth_m: float,
    radius: int,
    color_override: tuple[int, int, int] | None = None,
) -> int:
    xs, ys, depths = project_points(
        points_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        image_shape=image.shape,
        max_depth_m=max_depth_m,
    )
    if len(xs) == 0:
        return 0
    if color_override is None:
        near = float(np.percentile(depths, 1))
        far = float(np.percentile(depths, 99))
        if far <= near:
            far = near + 1.0
        normalized = np.clip((depths - near) / (far - near), 0.0, 1.0)
        colors = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        ).reshape(-1, 3)
    else:
        colors = np.tile(np.asarray(color_override, dtype=np.uint8).reshape(1, 3), (len(xs), 1))
    r = max(1, int(radius))
    for x, y, color in zip(xs, ys, colors, strict=False):
        cv2.circle(image, (int(x), int(y)), r, color.tolist(), -1, cv2.LINE_AA)
    return int(len(xs))


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
    thickness: int,
) -> tuple[bool, tuple[int, int] | None]:
    height, width = image.shape[:2]
    uv, depth = project_corners(
        corners_ego,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
    )
    drawn = False
    for edge in box_edges_polylinez(corners_ego):
        local_indices = []
        for point in edge:
            distances = np.linalg.norm(np.asarray(corners_ego) - point.reshape(1, 3), axis=1)
            local_indices.append(int(np.argmin(distances)))
        a, b = local_indices
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
        cv2.line(image, pa, pb, color, int(thickness), cv2.LINE_AA)
        drawn = True
    valid = (depth > 0.05) & np.isfinite(uv).all(axis=1)
    anchor = None
    if valid.any():
        pt = np.rint(uv[valid].mean(axis=0)).astype(int)
        anchor = (int(pt[0]), int(pt[1]))
    return drawn, anchor


def draw_text(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    height, width = image.shape[:2]
    x = int(np.clip(anchor[0] + 8, 8, width - 360))
    y = int(np.clip(anchor[1] - 8, 24, height - 12))
    for line_index, line in enumerate(text.split("\n")):
        yy = y + line_index * int(22 * scale / 0.55)
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def put_title(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def _target_corners_ego(
    *,
    obj: dict[str, Any],
    target_box_world: np.ndarray,
    world_to_target_ego: np.ndarray,
) -> np.ndarray:
    return transform_points(
        box_corners_from_transform(obj["size"], target_box_world),
        world_to_target_ego,
    ).astype(
        np.float64
    )


def _source_corners_ego(
    *,
    obj: dict[str, Any],
    source_box_world: np.ndarray,
    world_to_source_ego: np.ndarray,
) -> np.ndarray:
    return transform_points(
        box_corners_from_transform(obj["size"], source_box_world),
        world_to_source_ego,
    ).astype(np.float64)


def visualize_frame(
    *,
    infos: list[dict[str, Any]],
    tracks: dict[str, Any],
    frame_index: int,
    dataset_root: Path,
    camera: str,
    output_dir: Path,
    reference_cameras: tuple[str, ...],
    max_depth_m: float,
    point_radius: int,
    box_margin_m: float,
    min_object_points: int,
) -> bool:
    info = infos[int(frame_index)]
    cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
    lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception")
    if not isinstance(cam_data, dict) or not isinstance(lidar_data, dict):
        return False
    image_path = resolve_sensor_file(
        dataset_root,
        str(cam_data["data_path"]),
        CAMERA_TO_IMAGE_FOLDER.get(camera, camera),
    )
    pcd_path = resolve_sensor_file(dataset_root, str(lidar_data["data_path"]), "perception")
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    cloud = read_pcd_cloud(pcd_path)
    xyz = cloud.xyz
    timestamps = cloud.timestamps
    if timestamps is None:
        return False

    reference_time_s = reference_time_seconds_from_cameras(info, reference_cameras)
    target_ego_to_world = _interpolate_ego_transform(infos, reference_time_s)
    if target_ego_to_world is None:
        target_ego_to_world = ego_world_transform(info)
    world_to_target_ego = np.linalg.inv(target_ego_to_world)
    source_ego_to_world = ego_world_transform(info)

    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
    boxes_overlay = image.copy()
    before_projection = image.copy()
    after_projection = image.copy()
    moved = np.zeros(len(xyz), dtype=bool)
    object_count = 0
    before_point_count = 0
    after_point_count = 0
    for obj in dynamic_objects_for_info(info):
        mask = points_in_box_mask(xyz, obj, margin_m=box_margin_m) & ~moved
        count = int(np.count_nonzero(mask))
        if count < int(min_object_points):
            continue
        oid = object_id(obj)
        target_box_world = _interpolate_track_transform(tracks.get(oid, []), reference_time_s)
        if target_box_world is None:
            continue
        source_time_s = float(np.mean(timestamps[mask]))
        source_box_world = _interpolate_track_transform(tracks.get(oid, []), source_time_s)
        if source_box_world is None:
            continue
        before_points = xyz[mask]
        after_points = compensate_points_between_box_poses(
            before_points,
            source_box_to_world=source_box_world,
            target_box_to_world=target_box_world,
            world_to_output_ego=world_to_target_ego,
            source_ego_to_world=source_ego_to_world,
        )
        before_corners = box_corners_ego(obj).astype(np.float64)
        after_corners = _target_corners_ego(
            obj=obj,
            target_box_world=target_box_world,
            world_to_target_ego=world_to_target_ego,
        )
        delta = box_correction_delta_ego(before_corners, after_corners)
        delta_norm = float(np.linalg.norm(delta))

        drawn_before = draw_points(
            before_projection,
            before_points,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            max_depth_m=max_depth_m,
            radius=point_radius,
        )
        drawn_after = draw_points(
            after_projection,
            after_points,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            max_depth_m=max_depth_m,
            radius=point_radius,
        )
        before_box_drawn, before_anchor = draw_box(
            boxes_overlay,
            before_corners,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 0, 255),
            thickness=1,
        )
        after_box_drawn, after_anchor = draw_box(
            boxes_overlay,
            after_corners,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 255, 0),
            thickness=1,
        )
        label = (
            f"id={oid} {obj.get('type', '')} n={count}\n"
            f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})m |d|={delta_norm:.3f}m"
        )
        if before_box_drawn and after_box_drawn and before_anchor is not None and after_anchor is not None:
            midpoint = (
                int(round((before_anchor[0] + after_anchor[0]) / 2.0)),
                int(round((before_anchor[1] + after_anchor[1]) / 2.0)),
            )
            draw_text(boxes_overlay, label, midpoint, color=(255, 255, 255), scale=0.5)
        elif before_box_drawn and before_anchor is not None:
            draw_text(boxes_overlay, label, before_anchor, color=(255, 255, 255), scale=0.5)
        elif after_box_drawn and after_anchor is not None:
            draw_text(boxes_overlay, label, after_anchor, color=(255, 255, 255), scale=0.5)
        if drawn_before or drawn_after:
            moved[mask] = True
            object_count += 1
            before_point_count += int(drawn_before)
            after_point_count += int(drawn_after)

    put_title(
        boxes_overlay,
        f"boxes only | before red / after green | objects={object_count}",
        (255, 255, 255),
    )
    put_title(
        before_projection,
        f"before projection | box-internal raw points only | objects={object_count} points={before_point_count}",
        (0, 0, 255),
    )
    put_title(
        after_projection,
        f"after projection | compensated box-internal points only | objects={object_count} points={after_point_count}",
        (0, 255, 0),
    )
    canvas = np.concatenate([boxes_overlay, before_projection, after_projection], axis=1)
    base_dir = Path(output_dir)
    stem = f"{camera}_{frame_index:06d}_{image_path.stem}"
    output_path = base_dir / f"{stem}_box_points_projection_qc.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    start = int(args.start)
    end = min(len(infos), int(args.end))
    track_start = max(0, start - 1)
    track_end = min(len(infos), end + 1)
    frame_point_clouds = load_frame_point_clouds(
        infos,
        dataset_root=dataset_root,
        resolve_sensor_file=resolve_sensor_file,
        start=track_start,
        end=track_end,
    )
    tracks = build_object_tracks(
        infos[track_start:track_end],
        frame_point_clouds,
        min_object_points=int(args.min_object_points),
        box_margin_m=float(args.box_margin_m),
    )
    cameras = tuple(item.strip() for item in str(args.cameras).split(",") if item.strip())
    reference_cameras = tuple(
        item.strip() for item in str(args.reference_cameras).split(",") if item.strip()
    )
    total = 0
    for camera in cameras:
        out_dir = Path(args.output_dir) / camera
        rendered = 0
        for frame_index in range(start, end):
            if visualize_frame(
                infos=infos,
                tracks=tracks,
                frame_index=frame_index,
                dataset_root=dataset_root,
                camera=camera,
                output_dir=out_dir,
                reference_cameras=reference_cameras,
                max_depth_m=float(args.max_depth_m),
                point_radius=int(args.point_radius),
                box_margin_m=float(args.box_margin_m),
                min_object_points=int(args.min_object_points),
            ):
                rendered += 1
                total += 1
        print(f"[{camera}] rendered={rendered} -> {out_dir}")
    print(f"[done] total={total} -> {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize before/after dynamic compensation using only points inside "
            "dynamic 3D boxes, plus the 3D boxes and ego-frame correction labels."
        )
    )
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
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--box-margin-m", type=float, default=0.0)
    parser.add_argument("--min-object-points", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

