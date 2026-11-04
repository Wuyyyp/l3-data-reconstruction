from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    _interpolate_ego_transform,
    ego_world_transform,
    pcd_path_is_ego_frame,
    read_pcd_cloud,
    timestamp_to_seconds,
    transform_points,
)
from visualize_dynamic_box_points_compare import (
    CAMERA_TO_IMAGE_FOLDER,
    draw_box,
    draw_points,
    draw_text,
    put_title,
    resolve_sensor_file,
)


RAW_LIDAR_FOLDER_ALIASES = {
    "front_left_lidar": "top_left_lidar",
    "front_right_lidar": "top_right_lidar",
}


def load_xyz_cached(
    path: Path,
    cache: dict[Path, np.ndarray],
    loader: Callable[[Path], np.ndarray],
) -> np.ndarray:
    key = Path(path).resolve()
    cached = cache.get(key)
    if cached is None:
        cached = np.asarray(loader(Path(path)), dtype=np.float64).reshape(-1, 3)
        cache[key] = cached
    return cached


def resize_image_and_intrinsic(
    image: np.ndarray,
    intrinsic: np.ndarray,
    *,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image)
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    if int(max_width) <= 0 or source.shape[1] <= int(max_width):
        return source, matrix
    scale = float(max_width) / float(source.shape[1])
    resized = cv2.resize(
        source,
        (int(max_width), max(1, int(round(source.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    matrix[0, :] *= scale
    matrix[1, :] *= scale
    return resized, matrix


def _write_qc_image(path: Path, image: np.ndarray, jpeg_quality: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(
        cv2.imwrite(
            str(path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
    )


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def _inside_corners(points: np.ndarray, corners: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    center = c.mean(axis=0)
    axes = np.vstack([c[1] - c[0], c[3] - c[0], c[4] - c[0]])
    lengths = np.linalg.norm(axes, axis=1)
    unit = axes / np.clip(lengths.reshape(3, 1), 1e-9, None)
    half = lengths / 2.0 + float(margin_m)
    local = (values - center.reshape(1, 3)) @ unit.T
    return np.all(np.abs(local) <= half.reshape(1, 3) + 1e-9, axis=1)


def select_points_compensated_from_before_box(
    original_xyz: np.ndarray,
    compensated_xyz: np.ndarray,
    before_corners: np.ndarray,
    *,
    margin_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the same point indices before/after, selected by the original box.

    Dynamic compensation moves points that originally fell inside the annotated
    box.  QC must therefore visualize that same point subset after compensation,
    not re-select points from the compensated cloud using a possibly different
    after box.
    """

    original = np.asarray(original_xyz, dtype=np.float64).reshape(-1, 3)
    compensated = np.asarray(compensated_xyz, dtype=np.float64).reshape(-1, 3)
    if len(original) != len(compensated):
        raise ValueError(
            "before/after QC clouds must have the same point count/order to visualize "
            "compensated points selected by the before box"
        )
    before_mask = _inside_corners(original, before_corners, margin_m=margin_m)
    return original[before_mask], compensated[before_mask], before_mask


def _raw_lidar_xyz_for_qc(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    lidar_name: str,
    xyz_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray | None:
    lidar_data = info.get("sensors", {}).get("lidar", {}).get(lidar_name)
    if not isinstance(lidar_data, dict):
        return None
    folder = RAW_LIDAR_FOLDER_ALIASES.get(lidar_name, lidar_name)
    lidar_path = str(lidar_data.get("data_path") or lidar_data.get("aws_path") or "")
    if not lidar_path:
        return None
    original_pcd = resolve_sensor_file(dataset_root, lidar_path, folder)

    def load_points(path: Path) -> np.ndarray:
        points = read_pcd_cloud(path).xyz
        if not pcd_path_is_ego_frame(path):
            points = transform_points(
                points,
                np.asarray(lidar_data["extrinsic"], dtype=np.float64).reshape(4, 4),
            )
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    if xyz_cache is None:
        return load_points(original_pcd)
    return load_xyz_cached(original_pcd, xyz_cache, load_points)


def transform_source_ego_to_target_ego(
    points_source_ego: np.ndarray,
    *,
    source_ego_to_world: np.ndarray,
    target_ego_to_world: np.ndarray,
) -> np.ndarray:
    points_world = transform_points(points_source_ego, source_ego_to_world)
    return transform_points(points_world, np.linalg.inv(target_ego_to_world))


def _camera_time_target_ego_to_world(
    *,
    infos: list[dict[str, Any]],
    info: dict[str, Any],
    cam_data: dict[str, Any],
    reference_time_offset_s: float,
) -> tuple[np.ndarray, float, bool]:
    camera_time_s = timestamp_to_seconds(cam_data["timestamp"])
    target_time_s = camera_time_s + float(reference_time_offset_s)
    target_ego_to_world = _interpolate_ego_transform(infos, target_time_s)
    fallback = target_ego_to_world is None
    if fallback:
        target_ego_to_world = ego_world_transform(info)
    return target_ego_to_world, target_time_s, fallback


def _original_xyz_for_qc(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    box_payload: dict[str, Any],
    lidar_name: str | None,
    xyz_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray | None:
    if lidar_name:
        return _raw_lidar_xyz_for_qc(
            dataset_root=dataset_root,
            info=info,
            lidar_name=lidar_name,
            xyz_cache=xyz_cache,
        )

    lidar_names = [
        str(item)
        for item in box_payload.get("lidar_names", [])
        if str(item)
    ]
    if lidar_names:
        chunks = []
        for raw_lidar_name in lidar_names:
            xyz = _raw_lidar_xyz_for_qc(
                dataset_root=dataset_root,
                info=info,
                lidar_name=raw_lidar_name,
                xyz_cache=xyz_cache,
            )
            if xyz is not None:
                chunks.append(xyz)
        if chunks:
            return np.concatenate(chunks, axis=0)

    perception_data = info.get("sensors", {}).get("lidar", {}).get("perception")
    if not isinstance(perception_data, dict):
        return None
    original_pcd = resolve_sensor_file(
        dataset_root,
        str(perception_data["data_path"]),
        "perception",
    )
    if xyz_cache is None:
        return np.asarray(read_pcd_cloud(original_pcd).xyz, dtype=np.float64).reshape(-1, 3)
    return load_xyz_cached(
        original_pcd,
        xyz_cache,
        lambda path: read_pcd_cloud(path).xyz,
    )


def _draw_saved_frame(
    *,
    dataset_root: Path,
    infos: list[dict[str, Any]],
    frame_index: int,
    camera: str,
    box_payload: dict[str, Any],
    compensated_pcd_dir: Path,
    output_dir: Path,
    max_depth_m: float,
    point_radius: int,
    box_margin_m: float,
    lidar_name: str | None = None,
    panel_width: int = 1280,
    jpeg_quality: int = 90,
    xyz_cache: dict[Path, np.ndarray] | None = None,
    point_selection_cache: dict[tuple[int, str, str, float], tuple[np.ndarray, np.ndarray]] | None = None,
    write_executor: ThreadPoolExecutor | None = None,
    write_futures: list[Future] | None = None,
) -> bool:
    info = infos[frame_index]
    cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
    lidar_key = lidar_name or "perception"
    lidar_data = info.get("sensors", {}).get("lidar", {}).get(lidar_key)
    perception_data = info.get("sensors", {}).get("lidar", {}).get("perception")
    if not isinstance(cam_data, dict) or not isinstance(lidar_data, dict):
        return False
    if not isinstance(perception_data, dict):
        return False
    image_path = resolve_sensor_file(
        dataset_root,
        str(cam_data["data_path"]),
        CAMERA_TO_IMAGE_FOLDER.get(camera, camera),
    )
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    if lidar_name:
        perception_pcd_name = Path(str(perception_data.get("data_path", ""))).name
        compensated_pcd = compensated_pcd_dir / lidar_name / perception_pcd_name
    else:
        perception_pcd_name = Path(str(perception_data.get("data_path", ""))).name
        compensated_pcd = compensated_pcd_dir / perception_pcd_name
    if not compensated_pcd.exists():
        return False
    if xyz_cache is None:
        compensated_xyz = np.asarray(
            read_pcd_cloud(compensated_pcd).xyz, dtype=np.float64
        ).reshape(-1, 3)
    else:
        compensated_xyz = load_xyz_cached(
            compensated_pcd,
            xyz_cache,
            lambda path: read_pcd_cloud(path).xyz,
        )
    original_xyz = _original_xyz_for_qc(
        dataset_root=dataset_root,
        info=info,
        box_payload=box_payload,
        lidar_name=lidar_name,
        xyz_cache=xyz_cache,
    )
    if original_xyz is None:
        return False
    reference_time_offset_s = float(box_payload.get("reference_time_offset_s") or 0.0)
    source_ego_to_world = ego_world_transform(info)
    target_ego_to_world, target_camera_time_s, target_fallback = _camera_time_target_ego_to_world(
        infos=infos,
        info=info,
        cam_data=cam_data,
        reference_time_offset_s=reference_time_offset_s,
    )
    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    image, intrinsic = resize_image_and_intrinsic(
        image,
        np.asarray(cam_data["cam_intrinsic"], dtype=np.float64),
        max_width=int(panel_width),
    )

    panel_boxes = image.copy()
    panel_before = image.copy()
    panel_after = image.copy()
    before_points_drawn = 0
    after_points_drawn = 0
    object_count = 0
    before_by_id = {str(item.get("object_id", "")): item for item in box_payload.get("before", [])}
    after_by_id = {str(item.get("object_id", "")): item for item in box_payload.get("after", [])}
    for oid, before in before_by_id.items():
        after = after_by_id.get(oid)
        if after is None:
            continue
        before_corners = np.asarray(before["corners"], dtype=np.float64).reshape(8, 3)
        after_corners = np.asarray(after["corners"], dtype=np.float64).reshape(8, 3)
        before_corners_target = transform_source_ego_to_target_ego(
            before_corners,
            source_ego_to_world=source_ego_to_world,
            target_ego_to_world=target_ego_to_world,
        )
        after_corners_target = transform_source_ego_to_target_ego(
            after_corners,
            source_ego_to_world=source_ego_to_world,
            target_ego_to_world=target_ego_to_world,
        )
        before_drawn, before_anchor = draw_box(
            panel_boxes,
            before_corners_target,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 0, 255),
            thickness=1,
        )
        after_drawn, after_anchor = draw_box(
            panel_boxes,
            after_corners_target,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            color=(0, 255, 0),
            thickness=1,
        )
        selection_key = (frame_index, lidar_key, oid, float(box_margin_m))
        selected = (
            point_selection_cache.get(selection_key)
            if point_selection_cache is not None
            else None
        )
        if selected is None:
            try:
                before_points, after_points, _before_mask = select_points_compensated_from_before_box(
                    original_xyz,
                    compensated_xyz,
                    before_corners,
                    margin_m=box_margin_m,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{camera} frame={frame_index} lidar={lidar_key}: {exc}"
                ) from exc
            if point_selection_cache is not None:
                point_selection_cache[selection_key] = (before_points, after_points)
        else:
            before_points, after_points = selected
        before_points_drawn += draw_points(
            panel_before,
            transform_source_ego_to_target_ego(
                before_points,
                source_ego_to_world=source_ego_to_world,
                target_ego_to_world=target_ego_to_world,
            ),
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            max_depth_m=max_depth_m,
            radius=point_radius,
        )
        after_points_drawn += draw_points(
            panel_after,
            transform_source_ego_to_target_ego(
                after_points,
                source_ego_to_world=source_ego_to_world,
                target_ego_to_world=target_ego_to_world,
            ),
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            max_depth_m=max_depth_m,
            radius=point_radius,
        )
        delta = after_corners.mean(axis=0) - before_corners.mean(axis=0)
        label = (
            f"id={oid} {before.get('object_type', '')}\n"
            f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})m |d|={np.linalg.norm(delta):.3f}m"
        )
        anchor = after_anchor or before_anchor
        if (before_drawn or after_drawn) and anchor is not None:
            draw_text(panel_boxes, label, anchor, color=(255, 255, 255), scale=0.5)
        object_count += 1

    lidar_label = f" | lidar={lidar_name}" if lidar_name else ""
    put_title(
        panel_boxes,
        f"saved boxes{lidar_label} | before red / after green | objects={object_count} | "
        f"cam ego offset={reference_time_offset_s*1000:.1f}ms",
        (255, 255, 255),
    )
    put_title(
        panel_before,
        f"before projection{lidar_label} | before-box points={before_points_drawn} | "
        f"cam_time={target_camera_time_s:.6f}s",
        (0, 0, 255),
    )
    put_title(
        panel_after,
        f"after projection{lidar_label} | same before-box points after compensation={after_points_drawn}"
        + (" | interp fallback" if target_fallback else ""),
        (0, 255, 0),
    )
    canvas = np.concatenate([panel_boxes, panel_before, panel_after], axis=1)
    prefix = f"{lidar_name}_" if lidar_name else ""
    output_path = output_dir / f"{prefix}{camera}_{frame_index:06d}_{image_path.stem}_saved_dynamic_qc.jpg"
    if write_executor is None:
        return _write_qc_image(output_path, canvas, jpeg_quality)
    if write_futures is None:
        raise ValueError("write_futures is required with write_executor")
    write_futures.append(
        write_executor.submit(_write_qc_image, output_path, canvas, jpeg_quality)
    )
    return True


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    box_dir = Path(args.box_dir)
    compensated_pcd_dir = Path(args.compensated_pcd_dir)
    box_by_lidar_dir = Path(args.box_by_lidar_dir) if args.box_by_lidar_dir else None
    output_root = Path(args.output_dir)
    cameras = tuple(item.strip() for item in str(args.cameras).split(",") if item.strip())
    lidars = tuple(item.strip() for item in str(args.lidars).split(",") if item.strip())
    sample_ratio = float(args.sample_ratio)
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1]")
    rng = np.random.default_rng(int(args.sample_seed))
    total = 0
    xyz_cache: dict[Path, np.ndarray] = {}
    point_selection_cache: dict[
        tuple[int, str, str, float], tuple[np.ndarray, np.ndarray]
    ] = {}
    write_futures: list[Future] = []
    write_workers = max(1, int(args.write_workers))
    with ThreadPoolExecutor(max_workers=write_workers) as write_executor:
        for json_path in sorted(box_dir.glob("frame_*_boxes.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            frame_index = int(payload["frame_index"])
            if frame_index < int(args.start) or (
                int(args.end) >= 0 and frame_index >= int(args.end)
            ):
                continue
            if sample_ratio < 1.0 and float(rng.random()) > sample_ratio:
                continue
            if lidars:
                if box_by_lidar_dir is None:
                    raise ValueError("--box-by-lidar-dir is required when --lidars is set")
                for lidar in lidars:
                    lidar_json = box_by_lidar_dir / lidar / json_path.name
                    if not lidar_json.exists():
                        continue
                    lidar_payload = json.loads(lidar_json.read_text(encoding="utf-8"))
                    for camera in cameras:
                        out_dir = output_root / lidar / camera
                        if _draw_saved_frame(
                            dataset_root=dataset_root,
                            infos=infos,
                            frame_index=frame_index,
                            camera=camera,
                            box_payload=lidar_payload,
                            compensated_pcd_dir=compensated_pcd_dir,
                            output_dir=out_dir,
                            max_depth_m=float(args.max_depth_m),
                            point_radius=int(args.point_radius),
                            box_margin_m=float(args.box_margin_m),
                            lidar_name=lidar,
                            panel_width=int(args.panel_width),
                            jpeg_quality=int(args.jpeg_quality),
                            xyz_cache=xyz_cache,
                            point_selection_cache=point_selection_cache,
                            write_executor=write_executor,
                            write_futures=write_futures,
                        ):
                            total += 1
            else:
                for camera in cameras:
                    out_dir = output_root / camera
                    if _draw_saved_frame(
                        dataset_root=dataset_root,
                        infos=infos,
                        frame_index=frame_index,
                        camera=camera,
                        box_payload=payload,
                        compensated_pcd_dir=compensated_pcd_dir,
                        output_dir=out_dir,
                        max_depth_m=float(args.max_depth_m),
                        point_radius=int(args.point_radius),
                        box_margin_m=float(args.box_margin_m),
                        panel_width=int(args.panel_width),
                        jpeg_quality=int(args.jpeg_quality),
                        xyz_cache=xyz_cache,
                        point_selection_cache=point_selection_cache,
                        write_executor=write_executor,
                        write_futures=write_futures,
                    ):
                        total += 1
            while len(write_futures) >= 2 * write_workers:
                if not bool(write_futures.pop(0).result()):
                    raise OSError("failed to write dynamic compensation QC image")
        for future in write_futures:
            if not bool(future.result()):
                raise OSError("failed to write dynamic compensation QC image")
    print(f"[done] visualizations={total} -> {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize saved raw-lidar dynamic compensation outputs."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--box-dir", type=Path, required=True)
    parser.add_argument("--box-by-lidar-dir", type=Path)
    parser.add_argument("--compensated-pcd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument(
        "--cameras",
        default="center_camera_fov120,left_front_camera,right_front_camera",
    )
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--box-margin-m", type=float, default=0.0)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--panel-width", type=int, default=1280)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--lidars", default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

