from __future__ import annotations

import argparse
import json
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
from compensate_dynamic_perception_pcd import resolve_sensor_file
from visualize_dynamic_box_points_compare import (
    CAMERA_TO_IMAGE_FOLDER,
    draw_box,
    draw_points,
    put_title,
)
from visualize_saved_dynamic_compensation_qc import (
    _original_xyz_for_qc,
    select_points_compensated_from_before_box,
)


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def parse_offset_output_specs(values: list[str]) -> list[tuple[float, Path]]:
    specs: list[tuple[float, Path]] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--offset-output must be formatted as <offset_ms>=<output_root>")
        offset_text, path_text = item.split("=", maxsplit=1)
        specs.append((float(offset_text), Path(path_text)))
    if not specs:
        raise ValueError("at least one --offset-output is required")
    return sorted(specs, key=lambda row: row[0])


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _target_ego_to_world(
    *,
    infos: list[dict[str, Any]],
    info: dict[str, Any],
    cam_data: dict[str, Any],
    offset_ms: float,
) -> tuple[np.ndarray, bool]:
    target_time_s = timestamp_to_seconds(cam_data["timestamp"]) + float(offset_ms) * 1.0e-3
    target = _interpolate_ego_transform(infos, target_time_s)
    if target is None:
        return ego_world_transform(info), True
    return target, False


def _load_box_payload(root: Path, frame_index: int) -> dict[str, Any] | None:
    box_dir = root / "dynamic_3d_boxes_json"
    matches = sorted(box_dir.glob(f"frame_{int(frame_index):06d}_*_boxes.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _compensated_pcd_path(root: Path, info: dict[str, Any]) -> Path:
    perception = info.get("sensors", {}).get("lidar", {}).get("perception", {})
    pcd_name = Path(str(perception.get("data_path", ""))).name
    return root / "compensated_perception" / pcd_name


def original_xyz_for_offset_qc(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    box_payload: dict[str, Any],
) -> np.ndarray | None:
    """Return before points in the same order as compensated offset output."""
    return _original_xyz_for_qc(
        dataset_root=dataset_root,
        info=info,
        box_payload=box_payload,
        lidar_name=None,
    )


def _draw_selected_objects(
    *,
    panel: np.ndarray,
    before_points: np.ndarray,
    after_points: np.ndarray,
    box_payload: dict[str, Any],
    source_ego_to_world: np.ndarray,
    target_ego_to_world: np.ndarray,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    max_depth_m: float,
    point_radius: int,
    draw_after_boxes: bool,
) -> tuple[int, int]:
    before_by_id = {str(item.get("object_id", "")): item for item in box_payload.get("before", [])}
    after_by_id = {str(item.get("object_id", "")): item for item in box_payload.get("after", [])}
    selected_before: list[np.ndarray] = []
    selected_after: list[np.ndarray] = []
    object_count = 0
    for oid, before in before_by_id.items():
        after = after_by_id.get(oid)
        if after is None:
            continue
        before_corners = np.asarray(before["corners"], dtype=np.float64).reshape(8, 3)
        after_corners = np.asarray(after["corners"], dtype=np.float64).reshape(8, 3)
        try:
            object_before, object_after, _ = select_points_compensated_from_before_box(
                before_points,
                after_points,
                before_corners,
                margin_m=0.0,
            )
        except ValueError:
            continue
        if len(object_before):
            selected_before.append(object_before)
            selected_after.append(object_after)
        if draw_after_boxes:
            after_corners_target = transform_points(
                transform_points(after_corners, source_ego_to_world),
                np.linalg.inv(target_ego_to_world),
            )
            draw_box(
                panel,
                after_corners_target,
                ego_to_camera=ego_to_camera,
                intrinsic=intrinsic,
                color=(0, 255, 0),
                thickness=1,
            )
        object_count += 1
    if not selected_after:
        return 0, object_count
    stacked_after = np.concatenate(selected_after, axis=0)
    after_target = transform_points(
        transform_points(stacked_after, source_ego_to_world),
        np.linalg.inv(target_ego_to_world),
    )
    drawn = draw_points(
        panel,
        after_target,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        radius=point_radius,
    )
    return int(drawn), object_count


def render_frame_camera(
    *,
    infos: list[dict[str, Any]],
    dataset_root: Path,
    frame_index: int,
    camera: str,
    offset_outputs: list[tuple[float, Path]],
    output_dir: Path,
    max_depth_m: float,
    point_radius: int,
    original_xyz_cache: dict[int, np.ndarray] | None = None,
    compensated_xyz_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> bool:
    info = infos[frame_index]
    cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
    perception = info.get("sensors", {}).get("lidar", {}).get("perception")
    if not isinstance(cam_data, dict) or not isinstance(perception, dict):
        return False
    image_path = resolve_sensor_file(
        dataset_root,
        str(cam_data["data_path"]),
        CAMERA_TO_IMAGE_FOLDER.get(camera, camera),
    )
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    source_ego_to_world = ego_world_transform(info)
    ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)

    panels: list[np.ndarray] = []
    first_payload = _load_box_payload(offset_outputs[0][1], frame_index)
    if first_payload is None:
        return False
    original_xyz = None
    if original_xyz_cache is not None:
        original_xyz = original_xyz_cache.get(int(frame_index))
    if original_xyz is None:
        original_xyz = original_xyz_for_offset_qc(
            dataset_root=dataset_root,
            info=info,
            box_payload=first_payload,
        )
        if original_xyz is not None and original_xyz_cache is not None:
            original_xyz_cache[int(frame_index)] = original_xyz
    if original_xyz is None:
        return False
    before_panel = image.copy()
    raw_drawn, raw_objects = _draw_selected_objects(
        panel=before_panel,
        before_points=original_xyz,
        after_points=original_xyz,
        box_payload=first_payload,
        source_ego_to_world=source_ego_to_world,
        target_ego_to_world=source_ego_to_world,
        ego_to_camera=ego_to_camera,
        intrinsic=intrinsic,
        max_depth_m=max_depth_m,
        point_radius=point_radius,
        draw_after_boxes=False,
    )
    put_title(
        before_panel,
        f"raw before-box points | drawn={raw_drawn} | objects={raw_objects}",
        (0, 0, 255),
    )
    panels.append(before_panel)

    for offset_ms, root in offset_outputs:
        payload = _load_box_payload(root, frame_index)
        if payload is None:
            continue
        pcd_path = _compensated_pcd_path(root, info)
        if not pcd_path.exists():
            continue
        cache_key = (str(root), int(frame_index))
        compensated_xyz = None
        if compensated_xyz_cache is not None:
            compensated_xyz = compensated_xyz_cache.get(cache_key)
        if compensated_xyz is None:
            compensated_xyz = np.asarray(read_pcd_cloud(pcd_path).xyz, dtype=np.float64).reshape(-1, 3)
            if compensated_xyz_cache is not None:
                compensated_xyz_cache[cache_key] = compensated_xyz
        target_ego_to_world, fallback = _target_ego_to_world(
            infos=infos,
            info=info,
            cam_data=cam_data,
            offset_ms=offset_ms,
        )
        panel = image.copy()
        drawn, objects = _draw_selected_objects(
            panel=panel,
            before_points=original_xyz,
            after_points=compensated_xyz,
            box_payload=payload,
            source_ego_to_world=source_ego_to_world,
            target_ego_to_world=target_ego_to_world,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            max_depth_m=max_depth_m,
            point_radius=point_radius,
            draw_after_boxes=True,
        )
        suffix = " | interp fallback" if fallback else ""
        put_title(panel, f"offset={offset_ms:g}ms | box points={drawn} | objects={objects}{suffix}", (0, 255, 0))
        panels.append(panel)

    if len(panels) <= 1:
        return False
    height = max(panel.shape[0] for panel in panels)
    canvas = np.concatenate([_resize_to_height(panel, height) for panel in panels], axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera}_{frame_index:06d}_{image_path.stem}_dynamic_offset_sweep_box_points.jpg"
    return bool(cv2.imwrite(str(output_path), canvas))


def select_frame_indices(
    *, start: int, end: int, sample_ratio: float, sample_seed: int
) -> list[int]:
    indices = list(range(int(start), int(end)))
    ratio = float(sample_ratio)
    if ratio >= 1.0:
        return indices
    rng = np.random.default_rng(int(sample_seed))
    keep_count = max(1, int(round(len(indices) * ratio)))
    selected = rng.choice(np.asarray(indices, dtype=np.int64), size=keep_count, replace=False)
    return sorted(int(value) for value in selected)


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    pkl_path = Path(args.pkl)
    if not pkl_path.is_absolute():
        pkl_path = dataset_root / pkl_path
    infos = load_infos(pkl_path)
    cameras = [item.strip() for item in str(args.cameras).split(",") if item.strip()]
    offset_outputs = parse_offset_output_specs(list(args.offset_output))
    output_root = Path(args.output_dir)
    frame_end = min(len(infos), int(args.end))
    frames = select_frame_indices(
        start=int(args.start),
        end=frame_end,
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
    )
    total = 0
    original_xyz_cache: dict[int, np.ndarray] = {}
    compensated_xyz_cache: dict[tuple[str, int], np.ndarray] = {}
    for frame_index in frames:
        for camera in cameras:
            if render_frame_camera(
                infos=infos,
                dataset_root=dataset_root,
                frame_index=frame_index,
                camera=camera,
                offset_outputs=offset_outputs,
                output_dir=output_root / camera,
                max_depth_m=float(args.max_depth_m),
                point_radius=int(args.point_radius),
                original_xyz_cache=original_xyz_cache,
                compensated_xyz_cache=compensated_xyz_cache,
            ):
                total += 1
        print(f"[done] frame={frame_index:04d}")
    print(f"[done] visualizations={total} -> {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare dynamic object compensation projections across exposure offsets."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset-output", action="append", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--cameras",
        default="center_camera_fov120,center_camera_fov30,left_front_camera,left_rear_camera,rear_camera,right_front_camera,right_rear_camera",
    )
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--point-radius", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())


