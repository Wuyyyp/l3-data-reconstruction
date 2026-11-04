from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    DEFAULT_REFERENCE_CAMERAS_6,
    _interpolate_ego_transform,
    _interpolate_track_transform,
    box_corners_from_transform,
    box_corners_ego,
    build_object_tracks,
    dynamic_objects_for_info,
    ego_world_transform,
    is_dynamic_object,
    load_frame_point_clouds,
    object_id,
    points_in_box_mask,
    read_pcd_cloud,
    reference_time_seconds_from_cameras,
    transform_points,
    write_polylinez_shapefile,
)


def load_infos(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def resolve_sensor_file(dataset_root: Path, data_path: str, fallback_folder: str) -> Path:
    name = Path(str(data_path)).name
    candidate = dataset_root / fallback_folder / name
    if candidate.exists():
        return candidate
    raw = Path(str(data_path))
    if raw.exists():
        return raw
    raise FileNotFoundError(candidate)


def _yaw_from_object(obj: dict[str, Any]) -> float:
    rotation = np.asarray(obj.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    if rotation.size >= 3:
        return float(rotation[2])
    return 0.0


def _box_feature(
    *,
    frame_index: int,
    obj: dict[str, Any],
    timestamp_s: float,
    corners: np.ndarray,
    status: str,
) -> dict[str, object]:
    size = np.asarray(obj.get("size", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return {
        "frame": int(frame_index),
        "object_id": object_id(obj),
        "object_type": str(obj.get("type", "")),
        "timestamp_s": float(timestamp_s),
        "corners": np.asarray(corners, dtype=np.float64).reshape(8, 3),
        "length": float(size[0]),
        "width": float(size[1]),
        "height": float(size[2]),
        "yaw": _yaw_from_object(obj),
        "status": status,
    }


def boxes_for_frame(
    *,
    infos: list[dict[str, Any]],
    tracks: dict[str, Any],
    cloud_xyz: np.ndarray,
    point_timestamps: np.ndarray | None,
    frame_index: int,
    reference_cameras: tuple[str, ...],
    dynamic_only: bool,
    min_object_points: int,
    box_margin_m: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    info = infos[int(frame_index)]
    frame_time_s = float(int(info["timestamp"]) / 1.0e9)
    reference_time_s = reference_time_seconds_from_cameras(info, reference_cameras)
    target_ego_to_world = _interpolate_ego_transform(infos, reference_time_s)
    if target_ego_to_world is None:
        target_ego_to_world = np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64)
    world_to_target_ego = np.linalg.inv(target_ego_to_world)

    before: list[dict[str, object]] = []
    after: list[dict[str, object]] = []
    claimed_points = np.zeros(len(cloud_xyz), dtype=bool)
    objects = dynamic_objects_for_info(info) if dynamic_only else (
        list(info.get("objects", []) or []) or dynamic_objects_for_info(info)
    )
    for obj in objects:
        dynamic = is_dynamic_object(obj)
        before_corners = box_corners_ego(obj)
        before_time_s = frame_time_s
        before_status = "before"
        after_corners = before_corners
        after_status = "static" if not dynamic else "no_track"
        if dynamic and point_timestamps is not None:
            mask = points_in_box_mask(cloud_xyz, obj, margin_m=box_margin_m) & ~claimed_points
            point_count = int(np.count_nonzero(mask))
            if point_count >= int(min_object_points):
                oid = object_id(obj)
                source_time_s = float(np.mean(point_timestamps[mask]))
                source_box_world = _interpolate_track_transform(tracks.get(oid, []), source_time_s)
                target_box_world = _interpolate_track_transform(tracks.get(oid, []), reference_time_s)
                if source_box_world is not None and target_box_world is not None:
                    after_corners = transform_points(
                        box_corners_from_transform(obj["size"], target_box_world),
                        world_to_target_ego,
                    )
                    after_status = "after_comp"
                    claimed_points[mask] = True
                else:
                    after_status = "no_track"
            else:
                after_status = "too_few_pts"
        before.append(
            _box_feature(
                frame_index=frame_index,
                obj=obj,
                timestamp_s=before_time_s,
                corners=before_corners,
                status=before_status,
            )
        )
        after.append(
            _box_feature(
                frame_index=frame_index,
                obj=obj,
                timestamp_s=reference_time_s,
                corners=after_corners,
                status=after_status,
            )
        )
    return before, after, reference_time_s


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
    reference_cameras = tuple(
        item.strip() for item in str(args.reference_cameras).split(",") if item.strip()
    )
    output_root = Path(args.output_dir)
    before_dir = output_root / "before"
    after_dir = output_root / "after"
    written = 0
    for frame_index in range(start, end):
        info = infos[frame_index]
        lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception", {})
        pcd_name = Path(str(lidar_data.get("data_path", f"frame_{frame_index:06d}.pcd"))).stem
        pcd_path = resolve_sensor_file(dataset_root, str(lidar_data["data_path"]), "perception")
        cloud = read_pcd_cloud(pcd_path)
        before, after, _reference_time_s = boxes_for_frame(
            infos=infos,
            tracks=tracks,
            cloud_xyz=cloud.xyz,
            point_timestamps=cloud.timestamps,
            frame_index=frame_index,
            reference_cameras=reference_cameras,
            dynamic_only=bool(args.dynamic_only),
            min_object_points=int(args.min_object_points),
            box_margin_m=float(args.box_margin_m),
        )
        stem = f"frame_{frame_index:06d}_{pcd_name}_boxes"
        write_polylinez_shapefile(before_dir / f"{stem}.shp", before)
        write_polylinez_shapefile(after_dir / f"{stem}.shp", after)
        written += 2
        print(
            f"[{frame_index:04d}] boxes={len(before)} -> "
            f"{before_dir / (stem + '.shp')} | {after_dir / (stem + '.shp')}"
        )
    print(f"[done] shapefiles={written} -> {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-frame ego-frame 3D boxes to PolyLineZ shapefiles before/after "
            "dynamic compensation."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, default=Path("result_interpolation_standard.pkl"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-cameras",
        default=",".join(DEFAULT_REFERENCE_CAMERAS_6),
    )
    parser.add_argument(
        "--dynamic-only",
        action="store_true",
        help="Export only dynamic objects. By default all objects are exported.",
    )
    parser.add_argument("--min-object-points", type=int, default=5)
    parser.add_argument("--box-margin-m", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

