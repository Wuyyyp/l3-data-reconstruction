from __future__ import annotations

import argparse
import copy
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dongfeng_registration.dynamic_object_compensation import (
    DEFAULT_REFERENCE_CAMERAS_6,
    interpolate_transform,
    reference_time_seconds_from_cameras,
    timestamp_to_seconds,
)


DEFAULT_OFFSET_FIELD = "ego2global_transformation_matrix_camera_reference_offset"
DEFAULT_OFFSET_TIMESTAMP_FIELD = "camera_reference_offset_timestamp"


def load_pickle(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), list):
        raise ValueError(f"{path} must contain a dict with list field 'infos'")
    return payload


def box_center_from_corners(corners: Any) -> list[float]:
    values = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    return values.mean(axis=0).astype(float).tolist()


def _as_python_matrix(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=np.float64).reshape(4, 4).astype(float).tolist()


def _object_after_by_id(box_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in box_payload.get("after", []) or []:
        if isinstance(row, dict) and row.get("object_id") is not None:
            rows[str(row["object_id"])] = row
    return rows


def interpolate_ego_from_precomputed(
    times_s: np.ndarray,
    ego_to_world: np.ndarray,
    time_s: float,
) -> np.ndarray | None:
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    transforms = np.asarray(ego_to_world, dtype=np.float64).reshape(-1, 4, 4)
    if len(times) < 2 or len(times) != len(transforms):
        return None
    index = int(np.searchsorted(times, float(time_s), side="right"))
    left = index - 1
    right = index
    if left < 0 or right >= len(times) or times[right] <= times[left]:
        return None
    alpha = (float(time_s) - float(times[left])) / float(times[right] - times[left])
    return interpolate_transform(transforms[left], transforms[right], alpha)


def update_info_for_dynamic_compensation(
    info: dict[str, Any],
    *,
    box_payload: dict[str, Any],
    compensated_pcd_relative_path: str,
    offset_ego_to_world: np.ndarray,
    offset_timestamp_ns: int,
    offset_field_name: str = DEFAULT_OFFSET_FIELD,
    offset_timestamp_field_name: str = DEFAULT_OFFSET_TIMESTAMP_FIELD,
    after_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(info)
    updated[offset_timestamp_field_name] = int(offset_timestamp_ns)
    updated[offset_field_name] = _as_python_matrix(offset_ego_to_world)
    updated["dynamic_compensation_reference_time_s"] = float(box_payload.get("reference_time_s"))
    updated["dynamic_compensation_reference_time_offset_s"] = float(
        box_payload.get("reference_time_offset_s") or 0.0
    )

    lidar = updated.setdefault("sensors", {}).setdefault("lidar", {})
    perception = lidar.setdefault("perception", {})
    original_data_path = perception.get("data_path")
    original_aws_path = perception.get("aws_path")
    if "original_data_path" not in perception:
        perception["original_data_path"] = original_data_path
    if "original_aws_path" not in perception:
        perception["original_aws_path"] = original_aws_path
    perception["data_path"] = str(compensated_pcd_relative_path)
    perception["aws_path"] = str(compensated_pcd_relative_path)
    perception["dynamic_compensated"] = True

    after_by_id = after_by_id or _object_after_by_id(box_payload)
    updated_gt_box_count = 0
    for obj in updated.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        after = after_by_id.get(str(obj.get("id", "")))
        if after is None:
            continue
        obj["location"] = box_center_from_corners(after["corners"])
        obj["size"] = [
            float(after.get("length", obj.get("size", [0.0, 0.0, 0.0])[0])),
            float(after.get("width", obj.get("size", [0.0, 0.0, 0.0])[1])),
            float(after.get("height", obj.get("size", [0.0, 0.0, 0.0])[2])),
        ]
        obj["rotation"] = [0.0, 0.0, float(after.get("yaw", 0.0))]
        obj["dynamic_compensation"] = {
            "enabled": True,
            "status": str(after.get("status", "")),
            "reference_time_s": float(box_payload.get("reference_time_s")),
            "reference_time_offset_s": float(box_payload.get("reference_time_offset_s") or 0.0),
            "object_timestamp_s": float(after.get("timestamp_s", box_payload.get("reference_time_s"))),
            "point_count": int(after.get("point_count") or 0),
            "lidar_name": str(after.get("lidar_name", "")),
            "source_time_s": after.get("source_time_s"),
        }
    gt_boxes = updated.get("gt_boxes")
    gt_ids = updated.get("ID")
    if not isinstance(gt_ids, (list, tuple)):
        gt_ids = updated.get("track_id")
    if isinstance(gt_boxes, list) and isinstance(gt_ids, (list, tuple)):
        for index, box in enumerate(gt_boxes):
            if index >= len(gt_ids):
                break
            after = after_by_id.get(str(gt_ids[index]))
            if after is None or str(after.get("status", "")) != "after_comp":
                continue
            values = list(box)
            if len(values) != 7:
                continue
            center = box_center_from_corners(after["corners"])
            values[:7] = [
                *center,
                float(after.get("length", values[3])),
                float(after.get("width", values[4])),
                float(after.get("height", values[5])),
                float(after.get("yaw", values[6])),
            ]
            gt_boxes[index] = values
            updated_gt_box_count += 1
    updated["dynamic_compensated_gt_box_count"] = updated_gt_box_count
    return updated


def _load_box_payload(box_dir: Path, frame_index: int, pcd_name: str) -> dict[str, Any]:
    expected = box_dir / f"frame_{frame_index:06d}_{Path(pcd_name).stem}_boxes.json"
    if expected.exists():
        return json.loads(expected.read_text(encoding="utf-8"))
    matches = sorted(box_dir.glob(f"frame_{frame_index:06d}_*_boxes.json"))
    if not matches:
        raise FileNotFoundError(f"missing box JSON for frame {frame_index}: {expected}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _pcd_name_for_info(info: dict[str, Any]) -> str:
    perception = info.get("sensors", {}).get("lidar", {}).get("perception")
    if not isinstance(perception, dict):
        raise ValueError("info missing sensors.lidar.perception")
    data_path = str(perception.get("data_path") or perception.get("aws_path") or "")
    name = Path(data_path).name
    if not name:
        raise ValueError("perception lidar data_path/aws_path has no filename")
    return name


def _relative_pcd_path(output_pkl: Path, compensated_pcd: Path) -> str:
    try:
        return str(compensated_pcd.resolve().relative_to(output_pkl.parent.resolve()))
    except ValueError:
        return str(compensated_pcd)


def generate_dynamic_compensated_result_interpolation_pkl(
    *,
    input_pkl: Path,
    box_dir: Path,
    compensated_pcd_dir: Path,
    output_pkl: Path,
    reference_cameras: tuple[str, ...] = DEFAULT_REFERENCE_CAMERAS_6,
    reference_time_offset_s: float = 0.0,
    offset_field_name: str = DEFAULT_OFFSET_FIELD,
    offset_timestamp_field_name: str = DEFAULT_OFFSET_TIMESTAMP_FIELD,
    manifest_json: Path | None = None,
) -> dict[str, Any]:
    payload = load_pickle(input_pkl)
    infos = payload["infos"]
    updated_payload = dict(payload)
    ego_times_s = np.asarray(
        [timestamp_to_seconds(info["timestamp"]) for info in infos],
        dtype=np.float64,
    )
    ego_to_world = np.stack(
        [
            np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64).reshape(4, 4)
            for info in infos
        ],
        axis=0,
    )
    updated_infos = []
    fallback_frames = []
    updated_objects = 0
    updated_gt_boxes = 0
    for frame_index, info in enumerate(infos):
        pcd_name = _pcd_name_for_info(info)
        compensated_pcd = Path(compensated_pcd_dir) / pcd_name
        if not compensated_pcd.exists():
            raise FileNotFoundError(compensated_pcd)
        box_payload = _load_box_payload(Path(box_dir), frame_index, pcd_name)
        reference_time_s = reference_time_seconds_from_cameras(info, reference_cameras) + float(
            reference_time_offset_s
        )
        offset_ego = interpolate_ego_from_precomputed(
            ego_times_s,
            ego_to_world,
            reference_time_s,
        )
        if offset_ego is None:
            offset_ego = np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64).reshape(
                4, 4
            )
            fallback_frames.append(frame_index)
        after_by_id = _object_after_by_id(box_payload)
        before_dynamic_count = sum(
            1
            for obj in info.get("objects", []) or []
            if str(obj.get("id", "")) in after_by_id
        )
        updated_objects += before_dynamic_count
        updated_infos.append(
            update_info_for_dynamic_compensation(
                info,
                box_payload=box_payload,
                compensated_pcd_relative_path=_relative_pcd_path(output_pkl, compensated_pcd),
                offset_ego_to_world=offset_ego,
                offset_timestamp_ns=int(round(reference_time_s * 1.0e9)),
                offset_field_name=offset_field_name,
                offset_timestamp_field_name=offset_timestamp_field_name,
                after_by_id=after_by_id,
            )
        )
        updated_gt_boxes += int(updated_infos[-1].get("dynamic_compensated_gt_box_count", 0))
    updated_payload["infos"] = updated_infos
    metadata = copy.deepcopy(updated_payload.get("metadata", {}))
    if not isinstance(metadata, dict):
        metadata = {"original_metadata": metadata}
    metadata["dynamic_object_compensation"] = {
        "enabled": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_pkl": str(input_pkl),
        "box_dir": str(box_dir),
        "compensated_pcd_dir": str(compensated_pcd_dir),
        "reference_cameras": list(reference_cameras),
        "reference_time_offset_s": float(reference_time_offset_s),
        "offset_field_name": str(offset_field_name),
        "offset_timestamp_field_name": str(offset_timestamp_field_name),
        "perception_path_updated": True,
        "objects_updated_from_after_boxes": True,
        "gt_boxes_updated_from_after_boxes": True,
        "original_timestamp_and_ego2global_preserved": True,
        "fallback_frame_indices_for_offset_ego": fallback_frames,
    }
    updated_payload["metadata"] = metadata
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(updated_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "output_pkl": str(output_pkl),
        "frame_count": len(updated_infos),
        "updated_object_count": updated_objects,
        "updated_gt_box_count": updated_gt_boxes,
        "fallback_frame_count": len(fallback_frames),
        "fallback_frame_indices": fallback_frames,
        "offset_field_name": str(offset_field_name),
        "offset_timestamp_field_name": str(offset_timestamp_field_name),
        "reference_time_offset_s": float(reference_time_offset_s),
    }
    if manifest_json is not None:
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _parse_cameras(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("reference cameras cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deliverable result_interpolation.pkl with dynamic compensated PCD/boxes."
    )
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--box-dir", type=Path, required=True)
    parser.add_argument("--compensated-pcd-dir", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--reference-cameras", default=",".join(DEFAULT_REFERENCE_CAMERAS_6))
    parser.add_argument("--reference-time-offset-s", type=float, default=0.0)
    parser.add_argument("--offset-field-name", default=DEFAULT_OFFSET_FIELD)
    parser.add_argument("--offset-timestamp-field-name", default=DEFAULT_OFFSET_TIMESTAMP_FIELD)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_dynamic_compensated_result_interpolation_pkl(
        input_pkl=Path(args.input_pkl),
        box_dir=Path(args.box_dir),
        compensated_pcd_dir=Path(args.compensated_pcd_dir),
        output_pkl=Path(args.output_pkl),
        manifest_json=Path(args.manifest_json) if args.manifest_json else None,
        reference_cameras=_parse_cameras(args.reference_cameras),
        reference_time_offset_s=float(args.reference_time_offset_s),
        offset_field_name=str(args.offset_field_name),
        offset_timestamp_field_name=str(args.offset_timestamp_field_name),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
