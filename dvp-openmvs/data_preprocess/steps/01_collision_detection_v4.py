#!/usr/bin/env python3
"""Single-PKL BEV collision and center-lane crossing checker.

V3 keeps the collision interface from check_collision_v2 and adds detection for
vehicle footprints crossing selected lane-line colors.  Yellow lane lines are
checked by default because they normally separate opposing traffic.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_INPUT = "./cutin_obj222.pkl"
DEFAULT_LANE_COLORS = ("YELLOW",)

VEHICLE_TYPE_HINTS = {
    "car",
    "suv",
    "van",
    "truck",
    "bus",
    "vehicle_else",
    "huge_vehicle",
    "trailer",
    "construction_vehicle",
    "policecar",
    "firetruck",
    "ambulance",
    "sprinkler",
}


@dataclass
class ObjectBox:
    object_id: str
    object_type: str
    center: np.ndarray
    size: np.ndarray
    yaw: float
    corners: np.ndarray


def load_pkl(path: str | Path) -> Any:
    import pickle

    with open(path, "rb") as file:
        try:
            return pickle.load(file)
        except Exception:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def is_vehicle_type(object_type: str) -> bool:
    return str(object_type).lower() in VEHICLE_TYPE_HINTS


def resolve_frame_range(start_frame: int | None, end_frame: int | None, frame_count: int) -> tuple[int, int]:
    start = 0 if start_frame is None else int(start_frame)
    end = frame_count - 1 if end_frame is None else int(end_frame)
    if start < 0 or end >= frame_count or start > end:
        raise IndexError(f"Invalid frame range {start}-{end} for {frame_count} frames")
    return start, end


def corners_from_center(center: np.ndarray, size: np.ndarray, yaw: float, extra_margin: float = 0.0) -> np.ndarray:
    length = max(0.01, float(size[0]) + 2.0 * extra_margin)
    width = max(0.01, float(size[1]) + 2.0 * extra_margin)
    local = np.asarray(
        [
            [length * 0.5, -width * 0.5],
            [length * 0.5, width * 0.5],
            [-length * 0.5, width * 0.5],
            [-length * 0.5, -width * 0.5],
        ],
        dtype=np.float64,
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return local @ rotation.T + center[:2]


def object_box_from_dict(obj: dict[str, Any], extra_margin: float) -> ObjectBox | None:
    try:
        location = np.asarray(obj["location"], dtype=np.float64)[:3]
        size = np.asarray(obj["size"], dtype=np.float64)[:3]
        yaw = float((obj.get("rotation") or [0.0, 0.0, 0.0])[2])
    except Exception:
        return None
    if size.shape[0] < 2 or not np.all(np.isfinite(location[:2])) or not np.all(np.isfinite(size[:2])):
        return None
    center = location[:2]
    size_bev = size[:2]
    return ObjectBox(
        object_id=str(obj.get("id", "unknown")),
        object_type=str(obj.get("type", "unknown")),
        center=center,
        size=size_bev,
        yaw=yaw,
        corners=corners_from_center(center, size_bev, yaw, extra_margin),
    )


def collect_object_boxes(info: dict[str, Any], vehicle_only: bool, extra_margin: float) -> list[ObjectBox]:
    boxes = []
    for index, obj in enumerate(info.get("objects", []) or []):
        if not isinstance(obj, dict):
            continue
        if "id" not in obj:
            obj = {**obj, "id": f"missing_id_{index}"}
        if vehicle_only and not is_vehicle_type(str(obj.get("type", "unknown"))):
            continue
        box = object_box_from_dict(obj, extra_margin)
        if box is not None:
            boxes.append(box)
    return boxes


def make_ego_box(
    ego_length: float,
    ego_width: float,
    ego_center_x: float,
    ego_center_y: float,
    ego_yaw: float,
    extra_margin: float,
) -> ObjectBox:
    center = np.asarray([ego_center_x, ego_center_y], dtype=np.float64)
    size = np.asarray([ego_length, ego_width], dtype=np.float64)
    return ObjectBox(
        object_id="ego",
        object_type="ego_vehicle",
        center=center,
        size=size,
        yaw=float(ego_yaw),
        corners=corners_from_center(center, size, float(ego_yaw), extra_margin),
    )


def polygon_axes(corners: np.ndarray) -> list[np.ndarray]:
    axes = []
    for index in (0, 1):
        edge = corners[(index + 1) % 4] - corners[index]
        normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-9:
            axes.append(normal / norm)
    return axes


def project_polygon(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = corners @ axis
    return float(np.min(values)), float(np.max(values))


def sat_collision(a: ObjectBox, b: ObjectBox, overlap_epsilon: float) -> tuple[bool, float]:
    min_overlap = float("inf")
    for axis in polygon_axes(a.corners) + polygon_axes(b.corners):
        min_a, max_a = project_polygon(a.corners, axis)
        min_b, max_b = project_polygon(b.corners, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= overlap_epsilon:
            return False, 0.0
        min_overlap = min(min_overlap, overlap)
    return True, float(min_overlap)


def ordered_pair(a: ObjectBox, b: ObjectBox) -> tuple[ObjectBox, ObjectBox]:
    if a.object_id == "ego":
        return a, b
    if b.object_id == "ego":
        return b, a
    return (a, b) if str(a.object_id) <= str(b.object_id) else (b, a)


def collision_row(frame_idx: int, scope: str, a: ObjectBox, b: ObjectBox, penetration: float) -> dict[str, Any]:
    first, second = ordered_pair(a, b)
    return {
        "frame_idx": int(frame_idx),
        "scope": scope,
        "object_id_a": first.object_id,
        "object_id_b": second.object_id,
        "object_type_a": first.object_type,
        "object_type_b": second.object_type,
        "center_distance_m": float(np.linalg.norm(first.center - second.center)),
        "penetration_m": float(penetration),
    }


def detect_frame_collisions(
    info: dict[str, Any],
    frame_idx: int,
    vehicle_only: bool,
    overlap_epsilon: float,
    box_extra_margin: float,
    include_ego: bool,
    ego_length: float,
    ego_width: float,
    ego_center_x: float,
    ego_center_y: float,
    ego_yaw: float,
) -> list[dict[str, Any]]:
    boxes = collect_object_boxes(info, vehicle_only=vehicle_only, extra_margin=box_extra_margin)
    collisions = []
    for index, a in enumerate(boxes):
        for b in boxes[index + 1 :]:
            has_collision, penetration = sat_collision(a, b, overlap_epsilon)
            if has_collision:
                collisions.append(collision_row(frame_idx, "object", a, b, penetration))
    if include_ego:
        ego = make_ego_box(
            ego_length, ego_width, ego_center_x, ego_center_y, ego_yaw, box_extra_margin
        )
        for box in boxes:
            if box.object_id == "ego":
                continue
            has_collision, penetration = sat_collision(ego, box, overlap_epsilon)
            if has_collision:
                collisions.append(collision_row(frame_idx, "ego", ego, box, penetration))
    return collisions


def summarize_collision_pairs(collisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in collisions:
        key = (str(row["scope"]), str(row["object_id_a"]), str(row["object_id_b"]))
        groups.setdefault(key, []).append(row)
    pair_rows = []
    for (scope, object_id_a, object_id_b), rows in groups.items():
        frames = sorted({int(row["frame_idx"]) for row in rows})
        pair_rows.append(
            {
                "scope": scope,
                "object_id_a": object_id_a,
                "object_id_b": object_id_b,
                "object_type_a": rows[0]["object_type_a"],
                "object_type_b": rows[0]["object_type_b"],
                "collision_count": len(rows),
                "frame_start": frames[0],
                "frame_end": frames[-1],
                "frames": frames,
                "max_penetration_m": max(float(row["penetration_m"]) for row in rows),
                "min_center_distance_m": min(float(row["center_distance_m"]) for row in rows),
            }
        )
    pair_rows.sort(key=lambda row: (-int(row["collision_count"]), str(row["scope"]), str(row["object_id_a"])))
    return pair_rows


@dataclass
class LaneLine:
    line_id: str
    label: str
    line_type: str
    color: str
    points: np.ndarray
    source_field: str


@dataclass
class LineRelation:
    distance: float
    signed_distance: float
    tangent: np.ndarray


def normalize_lane_color(value: Any) -> str:
    """Normalize string and raw-enum lane colors to a stable name."""
    if isinstance(value, (int, np.integer)):
        # STPerceptionRoadLineColor: 2=white, 3=yellow.
        return {2: "WHITE", 3: "YELLOW"}.get(int(value), str(int(value)))
    color = str(value or "UNKNOWN").strip().upper()
    if "YELLOW" in color:
        return "YELLOW"
    if "WHITE" in color:
        return "WHITE"
    return color or "UNKNOWN"


def _lane_container(info: dict[str, Any]) -> tuple[list[Any], str | None]:
    for field in ("lanelines_annotation", "lane_at_ego"):
        value = info.get(field)
        if isinstance(value, dict) and isinstance(value.get("lane"), list):
            return value["lane"], field
    return [], None


def _line_points(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9))
    points = points[keep]
    return points if len(points) >= 2 else None


def collect_lane_lines(info: dict[str, Any], lane_colors: Iterable[str]) -> tuple[list[LaneLine], str | None]:
    wanted_colors = {normalize_lane_color(color) for color in lane_colors}
    raw_lines, source_field = _lane_container(info)
    lines: list[LaneLine] = []
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, dict):
            continue
        label = str(raw_line.get("label", "LANELINE")).strip().upper()
        if label not in {"", "LANELINE", "LANE_LINE"}:
            continue
        color = normalize_lane_color(raw_line.get("color"))
        if color not in wanted_colors:
            continue
        points = _line_points(raw_line.get("geo_3d"))
        if points is None:
            continue
        lines.append(
            LaneLine(
                line_id=str(raw_line.get("id", index)),
                label=label or "LANELINE",
                line_type=str(raw_line.get("type", "UNKNOWN")),
                color=color,
                points=points,
                source_field=str(source_field),
            )
        )
    return lines, source_field


def nearest_line_relation(point: np.ndarray, polyline: np.ndarray) -> LineRelation | None:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths_sq = np.sum(vectors * vectors, axis=1)
    valid = lengths_sq > 1e-12
    if not np.any(valid):
        return None
    starts = starts[valid]
    vectors = vectors[valid]
    lengths_sq = lengths_sq[valid]
    fractions = np.clip(np.sum((point - starts) * vectors, axis=1) / lengths_sq, 0.0, 1.0)
    nearest = starts + vectors * fractions[:, None]
    deltas = point - nearest
    distances_sq = np.sum(deltas * deltas, axis=1)
    index = int(np.argmin(distances_sq))
    tangent = vectors[index] / math.sqrt(float(lengths_sq[index]))
    signed_distance = float(tangent[0] * deltas[index, 1] - tangent[1] * deltas[index, 0])
    return LineRelation(
        distance=math.sqrt(float(distances_sq[index])),
        signed_distance=signed_distance,
        tangent=tangent,
    )


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray, epsilon: float = 1e-9) -> bool:
    if abs(_orientation(start, end, point)) > epsilon:
        return False
    return bool(
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and (
        (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    ):
        return True
    return any(
        (
            abs(value) <= 1e-9
            and _point_on_segment(point, seg_start, seg_end)
        )
        for value, point, seg_start, seg_end in (
            (o1, c, a, b),
            (o2, d, a, b),
            (o3, a, c, d),
            (o4, b, c, d),
        )
    )


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    signs = np.asarray(
        [_orientation(polygon[i], polygon[(i + 1) % len(polygon)], point) for i in range(len(polygon))]
    )
    return bool(np.all(signs >= -1e-9) or np.all(signs <= 1e-9))


def polyline_intersects_box(polyline: np.ndarray, corners: np.ndarray) -> bool:
    for point in polyline:
        if point_in_convex_polygon(point, corners):
            return True
    for line_start, line_end in zip(polyline[:-1], polyline[1:]):
        for index, box_start in enumerate(corners):
            box_end = corners[(index + 1) % len(corners)]
            if segments_intersect(line_start, line_end, box_start, box_end):
                return True
    return False


def collect_lane_subjects(
    info: dict[str, Any],
    vehicle_only: bool,
    box_extra_margin: float,
    lane_crossing_margin: float,
    include_ego: bool,
    ego_length: float,
    ego_width: float,
    ego_center_x: float,
    ego_center_y: float,
    ego_yaw: float,
) -> list[ObjectBox]:
    margin = box_extra_margin + lane_crossing_margin
    subjects = collect_object_boxes(info, vehicle_only=vehicle_only, extra_margin=margin)
    if include_ego:
        subjects.append(
            make_ego_box(
                ego_length,
                ego_width,
                ego_center_x,
                ego_center_y,
                ego_yaw,
                margin,
            )
        )
    return subjects


def _event_key(frame_idx: int, subject_id: str, line_id: str) -> tuple[int, str, str]:
    return int(frame_idx), str(subject_id), str(line_id)


def _new_lane_event(
    frame_idx: int,
    subject: ObjectBox,
    line: LaneLine,
    relation: LineRelation,
) -> dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "subject_id": subject.object_id,
        "subject_type": subject.object_type,
        "lane_line_id": line.line_id,
        "lane_line_type": line.line_type,
        "lane_line_color": line.color,
        "lane_source_field": line.source_field,
        "footprint_intersection": False,
        "center_side_change": False,
        "distance_to_line_m": float(relation.distance),
        "signed_distance_m": float(relation.signed_distance),
    }


def summarize_lane_crossings(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (str(event["subject_id"]), str(event["lane_line_id"]))
        groups.setdefault(key, []).append(event)

    summaries = []
    for (subject_id, line_id), rows in groups.items():
        rows.sort(key=lambda row: int(row["frame_idx"]))
        frames = sorted({int(row["frame_idx"]) for row in rows})
        summaries.append(
            {
                "subject_id": subject_id,
                "subject_type": rows[0]["subject_type"],
                "lane_line_id": line_id,
                "lane_line_type": rows[0]["lane_line_type"],
                "lane_line_color": rows[0]["lane_line_color"],
                "event_count": len(rows),
                "footprint_intersection_count": sum(bool(row["footprint_intersection"]) for row in rows),
                "center_side_change_count": sum(bool(row["center_side_change"]) for row in rows),
                "frame_start": frames[0],
                "frame_end": frames[-1],
                "frames": frames,
                "min_distance_to_line_m": min(float(row["distance_to_line_m"]) for row in rows),
            }
        )
    summaries.sort(key=lambda row: (-int(row["event_count"]), str(row["subject_id"]), str(row["lane_line_id"])))
    return summaries


def detect_pkl_collisions(
    input_pkl: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    overlap_epsilon: float = 0.0,
    box_extra_margin: float = 0.0,
    include_non_vehicles: bool = False,
    include_ego: bool = False,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    ego_center_x: float = 0.0,
    ego_center_y: float = 0.0,
    ego_yaw: float = math.pi / 2.0,
    include_events: bool = False,
    check_lane_crossing: bool = True,
    lane_colors: Iterable[str] = DEFAULT_LANE_COLORS,
    lane_crossing_margin: float = 0.0,
    side_change_epsilon: float = 0.05,
    side_change_max_distance: float = 5.0,
) -> dict[str, Any]:
    """Return collision results plus yellow center-line crossing results."""
    if lane_crossing_margin < 0.0:
        raise ValueError("lane_crossing_margin must be >= 0")
    if side_change_epsilon < 0.0 or side_change_max_distance <= 0.0:
        raise ValueError("side-change thresholds must be positive")

    data = load_pkl(input_pkl)
    if not isinstance(data, dict) or not isinstance(data.get("infos"), list) or not data["infos"]:
        raise ValueError("Input PKL must be a dict with non-empty data['infos']")

    infos = data["infos"]
    start, end = resolve_frame_range(start_frame, end_frame, len(infos))
    vehicle_only = not include_non_vehicles
    normalized_colors = tuple(sorted({normalize_lane_color(color) for color in lane_colors}))
    if not normalized_colors:
        raise ValueError("At least one lane color must be supplied")

    collisions: list[dict[str, Any]] = []
    lane_events_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    previous_relations: dict[tuple[str, str], tuple[int, LineRelation]] = {}
    lane_line_observation_count = 0
    lane_subject_observation_count = 0
    frames_with_lane_lines = 0
    lane_source_fields: set[str] = set()

    for frame_idx in range(start, end + 1):
        info = infos[frame_idx]
        if not isinstance(info, dict):
            continue
        collisions.extend(
            detect_frame_collisions(
                info,
                frame_idx,
                vehicle_only,
                overlap_epsilon,
                box_extra_margin,
                include_ego,
                ego_length,
                ego_width,
                ego_center_x,
                ego_center_y,
                ego_yaw,
            )
        )
        if not check_lane_crossing:
            continue

        lines, source_field = collect_lane_lines(info, normalized_colors)
        if source_field:
            lane_source_fields.add(source_field)
        lane_line_observation_count += len(lines)
        frames_with_lane_lines += int(bool(lines))
        subjects = collect_lane_subjects(
            info,
            vehicle_only,
            box_extra_margin,
            lane_crossing_margin,
            include_ego,
            ego_length,
            ego_width,
            ego_center_x,
            ego_center_y,
            ego_yaw,
        )
        lane_subject_observation_count += len(subjects)

        current_relation_keys: set[tuple[str, str]] = set()
        for subject in subjects:
            for line in lines:
                relation = nearest_line_relation(subject.center, line.points)
                if relation is None:
                    continue
                relation_key = (subject.object_id, line.line_id)
                current_relation_keys.add(relation_key)
                event_key = _event_key(frame_idx, subject.object_id, line.line_id)
                footprint_intersection = polyline_intersects_box(line.points, subject.corners)

                previous = previous_relations.get(relation_key)
                center_side_change = False
                if previous is not None and previous[0] == frame_idx - 1:
                    previous_relation = previous[1]
                    if float(np.dot(previous_relation.tangent, relation.tangent)) < 0.0:
                        relation = LineRelation(
                            distance=relation.distance,
                            signed_distance=-relation.signed_distance,
                            tangent=-relation.tangent,
                        )
                    center_side_change = bool(
                        abs(previous_relation.signed_distance) > side_change_epsilon
                        and abs(relation.signed_distance) > side_change_epsilon
                        and previous_relation.signed_distance * relation.signed_distance < 0.0
                        and max(previous_relation.distance, relation.distance) <= side_change_max_distance
                    )

                if footprint_intersection or center_side_change:
                    event = lane_events_by_key.get(event_key)
                    if event is None:
                        event = _new_lane_event(frame_idx, subject, line, relation)
                        lane_events_by_key[event_key] = event
                    event["footprint_intersection"] = bool(event["footprint_intersection"] or footprint_intersection)
                    event["center_side_change"] = bool(event["center_side_change"] or center_side_change)

                previous_relations[relation_key] = (frame_idx, relation)

        # Do not compare across a frame where either the subject or line was absent.
        for relation_key in list(previous_relations):
            if previous_relations[relation_key][0] < frame_idx and relation_key not in current_relation_keys:
                del previous_relations[relation_key]

    collision_pairs = summarize_collision_pairs(collisions)
    collision_ids = [f"{row['object_id_a']}:{row['object_id_b']}" for row in collision_pairs]
    colliding_object_ids = sorted(
        {str(row["object_id_a"]) for row in collision_pairs} | {str(row["object_id_b"]) for row in collision_pairs}
    )
    lane_events = sorted(
        lane_events_by_key.values(),
        key=lambda row: (int(row["frame_idx"]), str(row["subject_id"]), str(row["lane_line_id"])),
    )
    lane_crossing_pairs = summarize_lane_crossings(lane_events)
    lane_crossing_subject_ids = sorted({str(row["subject_id"]) for row in lane_events})
    warnings = []
    if check_lane_crossing and lane_line_observation_count == 0:
        warnings.append(
            "No matching lane lines were found. Check lanelines_annotation/lane_at_ego and lane color annotations."
        )
    elif check_lane_crossing and lane_subject_observation_count == 0:
        warnings.append("No object boxes were available for lane checks; use --include-ego to check the ego vehicle.")

    result = {
        "input_pkl": str(input_pkl),
        "collision_check_passed": not bool(collisions),
        "has_collision": bool(collisions),
        "collision_count": len(collisions),
        "collision_ids": collision_ids,
        "colliding_object_ids": colliding_object_ids,
        "collision_pairs": collision_pairs,
        "has_lane_crossing": bool(lane_events),
        "has_center_side_change": any(bool(row["center_side_change"]) for row in lane_events),
        "lane_crossing_count": len(lane_events),
        "lane_crossing_subject_ids": lane_crossing_subject_ids,
        "lane_crossing_pairs": lane_crossing_pairs,
        "lane_crossing_checked": bool(check_lane_crossing),
        "lane_colors": list(normalized_colors),
        "lane_line_observation_count": int(lane_line_observation_count),
        "lane_subject_observation_count": int(lane_subject_observation_count),
        "frames_with_lane_lines": int(frames_with_lane_lines),
        "lane_source_fields": sorted(lane_source_fields),
        "warnings": warnings,
        "frame_start": int(start),
        "frame_end": int(end),
        "num_frames": len(infos),
        "checked_frame_count": int(end - start + 1),
        "vehicle_only": vehicle_only,
        "include_ego": bool(include_ego),
        "ego_yaw_rad": float(ego_yaw) if include_ego else None,
        "overlap_epsilon_m": float(overlap_epsilon),
        "box_extra_margin_m": float(box_extra_margin),
        "lane_crossing_margin_m": float(lane_crossing_margin),
        "side_change_epsilon_m": float(side_change_epsilon),
        "side_change_max_distance_m": float(side_change_max_distance),
    }
    if include_events:
        result["collisions"] = collisions
        result["lane_crossings"] = lane_events
    return json_safe(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checker for BEV collisions and center-lane crossings inside one PKL."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input PKL path")
    parser.add_argument("--start-frame", type=int, default=None, help="First frame to check")
    parser.add_argument("--end-frame", type=int, default=None, help="Last frame to check")
    parser.add_argument("--overlap-epsilon", type=float, default=0.0, help="SAT overlap tolerance in meters")
    parser.add_argument("--box-extra-margin", type=float, default=0.0, help="Extra BEV box margin in meters")
    parser.add_argument("--include-non-vehicles", action="store_true", help="Include non-vehicle object boxes")
    parser.add_argument("--include-ego", action="store_true", help="Check ego-object and ego-lane interactions")
    parser.add_argument("--ego-length", type=float, default=4.8, help="Approximate ego footprint length")
    parser.add_argument("--ego-width", type=float, default=2.0, help="Approximate ego footprint width")
    parser.add_argument("--ego-center-x", type=float, default=0.0, help="Ego footprint center x in ego frame")
    parser.add_argument("--ego-center-y", type=float, default=0.0, help="Ego footprint center y in ego frame")
    parser.add_argument(
        "--ego-yaw",
        type=float,
        default=math.pi / 2.0,
        help="Ego footprint yaw. Default pi/2 means ego length points toward +y.",
    )
    parser.add_argument("--skip-lane-crossing", action="store_true", help="Disable lane crossing checks")
    parser.add_argument(
        "--lane-color",
        action="append",
        default=None,
        help="Lane color to check; repeat for multiple colors. Default: YELLOW.",
    )
    parser.add_argument(
        "--lane-crossing-margin",
        type=float,
        default=0.0,
        help="Extra footprint margin used only for lane crossing checks",
    )
    parser.add_argument(
        "--side-change-epsilon",
        type=float,
        default=0.05,
        help="Ignore center-side changes within this signed-distance deadband",
    )
    parser.add_argument(
        "--side-change-max-distance",
        type=float,
        default=5.0,
        help="Maximum center distance on both frames for a side-change event",
    )
    parser.add_argument("--include-events", action="store_true", help="Include per-frame collision/crossing rows")
    parser.add_argument("--output-json", default=None, help="Optional path for the same JSON result")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print stdout JSON")
    parser.add_argument("--fail-on-collision", action="store_true", help="Exit 2 when collision is detected")
    parser.add_argument("--fail-on-lane-crossing", action="store_true", help="Exit 3 when lane crossing is detected")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = detect_pkl_collisions(
        input_pkl=args.input,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        overlap_epsilon=args.overlap_epsilon,
        box_extra_margin=args.box_extra_margin,
        include_non_vehicles=args.include_non_vehicles,
        include_ego=args.include_ego,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
        ego_center_x=args.ego_center_x,
        ego_center_y=args.ego_center_y,
        ego_yaw=args.ego_yaw,
        include_events=args.include_events,
        check_lane_crossing=not args.skip_lane_crossing,
        lane_colors=args.lane_color or DEFAULT_LANE_COLORS,
        lane_crossing_margin=args.lane_crossing_margin,
        side_change_epsilon=args.side_change_epsilon,
        side_change_max_distance=args.side_change_max_distance,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    if args.fail_on_collision and result["has_collision"]:
        raise SystemExit(2)
    if args.fail_on_lane_crossing and result["has_lane_crossing"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
