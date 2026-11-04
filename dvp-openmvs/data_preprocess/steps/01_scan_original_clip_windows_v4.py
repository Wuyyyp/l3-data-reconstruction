#!/usr/bin/env python3
"""Find non-overlapping 101-frame left/right reconstruction windows in L3 PKLs.

Each side is evaluated independently.  A passing window is emitted as either
``left`` or ``right``; this scanner never emits ``both``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_ROOT = "/mnt/nuplan/l3_data_test/L3-data-pkl-paths"
DEFAULT_CLIPS = f"{DEFAULT_ROOT}/available_clips.txt"
DEFAULT_PKL_LISTS = (
    f"{DEFAULT_ROOT}/available_pkl_paths_L3_V5data.txt",
    f"{DEFAULT_ROOT}/available_pkl_paths_L3_V5data_3.txt",
)

# Yellow dashed markings may separate lanes moving in the same direction and
# are legal to cross for this reconstruction use case.  Any unrecognized
# yellow type remains conservative and is treated as non-crossable.
CROSSABLE_YELLOW_LINE_TYPES = {
    "DASHED_LANE",
    "DOUBLE_DASHED",
    "THICK_DASHED",
}


def normalize_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def read_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def clip_name_from_path(path: str) -> str:
    return Path(path.rstrip("/")).parent.name


def localize_path(path: str) -> Path:
    if path.startswith("/mnt/duanxuhao_bingxing/"):
        return Path("/data/mnt/duanxuhao_bingixng") / path[len("/mnt/duanxuhao_bingxing/") :]
    if path.startswith("/mnt/"):
        return Path("/data") / path.lstrip("/")
    return Path(path)


def resolve_pkl_path(source_path: str, clip_name: str, current_pkl_root: str) -> Path:
    """Resolve legacy list paths against the currently mounted V5 source."""
    legacy = localize_path(source_path)
    if legacy.exists():
        return legacy
    current_root = Path(current_pkl_root)
    matches = sorted(
        current_root.glob(
            f"batch_*/V5_20260515/{clip_name}/result_pkl_all/{clip_name}.pkl"
        )
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple current PKLs found for {clip_name}: {matches}")
    raise FileNotFoundError(f"Neither legacy nor current PKL exists for {clip_name}: {legacy}")


def build_pkl_index(path_lists: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for list_path in path_lists:
        for raw_path in read_lines(list_path):
            name = clip_name_from_path(raw_path)
            previous = result.get(name)
            if previous is not None and previous != raw_path:
                raise ValueError(f"Duplicate clip with different PKLs: {name}")
            result[name] = raw_path
    return result


def load_detector(path: str):
    spec = importlib.util.spec_from_file_location("collision_detection_v4_scan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import detector: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lane_container(info: dict[str, Any]) -> tuple[list[Any], str | None]:
    for field in ("lanelines_annotation", "lane_at_ego"):
        value = info.get(field)
        if isinstance(value, dict) and isinstance(value.get("lane"), list):
            return value["lane"], field
    return [], None


def collect_lines(
    detector: Any,
    info: dict[str, Any],
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray], bool]:
    raw_lines, source = lane_container(info)
    all_yellow_lines, _ = detector.collect_lane_lines(info, ("YELLOW",))
    yellow_lines = [
        line
        for line in all_yellow_lines
        if str(line.line_type or "UNKNOWN").strip().upper()
        not in CROSSABLE_YELLOW_LINE_TYPES
    ]
    roadsides: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []
    usable_road_geometry = False
    for raw in raw_lines:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", "")).strip().upper()
        if label not in {"LANELINE", "LANE_LINE", "ROADSIDE", "CENTERLINE"}:
            continue
        points = detector._line_points(raw.get("geo_3d"))
        if points is None:
            continue
        usable_road_geometry = True
        if label == "ROADSIDE":
            roadsides.append(points)
        elif label == "CENTERLINE":
            centerlines.append(points)
    annotation_available = source is not None and usable_road_geometry
    return yellow_lines, roadsides, centerlines, annotation_available


def box_hits_any_line(detector: Any, box: Any, lines: Iterable[Any]) -> bool:
    for line in lines:
        points = line.points if hasattr(line, "points") else line
        if detector.polyline_intersects_box(points, box.corners):
            return True
    return False


def boxes_collide(detector: Any, ego: Any, objects: list[Any], overlap_epsilon: float) -> bool:
    ego_min = ego.corners.min(axis=0)
    ego_max = ego.corners.max(axis=0)
    for obj in objects:
        obj_min = obj.corners.min(axis=0)
        obj_max = obj.corners.max(axis=0)
        if (
            ego_max[0] <= obj_min[0]
            or obj_max[0] <= ego_min[0]
            or ego_max[1] <= obj_min[1]
            or obj_max[1] <= ego_min[1]
        ):
            continue
        if detector.sat_collision(ego, obj, overlap_epsilon)[0]:
            return True
    return False


def nearest_centerline_relation(detector: Any, point: np.ndarray, centerlines: Iterable[np.ndarray]) -> Any | None:
    relations = [detector.nearest_line_relation(point, line) for line in centerlines]
    relations = [relation for relation in relations if relation is not None]
    return min(relations, key=lambda relation: float(relation.distance), default=None)


def target_lane_failures(
    detector: Any,
    centerlines: list[np.ndarray],
    target_x: float,
    max_distance: float,
    opposing_dot_threshold: float,
) -> tuple[bool, bool, bool]:
    """Return (local_evidence_missing, target_not_in_lane, target_opposing).

    Missing or distant centerline geometry is unsafe, not a pass.  The old
    fail-open behavior allowed a target to leave the road whenever unrelated
    lane/road polylines existed elsewhere in the frame but the geometry near
    the ego vehicle was absent or truncated.
    """
    if not centerlines:
        return True, False, False
    origin = nearest_centerline_relation(detector, np.asarray([0.0, 0.0]), centerlines)
    if origin is None or float(origin.distance) > max_distance:
        return True, False, False
    target = nearest_centerline_relation(detector, np.asarray([target_x, 0.0]), centerlines)
    if target is None or float(target.distance) > max_distance:
        return False, True, False
    direction_dot = float(np.dot(origin.tangent, target.tangent))
    return False, False, direction_dot < opposing_dot_threshold


def make_probe_boxes(detector: Any, args: dict[str, Any]) -> dict[str, Any]:
    length = float(args["ego_length"])
    width = float(args["ego_width"])
    yaw = math.pi / 2.0
    margin = float(args["box_extra_margin"])
    boxes = {
        "left_collision": detector.make_ego_box(length, width, -3.0, 0.0, yaw, margin),
        "right_collision": detector.make_ego_box(length, width, 3.0, 0.0, yaw, margin),
        "left_corridor": detector.make_ego_box(length, width + 3.0, -1.5, 0.0, yaw, margin),
        "right_corridor": detector.make_ego_box(length, width + 3.0, 1.5, 0.0, yaw, margin),
    }
    return boxes


def non_overlapping_windows(frame_ok: list[bool], window_size: int) -> list[tuple[int, int]]:
    """Return the earliest maximal set of fixed-size non-overlapping windows."""
    selected: list[tuple[int, int]] = []
    start = 0
    frame_count = len(frame_ok)
    while start + window_size <= frame_count:
        end = start + window_size
        first_failure = next((idx for idx in range(start, end) if not frame_ok[idx]), None)
        if first_failure is None:
            selected.append((start, end - 1))
            start = end
        else:
            start = first_failure + 1
    return selected


def scan_clip(task: dict[str, Any]) -> dict[str, Any]:
    row = {
        "clip_index_0based": int(task["clip_index_0based"]),
        "clip_name": str(task["clip_name"]),
        "source_pkl_path": str(task.get("source_pkl_path") or ""),
        "local_pkl_path": "",
        "frame_count": 0,
        "segments": [],
        "failure_frame_counts": {},
        "observed_object_types": {},
        "error": "",
    }
    try:
        if not task.get("source_pkl_path"):
            raise FileNotFoundError("clip is absent from the PKL path lists")
        detector = load_detector(str(task["detector"]))
        path = resolve_pkl_path(
            str(task["source_pkl_path"]),
            str(task["clip_name"]),
            str(task["current_pkl_root"]),
        )
        row["local_pkl_path"] = str(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        data = detector.load_pkl(path)
        infos = data.get("infos") if isinstance(data, dict) else None
        if not isinstance(infos, list) or not infos:
            raise ValueError("PKL has no non-empty infos list")
        row["frame_count"] = len(infos)

        probes = make_probe_boxes(detector, task)
        left_ok: list[bool] = []
        right_ok: list[bool] = []
        failures: Counter[str] = Counter()
        observed_types: Counter[str] = Counter()

        for info in infos:
            if not isinstance(info, dict):
                left_ok.append(False)
                right_ok.append(False)
                failures["invalid_frame"] += 1
                continue
            raw_boxes = detector.collect_object_boxes(
                info,
                vehicle_only=False,
                extra_margin=float(task["box_extra_margin"]),
            )
            for box in raw_boxes:
                observed_types[normalize_type(box.object_type)] += 1
            # Every valid object box is a hard obstacle. A type allowlist caused
            # silent misses for SUV, road barriers, crash buckets, parking locks,
            # animals, school buses and other physical annotations.
            obstacles = raw_boxes
            yellow_lines, roadsides, centerlines, annotation_available = collect_lines(detector, info)

            left_collision = boxes_collide(
                detector, probes["left_collision"], obstacles, float(task["overlap_epsilon"])
            )
            right_collision = boxes_collide(
                detector, probes["right_collision"], obstacles, float(task["overlap_epsilon"])
            )
            left_yellow = box_hits_any_line(detector, probes["left_corridor"], yellow_lines)
            right_yellow = box_hits_any_line(detector, probes["right_corridor"], yellow_lines)
            left_roadside = box_hits_any_line(detector, probes["left_corridor"], roadsides)
            right_roadside = box_hits_any_line(detector, probes["right_corridor"], roadsides)
            left_local_missing, left_unlaned, left_opposing = target_lane_failures(
                detector,
                centerlines,
                -3.0,
                float(task["max_centerline_distance"]),
                float(task["opposing_direction_dot"]),
            )
            right_local_missing, right_unlaned, right_opposing = target_lane_failures(
                detector,
                centerlines,
                3.0,
                float(task["max_centerline_distance"]),
                float(task["opposing_direction_dot"]),
            )
            annotation_missing = bool(task["require_lane_annotation"]) and not annotation_available

            if left_collision:
                failures["left_3m_collision"] += 1
            if right_collision:
                failures["right_3m_collision"] += 1
            if left_yellow:
                failures["left_non_crossable_yellow_corridor"] += 1
            if right_yellow:
                failures["right_non_crossable_yellow_corridor"] += 1
            if left_roadside:
                failures["left_roadside_corridor"] += 1
            if right_roadside:
                failures["right_roadside_corridor"] += 1
            if left_local_missing:
                failures["left_local_lane_geometry_missing"] += 1
            if right_local_missing:
                failures["right_local_lane_geometry_missing"] += 1
            if left_unlaned:
                failures["left_target_not_in_annotated_lane"] += 1
            if right_unlaned:
                failures["right_target_not_in_annotated_lane"] += 1
            if left_opposing:
                failures["left_target_opposing_lane"] += 1
            if right_opposing:
                failures["right_target_opposing_lane"] += 1
            if annotation_missing:
                failures["missing_lane_annotation"] += 1

            left_ok.append(
                not (
                    left_collision
                    or left_yellow
                    or left_roadside
                    or left_local_missing
                    or left_unlaned
                    or left_opposing
                    or annotation_missing
                )
            )
            right_ok.append(
                not (
                    right_collision
                    or right_yellow
                    or right_roadside
                    or right_local_missing
                    or right_unlaned
                    or right_opposing
                    or annotation_missing
                )
            )

        window_size = int(task["window_size"])
        for side, flags in (("left", left_ok), ("right", right_ok)):
            for start, end in non_overlapping_windows(flags, window_size):
                row["segments"].append(
                    {
                        "side": side,
                        "frame_start": start,
                        "frame_end": end,
                        "frame_count": window_size,
                    }
                )
        row["segments"].sort(key=lambda item: (int(item["frame_start"]), str(item["side"])))
        row["failure_frame_counts"] = dict(sorted(failures.items()))
        row["observed_object_types"] = dict(sorted(observed_types.items()))
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def write_outputs(output: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    fields = [
        "segment_index",
        "clip_index_0based",
        "clip_name",
        "side",
        "frame_start",
        "frame_end",
        "frame_count",
        "source_pkl_path",
        "local_pkl_path",
    ]
    segment_rows: list[dict[str, Any]] = []
    for clip in rows:
        for segment in clip["segments"]:
            segment_rows.append(
                {
                    "segment_index": len(segment_rows),
                    "clip_index_0based": clip["clip_index_0based"],
                    "clip_name": clip["clip_name"],
                    **segment,
                    "source_pkl_path": clip["source_pkl_path"],
                    "local_pkl_path": clip["local_pkl_path"],
                }
            )
    with (output / "selected_101f_segments.tsv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(segment_rows)
    errors = [
        {"clip_index_0based": row["clip_index_0based"], "clip_name": row["clip_name"], "error": row["error"]}
        for row in rows
        if row["error"]
    ]
    (output / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "clip_diagnostics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "input_clips": len(rows),
        "processed_clips": sum(not row["error"] for row in rows),
        "error_clips": len(errors),
        "selected_segments": len(segment_rows),
        "selected_left": sum(row["side"] == "left" for row in segment_rows),
        "selected_right": sum(row["side"] == "right" for row in segment_rows),
        "clips_with_selected_segments": len({row["clip_name"] for row in segment_rows}),
        "window_size": args.window_size,
        "same_side_windows_overlap": False,
        "left_rule": "left_3m_all_boxes_clear AND left_corridor_roadside/yellow_clear AND local_origin/target_lane_valid",
        "right_rule": "right_3m_all_boxes_clear AND right_corridor_roadside/yellow_clear AND local_origin/target_lane_valid",
        "white_dashed_lane_is_allowed": True,
        "yellow_dashed_lane_is_allowed": True,
        "crossable_yellow_line_types": sorted(CROSSABLE_YELLOW_LINE_TYPES),
        "require_lane_annotation": not args.allow_missing_lane_annotations,
        "obstacle_policy": "all_valid_object_boxes",
        "max_centerline_distance": args.max_centerline_distance,
        "opposing_direction_dot": args.opposing_direction_dot,
        "local_lane_evidence_policy": "fail_closed",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-clips", default=DEFAULT_CLIPS)
    parser.add_argument("--pkl-path-lists", nargs="+", default=list(DEFAULT_PKL_LISTS))
    parser.add_argument(
        "--current-pkl-root",
        default="/mnt/l3-labeled-data/prd_data/ALL/Result",
        help="mounted Result directory used to resolve stale PKL-list paths",
    )
    parser.add_argument(
        "--clip-name",
        action="append",
        help="only scan this clip; repeat to scan more than one clip",
    )
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=101)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ego-length", type=float, default=4.8)
    parser.add_argument("--ego-width", type=float, default=2.0)
    parser.add_argument("--overlap-epsilon", type=float, default=0.0)
    parser.add_argument("--box-extra-margin", type=float, default=0.0)
    parser.add_argument("--max-centerline-distance", type=float, default=2.5)
    parser.add_argument("--opposing-direction-dot", type=float, default=-0.2)
    parser.add_argument("--allow-missing-lane-annotations", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.window_size <= 0 or args.workers <= 0:
        raise ValueError("window-size and workers must be positive")
    if args.ego_length <= 0.0 or args.ego_width <= 0.0 or args.max_centerline_distance <= 0.0:
        raise ValueError("ego dimensions and max-centerline-distance must be positive")
    if not -1.0 <= args.opposing_direction_dot <= 1.0:
        raise ValueError("opposing-direction-dot must be in [-1, 1]")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pkl_index = build_pkl_index(args.pkl_path_lists)
    clips = [Path(value.rstrip("/")).name for value in read_lines(args.main_clips)]
    if args.limit is not None:
        clips = clips[: args.limit]
    indexed_clips = list(enumerate(clips))
    if args.clip_name:
        requested = set(args.clip_name)
        indexed_clips = [(index, clip) for index, clip in indexed_clips if clip in requested]
        missing = requested - {clip for _, clip in indexed_clips}
        if missing:
            raise ValueError(f"clip names absent from main list: {sorted(missing)}")
    tasks = [
        {
            "clip_index_0based": index,
            "clip_name": clip,
            "source_pkl_path": pkl_index.get(clip),
            "detector": args.detector,
            "window_size": args.window_size,
            "ego_length": args.ego_length,
            "ego_width": args.ego_width,
            "overlap_epsilon": args.overlap_epsilon,
            "box_extra_margin": args.box_extra_margin,
            "require_lane_annotation": not args.allow_missing_lane_annotations,
            "max_centerline_distance": args.max_centerline_distance,
            "opposing_direction_dot": args.opposing_direction_dot,
            "current_pkl_root": args.current_pkl_root,
        }
        for index, clip in indexed_clips
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(scan_clip, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"progress={completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: int(row["clip_index_0based"]))
    summary = write_outputs(output, rows, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["error_clips"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
