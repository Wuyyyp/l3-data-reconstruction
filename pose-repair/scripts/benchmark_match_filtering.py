from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dongfeng_registration.staged_pipeline import (
    _filter_inter_cache_group_worker,
    _filter_intra_cache_group_worker,
    _retained_mask,
    evaluate_inter_camera_match_quality,
    evaluate_intra_camera_match_quality,
)


def _old_timing(root: Path, scope: str) -> dict[str, float | int]:
    paths = sorted(root.glob(f"{scope}_camera/**/pair_metrics_quality.csv"))
    columns = {
        "navigation_geometry_time_ms",
        "threshold_annotation_time_ms",
        "quality_filter_time_ms",
        "evaluation_total_time_ms",
    }
    totals = {column: 0.0 for column in columns}
    rows = 0
    for path in paths:
        frame = pd.read_csv(path)
        rows += len(frame)
        for column in columns:
            if column in frame:
                totals[column] += float(frame[column].sum())
    return {"files": len(paths), "rows": rows, **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark match filtering on existing raw caches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--match-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("intra", "inter", "all"), default="all")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=0)
    args = parser.parse_args()

    scopes = ("intra", "inter") if args.scope == "all" else (args.scope,)
    if args.parallel_workers > 0:
        if args.max_pairs is not None:
            parser.error("--max-pairs is not supported with --parallel-workers")
        grouped_paths: list[tuple[str, tuple[Path, ...]]] = []
        for scope in scopes:
            groups: dict[Path, list[Path]] = defaultdict(list)
            for raw_path in sorted(
                args.match_root.glob(f"{scope}_camera/**/raw_matches.parquet")
            ):
                groups[raw_path.parent.parent].append(raw_path.parent)
            grouped_paths.extend(
                (scope, tuple(cache_dirs)) for cache_dirs in groups.values()
            )
        started = time.perf_counter()
        summaries: list[dict[str, int]] = []
        with ProcessPoolExecutor(
            max_workers=int(args.parallel_workers),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = []
            for scope, cache_dirs in grouped_paths:
                worker = (
                    _filter_intra_cache_group_worker
                    if scope == "intra"
                    else _filter_inter_cache_group_worker
                )
                futures.append(
                    executor.submit(
                        worker,
                        args.config,
                        args.dataset_root,
                        cache_dirs,
                        0,
                        None,
                        False,
                    )
                )
            for future in futures:
                summaries.append(future.result())
        print(
            json.dumps(
                {
                    "parallel_workers": int(args.parallel_workers),
                    "group_count": len(grouped_paths),
                    "wall_time_ms": (time.perf_counter() - started) * 1000.0,
                    "processed_dirs": sum(item["processed_dirs"] for item in summaries),
                    "raw_matches": sum(item["raw_matches"] for item in summaries),
                    "retained_matches": sum(
                        item["retained_matches"] for item in summaries
                    ),
                },
                indent=2,
            )
        )
        return

    report: dict[str, object] = {}
    for scope in scopes:
        raw_paths = sorted(args.match_root.glob(f"{scope}_camera/**/raw_matches.parquet"))
        old = _old_timing(args.match_root, scope)
        started = time.perf_counter()
        raw_rows = 0
        retained_rows = 0
        new_rows = 0
        new_recorded_total_ms = 0.0
        new_recorded_navigation_ms = 0.0
        new_recorded_threshold_ms = 0.0
        new_recorded_quality_filter_ms = 0.0
        selected_pairs = 0
        remaining_pairs = args.max_pairs
        for path in raw_paths:
            raw = pd.read_parquet(path)
            raw.attrs.clear()
            if args.max_pairs is not None:
                group_columns = (
                    ["camera", "frame1", "frame2"]
                    if scope == "intra"
                    else ["camera1", "camera2", "frame1", "frame2"]
                )
                selected_groups = raw[group_columns].drop_duplicates().head(remaining_pairs)
                raw = raw.merge(selected_groups, on=group_columns, how="inner")
                remaining_pairs -= len(selected_groups)
            raw_rows += len(raw)
            if raw.empty:
                continue
            selected_pairs += len(raw.groupby(
                ["camera", "frame1", "frame2"]
                if scope == "intra"
                else ["camera1", "camera2", "frame1", "frame2"]
            ))
            if scope == "intra":
                evaluated = evaluate_intra_camera_match_quality(
                    config_path=args.config,
                    dataset_root=args.dataset_root,
                    matches=raw,
                )
            else:
                evaluated = evaluate_inter_camera_match_quality(
                    config_path=args.config,
                    dataset_root=args.dataset_root,
                    matches=raw,
                )
            new_rows += len(evaluated)
            retained_rows += int(_retained_mask(evaluated).sum())
            pair_metrics = evaluated.attrs.get("pair_metrics")
            if isinstance(pair_metrics, pd.DataFrame):
                for column, target in (
                    ("evaluation_total_time_ms", "new_recorded_total_ms"),
                    ("navigation_geometry_time_ms", "new_recorded_navigation_ms"),
                    ("threshold_annotation_time_ms", "new_recorded_threshold_ms"),
                    ("quality_filter_time_ms", "new_recorded_quality_filter_ms"),
                ):
                    if column in pair_metrics:
                        value = float(pair_metrics[column].sum())
                        if target == "new_recorded_total_ms":
                            new_recorded_total_ms += value
                        elif target == "new_recorded_navigation_ms":
                            new_recorded_navigation_ms += value
                        elif target == "new_recorded_threshold_ms":
                            new_recorded_threshold_ms += value
                        else:
                            new_recorded_quality_filter_ms += value
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        report[scope] = {
            "files": len(raw_paths),
            "raw_rows": raw_rows,
            "evaluated_rows": new_rows,
            "retained_rows": retained_rows,
            "selected_pairs": selected_pairs,
            "new_wall_time_ms": elapsed_ms,
            "new_recorded_total_time_ms": new_recorded_total_ms,
            "new_recorded_navigation_ms": new_recorded_navigation_ms,
            "new_recorded_threshold_ms": new_recorded_threshold_ms,
            "new_recorded_quality_filter_ms": new_recorded_quality_filter_ms,
            "old_recorded_total_time_ms": old.get("evaluation_total_time_ms", 0.0),
            "old_recorded_navigation_ms": old.get("navigation_geometry_time_ms", 0.0),
            "old_recorded_threshold_ms": old.get("threshold_annotation_time_ms", 0.0),
            "old_recorded_quality_filter_ms": old.get("quality_filter_time_ms", 0.0),
            "old_recorded_rows": old.get("rows", 0),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
