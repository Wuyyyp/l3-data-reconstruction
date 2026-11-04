"""
Batch result aggregation helpers.

Post-processing writes one JSON summary per scene and split. This module reads
those summaries, turns them into CSV rows, optionally chooses the best repeat by
PSNR, and can rebuild total CSV files from already existing outputs.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from utils.batch_training.config import (
    build_output_name,
    build_scene_output_stem,
    build_scene_paths,
    build_train_payload,
)
from utils.experiment_utils import archive_existing_output, load_json


def load_result_summary(output_dir: str, split_name: str) -> Optional[dict[str, Any]]:
    """
        Load one `result_<split>.json` file if it exists and contains data.
    """
    summary_path = os.path.join(output_dir, "result_{}.json".format(split_name))
    if not os.path.exists(summary_path):
        return None
    payload = load_json(summary_path, default={})
    return payload if isinstance(payload, dict) and payload else None


def collect_scene_result_rows(
    config_path: str,
    scene: dict[str, Any],
    run_index: int,
    output_dir: str,
    train_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
        Convert one scene output directory into train/test CSV rows.
    """
    common_row = {
        "config_name": Path(config_path).stem,
        "scene_name": str(scene["name"]),
        "run_index": int(run_index + 1),
        "output_name": os.path.basename(output_dir.rstrip(os.sep)),
        "model_path": output_dir,
        "training_method": str(train_payload.get("training_method", "3dgs")).lower(),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ("train", "test"):
        summary = load_result_summary(output_dir, split_name)
        if summary is None:
            continue
        row = dict(common_row)
        row["split"] = split_name
        row.update(summary)
        rows_by_split[split_name].append(row)
    return rows_by_split


def write_result_total_csv(output_root: str, config_path: str, split_name: str, rows: list[dict[str, Any]]) -> Optional[str]:
    """
        Write rows for one split into the config-level total CSV file.
    """
    if not rows:
        return None
    csv_path = os.path.join(output_root, "{}_result_{}_total.csv".format(Path(config_path).stem, split_name))
    ordered_fieldnames = [
        "config_name",
        "split",
        "iteration",
        "scene_name",
        "PSNR",
        "SSIM",
        "LPIPS",
        "FPS",
        "NUM",
        "Training_time",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _parse_metric_float(metric_value: Any) -> float:
    """
        Parse a metric for sorting, treating missing values as negative infinity.
    """
    try:
        return float(metric_value)
    except (TypeError, ValueError):
        return float("-inf")


def should_select_best_repeat_by_psnr(config: dict[str, Any]) -> bool:
    """
        Check whether repeat outputs should be collapsed to the best PSNR run.
    """
    return bool(config.get("select_best_repeat_by_psnr", False)) and int(config.get("repeat", 1)) > 1


def select_best_repeat_outputs_by_psnr(
    config: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    scenes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, str]:
    """
        Choose the winning repeat output name for each scene by PSNR.

        Test rows are preferred when available; train rows are used as a fallback
        so selection can still work if a run only produced train metrics.
    """
    repeat_count = int(config.get("repeat", 1))
    winners: dict[str, str] = {}
    for scene in scenes if scenes is not None else config.get("scenes", []):
        base_output_name = build_scene_output_stem(config, scene)
        candidate_output_names = [build_output_name(config, scene, run_index) for run_index in range(repeat_count)]

        metric_rows: dict[str, dict[str, Any]] = {}
        for split_name in ("test", "train"):
            metric_rows = {
                str(row["output_name"]): row
                for row in rows_by_split.get(split_name, [])
                if str(row.get("output_name", "")) in candidate_output_names
            }
            if metric_rows:
                break
        if not metric_rows:
            raise ValueError(
                "select_best_repeat_by_psnr requires available result rows for scene {}, but none were found.".format(
                    scene["name"]
                )
            )

        best_output_name = max(
            candidate_output_names,
            key=lambda output_name: (
                _parse_metric_float(metric_rows.get(output_name, {}).get("PSNR")),
                -candidate_output_names.index(output_name),
            ),
        )
        if best_output_name not in metric_rows:
            raise ValueError(
                "select_best_repeat_by_psnr could not find a valid PSNR row for scene {}.".format(scene["name"])
            )
        winners[base_output_name] = best_output_name
    return winners


def apply_best_repeat_selection_by_psnr(
    config: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    scenes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
        Keep only best-repeat outputs on disk and return rows for the winners.

        Non-winning repeat folders are removed. The winning folder is renamed to
        the base scene output name, archiving an existing target if needed.
    """
    output_root = str(config["output_root"])
    scenes_to_select = scenes if scenes is not None else config.get("scenes", [])
    best_outputs = select_best_repeat_outputs_by_psnr(config, rows_by_split, scenes_to_select)
    filtered_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}

    for scene in scenes_to_select:
        base_output_name = build_scene_output_stem(config, scene)
        best_output_name = best_outputs[base_output_name]
        best_source_dir = os.path.join(output_root, best_output_name)
        best_target_dir = os.path.join(output_root, base_output_name)

        if not os.path.isdir(best_source_dir):
            raise FileNotFoundError("Best repeat output directory does not exist: {}".format(best_source_dir))

        for run_index in range(int(config.get("repeat", 1))):
            candidate_output_name = build_output_name(config, scene, run_index)
            candidate_output_dir = os.path.join(output_root, candidate_output_name)
            if candidate_output_name != best_output_name and os.path.isdir(candidate_output_dir):
                shutil.rmtree(candidate_output_dir)

        if best_output_name != base_output_name:
            archived_path = archive_existing_output(best_target_dir)
            if archived_path is not None:
                print("Archived existing winner target output to {}".format(archived_path))
            if os.path.exists(best_target_dir):
                raise FileExistsError(
                    "Winner target output directory already exists and could not be archived: {}".format(best_target_dir)
                )
            os.rename(best_source_dir, best_target_dir)

        for split_name in ("train", "test"):
            matching_rows = [
                dict(row)
                for row in rows_by_split.get(split_name, [])
                if str(row.get("output_name", "")) == best_output_name
            ]
            if matching_rows:
                selected_row = matching_rows[0]
                selected_row["output_name"] = base_output_name
                selected_row["model_path"] = best_target_dir
                filtered_rows_by_split[split_name].append(selected_row)
    return filtered_rows_by_split


def extend_result_rows(
    result_rows_by_split: dict[str, list[dict[str, Any]]],
    scene_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """
        Append newly collected rows into the split-indexed row accumulator.
    """
    for split_name, rows in scene_rows.items():
        result_rows_by_split[split_name].extend(rows)


def collect_existing_total_rows(config_path: str, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
        Read result rows from existing output folders without launching jobs.
    """
    result_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    repeat_count = int(config.get("repeat", 1))
    use_selected_repeat_outputs = should_select_best_repeat_by_psnr(config)
    for scene in config.get("scenes", []):
        if use_selected_repeat_outputs:
            base_output_dir = os.path.join(config["output_root"], build_scene_output_stem(config, scene))
            if os.path.isdir(base_output_dir):
                data_dir = os.path.join(config["data_root"], scene["name"])
                train_payload = build_train_payload(config, scene, data_dir, base_output_dir)
                scene_rows = collect_scene_result_rows(config_path, scene, 0, base_output_dir, train_payload)
                extend_result_rows(result_rows_by_split, scene_rows)
                continue
        for run_index in range(repeat_count):
            data_dir, output_dir = build_scene_paths(config, scene, run_index)
            train_payload = build_train_payload(config, scene, data_dir, output_dir)
            scene_rows = collect_scene_result_rows(config_path, scene, run_index, output_dir, train_payload)
            extend_result_rows(result_rows_by_split, scene_rows)
    return result_rows_by_split


def rebuild_result_total_csv(config_path: str, config: dict[str, Any]) -> None:
    """
        Recreate total CSV files from existing per-scene result summaries.
    """
    rows_by_split = collect_existing_total_rows(config_path, config)
    for split_name, rows in rows_by_split.items():
        csv_path = write_result_total_csv(config["output_root"], config_path, split_name, rows)
        if csv_path is not None:
            print("Rebuilt {} total CSV to {}".format(split_name, csv_path))
