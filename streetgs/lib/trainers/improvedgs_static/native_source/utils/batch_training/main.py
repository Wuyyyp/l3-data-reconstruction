"""
Batch training workflow.

For each config, the runner expands scenes and repeats, selects a GPU, launches
`train.py`, optionally launches `postprocess.py`, then aggregates result JSON
files into total CSV files. It also supports rebuilding CSV files from existing
outputs without rerunning training.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from utils.batch_training.config import (
    build_postprocess_payload,
    build_python_command,
    build_scene_paths,
    build_train_payload,
    filter_config_scenes,
    load_batch_config,
    parse_args,
    parse_scene_filter,
    resolve_config_paths,
)
from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import install_signal_handlers, run_command
from utils.batch_training.results import (
    apply_best_repeat_selection_by_psnr,
    collect_scene_result_rows,
    extend_result_rows,
    rebuild_result_total_csv,
    should_select_best_repeat_by_psnr,
    write_result_total_csv,
)
from utils.experiment_utils import backup_batch_training_config


def run_single_config(config_path: str, dry_run: bool, scene_filter: set[str], rebuild_total: bool) -> int:
    """
        Execute or rebuild one batch config.

        The function owns the top-level order for one config: validate scenes,
        prepare output root, run each repeat/scene pair, collect metrics, select
        best repeats when requested, and write total CSV files.
    """
    config = load_batch_config(config_path)
    if not isinstance(config.get("scenes"), list) or len(config["scenes"]) == 0:
        raise ValueError("Config {} does not contain any scenes.".format(config_path))
    if rebuild_total:
        if dry_run:
            print("Dry run: would rebuild total CSV for {}".format(config_path))
            return 0
        Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
        rebuild_result_total_csv(config_path, config)
        return 0

    scenes_to_run = filter_config_scenes(config, scene_filter)
    if not dry_run:
        Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
        backup_path = backup_batch_training_config(config, config_path)
        if backup_path is not None:
            print("Backed up batch training config to {}".format(backup_path))

    python_executable = config.get("python_executable", sys.executable)
    stop_on_error = bool(config.get("stop_on_error", True))
    auto_select_best_repeat = should_select_best_repeat_by_psnr(config)
    run_postprocess = bool(config.get("run_postprocess", True) or auto_select_best_repeat)
    gpu_id = config.get("gpu_id")
    result_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    first_failure_code = 0

    try:
        for run_index in range(int(config.get("repeat", 1))):
            for scene in scenes_to_run:
                data_dir, output_dir = build_scene_paths(config, scene, run_index)
                lock_path = None
                try:
                    selected_gpu, lock_path = select_scene_gpu(config, scene["name"], gpu_id, dry_run)

                    train_payload = build_train_payload(config, scene, data_dir, output_dir)
                    train_command = build_python_command(python_executable, "train.py", train_payload)
                    return_code = run_command(train_command, gpu_id=selected_gpu, dry_run=dry_run)
                    if return_code != 0:
                        if stop_on_error:
                            return return_code
                        if first_failure_code == 0:
                            first_failure_code = return_code
                        continue

                    if not run_postprocess:
                        continue

                    postprocess_payload = build_postprocess_payload(config, scene, data_dir, output_dir)
                    postprocess_command = build_python_command(python_executable, "postprocess.py", postprocess_payload)
                    return_code = run_command(postprocess_command, gpu_id=selected_gpu, dry_run=dry_run)
                    if return_code != 0:
                        if stop_on_error:
                            return return_code
                        if first_failure_code == 0:
                            first_failure_code = return_code
                        continue
                    if not dry_run:
                        scene_rows = collect_scene_result_rows(config_path, scene, run_index, output_dir, train_payload)
                        extend_result_rows(result_rows_by_split, scene_rows)
                finally:
                    release_gpu_lock(lock_path)
    except KeyboardInterrupt:
        return 130

    if not dry_run:
        if auto_select_best_repeat:
            selected_scenes = scenes_to_run if scene_filter else None
            result_rows_by_split = apply_best_repeat_selection_by_psnr(config, result_rows_by_split, selected_scenes)
        if len(config.get("scenes", [])) > 1 and not scene_filter:
            for split_name, rows in result_rows_by_split.items():
                csv_path = write_result_total_csv(config["output_root"], config_path, split_name, rows)
                if csv_path is not None:
                    print("Saved {} total CSV to {}".format(split_name, csv_path))
        if len(config.get("scenes", [])) > 1 and scene_filter:
            rebuild_result_total_csv(config_path, config)

    return first_failure_code


def main() -> int:
    """
        Parse batch arguments and process every resolved config path.
    """
    install_signal_handlers()
    args = parse_args()
    scene_filter = parse_scene_filter(args.only_scene)
    config_paths = resolve_config_paths(args.config)
    for config_path in config_paths:
        print("Processing config {}".format(config_path))
        return_code = run_single_config(config_path, args.dry_run, scene_filter, bool(args.rebuild_total))
        if return_code != 0:
            return return_code
    return 0
