"""
Batch-training configuration helpers.

The batch runner stores shared defaults at the config level and per-scene
overrides inside each scene entry. These helpers merge those dictionaries,
resolve data/output paths, and convert the final payloads into command-line
arguments for `train.py` and `postprocess.py`.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

from utils.experiment_utils import load_json


DEFAULT_CONFIG_PATH = "training_config.json"


def parse_args() -> argparse.Namespace:
    """
        Parse command-line options for the batch runner.
    """
    parser = argparse.ArgumentParser(description="Config-driven batch training")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only_scene", type=str, default="")
    parser.add_argument("--rebuild_total", action="store_true")
    return parser.parse_args()


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
        Deep-merge two dictionaries without mutating either input.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_batch_config(config_path: str) -> dict[str, Any]:
    """
        Load one batch JSON config, returning an empty dict for missing content.
    """
    return load_json(config_path, default={})


def parse_scene_filter(only_scene: str) -> set[str]:
    """
        Convert a comma-separated scene filter into a set of scene names.
    """
    return {item.strip() for item in str(only_scene).split(",") if item.strip()}


def filter_config_scenes(config: dict[str, Any], scene_filter: set[str]) -> list[dict[str, Any]]:
    """
        Return only scenes requested by the filter and report unknown names early.
    """
    scenes = list(config.get("scenes", []))
    if not scene_filter:
        return scenes
    filtered_scenes = [scene for scene in scenes if str(scene.get("name", "")) in scene_filter]
    missing_scenes = sorted(scene_filter - {str(scene.get("name", "")) for scene in filtered_scenes})
    if missing_scenes:
        raise ValueError("Requested scenes not found in config: {}".format(", ".join(missing_scenes)))
    return filtered_scenes


def resolve_config_paths(config_path: str) -> list[str]:
    """
        Resolve either a single config file or all runnable configs in a folder.
    """
    resolved_path = Path(config_path)
    if resolved_path.is_file():
        return [str(resolved_path)]
    if resolved_path.is_dir():
        config_files: list[str] = []
        for path in sorted(resolved_path.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".json" or path.name.endswith(".schema.json"):
                continue
            payload = load_json(str(path), default={})
            if isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
                config_files.append(str(path))
        if not config_files:
            raise FileNotFoundError(
                "No runnable config files with `scenes` found in config directory: {}".format(config_path)
            )
        return config_files
    raise FileNotFoundError("Config path does not exist: {}".format(config_path))


def dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    """
        Convert a payload dictionary into argparse-style `--key value` tokens.
    """
    cli_args = []
    for key, value in payload.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            cli_args.extend([flag, str(value).lower()])
        elif value is None:
            continue
        elif isinstance(value, list):
            if value:
                cli_args.append(flag)
                cli_args.extend(str(item) for item in value)
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


def build_scene_output_stem(config: dict[str, Any], scene: dict[str, Any]) -> str:
    """
        Build the base output name from scene name plus global and scene suffixes.
    """
    output_name = scene["name"]
    suffix_parts = [part for part in [config.get("output_suffix", ""), scene.get("output_suffix", "")] if part]
    if suffix_parts:
        output_name = "{}-{}".format(output_name, "-".join(suffix_parts))
    return output_name


def build_output_name(config: dict[str, Any], scene: dict[str, Any], run_index: int) -> str:
    """
        Add a repeat index to the output name when the config runs repeats.
    """
    output_name = build_scene_output_stem(config, scene)
    if int(config.get("repeat", 1)) > 1:
        output_name = "{}-{}".format(output_name, run_index + 1)
    return output_name


def build_scene_paths(config: dict[str, Any], scene: dict[str, Any], run_index: int) -> tuple[str, str]:
    """
        Resolve the input data folder and output folder for one scene run.
    """
    output_dir = os.path.join(config["output_root"], build_output_name(config, scene, run_index))
    data_dir = os.path.join(config["data_root"], scene["name"])
    return data_dir, output_dir


def build_train_payload(config: dict[str, Any], scene: dict[str, Any], data_dir: str, output_dir: str) -> dict[str, Any]:
    """
        Merge train arguments and inject source/model paths for one scene run.
    """
    train_payload = merge_dict(config.get("train_args", {}), scene.get("train_args", {}))
    train_payload["source_path"] = data_dir
    train_payload["model_path"] = output_dir
    checkpoint_args = merge_dict(config.get("checkpoint_args", {}), scene.get("checkpoint_args", {}))
    checkpoint_iterations = checkpoint_args.get("checkpoint_iterations", [])
    if checkpoint_iterations:
        train_payload["checkpoint_iterations"] = checkpoint_iterations
    return train_payload


def build_postprocess_payload(config: dict[str, Any], scene: dict[str, Any], data_dir: str, output_dir: str) -> dict[str, Any]:
    """
        Build post-processing arguments from config and matching train settings.

        Dataset-related train options are copied so post-processing reconstructs
        the same scene and camera setup used during training.
    """
    postprocess_payload = merge_dict(config.get("postprocess_args", {}), scene.get("postprocess_args", {}))
    train_payload = merge_dict(config.get("train_args", {}), scene.get("train_args", {}))
    for key in (
        "sh_degree",
        "images",
        "resolution",
        "white_background",
        "data_device",
        "eval",
        "train_test_exp",
        "debug",
        "antialiasing",
        "depth_ratio",
    ):
        if key in train_payload:
            postprocess_payload[key] = train_payload[key]
    postprocess_payload["source_path"] = data_dir
    postprocess_payload["model_path"] = output_dir
    return postprocess_payload


def build_python_command(python_executable: str, script_name: str, payload: dict[str, Any]) -> list[str]:
    """
        Build the final Python command list passed to subprocess.
    """
    return [python_executable or sys.executable, script_name, *dict_to_cli_args(payload)]
