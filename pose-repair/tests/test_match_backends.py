from pathlib import Path
from textwrap import indent

import pandas as pd
import pytest

from dongfeng_registration.config import BACKEND_NAMES, load_config
from dongfeng_registration.match_backends import (
    UNIFIED_MATCH_COLUMNS,
    backend_names,
    normalize_match_records,
)


def write_config(path: Path, matching: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset_root: ./data",
                "output_root: ./out",
                "run_name: test",
                "cameras: [front_camera]",
                "matching:",
                indent(matching, "  "),
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_full_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_parses_sift_backend(tmp_path):
    config_path = write_config(
        tmp_path / "sift.yaml",
        "\n".join(
            [
                "backend: sift_lk",
                'lk_mode: "on"',
            ]
        ),
    )

    config = load_config(config_path)

    assert config.matching.backend == "sift_lk"
    assert config.matching.lk_mode == "on"


def test_load_config_parses_deep_backend_and_related_settings(tmp_path):
    config_path = write_config(
        tmp_path / "deep.yaml",
        "\n".join(
            [
                "backend: superpoint_lightglue",
                'lk_mode: "off"',
                "model_device: cpu",
                "deep_image_size: 1024",
                "deep_max_keypoints: 512",
                "deep_score_threshold: 0.25",
                "deep_score_threshold_intra: 0.3",
                "deep_score_threshold_inter: 0.1",
                "deep_batch_size: 6",
                "loma_process_count: 2",
            ]
        ),
    )

    config = load_config(config_path)

    assert config.matching.backend == "superpoint_lightglue"
    assert config.matching.lk_mode == "off"
    assert config.matching.model_device == "cpu"
    assert config.matching.deep_image_size == 1024
    assert config.matching.deep_max_keypoints == 512
    assert config.matching.deep_score_threshold == 0.25
    assert config.matching.deep_score_threshold_intra == 0.3
    assert config.matching.deep_score_threshold_inter == 0.1
    assert config.matching.deep_batch_size == 6
    assert config.matching.loma_process_count == 2


def test_load_config_parses_loma_backend(tmp_path):
    config_path = write_config(
        tmp_path / "loma.yaml",
        "\n".join(
            [
                "backend: loma",
                'lk_mode: "off"',
                "model_device: cuda:0",
                "deep_image_size: 1600",
                "deep_max_keypoints: 2048",
                "deep_score_threshold: 0.2",
            ]
        ),
    )

    config = load_config(config_path)

    assert config.matching.backend == "loma"
    assert config.matching.lk_mode == "off"


def test_backend_names_lists_all_supported_backends():
    assert backend_names() == list(BACKEND_NAMES)


def test_load_config_resolves_relative_paths_against_config_directory(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = write_full_config(
        config_dir / "paths.yaml",
        "\n".join(
            [
                "dataset_root: ../data",
                "output_root: ../outputs",
                "run_name: test",
                "cameras: [front_camera]",
            ]
        ),
    )

    config = load_config(config_path)

    assert config.dataset_root == (config_dir / "../data").resolve()
    assert config.output_root == (config_dir / "../outputs").resolve()


@pytest.mark.parametrize("yaml_text", ["", "[]"])
def test_load_config_rejects_empty_or_non_mapping_yaml(tmp_path, yaml_text):
    config_path = write_full_config(tmp_path / "bad.yaml", yaml_text)

    with pytest.raises(ValueError, match="configuration file must contain a mapping"):
        load_config(config_path)


def test_invalid_backend_is_rejected(tmp_path):
    config_path = write_config(
        tmp_path / "invalid.yaml",
        "\n".join(
            [
                "backend: unknown_backend",
            ]
        ),
    )

    with pytest.raises(ValueError):
        load_config(config_path)


def test_normalize_match_records_fills_sparse_row_defaults():
    df = normalize_match_records(
        [
            {
                "nx1": 0.1,
                "ny1": 0.2,
                "nx2": 0.3,
                "ny2": 0.4,
                "x1": 10.5,
                "y1": 20.25,
                "x2": 30.75,
                "y2": 40.125,
                "backend": "sift_lk",
            }
        ]
    )

    assert list(df.columns) == UNIFIED_MATCH_COLUMNS
    assert df.loc[0, "kp1"] == -1
    assert df.loc[0, "kp2"] == -1
    assert pd.isna(df.loc[0, "descriptor_distance"])
    assert pd.isna(df.loc[0, "ratio"])
    assert pd.isna(df.loc[0, "lk_fb_error"])
    assert pd.isna(df.loc[0, "backend_score"])
    assert df.loc[0, "backend"] == "sift_lk"
    assert df.loc[0, "x1"] == 10.5
    assert df.loc[0, "y2"] == 40.125
    assert df.loc[0, "lk_valid"]
    assert not df.loc[0, "ransac_inlier"]


def test_normalize_match_records_preserves_deep_row_values():
    record = {
        "kp1": 7,
        "kp2": 11,
        "x1": 1.1,
        "y1": 2.2,
        "x2": 3.3,
        "y2": 4.4,
        "nx1": 0.11,
        "ny1": 0.22,
        "nx2": 0.33,
        "ny2": 0.44,
        "descriptor_distance": 0.9,
        "ratio": 0.8,
        "lk_fb_error": 0.7,
        "lk_valid": False,
        "ransac_inlier": True,
        "backend": "superpoint_lightglue",
        "backend_score": 0.6,
        "extra_ignored": "kept out of frame",
    }

    df = normalize_match_records([record])

    assert list(df.columns) == UNIFIED_MATCH_COLUMNS
    for key, value in record.items():
        if key in UNIFIED_MATCH_COLUMNS:
            assert df.loc[0, key] == value
    assert "extra_ignored" not in df.columns
    assert not df.loc[0, "lk_valid"]
    assert df.loc[0, "ransac_inlier"]


def test_normalize_match_records_accepts_loma_backend():
    df = normalize_match_records(
        [
            {
                "backend": "loma",
                "backend_score": 0.9,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 0.1,
                "ny1": 0.2,
                "nx2": 0.3,
                "ny2": 0.4,
            }
        ]
    )

    assert df.loc[0, "backend"] == "loma"
    assert df.loc[0, "backend_score"] == 0.9


@pytest.mark.parametrize(
    "record, match",
    [
        ({}, "backend is required"),
        ({"backend": "does_not_exist"}, "unknown backend"),
        (
            {
                "backend": "sift_lk",
                "x1": "bad",
                "y1": 1.0,
                "x2": 2.0,
                "y2": 3.0,
                "nx1": 4.0,
                "ny1": 5.0,
                "nx2": 6.0,
                "ny2": 7.0,
            },
            "x1 must be numeric",
        ),
        (
            {
                "backend": "sift_lk",
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 5.0,
                "ny1": 6.0,
                "nx2": 7.0,
                "ny2": 8.0,
                "lk_valid": "yes",
            },
            "lk_valid must be bool",
        ),
        (
            {
                "backend": "sift_lk",
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "nx1": 5.0,
                "ny1": 6.0,
                "nx2": 7.0,
                "ny2": 8.0,
                "backend_score": "high",
            },
            "backend_score must be numeric",
        ),
    ],
)
def test_normalize_match_records_rejects_malformed_rows(record, match):
    with pytest.raises(ValueError, match=match):
        normalize_match_records([record])
