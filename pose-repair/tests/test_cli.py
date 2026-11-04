from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from dongfeng_registration.cli import app
from dongfeng_registration.config import load_config
from dongfeng_registration.delivery_pipeline import build_delivery_stages
from dongfeng_registration.pipeline import run_camera
from tests.conftest import real_t3_data_available


runner = CliRunner()

DATA_ROOT = Path(
    "2026_03_05_16_38_30_dlp_pilotGtParser"
    "/2026_03_05_16_38_30_dlp_pilotGtParser"
)

requires_real_t3_data = pytest.mark.skipif(
    not real_t3_data_available(DATA_ROOT),
    reason=f"optional real T3 dataset not available: {DATA_ROOT}",
)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/t3.yaml",
        "configs/t3_windows_full37.yaml",
        "configs/t3_windows_subset21.yaml",
        "configs/l29.yaml",
        "configs/l29_windows.yaml",
        "configs/m18.yaml",
        "configs/m18_windows.yaml",
    ],
)
def test_vehicle_configs_enable_bounded_dynamic_compensation_cache(config_path: str):
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dynamic = payload["dynamic_compensation"]

    assert dynamic["track_workers"] >= 1
    assert dynamic["cloud_cache_max_gb"] == 8.0
    assert dynamic["single_lidar_output_mode"] == "hardlink"


def _write_minimal_cli_config(path: Path, *, optimization: dict[str, object] | None = None) -> None:
    lines = [
        f"dataset_root: {path.parent.as_posix()}",
        f"output_root: {(path.parent / 'outputs').as_posix()}",
        "cameras: [camera_a, camera_b]",
    ]
    if optimization:
        lines.append("optimization:")
        for key, value in optimization.items():
            lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_inspect_cli_reports_t3_mainline_seven_cameras(monkeypatch):
    cameras = load_config(Path("configs/t3.yaml")).cameras
    report = pd.DataFrame(
        {
            "camera": cameras,
            "frame_count": [600] * len(cameras),
            "timestamps_monotonic": [True] * len(cameras),
            "all_images_exist": [True] * len(cameras),
            "image_width": [1920] * len(cameras),
            "image_height": [1080] * len(cameras),
            "image_size_matches": [True] * len(cameras),
            "model": ["pinhole"] * len(cameras),
            "calibration_valid": [True] * len(cameras),
        }
    )
    monkeypatch.setattr(
        "dongfeng_registration.cli.inspect_dataset",
        lambda _root, configured_cameras: report.loc[
            report["camera"].isin(configured_cameras)
        ].copy(),
    )

    result = runner.invoke(app, ["inspect", "--config", "configs/t3.yaml"])
    assert result.exit_code == 0, result.output
    assert "center_camera_fov30" in result.output
    assert "center_camera_fov120" in result.output
    assert "left_front_camera" in result.output
    assert "7 camera(s) checked" in result.output


@requires_real_t3_data
def test_small_real_run_writes_complete_output_contract(tmp_path):
    config = load_config(Path("configs/t3.yaml"))
    config.dataset_root = DATA_ROOT.resolve()
    config.output_root = tmp_path
    output = run_camera(
        config,
        camera="center_camera_fov30",
        start_frame=285,
        frame_count=3,
        frame_offset=1,
        run_name="test-small",
    )
    matches_path = output / "observations/center_camera_fov30_matches.parquet"
    tracks_path = output / "observations/center_camera_fov30_tracks.parquet"
    pair_metrics_path = output / "metrics/pair_metrics.csv"
    assert matches_path.exists()
    assert tracks_path.exists()
    assert pair_metrics_path.exists()
    assert (output / "resolved_config.yaml").exists()
    assert (output / "assumptions.json").exists()
    assert (output / "visualizations/center_camera_fov30_error_over_time.png").exists()
    matches = pd.read_parquet(matches_path)
    required = {
        "camera", "frame1", "frame2", "timestamp1_ns", "timestamp2_ns",
        "x1", "y1", "x2", "y2", "track_id", "navigation_valid",
        "nav_point_to_epiline_px", "nav_symmetric_epipolar_px",
        "nav_sampson_px2", "normalized_algebraic_residual",
    }
    assert required <= set(matches.columns)
    assert len(matches) > 0


def test_staged_commands_are_registered():
    for command in (
        "segment-dataset",
        "match-dataset",
        "match-multicamera",
        "match-delivery",
        "optimize-multicamera-extrinsics",
        "export-deliverables",
        "reconstruct-colmap",
        "run-delivery",
        "diagnose-clip-quality",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output


def _write_delivery_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "delivery:",
                "  vehicle_model: M18",
                "  pose_backend: result_interpolation_pkl",
                "  raw_dataset_root: raw_m18",
                "  raw_pkl: result_interpolation.pkl",
                "  canonical_root: canonical_m18",
                "  base_config: configs/m18.yaml",
                "  dynamic_output_root: outputs/dynamic_m18",
                "  dynamic_pkl: outputs/dynamic_m18/result_interpolation_dynamic_compensated.pkl",
                "  reference_time_offset_s: 0.007",
                "  semantic_cache_root: data_cache/M18/semantic_masks",
                "  match_cache_root: data_cache/M18/matches",
                "  match_cache_key_root: data_cache/M18/matches/loma__mask2former",
                "  optimization_output_root: outputs/M18_joint",
                "  projection_qc_root: outputs/M18_projection_qc",
                "  deliverable_root: deliverables/M18_final",
                "  colmap_config: configs/colmap.yaml",
                "  cameras:",
                "    - front_camera_fov30",
                "    - left_front_camera",
                "  raw_lidars:",
                "    - front_lidar",
                "  reference_cameras:",
                "    - front_camera_fov30",
                "    - left_front_camera",
                "  intra_baselines_m: [2, 4, 6]",
                "  inter_pairs:",
                "    - front_camera_fov30__left_front_camera",
                "  inter_relations:",
                "    - same_index",
                "    - temporal_overlap",
                "  frame_count: 8",
                "  projection_frame_stride: 5",
                "  torch_device: cpu",
            ]
        ),
        encoding="utf-8",
    )


def test_run_delivery_dry_run_prints_full_mainline(tmp_path):
    config = tmp_path / "delivery.yaml"
    _write_delivery_config(config)

    result = runner.invoke(app, ["run-delivery", "--config", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    for stage in (
        "normalize",
        "dynamic-compensation",
        "export-dynamic-pkl",
        "persistent-match",
        "optimize",
        "projection-qc",
        "export-deliverables",
        "colmap-reconstruction",
    ):
        assert stage in result.output
    assert "normalize-dataset" in result.output
    assert str(Path("scripts") / "compensate_dynamic_raw_lidars.py") in result.output
    assert (
        str(Path("scripts") / "export_dynamic_compensated_result_interpolation_pkl.py")
        in result.output
    )
    assert "segment-dataset" not in result.output
    assert "match-delivery" in result.output
    assert "optimize-multicamera-extrinsics" in result.output
    assert str(Path("scripts") / "generate_reprojection_qc.py") in result.output
    assert "export-deliverables" in result.output
    assert "reconstruct-colmap" in result.output
    assert "--remove-compensated-perception-source" in result.output

    dynamic_command = next(
        " ".join(stage.command)
        for stage in build_delivery_stages(config)
        if stage.name == "dynamic-compensation"
    )
    assert f"--config {config.resolve()}" in dynamic_command
    assert "--timing-json" in dynamic_command


def test_run_delivery_supports_pair_specific_inter_relations(tmp_path):
    config = tmp_path / "delivery.yaml"
    _write_delivery_config(config)
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n  inter_relations_by_pair:\n"
        + "    front_camera_fov30__left_front_camera:\n"
        + "      - same_index\n",
        encoding="utf-8",
    )

    commands = {
        stage.name: " ".join(stage.command)
        for stage in build_delivery_stages(config)
        if stage.name in {"persistent-match", "optimize"}
    }

    assert "match-delivery" in commands["persistent-match"]
    assert (
        "--inter-pair-relation front_camera_fov30__left_front_camera=same_index"
        in commands["optimize"]
    )


def test_run_delivery_keeps_semantic_and_preflight_inside_persistent_stage(tmp_path):
    config = tmp_path / "delivery.yaml"
    _write_delivery_config(config)
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\nsemantic_segmentation:\n  processor_image_size: 1024\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run-delivery", "--config", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "segment-dataset" not in result.output
    assert "diagnose-clip-quality" not in result.output
    assert "match-delivery" in result.output


def test_delivery_run_output_root_groups_non_cache_artifacts(tmp_path):
    config = tmp_path / "delivery.yaml"
    _write_delivery_config(config)
    run_root = tmp_path / "run_output"
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("  dynamic_output_root: outputs/dynamic_m18\n", "")
        .replace(
            "  dynamic_pkl: outputs/dynamic_m18/result_interpolation_dynamic_compensated.pkl\n",
            "",
        )
        .replace("  optimization_output_root: outputs/M18_joint\n", "")
        .replace("  projection_qc_root: outputs/M18_projection_qc\n", "")
        .replace("  deliverable_root: deliverables/M18_final\n", "")
        .replace(
            "  reference_time_offset_s: 0.007\n",
            f"  reference_time_offset_s: 0.007\n  run_output_root: {run_root.as_posix()}\n",
        ),
        encoding="utf-8",
    )

    commands = {
        stage.name: " ".join(stage.command) for stage in build_delivery_stages(config)
    }
    assert str(run_root / "dynamic_compensation") in commands["dynamic-compensation"]
    assert str(run_root / "dynamic_compensation") in commands["export-dynamic-pkl"]
    assert str(run_root / "joint_optimization") in commands["optimize"]
    assert str(run_root / "quality_control" / "dynamic_pointcloud") in commands[
        "dynamic-pointcloud-qc"
    ]
    assert str(run_root / "quality_control" / "reprojection") in commands[
        "projection-qc"
    ]
    assert str(run_root / "deliverables") in commands["export-deliverables"]
    assert f"--run-output-root {run_root}" in commands["colmap-reconstruction"]
    assert "match-delivery" in commands["persistent-match"]
    assert str(Path.cwd() / "data_cache" / "M18" / "matches") not in commands[
        "persistent-match"
    ]


def test_run_delivery_can_slice_stages_in_dry_run(tmp_path):
    config = tmp_path / "delivery.yaml"
    _write_delivery_config(config)

    result = runner.invoke(
        app,
        [
            "run-delivery",
            "--config",
            str(config),
            "--dry-run",
            "--from-stage",
            "persistent-match",
            "--to-stage",
            "optimize",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "normalize-dataset" not in result.output
    assert "scripts\\compensate_dynamic_raw_lidars.py" not in result.output
    assert "match-delivery" in result.output
    assert "optimize-multicamera-extrinsics" in result.output
    assert "export-deliverables" not in result.output


def test_t3_config_uses_2_4_6m_intra_baselines_and_adaptive_thresholds():
    settings = load_config(Path("configs/t3.yaml"))

    assert "match_quality_frame_offsets" not in settings.assumptions
    assert "match_quality_reprojection_thresholds_px" not in settings.assumptions
    assert "pose_quality_frame_offsets" not in settings.assumptions
    assert "pose_quality_reprojection_thresholds_px" not in settings.assumptions
    assert settings.assumptions["intra_camera_target_baselines_m"] == [2.0, 4.0, 6.0]
    assert settings.assumptions["quality_threshold_mode"] == "adaptive_mad"
    assert settings.assumptions["quality_threshold_mad_k"] == 2.0
    assert settings.assumptions["quality_threshold_min_candidates"] == 10
    assert settings.assumptions["quality_threshold_insufficient_reprojection_px"] == 5.0
    assert settings.assumptions["quality_threshold_insufficient_ray_closest_distance_m"] == 0.10
    assert "reprojection_threshold_mode" not in settings.assumptions
    assert "ray_closest_distance_mode" not in settings.assumptions
    assert "baseline_reprojection_threshold_px_2" not in settings.assumptions
    assert "reprojection_threshold_max_px_by_baseline" not in settings.assumptions
    assert "ray_closest_distance_min_m" not in settings.assumptions
    assert "ray_closest_distance_max_m" not in settings.assumptions


def test_segment_dataset_cli_defaults_preview_ratio_to_point_one(monkeypatch, tmp_path):
    captured = {}

    fake_config = type(
        "Config",
        (),
        {"dataset_root": tmp_path},
    )()

    class FakeMaskConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _write_minimal_cli_config(tmp_path / "config.yaml")
    monkeypatch.setattr("dongfeng_registration.cli.load_config", lambda _path: fake_config)
    monkeypatch.setattr("dongfeng_registration.cli.DynamicMaskConfig", FakeMaskConfig)
    monkeypatch.setattr(
        "dongfeng_registration.cli.segment_dataset_stage",
        lambda **kwargs: tmp_path / "semantic",
    )

    result = runner.invoke(
        app,
        [
            "segment-dataset",
            "--config",
            str(tmp_path / "config.yaml"),
            "--camera",
            "center_camera_fov30",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["preview_ratio"] == 0.1
    assert captured["device"] == "cuda:0"
    assert captured["batch_size"] == 4
    assert captured["inference_dtype"] == "fp16"
    assert captured["use_fast_processor"] is True
    assert captured["source_validation"] == "none"
    assert captured["image_cache_mode"] == "preload"


def test_match_multicamera_cli_accepts_explicit_temporal_overlap_relation(monkeypatch, tmp_path):
    captured = {}

    fake_config = type(
        "Config",
        (),
        {"dataset_root": tmp_path},
    )()

    def fake_match_multicamera(**kwargs):
        captured.update(kwargs)
        return tmp_path / "matches"

    _write_minimal_cli_config(tmp_path / "config.yaml")
    monkeypatch.setattr("dongfeng_registration.cli.load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        "dongfeng_registration.cli.match_multicamera_dataset_stage",
        fake_match_multicamera,
    )

    result = runner.invoke(
        app,
        [
            "match-multicamera",
            "--config",
            str(tmp_path / "config.yaml"),
            "--cameras",
            "center_camera_fov30",
            "--cameras",
            "left_front_camera",
            "--relation",
            "temporal_overlap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["relations"] == ("temporal_overlap",)


def test_match_multicamera_cli_rejects_unknown_relation(monkeypatch, tmp_path):
    captured = {}

    fake_config = type(
        "Config",
        (),
        {"dataset_root": tmp_path},
    )()

    def fake_match_multicamera(**kwargs):
        captured.update(kwargs)
        return tmp_path / "matches"

    _write_minimal_cli_config(tmp_path / "config.yaml")
    monkeypatch.setattr("dongfeng_registration.cli.load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        "dongfeng_registration.cli.match_multicamera_dataset_stage",
        fake_match_multicamera,
    )

    result = runner.invoke(
        app,
        [
            "match-multicamera",
            "--config",
            str(tmp_path / "config.yaml"),
            "--cameras",
            "center_camera_fov30",
            "--cameras",
            "left_front_camera",
            "--relation",
            "unknown_relation",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported" in result.output
    assert captured == {}


def test_match_multicamera_cli_defaults_to_same_index_only(monkeypatch, tmp_path):
    captured = {}

    fake_config = type(
        "Config",
        (),
        {"dataset_root": tmp_path},
    )()

    def fake_match_multicamera(**kwargs):
        captured.update(kwargs)
        return tmp_path / "matches"

    _write_minimal_cli_config(tmp_path / "config.yaml")
    monkeypatch.setattr("dongfeng_registration.cli.load_config", lambda _path: fake_config)
    monkeypatch.setattr(
        "dongfeng_registration.cli.match_multicamera_dataset_stage",
        fake_match_multicamera,
    )

    result = runner.invoke(
        app,
        [
            "match-multicamera",
            "--config",
            str(tmp_path / "config.yaml"),
            "--cameras",
            "left_front_camera",
            "--cameras",
            "left_rear_camera",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["relations"] == ("same_index",)


def test_optimize_multicamera_extrinsics_cli_passes_arguments(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return tmp_path / "joint"

    _write_minimal_cli_config(tmp_path / "config.yaml")
    monkeypatch.setattr(
        "dongfeng_registration.cli.optimize_multi_camera_extrinsics_from_match_cache",
        fake_optimize,
    )

    result = runner.invoke(
        app,
        [
            "optimize-multicamera-extrinsics",
            "--config",
            str(tmp_path / "config.yaml"),
            "--match-cache-root",
            str(tmp_path / "data_cache" / "matches" / "cache"),
            "--output",
            str(tmp_path / "output"),
            "--camera",
            "camera_a",
            "--camera",
            "camera_b",
            "--intra-offset",
            "5",
            "--inter-pair",
            "camera_a__camera_b",
            "--max-observations",
            "123",
            "--pair-sample-max-matches",
            "30",
            "--pair-sample-seed",
            "99",
            "--pair-sample-strategy",
            "spatial_grid_random",
            "--pair-sample-grid-cols",
            "3",
            "--pair-sample-grid-rows",
            "3",
            "--max-nfev",
            "9",
            "--evaluation-max-observations",
            "77",
            "--torch-device",
            "cpu",
            "--torch-learning-rate",
            "0.02",
            "--torch-lr-schedule",
            "step",
            "--torch-lr-step-size",
            "4",
            "--torch-lr-decay",
            "0.25",
            "--torch-lr-min-factor",
            "0.125",
            "--adaptive-weight-mode",
            "group_static",
            "--adaptive-weight-alpha",
            "0.6",
            "--adaptive-weight-min",
            "0.4",
            "--adaptive-weight-max",
            "2.5",
            "--adaptive-weight-long-tail-ratio-guard",
            "7.0",
            "--adaptive-weight-min-group-count",
            "123",
            "--intra-weight",
            "1.0",
            "--inter-weight",
            "3.0",
            "--optimize-time-varying-disturbance",
            "--time-varying-disturbance-dof",
            "6",
            "--optimize-body-time-varying-disturbance",
            "--body-time-varying-disturbance-dof",
            "6",
            "--time-varying-rotation-bound-deg",
            "1.5",
            "--time-varying-translation-bound-m",
            "0.05",
            "--fixed-translation-l2-weight",
            "0.2",
            "--fixed-rotation-l2-weight",
            "0.3",
            "--time-varying-translation-l2-weight",
            "0.4",
            "--time-varying-rotation-l2-weight",
            "0.5",
            "--time-varying-translation-smooth-l2-weight",
            "0.6",
            "--time-varying-rotation-smooth-l2-weight",
            "0.7",
            "--time-varying-translation-mean-l2-weight",
            "0.8",
            "--time-varying-rotation-mean-l2-weight",
            "0.9",
            "--time-varying-frame-common-translation-l2-weight",
            "1.1",
            "--time-varying-frame-common-rotation-l2-weight",
            "1.2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config_path"] == tmp_path / "config.yaml"
    assert captured["match_cache_root"] == [tmp_path / "data_cache" / "matches" / "cache"]
    assert captured["output"] == tmp_path / "output"
    assert captured["cameras"] == ["camera_a", "camera_b"]
    assert captured["intra_offsets"] == [5]
    assert captured["inter_pairs"] == ["camera_a__camera_b"]
    assert captured["max_observations"] == 123
    assert captured["pair_sample_max_matches"] == 30
    assert captured["pair_sample_seed"] == 99
    assert captured["pair_sample_strategy"] == "spatial_grid_random"
    assert captured["pair_sample_grid_cols"] == 3
    assert captured["pair_sample_grid_rows"] == 3
    assert captured["max_nfev"] == 9
    assert captured["evaluation_max_observations"] == 77
    assert captured["torch_device"] == "cpu"
    assert captured["torch_learning_rate"] == 0.02
    assert captured["torch_lr_schedule"] == "step"
    assert captured["torch_lr_step_size"] == 4
    assert captured["torch_lr_decay"] == 0.25
    assert captured["torch_lr_min_factor"] == 0.125
    assert captured["adaptive_weight_mode"] == "group_static"
    assert captured["adaptive_weight_alpha"] == 0.6
    assert captured["adaptive_weight_min"] == 0.4
    assert captured["adaptive_weight_max"] == 2.5
    assert captured["adaptive_weight_long_tail_ratio_guard"] == 7.0
    assert captured["adaptive_weight_min_group_count"] == 123
    assert captured["intra_weight"] == 1.0
    assert captured["inter_weight"] == 3.0
    assert captured["optimize_time_varying_disturbance"] is True
    assert captured["time_varying_disturbance_dof"] == 6
    assert captured["optimize_body_time_varying_disturbance"] is True
    assert captured["body_time_varying_disturbance_dof"] == 6
    assert captured["time_varying_rotation_bound_deg"] == 1.5
    assert captured["time_varying_translation_bound_m"] == 0.05
    assert captured["fixed_translation_l2_weight"] == pytest.approx(0.2)
    assert captured["fixed_rotation_l2_weight"] == pytest.approx(0.3)
    assert captured["time_varying_translation_l2_weight"] == pytest.approx(0.4)
    assert captured["time_varying_rotation_l2_weight"] == pytest.approx(0.5)
    assert captured["time_varying_translation_smooth_l2_weight"] == pytest.approx(0.6)
    assert captured["time_varying_rotation_smooth_l2_weight"] == pytest.approx(0.7)
    assert captured["time_varying_translation_mean_l2_weight"] == pytest.approx(0.8)
    assert captured["time_varying_rotation_mean_l2_weight"] == pytest.approx(0.9)
    assert captured["time_varying_frame_common_translation_l2_weight"] == pytest.approx(1.1)
    assert captured["time_varying_frame_common_rotation_l2_weight"] == pytest.approx(1.2)
    assert captured["absorb_camera_common_mode"] is True
    assert captured["camera_common_mode_min_cameras"] == 3
    assert captured["camera_common_mode_min_median_cos"] == pytest.approx(0.5)
    assert captured["camera_common_mode_min_positive_projection_fraction"] == pytest.approx(0.8)
    assert captured["absorb_camera_long_term_mode"] is True
    assert captured["camera_long_term_mode_min_frames"] == 30
    assert captured["camera_long_term_mode_min_median_cos"] == pytest.approx(0.5)
    assert captured["camera_long_term_mode_min_positive_projection_fraction"] == pytest.approx(0.8)
    assert captured["camera_long_term_mode_min_mean_to_rms"] == pytest.approx(0.35)
    assert "translation_prior_sigma_m" not in captured
    assert "rotation_prior_sigma_deg" not in captured
    assert "time_varying_prior_sigma_deg" not in captured
    assert "time_varying_translation_prior_sigma_m" not in captured
    assert "time_varying_smooth_sigma_deg" not in captured
    assert "time_varying_translation_smooth_sigma_m" not in captured


def test_optimize_multicamera_extrinsics_cli_defaults_to_full_optimization_sample(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return tmp_path / "joint"

    _write_minimal_cli_config(
        tmp_path / "config.yaml",
        optimization={
            "pair_sample_strategy": "spatial_grid_random",
            "pair_sample_grid_cols": 4,
            "pair_sample_grid_rows": 2,
        },
    )
    monkeypatch.setattr(
        "dongfeng_registration.cli.optimize_multi_camera_extrinsics_from_match_cache",
        fake_optimize,
    )

    result = runner.invoke(
        app,
        [
            "optimize-multicamera-extrinsics",
            "--config",
            str(tmp_path / "config.yaml"),
            "--match-cache-root",
            str(tmp_path / "data_cache" / "matches" / "cache"),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cameras"] == ["camera_a", "camera_b"]
    assert captured["inter_relations"] is None
    assert captured["max_observations"] is None
    assert captured["pair_sample_seed"] == 20260704
    assert captured["pair_sample_strategy"] == "spatial_grid_random"
    assert captured["pair_sample_grid_cols"] == 4
    assert captured["pair_sample_grid_rows"] == 2
    assert captured["max_nfev"] == 1500
    assert captured["torch_learning_rate"] == 0.003
    assert captured["torch_lr_schedule"] == "step"
    assert captured["torch_lr_step_size"] == 100
    assert captured["torch_lr_decay"] == 0.5
    assert captured["torch_lr_min_factor"] == 0.1
    assert captured["torch_dtype"] == "float32"
    assert captured["torch_early_stopping_patience"] == 100
    assert captured["torch_early_stopping_min_delta"] == pytest.approx(1e-3)
    assert captured["torch_early_stopping_min_steps"] == 500
    assert captured["adaptive_weight_mode"] == "none"
    assert captured["adaptive_weight_max"] == 2.0
    assert captured["translation_bound_m"] == pytest.approx(1.0)
    assert captured["rotation_bound_deg"] == pytest.approx(10.0)
    assert captured["fixed_translation_l2_weight"] == pytest.approx(50.0)
    assert captured["fixed_rotation_l2_weight"] == pytest.approx(300.0)
    assert captured["time_varying_disturbance_dof"] == 6
    assert captured["time_varying_rotation_bound_deg"] == 15.0
    assert captured["time_varying_translation_bound_m"] == pytest.approx(0.15)
    assert captured["time_varying_translation_l2_weight"] == pytest.approx(200.0)
    assert captured["time_varying_rotation_l2_weight"] == pytest.approx(5.0)
    assert captured["time_varying_translation_smooth_l2_weight"] == pytest.approx(20.0)
    assert captured["time_varying_rotation_smooth_l2_weight"] == pytest.approx(0.5)
    assert captured["time_varying_translation_mean_l2_weight"] == pytest.approx(5000.0)
    assert captured["time_varying_rotation_mean_l2_weight"] == pytest.approx(2500.0)
    assert captured["time_varying_frame_common_translation_l2_weight"] == pytest.approx(5000.0)
    assert captured["time_varying_frame_common_rotation_l2_weight"] == pytest.approx(2500.0)
    assert captured["absorb_camera_common_mode"] is True
    assert captured["camera_common_mode_min_cameras"] == 3
    assert captured["camera_common_mode_min_median_cos"] == pytest.approx(0.5)
    assert captured["camera_common_mode_min_positive_projection_fraction"] == pytest.approx(0.8)
    assert captured["absorb_camera_long_term_mode"] is True
    assert captured["camera_long_term_mode_min_frames"] == 30
    assert captured["camera_long_term_mode_min_median_cos"] == pytest.approx(0.5)
    assert captured["camera_long_term_mode_min_positive_projection_fraction"] == pytest.approx(0.8)
    assert captured["camera_long_term_mode_min_mean_to_rms"] == pytest.approx(0.35)
    assert "translation_prior_sigma_m" not in captured
    assert "rotation_prior_sigma_deg" not in captured
    assert "time_varying_prior_sigma_deg" not in captured
    assert "time_varying_translation_prior_sigma_m" not in captured
    assert "time_varying_smooth_sigma_deg" not in captured
    assert "time_varying_translation_smooth_sigma_m" not in captured


def test_optimize_multicamera_extrinsics_cli_reads_l2_weights_from_config(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return tmp_path / "joint"

    _write_minimal_cli_config(
        tmp_path / "config.yaml",
        optimization={
            "max_nfev": 5000,
            "torch_learning_rate": 0.001,
            "torch_lr_schedule": "constant",
            "torch_lr_step_size": 200,
            "torch_lr_decay": 0.25,
            "torch_lr_min_factor": 0.2,
            "torch_early_stopping_min_delta": 1e-4,
            "torch_early_stopping_min_steps": 1000,
            "inter_weight": 2.0,
            "fixed_translation_l2_weight": 11.0,
            "fixed_rotation_l2_weight": 12.0,
            "time_varying_translation_l2_weight": 13.0,
            "time_varying_rotation_l2_weight": 14.0,
            "time_varying_translation_smooth_l2_weight": 15.0,
            "time_varying_rotation_smooth_l2_weight": 16.0,
            "time_varying_translation_mean_l2_weight": 17.0,
            "time_varying_rotation_mean_l2_weight": 18.0,
            "time_varying_frame_common_translation_l2_weight": 19.0,
            "time_varying_frame_common_rotation_l2_weight": 20.0,
            "mode_projection_enabled": True,
            "mode_projection_interval_steps": 25,
            "mode_projection_absorb_frame_common": False,
            "mode_projection_absorb_camera_long_term": True,
            "mode_projection_reset_optimizer_state": False,
            "absorb_camera_common_mode": False,
            "camera_common_mode_min_cameras": 4,
            "camera_common_mode_min_median_cos": 0.75,
            "camera_common_mode_min_positive_projection_fraction": 0.9,
            "absorb_camera_long_term_mode": False,
            "camera_long_term_mode_min_frames": 40,
            "camera_long_term_mode_min_median_cos": 0.7,
            "camera_long_term_mode_min_positive_projection_fraction": 0.85,
            "camera_long_term_mode_min_mean_to_rms": 0.45,
        },
    )
    monkeypatch.setattr(
        "dongfeng_registration.cli.optimize_multi_camera_extrinsics_from_match_cache",
        fake_optimize,
    )

    result = runner.invoke(
        app,
        [
            "optimize-multicamera-extrinsics",
            "--config",
            str(tmp_path / "config.yaml"),
            "--match-cache-root",
            str(tmp_path / "data_cache" / "matches" / "cache"),
            "--output",
            str(tmp_path / "output"),
            "--camera",
            "camera_a",
            "--camera",
            "camera_b",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["max_nfev"] == 5000
    assert captured["torch_learning_rate"] == pytest.approx(0.001)
    assert captured["torch_lr_schedule"] == "constant"
    assert captured["torch_lr_step_size"] == 200
    assert captured["torch_lr_decay"] == pytest.approx(0.25)
    assert captured["torch_lr_min_factor"] == pytest.approx(0.2)
    assert captured["torch_early_stopping_min_delta"] == pytest.approx(1e-4)
    assert captured["torch_early_stopping_min_steps"] == 1000
    assert captured["inter_weight"] == pytest.approx(2.0)
    assert captured["fixed_translation_l2_weight"] == pytest.approx(11.0)
    assert captured["fixed_rotation_l2_weight"] == pytest.approx(12.0)
    assert captured["time_varying_translation_l2_weight"] == pytest.approx(13.0)
    assert captured["time_varying_rotation_l2_weight"] == pytest.approx(14.0)
    assert captured["time_varying_translation_smooth_l2_weight"] == pytest.approx(15.0)
    assert captured["time_varying_rotation_smooth_l2_weight"] == pytest.approx(16.0)
    assert captured["time_varying_translation_mean_l2_weight"] == pytest.approx(17.0)
    assert captured["time_varying_rotation_mean_l2_weight"] == pytest.approx(18.0)
    assert captured["time_varying_frame_common_translation_l2_weight"] == pytest.approx(19.0)
    assert captured["time_varying_frame_common_rotation_l2_weight"] == pytest.approx(20.0)
    assert captured["mode_projection_enabled"] is True
    assert captured["mode_projection_interval_steps"] == 25
    assert captured["mode_projection_absorb_frame_common"] is False
    assert captured["mode_projection_absorb_camera_long_term"] is True
    assert captured["mode_projection_reset_optimizer_state"] is False
    assert captured["absorb_camera_common_mode"] is False
    assert captured["camera_common_mode_min_cameras"] == 4
    assert captured["camera_common_mode_min_median_cos"] == pytest.approx(0.75)
    assert captured["camera_common_mode_min_positive_projection_fraction"] == pytest.approx(0.9)
    assert captured["absorb_camera_long_term_mode"] is False
    assert captured["camera_long_term_mode_min_frames"] == 40
    assert captured["camera_long_term_mode_min_median_cos"] == pytest.approx(0.7)
    assert captured["camera_long_term_mode_min_positive_projection_fraction"] == pytest.approx(0.85)
    assert captured["camera_long_term_mode_min_mean_to_rms"] == pytest.approx(0.45)


def test_optimize_multicamera_extrinsics_cli_reads_split_body_constraints_from_config(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return tmp_path / "joint"

    _write_minimal_cli_config(
        tmp_path / "config.yaml",
        optimization={
            "translation_bound_m": 0.8,
            "rotation_bound_deg": 8.0,
            "fixed_translation_l2_weight": 21.0,
            "fixed_rotation_l2_weight": 22.0,
            "optimize_time_varying_disturbance": True,
            "time_varying_disturbance_dof": 6,
            "time_varying_rotation_bound_deg": 9.0,
            "time_varying_translation_bound_m": 0.09,
            "time_varying_translation_l2_weight": 23.0,
            "time_varying_rotation_l2_weight": 24.0,
            "time_varying_translation_smooth_l2_weight": 25.0,
            "time_varying_rotation_smooth_l2_weight": 26.0,
            "time_varying_translation_mean_l2_weight": 27.0,
            "time_varying_rotation_mean_l2_weight": 28.0,
            "time_varying_frame_common_translation_l2_weight": 29.0,
            "time_varying_frame_common_rotation_l2_weight": 30.0,
            "optimize_body_time_varying_disturbance": True,
            "body_time_varying_disturbance_dof": 6,
            "body_time_varying_rotation_bound_deg": 4.0,
            "body_time_varying_translation_bound_m": 0.04,
            "body_time_varying_translation_l2_weight": 31.0,
            "body_time_varying_rotation_l2_weight": 32.0,
            "body_time_varying_translation_smooth_l2_weight": 33.0,
            "body_time_varying_rotation_smooth_l2_weight": 34.0,
        },
    )
    monkeypatch.setattr(
        "dongfeng_registration.cli.optimize_multi_camera_extrinsics_from_match_cache",
        fake_optimize,
    )

    result = runner.invoke(
        app,
        [
            "optimize-multicamera-extrinsics",
            "--config",
            str(tmp_path / "config.yaml"),
            "--match-cache-root",
            str(tmp_path / "data_cache" / "matches" / "cache"),
            "--output",
            str(tmp_path / "output"),
            "--camera",
            "camera_a",
            "--camera",
            "camera_b",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["translation_bound_m"] == pytest.approx(0.8)
    assert captured["rotation_bound_deg"] == pytest.approx(8.0)
    assert captured["fixed_translation_l2_weight"] == pytest.approx(21.0)
    assert captured["fixed_rotation_l2_weight"] == pytest.approx(22.0)
    assert captured["optimize_time_varying_disturbance"] is True
    assert captured["time_varying_disturbance_dof"] == 6
    assert captured["time_varying_rotation_bound_deg"] == pytest.approx(9.0)
    assert captured["time_varying_translation_bound_m"] == pytest.approx(0.09)
    assert captured["time_varying_translation_l2_weight"] == pytest.approx(23.0)
    assert captured["time_varying_rotation_l2_weight"] == pytest.approx(24.0)
    assert captured["time_varying_translation_smooth_l2_weight"] == pytest.approx(25.0)
    assert captured["time_varying_rotation_smooth_l2_weight"] == pytest.approx(26.0)
    assert captured["time_varying_translation_mean_l2_weight"] == pytest.approx(27.0)
    assert captured["time_varying_rotation_mean_l2_weight"] == pytest.approx(28.0)
    assert captured["time_varying_frame_common_translation_l2_weight"] == pytest.approx(29.0)
    assert captured["time_varying_frame_common_rotation_l2_weight"] == pytest.approx(30.0)
    assert captured["optimize_body_time_varying_disturbance"] is True
    assert captured["body_time_varying_disturbance_dof"] == 6
    assert captured["body_time_varying_rotation_bound_deg"] == pytest.approx(4.0)
    assert captured["body_time_varying_translation_bound_m"] == pytest.approx(0.04)
    assert captured["body_time_varying_translation_l2_weight"] == pytest.approx(31.0)
    assert captured["body_time_varying_rotation_l2_weight"] == pytest.approx(32.0)
    assert captured["body_time_varying_translation_smooth_l2_weight"] == pytest.approx(33.0)
    assert captured["body_time_varying_rotation_smooth_l2_weight"] == pytest.approx(34.0)


def test_optimize_multicamera_alternating_cli_passes_arguments(monkeypatch, tmp_path):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return tmp_path / "alternating"

    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "dongfeng_registration.cli.optimize_multi_camera_alternating_from_match_cache",
        fake_optimize,
    )

    result = runner.invoke(
        app,
        [
            "optimize-multicamera-alternating",
            "--config",
            str(tmp_path / "config.yaml"),
            "--match-cache-root",
            str(tmp_path / "data_cache" / "matches" / "cache"),
            "--output",
            str(tmp_path / "output"),
            "--camera",
            "camera_a",
            "--camera",
            "camera_b",
            "--intra-offset",
            "5",
            "--inter-pair",
            "camera_a__camera_b",
            "--block-rounds",
            "1",
            "--block-max-nfev",
            "7",
            "--global-max-nfev",
            "11",
            "--block-inter-weight",
            "0.25",
            "--no-global-refine",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cameras"] == ["camera_a", "camera_b"]
    assert captured["intra_offsets"] == [5]
    assert captured["inter_pairs"] == ["camera_a__camera_b"]
    assert captured["block_rounds"] == 1
    assert captured["block_max_nfev"] == 7
    assert captured["global_max_nfev"] == 11
    assert captured["block_inter_weight"] == 0.25
    assert captured["run_global_refine"] is False
