from pathlib import Path

import pytest

from dongfeng_registration.config import load_config
from dongfeng_registration.delivery_pipeline import build_delivery_stages
from dongfeng_registration import staged_pipeline


ROOT = Path(__file__).resolve().parents[1]


def _stages(profile: str, name: str):
    config = ROOT / "configs" / profile
    return [stage for stage in build_delivery_stages(config) if stage.name == name]


def _option_values(command: tuple[str, ...], option: str) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command[:-1]) if value == option]


def test_t3_full37_profile_excludes_degenerate_front_same_index_relation():
    persistent = _stages("t3_windows_full37.yaml", "persistent-match")
    optimize = _stages("t3_windows_full37.yaml", "optimize")[0].command
    relations = _option_values(optimize, "--inter-pair-relation")

    assert len(persistent) == 1
    assert "match-delivery" in persistent[0].command
    assert len(relations) == 15
    assert (
        "center_camera_fov120__center_camera_fov30=same_index" not in relations
    )
    assert (
        "center_camera_fov120__center_camera_fov30=temporal_overlap_12m"
        in relations
    )


def test_t3_server_profile_excludes_degenerate_front_same_index_relation():
    optimize = _stages("t3.yaml", "optimize")[0].command
    relations = _option_values(optimize, "--inter-pair-relation")

    assert len(relations) == 15
    assert (
        "center_camera_fov120__center_camera_fov30=same_index" not in relations
    )
    assert (
        "center_camera_fov120__center_camera_fov30=temporal_overlap_12m"
        in relations
    )


def test_t3_subset21_profile_has_exact_group_breakdown():
    persistent = _stages("t3_windows_subset21.yaml", "persistent-match")
    optimize = _stages("t3_windows_subset21.yaml", "optimize")[0].command
    relations = _option_values(optimize, "--inter-pair-relation")

    assert len(persistent) == 1
    assert "match-delivery" in persistent[0].command
    assert len(relations) == 7
    assert (
        "center_camera_fov120__center_camera_fov30=same_index" not in relations
    )
    assert (
        "center_camera_fov120__center_camera_fov30=temporal_overlap_12m"
        in relations
    )


@pytest.mark.parametrize(
    "profile",
    ("t3.yaml", "t3_windows_full37.yaml", "t3_windows_subset21.yaml"),
)
def test_t3_front_fov_temporal_relation_allows_parallel_views(profile):
    config = load_config(ROOT / "configs" / profile)
    thresholds = config.assumptions[
        "inter_camera_temporal_overlap_max_view_dot_by_pair"
    ]

    assert thresholds["center_camera_fov120__center_camera_fov30"] == 1.0


@pytest.mark.parametrize(
    "profile",
    ("t3.yaml", "t3_windows_full37.yaml", "t3_windows_subset21.yaml"),
)
def test_t3_profiles_cache_all_images_in_cpu_memory(profile):
    config = load_config(ROOT / "configs" / profile)

    assert config.image_cache.max_items == 5000


@pytest.mark.parametrize(
    "profile",
    (
        "l29.yaml",
        "l29_windows.yaml",
        "m18.yaml",
        "m18_windows.yaml",
        "t3.yaml",
        "t3_windows_full37.yaml",
        "t3_windows_subset21.yaml",
    ),
)
def test_delivery_profiles_use_one_percent_compact_projection_qc(profile):
    stages = build_delivery_stages(ROOT / "configs" / profile)
    optimize = next(stage.command for stage in stages if stage.name == "optimize")
    projection_qc = next(
        stage.command for stage in stages if stage.name == "projection-qc"
    )

    assert _option_values(optimize, "--projection-qc-sample-ratio") == ["0.01"]
    assert _option_values(projection_qc, "--sample-ratio") == ["1.0"]
    assert _option_values(projection_qc, "--tile-width") == ["1280"]
    assert _option_values(projection_qc, "--tile-height") == ["720"]
    assert _option_values(projection_qc, "--workers") == ["4"]
    assert _option_values(projection_qc, "--jpeg-quality") == ["90"]


@pytest.mark.parametrize(
    "profile",
    (
        "l29.yaml",
        "l29_windows.yaml",
        "m18.yaml",
        "m18_windows.yaml",
        "t3.yaml",
        "t3_windows_full37.yaml",
        "t3_windows_subset21.yaml",
    ),
)
def test_delivery_profiles_use_one_percent_semantic_previews(profile):
    config = load_config(ROOT / "configs" / profile)

    assert config.semantic_segmentation.preview_ratio == pytest.approx(0.01)


def test_delivery_profiles_use_one_persistent_matching_stage():
    stages = build_delivery_stages(ROOT / "configs" / "t3_windows_subset21.yaml")

    persistent = [stage for stage in stages if stage.name == "persistent-match"]
    assert len(persistent) == 1
    assert not [stage for stage in stages if stage.name in {"intra-match", "inter-match"}]
    assert "match-delivery" in persistent[0].command


def test_persistent_matching_runs_intra_then_inter_in_one_call(monkeypatch, tmp_path):
    calls = []

    def fake_intra(**kwargs):
        calls.append(("intra", kwargs["camera"]))
        return tmp_path / "intra"

    def fake_inter(**kwargs):
        calls.append(("inter", kwargs["cameras"]))
        return tmp_path / "inter"

    monkeypatch.setattr(staged_pipeline, "match_dataset_stage", fake_intra)
    monkeypatch.setattr(staged_pipeline, "match_multicamera_dataset_stage", fake_inter)

    result = staged_pipeline.run_persistent_matching_stage(
        config_path=tmp_path / "delivery.yaml",
        dataset_root=tmp_path / "canonical",
        cameras=("camera_a", "camera_b"),
        semantic_cache_root=tmp_path / "semantic",
        match_cache_root=tmp_path / "matches",
        intra_baselines_m=(2.0, 6.0),
        inter_pairs=("camera_a__camera_b",),
        inter_relations=("same_index",),
    )

    assert result == tmp_path / "matches"
    assert calls == [
        ("intra", "camera_a"),
        ("intra", "camera_b"),
        ("inter", ("camera_a", "camera_b")),
    ]


def test_match_delivery_does_not_infer_empty_preflight_cache_from_output_root(
    monkeypatch, tmp_path
):
    from dongfeng_registration import cli
    from types import SimpleNamespace

    delivery_path = tmp_path / "delivery.yaml"
    delivery_path.write_text("placeholder", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        cli,
        "load_delivery_config",
        lambda _path: {
            "base_config": "base.yaml",
            "canonical_root": "canonical",
            "semantic_cache_root": "semantic",
            "match_cache_root": "matches",
            "run_output_root": "outputs",
            "cameras": ["camera_a"],
        },
    )

    def fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "matches"

    monkeypatch.setattr(cli, "run_persistent_matching_stage", fake_run)
    monkeypatch.setattr(
        cli,
        "run_preflight_from_delivery_config",
        lambda _path: {"overall_status": "PASS", "fail_action": "stop"},
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: SimpleNamespace(
            semantic_segmentation=SimpleNamespace(
                processor_image_size=16,
                confidence=0.25,
                dilation_pixels=1,
                batch_size=2,
                write_workers=1,
                preview_ratio=0.1,
            ),
            image_cache=SimpleNamespace(
                enabled=True,
                max_long_side=1600,
                max_items=23,
            ),
        ),
    )
    monkeypatch.setattr(cli, "segment_dataset_stage", lambda **_kwargs: tmp_path)
    cli.match_delivery(config=delivery_path)

    assert captured["preflight_cache_root"] is None


def test_match_delivery_runs_preflight_before_persistent_matching(
    monkeypatch, tmp_path
):
    from dongfeng_registration import cli
    from types import SimpleNamespace

    delivery_path = tmp_path / "delivery.yaml"
    delivery_path.write_text("placeholder", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        cli,
        "load_delivery_config",
        lambda _path: {
            "base_config": "base.yaml",
            "canonical_root": "canonical",
            "semantic_cache_root": "semantic",
            "match_cache_root": "matches",
            "cameras": ["camera_a"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_preflight_from_delivery_config",
        lambda _path: calls.append("preflight")
        or {"overall_status": "PASS", "fail_action": "stop"},
    )
    monkeypatch.setattr(
        cli,
        "run_persistent_matching_stage",
        lambda **_kwargs: calls.append("matching") or tmp_path / "matches",
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: SimpleNamespace(
            semantic_segmentation=SimpleNamespace(
                processor_image_size=16,
                confidence=0.25,
                dilation_pixels=1,
                batch_size=2,
                write_workers=1,
                preview_ratio=0.1,
            ),
            image_cache=SimpleNamespace(
                enabled=True,
                max_long_side=1600,
                max_items=23,
            ),
        ),
    )
    monkeypatch.setattr(
        cli,
        "segment_dataset_stage",
        lambda **kwargs: calls.append(
            ("semantic", kwargs["mask_config"].image_cache_max_items)
        )
        or tmp_path / "semantic",
    )

    cli.match_delivery(config=delivery_path)

    assert calls == ["preflight", ("semantic", 23), "matching"]

