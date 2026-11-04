from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from dongfeng_registration import colmap_reconstruction as colmap_module
from dongfeng_registration.colmap_reconstruction import (
    _build_patch_match_command,
    _build_bundle_adjustment_command,
    _run_dense_backend,
    _validate_reconstruction_executables,
    _rotation_to_qvec,
    _read_delivery_pose_records,
    _relative_image_name,
    _reset_output_dirs,
    build_feature_extractor_command,
    resize_masks_for_dense_workspace,
    build_static_valid_masks,
    load_colmap_config,
    _write_patch_match_sources,
)


def test_colmap_reconstruction_import_does_not_require_opencv() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins

real_import = builtins.__import__

def reject_opencv(name, *args, **kwargs):
    if name == "cv2":
        raise ModuleNotFoundError("cv2 intentionally unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_opencv
import dongfeng_registration.colmap_reconstruction
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_relative_image_name_rehomes_stale_absolute_delivery_path(tmp_path: Path) -> None:
    image_root = tmp_path / "canonical/images"
    image_path = image_root / "camera_a/100.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    record = {
        "camera": "camera_a",
        "image_filename": "100.jpg",
        "image_path": r"D:\old\canonical\images\camera_a\100.jpg",
    }

    assert _relative_image_name(record, image_root) == "camera_a/100.jpg"
    assert record["image_path"] == image_path


def test_count_colmap_points3d_reads_binary_header(tmp_path: Path) -> None:
    sparse_root = tmp_path / "sparse"
    sparse_root.mkdir()
    (sparse_root / "points3D.bin").write_bytes((42).to_bytes(8, "little"))

    assert colmap_module._count_colmap_points3d(sparse_root) == 42


def test_patch_match_command_uses_colmap_4_numeric_boolean_options(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: deliverables",
                "output_root: outputs",
                "colmap_executable: colmap",
                "patch_match_filter: true",
                "patch_match_write_consistency_graph: true",
            ]
        ),
        encoding="utf-8",
    )
    config = load_colmap_config(config_path)

    command = _build_patch_match_command(config, Path("dense"), geometric=True)

    assert "--PatchMatchStereo.geom_consistency=1" in command
    assert "--PatchMatchStereo.filter=1" in command
    assert "--PatchMatchStereo.write_consistency_graph=1" in command


def test_resize_masks_for_dense_workspace_matches_undistorted_image_size(tmp_path: Path) -> None:
    source_masks = tmp_path / "source_masks"
    dense_images = tmp_path / "dense" / "images"
    output_masks = tmp_path / "dense_masks"
    (source_masks / "camera_a").mkdir(parents=True)
    (dense_images / "camera_a").mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).resize(
        (8, 4), Image.Resampling.NEAREST
    ).save(source_masks / "camera_a" / "100.png")
    Image.new("RGB", (4, 2)).save(dense_images / "camera_a" / "100.jpg")

    result = resize_masks_for_dense_workspace(
        source_mask_root=source_masks,
        dense_image_root=dense_images,
        output_mask_root=output_masks,
    )

    output = output_masks / "camera_a" / "100.png"
    assert result == {"masks": 1, "resized": 1}
    assert Image.open(output).size == (4, 2)
    assert set(np.unique(np.asarray(Image.open(output)))) == {0, 255}


def test_build_static_valid_masks_writes_image_relative_masks(tmp_path: Path) -> None:
    semantic_root = tmp_path / "semantic"
    camera_root = semantic_root / "front_camera_fov30"
    class_masks = camera_root / "class_masks"
    metadata_dir = camera_root / "metadata"
    class_masks.mkdir(parents=True)
    metadata_dir.mkdir()
    class_mask_path = class_masks / "100.png"
    Image.fromarray(np.asarray([[1, 2], [3, 2]], dtype=np.uint16)).save(class_mask_path)
    (metadata_dir / "100.json").write_text(
        json.dumps(
            {
                "source": {"path": "images/front_camera_fov30/100.jpg"},
                "class_mask_path": str(class_mask_path),
                "class_id_to_name": {"1": "road", "2": "car", "3": "sky"},
            }
        ),
        encoding="utf-8",
    )

    result = build_static_valid_masks(
        semantic_root=semantic_root,
        output_root=tmp_path / "static_masks",
        dynamic_classes={"car"},
        required_relative_names={"front_camera_fov30/100.jpg"},
    )

    output_path = tmp_path / "static_masks" / "front_camera_fov30" / "100.png"
    assert result == {"masks": 1, "dynamic_pixels": 2}
    assert np.array_equal(
        np.asarray(Image.open(output_path)),
        np.asarray([[255, 0], [255, 0]], dtype=np.uint8),
    )


def test_build_static_valid_masks_rehomes_stale_windows_metadata_paths(
    tmp_path: Path,
) -> None:
    camera_root = tmp_path / "semantic/camera_a"
    class_masks = camera_root / "class_masks"
    metadata_dir = camera_root / "metadata"
    class_masks.mkdir(parents=True)
    metadata_dir.mkdir()
    Image.fromarray(np.asarray([[1, 2], [2, 1]], dtype=np.uint16)).save(
        class_masks / "1000.png"
    )
    (metadata_dir / "1000.json").write_text(
        json.dumps(
            {
                "source": {"path": r"D:\old\images\camera_a\100.jpg"},
                "class_mask_path": r"D:\old\cache\camera_a\class_masks\1000.png",
                "class_id_to_name": {"1": "road", "2": "car"},
            }
        ),
        encoding="utf-8",
    )

    result = build_static_valid_masks(
        semantic_root=tmp_path / "semantic",
        output_root=tmp_path / "static_masks",
        dynamic_classes={"car"},
        required_relative_names={"camera_a/100.jpg"},
    )

    assert result["masks"] == 1
    assert (tmp_path / "static_masks/camera_a/100.png").is_file()


def test_build_static_valid_masks_skips_unselected_images(tmp_path: Path) -> None:
    semantic_root = tmp_path / "semantic"
    camera_root = semantic_root / "camera_a"
    class_masks = camera_root / "class_masks"
    metadata_dir = camera_root / "metadata"
    class_masks.mkdir(parents=True)
    metadata_dir.mkdir()
    for timestamp in ("100", "200"):
        class_mask_path = class_masks / f"{timestamp}.png"
        Image.fromarray(np.zeros((1, 1), dtype=np.uint16)).save(class_mask_path)
        (metadata_dir / f"{timestamp}.json").write_text(
            json.dumps(
                {
                    "source": {"path": f"images/camera_a/{timestamp}.jpg"},
                    "class_mask_path": str(class_mask_path),
                    "class_id_to_name": {"0": "road"},
                }
            ),
            encoding="utf-8",
        )

    result = build_static_valid_masks(
        semantic_root=semantic_root,
        output_root=tmp_path / "static_masks",
        required_relative_names={"camera_a/200.jpg"},
    )

    assert result["masks"] == 1
    assert not (tmp_path / "static_masks/camera_a/100.png").exists()
    assert (tmp_path / "static_masks/camera_a/200.png").exists()


def test_feature_extractor_command_always_uses_static_mask_root() -> None:
    command = build_feature_extractor_command(
        colmap_executable=Path("colmap"),
        database_path=Path("matching/database.db"),
        image_root=Path("images"),
        image_list_path=Path("matching/front.txt"),
        camera_spec={"model": "PINHOLE", "params": [1000.0, 1001.0, 960.0, 540.0]},
        static_valid_mask_root=Path("matching/static_masks"),
        gpu_index="0",
        max_image_size=2400,
    )

    mask_index = command.index("--ImageReader.mask_path")
    assert Path(command[mask_index + 1]) == Path("matching/static_masks")
    assert command[command.index("--ImageReader.camera_model") + 1] == "PINHOLE"
    assert command[command.index("--ImageReader.camera_params") + 1] == "1000,1001,960,540"
    assert command[command.index("--ImageReader.single_camera") + 1] == "1"


def test_colmap_config_loads_separate_pose_and_dense_mask_classes(tmp_path: Path) -> None:
    config_path = tmp_path / "vehicle.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: deliverables",
                "output_root: outputs",
                "colmap_executable: colmap",
                "colmap_pose_mask_classes: [car, vegetation, sky]",
                "colmap_dense_mask_classes: [car, sky]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.pose_mask_classes == frozenset({"car", "vegetation", "sky"})
    assert config.dense_mask_classes == frozenset({"car", "sky"})


def test_dense_fusion_command_uses_dense_mask_root() -> None:
    from dongfeng_registration.colmap_reconstruction import build_stereo_fusion_command

    command = build_stereo_fusion_command(
        colmap_executable=Path("colmap"),
        dense_root=Path("dense"),
        fused_ply=Path("dense/fused.ply"),
        dense_max_image_size=1280,
        stereo_fusion_use_cache=True,
        stereo_fusion_cache_size_gb=16,
        mask_root=Path("matching/dense_valid_masks"),
    )

    assert Path(command[command.index("--StereoFusion.mask_path") + 1]) == Path(
        "matching/dense_valid_masks"
    )


def test_load_colmap_config_uses_after_delivery_pose_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: deliverables/T3_final",
                "output_root: outputs/T3_colmap",
                "colmap_executable: tools/colmap.exe",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.pose_variant == "after"
    assert config.pose_csv == (
        tmp_path / "deliverables/T3_final/image_extrinsics/image_camera_to_world_after.csv"
    )
    assert config.image_root == tmp_path / "canonical/images"


def test_load_colmap_config_derives_delivery_and_output_from_run_root(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "colmap_executable: tools/colmap.exe",
            ]
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"

    config = load_colmap_config(config_path, run_output_root=run_root)

    assert config.delivery_root == run_root / "deliverables"
    assert config.output_root == run_root / "deliverables/colmap-ba"
    assert config.pose_csv == (
        run_root / "deliverables/image_extrinsics/image_camera_to_world_after.csv"
    )
    assert config.calibration_pkl == (
        run_root / "deliverables/result_interpolation_optimized.pkl"
    )


def test_load_colmap_config_reads_embedded_vehicle_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "m18.yaml"
    config_path.write_text(
        "\n".join(
            [
                "delivery:",
                "  canonical_root: canonical",
                "  semantic_cache_root: data_cache/M18/semantic_masks",
                "  run_output_root: outputs/M18_run",
                "colmap:",
                "  enabled: true",
                "  colmap_executable: tools/colmap.exe",
                "  dense_max_image_size: 1280",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.canonical_root == tmp_path / "canonical"
    assert config.semantic_cache_root == tmp_path / "data_cache/M18/semantic_masks"
    assert config.output_root == tmp_path / "outputs/M18_run/deliverables/colmap-ba"
    assert config.dense_max_image_size == 1280


def test_load_colmap_config_allows_one_off_pose_and_output_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: tools/colmap.exe",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(
        config_path,
        pose_variant="before",
        output_root=tmp_path / "colmap_before",
    )

    assert config.pose_variant == "before"
    assert config.pose_csv == (
        tmp_path / "delivery/image_extrinsics/image_camera_to_world_before.csv"
    )
    assert config.output_root == tmp_path / "colmap_before"


def test_load_colmap_config_reads_bundle_adjustment_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: tools/colmap.exe",
                "run_bundle_adjustment: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.run_bundle_adjustment is False


def test_load_colmap_config_defaults_to_absolute_pose_prior(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.ba_mode == "absolute_pose_prior"


def test_load_colmap_config_defaults_to_colmap_mvs_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.mvs_backend == "colmap"
    assert config.openmvs_gpu_device is None


def test_load_colmap_config_reads_openmvs_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
                "mvs_backend: openmvs",
                "openmvs_interface_colmap_executable: tools/openmvs/InterfaceCOLMAP",
                "openmvs_densify_point_cloud_executable: tools/openmvs/DensifyPointCloud",
                "openmvs_resolution_level: 2",
                "openmvs_max_resolution: 1600",
                "openmvs_min_resolution: 800",
                "openmvs_number_views: 6",
                "openmvs_number_views_fuse: 4",
                "openmvs_gpu_device: '1'",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)

    assert config.mvs_backend == "openmvs"
    assert config.openmvs_interface_colmap_executable == (
        tmp_path / "tools/openmvs/InterfaceCOLMAP"
    )
    assert config.openmvs_densify_point_cloud_executable == (
        tmp_path / "tools/openmvs/DensifyPointCloud"
    )
    assert config.openmvs_resolution_level == 2
    assert config.openmvs_max_resolution == 1600
    assert config.openmvs_min_resolution == 800
    assert config.openmvs_number_views == 6
    assert config.openmvs_number_views_fuse == 4
    assert config.openmvs_gpu_device == "1"


def test_load_colmap_config_rejects_unknown_mvs_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
                "mvs_backend: unknown",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mvs_backend"):
        load_colmap_config(config_path)


def test_absolute_pose_prior_bundle_adjustment_command_uses_configured_sigmas(tmp_path: Path) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
                "ba_mode: absolute_pose_prior",
                "absolute_pose_prior_colmap_executable: tools/colmap_pose_prior/bin/colmap",
                "ba_position_prior_sigma_m: 0.12",
                "ba_rotation_prior_sigma_deg: 0.6",
            ]
        ),
        encoding="utf-8",
    )

    config = load_colmap_config(config_path)
    command = _build_bundle_adjustment_command(
        config,
        input_path=tmp_path / "input",
        output_path=tmp_path / "output",
    )

    assert command[1] == "absolute_pose_prior_bundle_adjuster"
    assert "--pose_csv" in command
    assert command[command.index("--position_prior_sigma_m") + 1] == "0.12"
    assert command[command.index("--rotation_prior_sigma_deg") + 1] == "0.6"


def test_rotation_to_qvec_preserves_non_axis_half_turn() -> None:
    axis = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    rotation = 2.0 * np.outer(axis, axis) - np.eye(3)

    qvec = _rotation_to_qvec(rotation)

    assert np.allclose(np.abs(qvec), np.asarray([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0]))


def test_read_delivery_pose_records_accepts_legacy_per_camera_after_csvs(tmp_path: Path) -> None:
    delivery_root = tmp_path / "delivery"
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: outputs",
                "colmap_executable: colmap.exe",
            ]
        ),
        encoding="utf-8",
    )
    legacy_csv = delivery_root / "image_extrinsics/after/camera_a_image_camera_to_world_after.csv"
    legacy_csv.parent.mkdir(parents=True)
    fields = ["camera", "frame", "timestamp_ns", "image_filename", "image_path"] + [
        f"T_cw_{row}{column}" for row in range(4) for column in range(4)
    ]
    row = {field: "0" for field in fields}
    row.update({"camera": "camera_a", "image_filename": "0.jpg", "image_path": "image.jpg"})
    for index in range(4):
        row[f"T_cw_{index}{index}"] = "1"
    with legacy_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    records = _read_delivery_pose_records(load_colmap_config(config_path))

    assert len(records) == 1
    assert records[0]["camera"] == "camera_a"


def test_reset_output_dirs_creates_empty_colmap_output_directories(tmp_path: Path) -> None:
    first = tmp_path / "sparse_input"
    second = tmp_path / "sparse_triangulated"
    first.mkdir()
    (first / "stale.txt").write_text("stale", encoding="utf-8")

    _reset_output_dirs(first, second)

    assert first.is_dir() and not any(first.iterdir())
    assert second.is_dir() and not any(second.iterdir())


def test_run_dense_backend_dispatches_to_openmvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
                "mvs_backend: openmvs",
            ]
        ),
        encoding="utf-8",
    )
    config = load_colmap_config(config_path)
    captured: dict[str, object] = {}

    def fake_openmvs(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"mvs_backend": "openmvs", "fused_ply": "dense/fused.ply"}

    def fail_colmap(**kwargs: object) -> dict[str, object]:
        raise AssertionError("COLMAP dense backend was called")

    monkeypatch.setattr("dongfeng_registration.colmap_reconstruction.run_openmvs_dense", fake_openmvs)
    monkeypatch.setattr("dongfeng_registration.colmap_reconstruction._run_colmap_dense", fail_colmap)

    result = _run_dense_backend(
        config=config,
        pose_model=tmp_path / "sparse_ba",
        dense_mask_root=tmp_path / "dense_masks",
        database_path=tmp_path / "matching/database.db",
    )

    assert result["mvs_backend"] == "openmvs"
    assert captured["sparse_model"] == tmp_path / "sparse_ba"
    assert captured["source_mask_root"] == tmp_path / "dense_masks"


def test_run_dense_backend_keeps_colmap_as_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                "colmap_executable: colmap",
            ]
        ),
        encoding="utf-8",
    )
    config = load_colmap_config(config_path)

    def fake_colmap(**kwargs: object) -> dict[str, object]:
        return {"mvs_backend": "colmap", "fused_ply": "dense/fused.ply"}

    monkeypatch.setattr("dongfeng_registration.colmap_reconstruction._run_colmap_dense", fake_colmap)
    monkeypatch.setattr(
        "dongfeng_registration.colmap_reconstruction.run_openmvs_dense",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("OpenMVS backend was called")),
    )

    result = _run_dense_backend(
        config=config,
        pose_model=tmp_path / "sparse_ba",
        dense_mask_root=tmp_path / "dense_masks",
        database_path=tmp_path / "matching/database.db",
    )

    assert result["mvs_backend"] == "colmap"


def test_openmvs_executables_are_required_only_for_enabled_openmvs_dense(
    tmp_path: Path,
) -> None:
    colmap = tmp_path / "colmap"
    colmap.write_text("binary", encoding="utf-8")
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                f"colmap_executable: {colmap.as_posix()}",
                "run_bundle_adjustment: false",
                "run_dense: true",
                "mvs_backend: openmvs",
                "openmvs_interface_colmap_executable: missing/InterfaceCOLMAP",
                "openmvs_densify_point_cloud_executable: missing/DensifyPointCloud",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="InterfaceCOLMAP"):
        _validate_reconstruction_executables(load_colmap_config(config_path))


@pytest.mark.parametrize(
    ("mvs_backend", "run_dense"),
    [("colmap", True), ("openmvs", False)],
)
def test_openmvs_executables_are_not_required_for_other_modes(
    tmp_path: Path, mvs_backend: str, run_dense: bool
) -> None:
    colmap = tmp_path / "colmap"
    colmap.write_text("binary", encoding="utf-8")
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "canonical_root: canonical",
                "semantic_cache_root: semantic",
                "delivery_root: delivery",
                "output_root: output",
                f"colmap_executable: {colmap.as_posix()}",
                "run_bundle_adjustment: false",
                f"run_dense: {'true' if run_dense else 'false'}",
                f"mvs_backend: {mvs_backend}",
                "openmvs_interface_colmap_executable: missing/InterfaceCOLMAP",
                "openmvs_densify_point_cloud_executable: missing/DensifyPointCloud",
            ]
        ),
        encoding="utf-8",
    )

    _validate_reconstruction_executables(load_colmap_config(config_path))


def test_write_patch_match_sources_replaces_auto_neighbors(tmp_path: Path) -> None:
    cfg = tmp_path / "patch-match.cfg"
    cfg.write_text("a.jpg\n__auto__, 12\nb.jpg\n__auto__, 12\n", encoding="utf-8")
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\n", encoding="utf-8")

    _write_patch_match_sources(cfg, pairs, maximum=12)

    assert cfg.read_text(encoding="utf-8") == "a.jpg\nb.jpg\n\nb.jpg\na.jpg\n"
