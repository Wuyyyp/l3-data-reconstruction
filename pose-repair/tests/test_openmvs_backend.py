from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from dongfeng_registration import openmvs_backend as openmvs_module
from dongfeng_registration.openmvs_backend import (
    OpenMVSPaths,
    OpenMVSSettings,
    build_densify_point_cloud_command,
    build_interface_colmap_command,
    count_ply_vertices,
    prepare_openmvs_masks,
    run_openmvs_dense,
)


def _settings() -> OpenMVSSettings:
    return OpenMVSSettings(
        interface_colmap_executable=Path("InterfaceCOLMAP"),
        densify_point_cloud_executable=Path("DensifyPointCloud"),
        resolution_level=1,
        max_resolution=1280,
        min_resolution=640,
        number_views=8,
        number_views_fuse=3,
        gpu_device=None,
    )


def _paths() -> OpenMVSPaths:
    root = Path("dense/openmvs")
    return OpenMVSPaths(
        root=root,
        colmap_workspace=root / "colmap_workspace",
        scene_path=root / "scene.mvs",
        dense_scene_path=root / "scene_dense.mvs",
        native_ply_path=root / "scene_dense.ply",
        fused_ply_path=Path("dense/fused.ply"),
    )


def test_interface_colmap_command_uses_undistorted_workspace() -> None:
    paths = _paths()

    command = build_interface_colmap_command(_settings(), paths)

    assert command == [
        "InterfaceCOLMAP",
        "-i",
        str(Path("dense/openmvs/colmap_workspace")),
        "-o",
        str(Path("dense/openmvs/scene.mvs")),
        "--image-folder",
        str(Path("dense/openmvs/colmap_workspace/images")),
    ]


def test_densify_command_uses_image_sidecar_masks_and_configured_limits_on_cpu() -> None:
    command = build_densify_point_cloud_command(_settings(), _paths())

    assert "--mask-path" not in command
    assert command[command.index("--ignore-mask-label") + 1] == "0"
    assert command[command.index("--resolution-level") + 1] == "1"
    assert command[command.index("--max-resolution") + 1] == "1280"
    assert command[command.index("--min-resolution") + 1] == "640"
    assert command[command.index("--number-views") + 1] == "8"
    assert command[command.index("--number-views-fuse") + 1] == "3"
    assert "--cuda-device" not in command


def test_densify_command_maps_configured_gpu_to_openmvs_cuda_device() -> None:
    settings = _settings()
    settings = OpenMVSSettings(
        interface_colmap_executable=settings.interface_colmap_executable,
        densify_point_cloud_executable=settings.densify_point_cloud_executable,
        resolution_level=settings.resolution_level,
        max_resolution=settings.max_resolution,
        min_resolution=settings.min_resolution,
        number_views=settings.number_views,
        number_views_fuse=settings.number_views_fuse,
        gpu_device="0",
    )

    command = build_densify_point_cloud_command(settings, _paths())

    assert command[command.index("--cuda-device") + 1] == "0"
    assert "--gpu-device" not in command


def test_prepare_openmvs_masks_matches_images_and_preserves_binary_values(
    tmp_path: Path,
) -> None:
    source_masks = tmp_path / "source_masks"
    undistorted_images = tmp_path / "workspace/images"
    (source_masks / "camera_a").mkdir(parents=True)
    (undistorted_images / "camera_a").mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(
        source_masks / "camera_a/100.png"
    )
    Image.new("RGB", (4, 2)).save(undistorted_images / "camera_a/100.jpg")

    result = prepare_openmvs_masks(
        source_mask_root=source_masks,
        undistorted_image_root=undistorted_images,
    )

    output = undistorted_images / "camera_a/100.jpg.mask.png"
    assert result == {"masks": 1, "resized": 1}
    assert output.is_file()
    assert Image.open(output).size == (4, 2)
    assert set(np.unique(np.asarray(Image.open(output)))) == {0, 255}


def test_prepare_openmvs_masks_requires_every_undistorted_image(tmp_path: Path) -> None:
    source_masks = tmp_path / "source_masks"
    undistorted_images = tmp_path / "workspace/images"
    (source_masks / "camera_a").mkdir(parents=True)
    (undistorted_images / "camera_a").mkdir(parents=True)
    Image.new("RGB", (4, 2)).save(undistorted_images / "camera_a/100.jpg")

    try:
        prepare_openmvs_masks(
            source_mask_root=source_masks,
            undistorted_image_root=undistorted_images,
        )
    except FileNotFoundError as exc:
        assert "camera_a" in str(exc)
        assert "100" in str(exc)
    else:
        raise AssertionError("missing OpenMVS mask did not fail")


def test_count_ply_vertices_reads_header_without_loading_points(tmp_path: Path) -> None:
    ply_path = tmp_path / "cloud.ply"
    ply_path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 42\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )

    assert count_ply_vertices(ply_path) == 42


def test_publish_openmvs_ply_removes_vertex_view_lists_for_cloudcompare(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene_dense.ply"
    destination = tmp_path / "fused.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property uint8 red\n"
        "property uint8 green\n"
        "property uint8 blue\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property list uint8 uint32 view_indices\n"
        "property list uint8 float32 view_weights\n"
        "end_header\n"
    ).encode("ascii")
    first = struct.pack("<3f3B3f", 1, 2, 3, 10, 20, 30, 0, 0, 1)
    first += struct.pack("<B2I", 2, 4, 7) + struct.pack("<B2f", 2, 0.6, 0.4)
    second = struct.pack("<3f3B3f", 4, 5, 6, 40, 50, 60, 0, 1, 0)
    second += struct.pack("<B3I", 3, 1, 2, 3) + struct.pack("<B3f", 3, 0.5, 0.3, 0.2)
    source.write_bytes(header + first + second)

    openmvs_module.publish_cloudcompare_compatible_ply(source, destination)

    output = destination.read_bytes()
    output_header_end = output.index(b"end_header\n") + len(b"end_header\n")
    output_header = output[:output_header_end].decode("ascii")
    assert "property list" not in output_header
    assert "property float32 nx" in output_header
    assert len(output) - output_header_end == 2 * 27
    assert output[output_header_end : output_header_end + 27] == first[:27]
    assert output[output_header_end + 27 :] == second[:27]


def _write_binary_ply(path: Path, vertex_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {vertex_count}\n".encode("ascii")
        + b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )


def test_run_openmvs_dense_runs_only_bridge_and_openmvs_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_masks = tmp_path / "source_masks"
    (source_masks / "camera_a").mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(
        source_masks / "camera_a/100.png"
    )
    paths = OpenMVSPaths.for_dense_root(tmp_path / "dense")
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is True
        commands.append(command)
        if len(command) > 1 and command[1] == "image_undistorter":
            image_path = paths.colmap_workspace / "images/camera_a/100.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (4, 2)).save(image_path)
        elif Path(command[0]).name == "InterfaceCOLMAP":
            paths.scene_path.write_bytes(b"scene")
        elif Path(command[0]).name == "DensifyPointCloud":
            paths.dense_scene_path.write_bytes(b"dense scene")
            _write_binary_ply(paths.native_ply_path, 42)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dongfeng_registration.openmvs_backend.subprocess.run", fake_run)

    report = run_openmvs_dense(
        settings=_settings(),
        paths=paths,
        colmap_executable=Path("colmap"),
        image_root=tmp_path / "images",
        sparse_model=tmp_path / "sparse_ba",
        source_mask_root=source_masks,
        max_image_size=1280,
    )

    assert [command[1] if command[0] == "colmap" else Path(command[0]).name for command in commands] == [
        "image_undistorter",
        "InterfaceCOLMAP",
        "DensifyPointCloud",
    ]
    assert not any("patch_match_stereo" in command for command in commands)
    assert not any("stereo_fusion" in command for command in commands)
    assert paths.fused_ply_path.read_bytes() == paths.native_ply_path.read_bytes()
    assert report["mvs_backend"] == "openmvs"
    assert report["dense_point_count"] == 42
    assert report["dense_file_size_bytes"] == paths.fused_ply_path.stat().st_size
    assert set(report["dense_stage_seconds"]) == {"undistort", "convert", "densify"}
    assert report["openmvs"]["masks"] == {"masks": 1, "resized": 1}


def test_run_openmvs_dense_fails_when_interface_scene_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_masks = tmp_path / "source_masks"
    (source_masks / "camera_a").mkdir(parents=True)
    Image.new("L", (2, 2), color=255).save(source_masks / "camera_a/100.png")
    paths = OpenMVSPaths.for_dense_root(tmp_path / "dense")

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        if len(command) > 1 and command[1] == "image_undistorter":
            image_path = paths.colmap_workspace / "images/camera_a/100.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image_path)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dongfeng_registration.openmvs_backend.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="scene.mvs"):
        run_openmvs_dense(
            settings=_settings(),
            paths=paths,
            colmap_executable=Path("colmap"),
            image_root=tmp_path / "images",
            sparse_model=tmp_path / "sparse_ba",
            source_mask_root=source_masks,
            max_image_size=1280,
        )


def test_run_openmvs_dense_rejects_zero_vertex_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_masks = tmp_path / "source_masks"
    (source_masks / "camera_a").mkdir(parents=True)
    Image.new("L", (2, 2), color=255).save(source_masks / "camera_a/100.png")
    paths = OpenMVSPaths.for_dense_root(tmp_path / "dense")

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        if len(command) > 1 and command[1] == "image_undistorter":
            image_path = paths.colmap_workspace / "images/camera_a/100.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image_path)
        elif Path(command[0]).name == "InterfaceCOLMAP":
            paths.scene_path.write_bytes(b"scene")
        elif Path(command[0]).name == "DensifyPointCloud":
            paths.dense_scene_path.write_bytes(b"dense scene")
            _write_binary_ply(paths.native_ply_path, 0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dongfeng_registration.openmvs_backend.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="zero vertices"):
        run_openmvs_dense(
            settings=_settings(),
            paths=paths,
            colmap_executable=Path("colmap"),
            image_root=tmp_path / "images",
            sparse_model=tmp_path / "sparse_ba",
            source_mask_root=source_masks,
            max_image_size=1280,
        )
