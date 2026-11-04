"""OpenMVS dense reconstruction command and workspace helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_PLY_SCALAR_BYTES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


@dataclass(frozen=True)
class OpenMVSSettings:
    interface_colmap_executable: Path
    densify_point_cloud_executable: Path
    resolution_level: int
    max_resolution: int
    min_resolution: int
    number_views: int
    number_views_fuse: int
    gpu_device: str | None


@dataclass(frozen=True)
class OpenMVSPaths:
    root: Path
    colmap_workspace: Path
    scene_path: Path
    dense_scene_path: Path
    native_ply_path: Path
    fused_ply_path: Path

    @classmethod
    def for_dense_root(cls, dense_root: Path) -> "OpenMVSPaths":
        dense_root = Path(dense_root)
        root = dense_root / "openmvs"
        return cls(
            root=root,
            colmap_workspace=root / "colmap_workspace",
            scene_path=root / "scene.mvs",
            dense_scene_path=root / "scene_dense.mvs",
            native_ply_path=root / "scene_dense.ply",
            fused_ply_path=dense_root / "fused.ply",
        )


def build_interface_colmap_command(
    settings: OpenMVSSettings,
    paths: OpenMVSPaths,
) -> list[str]:
    """Build the official COLMAP-to-OpenMVS conversion command."""
    return [
        str(settings.interface_colmap_executable),
        "-i",
        str(paths.colmap_workspace),
        "-o",
        str(paths.scene_path),
        "--image-folder",
        str(paths.colmap_workspace / "images"),
    ]


def build_densify_point_cloud_command(
    settings: OpenMVSSettings,
    paths: OpenMVSPaths,
) -> list[str]:
    """Build masked OpenMVS PatchMatch estimation and fusion command."""
    command = [
        str(settings.densify_point_cloud_executable),
        "-i",
        str(paths.scene_path),
        "-o",
        str(paths.dense_scene_path),
        "-w",
        str(paths.root),
        "--ignore-mask-label",
        "0",
        "--resolution-level",
        str(settings.resolution_level),
        "--max-resolution",
        str(settings.max_resolution),
        "--min-resolution",
        str(settings.min_resolution),
        "--number-views",
        str(settings.number_views),
        "--number-views-fuse",
        str(settings.number_views_fuse),
    ]
    if settings.gpu_device is not None:
        command.extend(["--cuda-device", settings.gpu_device])
    return command


def prepare_openmvs_masks(
    *,
    source_mask_root: Path,
    undistorted_image_root: Path,
) -> dict[str, int]:
    """Resize binary masks into OpenMVS sidecars beside each image."""
    source_mask_root = Path(source_mask_root)
    undistorted_image_root = Path(undistorted_image_root)
    masks = 0
    resized = 0
    for image_path in sorted(undistorted_image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(undistorted_image_root)
        source_path = source_mask_root / relative.with_suffix(".png")
        if not source_path.is_file():
            raise FileNotFoundError(f"OpenMVS source mask missing for image {relative}: {source_path}")
        output_path = image_path.with_name(f"{image_path.name}.mask.png")
        with Image.open(image_path) as image, Image.open(source_path) as source_mask:
            mask = source_mask.convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
                resized += 1
            values = np.asarray(mask)
            if not np.all((values == 0) | (values == 255)):
                raise ValueError(f"OpenMVS mask is not binary: {source_path}")
            mask.save(output_path)
        masks += 1
    if masks == 0:
        raise ValueError(f"no undistorted images found below {undistorted_image_root}")
    return {"masks": masks, "resized": resized}


def count_ply_vertices(path: Path) -> int:
    """Read the vertex count from a PLY header without loading point data."""
    path = Path(path)
    vertex_count: int | None = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", maxsplit=1)[1])
            if line == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex element: {path}")
    return vertex_count


def _read_exact(handle: object, size: int, description: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise RuntimeError(f"truncated OpenMVS PLY while reading {description}")
    return data


def publish_cloudcompare_compatible_ply(source: Path, destination: Path) -> None:
    """Publish OpenMVS points without per-vertex list properties.

    OpenMVS appends view indices and weights as list properties on the vertex
    element. CloudCompare classifies an element with an integer list as
    face-like, so the public PLY keeps only the scalar point attributes.
    """
    source = Path(source)
    destination = Path(destination)
    header_lines: list[bytes] = []
    vertex_properties: list[tuple[str, ...]] = []
    vertex_count: int | None = None
    in_vertex = False
    with source.open("rb") as input_handle:
        while True:
            line = input_handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {source}")
            header_lines.append(line)
            fields = tuple(line.decode("ascii", errors="strict").strip().split())
            if fields[:2] == ("format", "binary_little_endian"):
                pass
            elif fields[:2] == ("element", "vertex") and len(fields) == 3:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields and fields[0] == "property":
                vertex_properties.append(fields)
            if fields == ("end_header",):
                break

        list_indexes = [
            index
            for index, fields in enumerate(vertex_properties)
            if len(fields) >= 2 and fields[1] == "list"
        ]
        if not list_indexes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            return
        if vertex_count is None:
            raise ValueError(f"PLY header has no vertex element: {source}")
        if list_indexes != [len(vertex_properties) - 2, len(vertex_properties) - 1]:
            raise ValueError(f"unsupported OpenMVS PLY property layout: {source}")
        expected_lists = [
            ("property", "list", "uint8", "uint32", "view_indices"),
            ("property", "list", "uint8", "float32", "view_weights"),
        ]
        if [vertex_properties[index] for index in list_indexes] != expected_lists:
            raise ValueError(f"unsupported OpenMVS PLY list properties: {source}")

        scalar_properties = vertex_properties[: list_indexes[0]]
        try:
            scalar_record_bytes = sum(
                _PLY_SCALAR_BYTES[fields[1]]
                for fields in scalar_properties
                if len(fields) == 3
            )
        except KeyError as error:
            raise ValueError(f"unsupported OpenMVS PLY scalar type: {error.args[0]}") from error
        if len(scalar_properties) != sum(len(fields) == 3 for fields in scalar_properties):
            raise ValueError(f"unsupported OpenMVS PLY scalar property: {source}")

        list_header_lines = {
            b"property list uint8 uint32 view_indices\n",
            b"property list uint8 float32 view_weights\n",
        }
        output_header = b"".join(
            line for line in header_lines if line not in list_header_lines
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("wb") as output_handle:
                output_handle.write(output_header)
                for vertex_index in range(vertex_count):
                    output_handle.write(
                        _read_exact(
                            input_handle,
                            scalar_record_bytes,
                            f"vertex {vertex_index} scalar properties",
                        )
                    )
                    view_count = _read_exact(
                        input_handle, 1, f"vertex {vertex_index} view count"
                    )[0]
                    _read_exact(
                        input_handle,
                        view_count * 4,
                        f"vertex {vertex_index} view indices",
                    )
                    weight_count = _read_exact(
                        input_handle, 1, f"vertex {vertex_index} weight count"
                    )[0]
                    _read_exact(
                        input_handle,
                        weight_count * 4,
                        f"vertex {vertex_index} view weights",
                    )
                if input_handle.read(1):
                    raise ValueError(f"unexpected trailing data in OpenMVS PLY: {source}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _run_timed(command: list[str]) -> float:
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def _require_nonempty_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"OpenMVS {description} was not created or is empty: {path}")


def run_openmvs_dense(
    *,
    settings: OpenMVSSettings,
    paths: OpenMVSPaths,
    colmap_executable: Path,
    image_root: Path,
    sparse_model: Path,
    source_mask_root: Path,
    max_image_size: int,
) -> dict[str, object]:
    """Run COLMAP workspace preparation followed by masked OpenMVS densification."""
    if paths.root.exists():
        shutil.rmtree(paths.root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.fused_ply_path.parent.mkdir(parents=True, exist_ok=True)
    paths.fused_ply_path.unlink(missing_ok=True)

    undistort_seconds = _run_timed(
        [
            str(colmap_executable),
            "image_undistorter",
            "--image_path",
            str(image_root),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(paths.colmap_workspace),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(max_image_size),
        ]
    )
    mask_summary = prepare_openmvs_masks(
        source_mask_root=source_mask_root,
        undistorted_image_root=paths.colmap_workspace / "images",
    )

    convert_seconds = _run_timed(build_interface_colmap_command(settings, paths))
    _require_nonempty_file(paths.scene_path, "scene.mvs")

    densify_seconds = _run_timed(build_densify_point_cloud_command(settings, paths))
    _require_nonempty_file(paths.dense_scene_path, "dense scene")
    _require_nonempty_file(paths.native_ply_path, "dense PLY")
    vertex_count = count_ply_vertices(paths.native_ply_path)
    if vertex_count <= 0:
        raise RuntimeError(f"OpenMVS dense PLY contains zero vertices: {paths.native_ply_path}")

    publish_cloudcompare_compatible_ply(paths.native_ply_path, paths.fused_ply_path)
    return {
        "mvs_backend": "openmvs",
        "fused_ply": str(paths.fused_ply_path),
        "dense_point_count": vertex_count,
        "dense_file_size_bytes": paths.fused_ply_path.stat().st_size,
        "dense_stage_seconds": {
            "undistort": undistort_seconds,
            "convert": convert_seconds,
            "densify": densify_seconds,
        },
        "openmvs": {
            "scene": str(paths.scene_path),
            "dense_scene": str(paths.dense_scene_path),
            "native_ply": str(paths.native_ply_path),
            "masks": mask_summary,
        },
    }
