from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

try:
    import lzf as _native_lzf
except ImportError:  # Optional acceleration; the validated Python decoder remains available.
    _native_lzf = None


DEFAULT_REFERENCE_CAMERAS_6 = (
    "center_camera_fov30",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_front_camera",
    "right_rear_camera",
)

DYNAMIC_OBJECT_TYPES = {
    "car",
    "vehicle_car",
    "suv",
    "truck",
    "vehicle_truck",
    "bus",
    "vehicle_bus",
    "pedestrian",
    "person",
    "non_motor_rider",
    "cyclist",
    "motorcycle",
    "bicycle",
    "rider",
}


def canonical_object_type(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw.startswith("vehicle_"):
        stripped = raw.removeprefix("vehicle_")
        if stripped in DYNAMIC_OBJECT_TYPES:
            return stripped
    return raw


@dataclass(frozen=True)
class PcdCloud:
    header_lines: tuple[str, ...]
    header: dict[str, list[str]]
    data: np.ndarray
    dtype: np.dtype

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([self.data["x"], self.data["y"], self.data["z"]]).astype(
            np.float32,
            copy=False,
        )

    @property
    def timestamps(self) -> np.ndarray | None:
        names = self.data.dtype.names or ()
        if "timestamp" in names:
            return np.asarray(self.data["timestamp"], dtype=np.float64)
        if "timestamp_sec" in names and "timestamp_nsec" in names:
            seconds = np.asarray(self.data["timestamp_sec"], dtype=np.float64)
            nanoseconds = np.asarray(self.data["timestamp_nsec"], dtype=np.float64)
            return seconds + nanoseconds * 1.0e-9
        return None

    def with_xyz(self, xyz: np.ndarray) -> "PcdCloud":
        values = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if len(values) != len(self.data):
            raise ValueError("updated xyz must have the same point count as the PCD")
        updated = self.data.copy()
        updated["x"] = values[:, 0]
        updated["y"] = values[:, 1]
        updated["z"] = values[:, 2]
        return PcdCloud(
            header_lines=self.header_lines,
            header=self.header,
            data=updated,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class CompensationObjectReport:
    frame_index: int
    object_id: str
    object_type: str
    point_count: int
    source_time_s: float
    reference_time_s: float
    motion_translation_m: float
    status: str


@dataclass(frozen=True)
class FrameCompensationResult:
    cloud: PcdCloud
    reports: tuple[CompensationObjectReport, ...]
    moved_point_count: int
    reference_time_s: float


@dataclass(frozen=True)
class ObjectTrackSample:
    frame_index: int
    time_s: float
    transform_world_from_box: np.ndarray
    obj: dict[str, Any]
    points_world: np.ndarray | None = None


@dataclass(frozen=True)
class IcpMotionConfig:
    enabled: bool = False
    translation_only: bool = False
    min_points: int = 10
    max_correspondence_distance_m: float = 1.5
    max_iterations: int = 60
    fitness_min: float = 0.05
    rmse_max_m: float = 1.5
    max_icp_translation_delta_m: float = 5.0
    max_icp_rotation_delta_deg: float = 45.0
    max_points: int = 3000
    random_seed: int = 0


@dataclass(frozen=True)
class IcpMotionEstimate:
    transform_target_from_source: np.ndarray
    used_icp: bool
    status: str
    fitness: float
    rmse_m: float
    inlier_count: int
    translation_delta_from_box_m: float
    rotation_delta_from_box_deg: float


@dataclass(frozen=True)
class ObjectIcpMotionCandidate:
    object_id: str
    lidar_name: str
    transform_target_from_source: np.ndarray
    source_time_s: float
    target_time_s: float
    fitness: float
    rmse_m: float
    inlier_count: int


@dataclass(frozen=True)
class ObjectMotionVelocity:
    object_id: str
    velocity_world_mps: np.ndarray
    source: str
    candidate_count: int
    candidate_lidars: tuple[str, ...]
    score: float


def select_object_motion_velocity(
    candidates: Iterable[ObjectIcpMotionCandidate],
    *,
    min_candidates: int = 2,
    max_velocity_deviation_mps: float = 2.0,
    mode: str = "weighted_vote",
) -> ObjectMotionVelocity | None:
    """Select a robust object-level velocity from per-LiDAR ICP candidates."""

    rows_by_lidar: dict[str, tuple[ObjectIcpMotionCandidate, np.ndarray, float]] = {}
    for candidate in candidates:
        transform = np.asarray(candidate.transform_target_from_source, dtype=np.float64).reshape(4, 4)
        dt = float(candidate.target_time_s) - float(candidate.source_time_s)
        if abs(dt) < 1.0e-9:
            continue
        velocity = transform[:3, 3] / dt
        if not np.all(np.isfinite(velocity)):
            continue
        rmse = float(candidate.rmse_m)
        rmse_term = rmse if np.isfinite(rmse) and rmse > 0.0 else 1.0
        score = max(float(candidate.fitness), 0.0) * max(int(candidate.inlier_count), 1) / (
            rmse_term + 1.0e-6
        )
        lidar_name = str(candidate.lidar_name)
        row = (candidate, velocity.astype(np.float64), float(score))
        existing = rows_by_lidar.get(lidar_name)
        if existing is None or float(row[2]) > float(existing[2]):
            rows_by_lidar[lidar_name] = row
    rows = list(rows_by_lidar.values())
    if len(rows) < int(min_candidates):
        return None
    mode_value = str(mode).strip().lower()

    if mode_value == "best_supported_icp":
        best_row: tuple[ObjectIcpMotionCandidate, np.ndarray, float] | None = None
        best_supporters: list[tuple[ObjectIcpMotionCandidate, np.ndarray, float]] = []
        for row in rows:
            supporters = [
                other
                for other in rows
                if float(np.linalg.norm(other[1] - row[1])) <= float(max_velocity_deviation_mps)
            ]
            if len(supporters) < int(min_candidates):
                continue
            if best_row is None or float(row[2]) > float(best_row[2]):
                best_row = row
                best_supporters = supporters
        if best_row is None:
            return None
        candidate = best_row[0]
        return ObjectMotionVelocity(
            object_id=str(candidate.object_id),
            velocity_world_mps=best_row[1].astype(np.float64),
            source="object_icp_best_supported",
            candidate_count=len(best_supporters),
            candidate_lidars=tuple(str(row[0].lidar_name) for row in best_supporters),
            score=float(best_row[2]),
        )

    if mode_value != "weighted_vote":
        raise ValueError("object motion selection mode must be 'weighted_vote' or 'best_supported_icp'")

    velocities = np.stack([row[1] for row in rows], axis=0)
    center = np.median(velocities, axis=0)
    distances = np.linalg.norm(velocities - center.reshape(1, 3), axis=1)
    if len(distances) > 1:
        mad = float(np.median(np.abs(distances - np.median(distances))))
    else:
        mad = 0.0
    adaptive_threshold = 2.5 * 1.4826 * mad
    threshold = max(float(max_velocity_deviation_mps), adaptive_threshold)
    keep = distances <= threshold
    kept = [row for row, flag in zip(rows, keep) if bool(flag)]
    if len(kept) < int(min_candidates):
        return None

    weights = np.asarray([max(row[2], 1.0e-6) for row in kept], dtype=np.float64)
    kept_velocities = np.stack([row[1] for row in kept], axis=0)
    velocity = np.average(kept_velocities, axis=0, weights=weights)
    object_ids = [row[0].object_id for row in kept if row[0].object_id]
    object_id = object_ids[0] if object_ids else ""
    return ObjectMotionVelocity(
        object_id=str(object_id),
        velocity_world_mps=velocity.astype(np.float64),
        source="object_icp_voted" if len(kept) > 1 else "object_icp_best",
        candidate_count=len(kept),
        candidate_lidars=tuple(str(row[0].lidar_name) for row in kept),
        score=float(np.sum(weights)),
    )


def timestamp_to_seconds(value: int | float) -> float:
    raw = float(value)
    abs_raw = abs(raw)
    if abs_raw >= 1.0e18:
        return raw / 1.0e9
    if abs_raw >= 1.0e14:
        return raw / 1.0e6
    return raw


def pcd_path_is_ego_frame(path: Path | str) -> bool:
    """Return whether a local PCD path is already expressed in ego coordinates."""

    value = Path(path)
    folder = value.parent.name.lower()
    if folder in {"perception", "compensated_perception"}:
        return True
    return folder.endswith("_compensation")


def split_indices_by_timestamp_clusters(
    indices: np.ndarray | Iterable[int],
    timestamps: np.ndarray,
    *,
    min_points: int,
    max_gap_s: float = 0.030,
    max_gap_fraction: float = 0.30,
) -> list[np.ndarray]:
    """Split selected point indices into timestamp-contiguous clusters.

    A 360-degree spinning lidar can observe the same moving object near both the
    beginning and end of one scan.  In that case, points inside one annotated box
    are not a single-time rigid observation.  This helper splits the object points
    at large timestamp gaps and discards tiny fragments.
    """

    selected = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if len(selected) < int(min_points):
        return []
    all_times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(all_times) == 0:
        return []
    selected_times = all_times[selected]
    finite = np.isfinite(selected_times)
    if int(np.count_nonzero(finite)) < int(min_points):
        return []
    selected = selected[finite]
    selected_times = selected_times[finite]
    order = np.argsort(selected_times, kind="mergesort")
    selected = selected[order]
    selected_times = selected_times[order]
    if len(selected) < int(min_points):
        return []
    if len(selected) == 1:
        return [selected] if int(min_points) <= 1 else []

    span = float(selected_times[-1] - selected_times[0])
    gaps = np.diff(selected_times)
    relative_threshold = float(max_gap_fraction) * span if span > 0.0 else float("inf")
    split_mask = (gaps > float(max_gap_s)) | (gaps > relative_threshold)
    split_after = np.flatnonzero(split_mask) + 1
    starts = np.concatenate([[0], split_after])
    stops = np.concatenate([split_after, [len(selected)]])
    clusters: list[np.ndarray] = []
    for start, stop in zip(starts, stops, strict=True):
        cluster = selected[int(start) : int(stop)]
        if len(cluster) >= int(min_points):
            clusters.append(cluster.copy())
    return clusters


def pcd_numpy_dtype(size: int, type_code: str) -> str:
    code = str(type_code).upper()
    if len(code) > 1 and code[0] in {"F", "I", "U"} and code[1:].isdigit():
        code_size = int(code[1:])
        if int(size) == code_size:
            code = code[0]
    if code == "F" and size == 4:
        return "<f4"
    if code == "F" and size == 8:
        return "<f8"
    if code == "I" and size == 1:
        return "<i1"
    if code == "I" and size == 2:
        return "<i2"
    if code == "I" and size == 4:
        return "<i4"
    if code == "I" and size == 8:
        return "<i8"
    if code == "U" and size == 1:
        return "<u1"
    if code == "U" and size == 2:
        return "<u2"
    if code == "U" and size == 4:
        return "<u4"
    if code == "U" and size == 8:
        return "<u8"
    raise ValueError(f"unsupported PCD field type: TYPE={type_code}, SIZE={size}")


def pcd_struct_dtype(
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
) -> np.dtype:
    dtype_fields = []
    for field, size, type_code, count in zip(fields, sizes, types, counts, strict=True):
        dtype = np.dtype(pcd_numpy_dtype(size, type_code))
        dtype_fields.append((field, dtype, () if count == 1 else (count,)))
    return np.dtype(dtype_fields)


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
    if _native_lzf is not None:
        native_output = _native_lzf.decompress(data, int(expected_size))
        if native_output is not None:
            if len(native_output) != int(expected_size):
                raise ValueError(
                    "native LZF decompressed size mismatch: "
                    f"got {len(native_output)}, expected {expected_size}"
                )
            return native_output

    output = bytearray(expected_size)
    input_index = 0
    output_index = 0
    data_len = len(data)

    while input_index < data_len:
        ctrl = data[input_index]
        input_index += 1
        if ctrl < 32:
            length = ctrl + 1
            output[output_index : output_index + length] = data[input_index : input_index + length]
            input_index += length
            output_index += length
            continue

        length = ctrl >> 5
        reference = output_index - ((ctrl & 0x1F) << 8) - 1
        if length == 7:
            length += data[input_index]
            input_index += 1
        reference -= data[input_index]
        input_index += 1
        length += 2

        for _ in range(length):
            output[output_index] = output[reference]
            output_index += 1
            reference += 1

    if output_index != expected_size:
        raise ValueError(
            f"LZF decompressed size mismatch: got {output_index}, expected {expected_size}"
        )
    return bytes(output)


def pcd_binary_compressed_to_structured(
    payload: bytes,
    *,
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
    points: int,
) -> np.ndarray:
    """Decode PCL binary_compressed SoA payload into the normal structured AoS array."""

    if len(payload) < 8:
        raise ValueError("binary_compressed PCD payload is too short")
    compressed_size = int(np.frombuffer(payload[:4], dtype="<u4")[0])
    uncompressed_size = int(np.frombuffer(payload[4:8], dtype="<u4")[0])
    compressed = payload[8 : 8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError(
            f"binary_compressed PCD payload truncated: got {len(compressed)}, expected {compressed_size}"
        )
    decompressed = lzf_decompress(compressed, uncompressed_size)

    dtype = pcd_struct_dtype(fields, sizes, types, counts)
    expected_size = int(points) * int(dtype.itemsize)
    if len(decompressed) != expected_size:
        raise ValueError(
            f"binary_compressed PCD uncompressed size mismatch: got {len(decompressed)}, "
            f"expected {expected_size}"
        )

    data = np.zeros(int(points), dtype=dtype)
    offset = 0
    for field, size, type_code, count in zip(fields, sizes, types, counts, strict=True):
        item_dtype = np.dtype(pcd_numpy_dtype(size, type_code))
        value_count = int(points) * int(count)
        byte_count = value_count * int(size)
        values = np.frombuffer(
            decompressed[offset : offset + byte_count],
            dtype=item_dtype,
            count=value_count,
        )
        if int(count) == 1:
            data[field] = values
        else:
            data[field] = values.reshape(int(points), int(count))
        offset += byte_count
    return data


def read_pcd_header(path: Path) -> tuple[tuple[str, ...], dict[str, list[str]], int]:
    lines: list[str] = []
    header: dict[str, list[str]] = {}
    with Path(path).open("rb") as handle:
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError(f"missing DATA line in {path}")
            text = line_bytes.decode("ascii", errors="replace").rstrip("\r\n")
            lines.append(text)
            stripped = text.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, *values = stripped.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                return tuple(lines), header, handle.tell()


def read_pcd_cloud(path: Path | str) -> PcdCloud:
    path = Path(path)
    header_lines, header, data_offset = read_pcd_header(path)
    fields = header["FIELDS"]
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"{path} does not contain x/y/z fields")
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])
    dtype = pcd_struct_dtype(fields, sizes, types, counts)
    with path.open("rb") as handle:
        handle.seek(data_offset)
        payload = handle.read()
    data_kind = header.get("DATA", [""])[0].lower()
    if data_kind == "binary":
        data = np.frombuffer(payload, dtype=dtype, count=points).copy()
    elif data_kind == "binary_compressed":
        data = pcd_binary_compressed_to_structured(
            payload,
            fields=fields,
            sizes=sizes,
            types=types,
            counts=counts,
            points=points,
        )
    else:
        raise ValueError(f"unsupported PCD DATA kind for compensation: {data_kind} ({path})")
    return PcdCloud(header_lines=header_lines, header=header, data=data, dtype=dtype)


def load_frame_point_clouds(
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    dataset_root: Path | str,
    resolve_sensor_file,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Load perception PCD xyz/timestamps for every frame.

    ``resolve_sensor_file`` is injected so CLI scripts can keep their dataset-path
    resolution rules local while sharing the object-track construction semantics.
    """

    root = Path(dataset_root)
    clouds: list[tuple[np.ndarray, np.ndarray | None]] = []
    first = max(0, int(start))
    last = len(infos) if end is None else min(len(infos), int(end))
    for frame_index, info in enumerate(infos[first:last], start=first):
        lidar_data = info.get("sensors", {}).get("lidar", {}).get("perception")
        if not isinstance(lidar_data, dict):
            raise ValueError(f"frame {frame_index} is missing perception lidar")
        pcd_path = resolve_sensor_file(root, str(lidar_data["data_path"]), "perception")
        cloud = read_pcd_cloud(pcd_path)
        clouds.append((cloud.xyz, cloud.timestamps))
    return clouds


def write_pcd_cloud(path: Path | str, cloud: PcdCloud) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    point_count = len(cloud.data)
    lines = []
    for line in cloud.header_lines:
        key = line.strip().split(maxsplit=1)[0].upper() if line.strip() else ""
        if key == "WIDTH":
            lines.append(f"WIDTH {point_count}")
        elif key == "POINTS":
            lines.append(f"POINTS {point_count}")
        elif key == "DATA":
            lines.append("DATA binary")
        else:
            lines.append(line)
    with path.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        cloud.data.tofile(handle)


def _rotation_from_object(obj: dict[str, Any]) -> Rotation:
    rotation = np.asarray(obj.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    if rotation.size == 3:
        return Rotation.from_euler("xyz", rotation)
    if rotation.size == 4:
        return Rotation.from_quat(rotation)
    raise ValueError(f"unsupported object rotation shape: {rotation.shape}")


def box_transform_ego(obj: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_object(obj).as_matrix()
    transform[:3, 3] = np.asarray(obj["location"], dtype=np.float64).reshape(3)
    return transform


def box_corners_ego(obj: dict[str, Any]) -> np.ndarray:
    local = local_box_corners(obj["size"])
    return transform_points(local, box_transform_ego(obj)).astype(np.float64)


def local_box_corners(size: Iterable[float]) -> np.ndarray:
    size = np.asarray(list(size), dtype=np.float64).reshape(3)
    half = size / 2.0
    return np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )


def box_corners_from_transform(size: Iterable[float], transform: np.ndarray) -> np.ndarray:
    return transform_points(local_box_corners(size), transform).astype(np.float64)


BOX_EDGE_INDICES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def box_edges_polylinez(corners: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    return [values[[start, end], :].copy() for start, end in BOX_EDGE_INDICES]


def box_correction_delta_ego(before_corners: np.ndarray, after_corners: np.ndarray) -> np.ndarray:
    before = np.asarray(before_corners, dtype=np.float64).reshape(8, 3)
    after = np.asarray(after_corners, dtype=np.float64).reshape(8, 3)
    return after.mean(axis=0) - before.mean(axis=0)


def _flatten_parts(parts: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    starts: list[int] = []
    points: list[np.ndarray] = []
    cursor = 0
    for part in parts:
        values = np.asarray(part, dtype=np.float64).reshape(-1, 3)
        if len(values) == 0:
            continue
        starts.append(cursor)
        points.append(values)
        cursor += len(values)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.vstack(points), starts


def _shape_xy_box(points: np.ndarray) -> tuple[float, float, float, float]:
    if len(points) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _shape_z_box(points: np.ndarray) -> tuple[float, float]:
    if len(points) == 0:
        return (0.0, 0.0)
    return (float(np.min(points[:, 2])), float(np.max(points[:, 2])))


def _write_shapefile_header(
    handle,
    *,
    file_length_words: int,
    shape_type: int,
    xy_box: tuple[float, float, float, float],
    z_box: tuple[float, float],
) -> None:
    handle.write(struct.pack(">i", 9994))
    handle.write(b"\x00" * 20)
    handle.write(struct.pack(">i", int(file_length_words)))
    handle.write(struct.pack("<i", 1000))
    handle.write(struct.pack("<i", int(shape_type)))
    handle.write(struct.pack("<4d", *xy_box))
    handle.write(struct.pack("<2d", *z_box))
    handle.write(struct.pack("<2d", -1.0e38, -1.0e38))


def _polylinez_record_content(parts: list[np.ndarray]) -> bytes:
    points, starts = _flatten_parts(parts)
    xy_box = _shape_xy_box(points)
    z_min, z_max = _shape_z_box(points)
    content = bytearray()
    content += struct.pack("<i", 13)
    content += struct.pack("<4d", *xy_box)
    content += struct.pack("<2i", len(starts), len(points))
    for start in starts:
        content += struct.pack("<i", int(start))
    for x, y, _z in points:
        content += struct.pack("<2d", float(x), float(y))
    content += struct.pack("<2d", z_min, z_max)
    for _x, _y, z in points:
        content += struct.pack("<d", float(z))
    content += struct.pack("<2d", -1.0e38, -1.0e38)
    for _ in points:
        content += struct.pack("<d", -1.0e38)
    return bytes(content)


def _dbf_field_descriptor(name: str, field_type: str, size: int, decimals: int = 0) -> bytes:
    raw_name = str(name).encode("ascii", errors="ignore")[:10]
    descriptor = bytearray(32)
    descriptor[: len(raw_name)] = raw_name
    descriptor[11] = ord(field_type)
    descriptor[16] = int(size)
    descriptor[17] = int(decimals)
    return bytes(descriptor)


def _format_dbf_value(value: object, field_type: str, size: int, decimals: int = 0) -> bytes:
    if field_type == "C":
        text = "" if value is None else str(value)
        return text.encode("utf-8", errors="ignore")[:size].ljust(size, b" ")
    if field_type == "N":
        if value in (None, ""):
            return b" " * size
        number = float(value)
        if decimals <= 0:
            text = f"{int(round(number))}"
        else:
            text = f"{number:.{decimals}f}"
        if len(text) > size:
            text = text[:size]
        return text.rjust(size, " ").encode("ascii")
    raise ValueError(f"unsupported DBF field type: {field_type}")


def _write_dbf(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        ("FRAME", "N", 10, 0),
        ("OBJ_ID", "C", 32, 0),
        ("OBJ_TYPE", "C", 24, 0),
        ("TIME_S", "N", 18, 6),
        ("CX", "N", 18, 6),
        ("CY", "N", 18, 6),
        ("CZ", "N", 18, 6),
        ("LENGTH", "N", 12, 4),
        ("WIDTH", "N", 12, 4),
        ("HEIGHT", "N", 12, 4),
        ("YAW", "N", 12, 6),
        ("STATUS", "C", 16, 0),
    ]
    today = date.today()
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<BBBBIHH20x",
                3,
                today.year - 1900,
                today.month,
                today.day,
                len(rows),
                header_length,
                record_length,
            )
        )
        for field in fields:
            handle.write(_dbf_field_descriptor(*field))
        handle.write(b"\r")
        for row in rows:
            handle.write(b" ")
            for name, field_type, size, decimals in fields:
                handle.write(_format_dbf_value(row.get(name), field_type, size, decimals))
        handle.write(b"\x1a")


def write_polylinez_shapefile(path: Path | str, boxes: list[dict[str, object]]) -> Path:
    """Write one frame's 3D boxes as a PolyLineZ shapefile.

    Each feature is a wireframe 3D box made of 12 two-point line parts.  Coordinates
    are written exactly in the provided ego frame; no CRS transformation is applied.
    """

    shp_path = Path(path).with_suffix(".shp")
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[bytes] = []
    dbf_rows: list[dict[str, object]] = []
    all_points: list[np.ndarray] = []
    for box in boxes:
        corners = np.asarray(box["corners"], dtype=np.float64).reshape(8, 3)
        records.append(_polylinez_record_content(box_edges_polylinez(corners)))
        all_points.append(corners)
        center = corners.mean(axis=0)
        extent = corners.max(axis=0) - corners.min(axis=0)
        dbf_rows.append(
            {
                "FRAME": box.get("frame", 0),
                "OBJ_ID": box.get("object_id", ""),
                "OBJ_TYPE": box.get("object_type", ""),
                "TIME_S": box.get("timestamp_s", ""),
                "CX": center[0],
                "CY": center[1],
                "CZ": center[2],
                "LENGTH": box.get("length", extent[0]),
                "WIDTH": box.get("width", extent[1]),
                "HEIGHT": box.get("height", extent[2]),
                "YAW": box.get("yaw", 0.0),
                "STATUS": box.get("status", ""),
            }
        )
    if all_points:
        stacked = np.vstack(all_points)
    else:
        stacked = np.zeros((0, 3), dtype=np.float64)
    xy_box = _shape_xy_box(stacked)
    z_box = _shape_z_box(stacked)

    shp_length_words = 50 + sum((8 + len(record)) // 2 for record in records)
    with shp_path.open("wb") as handle:
        _write_shapefile_header(
            handle,
            file_length_words=shp_length_words,
            shape_type=13,
            xy_box=xy_box,
            z_box=z_box,
        )
        for index, record in enumerate(records, start=1):
            handle.write(struct.pack(">2i", index, len(record) // 2))
            handle.write(record)

    shx_path = shp_path.with_suffix(".shx")
    shx_length_words = 50 + len(records) * 4
    offset_words = 50
    with shx_path.open("wb") as handle:
        _write_shapefile_header(
            handle,
            file_length_words=shx_length_words,
            shape_type=13,
            xy_box=xy_box,
            z_box=z_box,
        )
        for record in records:
            handle.write(struct.pack(">2i", offset_words, len(record) // 2))
            offset_words += (8 + len(record)) // 2

    _write_dbf(shp_path.with_suffix(".dbf"), dbf_rows)
    shp_path.with_suffix(".prj").write_text(
        'LOCAL_CS["Ego vehicle coordinate system",UNIT["metre",1.0]]\n',
        encoding="ascii",
    )
    return shp_path


def ego_world_transform(info: dict[str, Any]) -> np.ndarray:
    return np.asarray(info["ego2global_transformation_matrix"], dtype=np.float64).reshape(4, 4)


def box_transform_world(info: dict[str, Any], obj: dict[str, Any]) -> np.ndarray:
    return ego_world_transform(info) @ box_transform_ego(obj)


def points_in_box_mask(
    points_ego: np.ndarray,
    obj: dict[str, Any],
    *,
    margin_m: float = 0.0,
) -> np.ndarray:
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    center = np.asarray(obj["location"], dtype=np.float64).reshape(3)
    half = np.asarray(obj["size"], dtype=np.float64).reshape(3) / 2.0 + float(margin_m)
    rotation = _rotation_from_object(obj).as_matrix()
    world_half = np.abs(rotation) @ half
    coarse = np.all(np.abs(points - center.reshape(1, 3)) <= world_half.reshape(1, 3), axis=1)
    if not coarse.any():
        return coarse
    local = (rotation.T @ (points[coarse] - center.reshape(1, 3)).T).T
    inside_local = np.all(np.abs(local) <= half.reshape(1, 3) + 1e-9, axis=1)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.flatnonzero(coarse)[inside_local]] = True
    return mask


def is_dynamic_object(obj: dict[str, Any]) -> bool:
    attributes = obj.get("attribute", {})
    if isinstance(attributes, dict):
        if int(attributes.get("ignore", 0) or 0) == 1:
            return False
        if int(attributes.get("motion", 0) or 0) == 1:
            return True
    return canonical_object_type(obj.get("type", "")) in DYNAMIC_OBJECT_TYPES


def object_id(obj: dict[str, Any]) -> str:
    return str(obj.get("id", ""))


def _sequence_value(values: Any, index: int, default: Any = None) -> Any:
    try:
        if values is None or len(values) <= index:
            return default
        return values[index]
    except TypeError:
        return default


def objects_from_gt_boxes(info: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
    if len(boxes) == 0:
        return []
    names = info.get("gt_names", [])
    ids = info.get("ID", [])
    velocities = info.get("vels", [])
    attributes = info.get("gt_attrs", [])
    num_points = info.get("num_points", [])
    objects: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        object_type = str(_sequence_value(names, index, ""))
        object_identifier = _sequence_value(ids, index, index)
        velocity = _sequence_value(velocities, index, [0.0, 0.0, 0.0])
        attribute = _sequence_value(attributes, index, {})
        points = _sequence_value(num_points, index, {})
        if not isinstance(attribute, dict):
            attribute = {}
        if not isinstance(points, dict):
            points = {}
        objects.append(
            {
                "id": int(object_identifier) if str(object_identifier).lstrip("-").isdigit() else object_identifier,
                "type": object_type,
                "location": box[:3].astype(float).tolist(),
                # gt_boxes uses [x, y, z, length, width, height, yaw].
                # project-local object dictionaries use size=[length, width, height].
                "size": [float(box[3]), float(box[4]), float(box[5])],
                "rotation": [0.0, 0.0, float(box[6])],
                "velocity": np.asarray(velocity, dtype=np.float64).reshape(-1)[:3].astype(float).tolist(),
                "attribute": attribute,
                "num_points": points,
                "info2d": [],
                "clip_points": None,
            }
        )
    return objects


def dynamic_objects_for_info(info: dict[str, Any]) -> list[dict[str, Any]]:
    objects = info.get("objects", [])
    if objects:
        return [obj for obj in objects if is_dynamic_object(obj)]
    return [obj for obj in objects_from_gt_boxes(info) if is_dynamic_object(obj)]


def reference_time_seconds_from_cameras(
    info: dict[str, Any],
    camera_names: Iterable[str],
) -> float:
    cams = info.get("sensors", {}).get("cams", {})
    values = []
    for camera in camera_names:
        cam = cams.get(str(camera))
        if isinstance(cam, dict) and "timestamp" in cam:
            values.append(timestamp_to_seconds(cam["timestamp"]))
    if not values:
        raise ValueError("none of the configured reference cameras has a timestamp")
    return float(np.mean(values))


def interpolate_transform(transform0: np.ndarray, transform1: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    t0 = np.asarray(transform0, dtype=np.float64).reshape(4, 4)
    t1 = np.asarray(transform1, dtype=np.float64).reshape(4, 4)
    if a <= 0.0:
        return t0.copy()
    if a >= 1.0:
        return t1.copy()
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (1.0 - a) * t0[:3, 3] + a * t1[:3, 3]
    rotations = Rotation.from_matrix(np.stack([t0[:3, :3], t1[:3, :3]], axis=0))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)([a]).as_matrix()[0]
    return out


def interpolate_relative_transform(transform: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = a * value[:3, 3]
    rotations = Rotation.from_matrix(np.stack([np.eye(3), value[:3, :3]], axis=0))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)([a]).as_matrix()[0]
    return out


def _sample_points_for_icp(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(values) <= int(max_points):
        return values
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(values), size=int(max_points), replace=False)
    return values[np.sort(indices)]


def _rotation_angle_deg(transform: np.ndarray) -> float:
    return float(
        np.rad2deg(
            Rotation.from_matrix(np.asarray(transform, dtype=np.float64).reshape(4, 4)[:3, :3]).magnitude()
        )
    )


def estimate_icp_corrected_motion(
    *,
    source_points_world: np.ndarray,
    target_points_world: np.ndarray,
    box_motion_target_from_source: np.ndarray,
    config: IcpMotionConfig,
) -> IcpMotionEstimate:
    """Estimate object motion with box motion as ICP initialization.

    The returned transform maps source-world points to target-world points.  If ICP
    is disabled, unavailable, or fails quality checks, the box motion is returned as
    a safe fallback and the status explains why.
    """

    box_motion = np.asarray(box_motion_target_from_source, dtype=np.float64).reshape(4, 4)
    source = np.asarray(source_points_world, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3)
    empty_stats = {
        "fitness": float("nan"),
        "rmse_m": float("nan"),
        "inlier_count": 0,
        "translation_delta_from_box_m": 0.0,
        "rotation_delta_from_box_deg": 0.0,
    }
    if not bool(config.enabled):
        return IcpMotionEstimate(box_motion, False, "box_only", **empty_stats)
    if len(source) < int(config.min_points) or len(target) < int(config.min_points):
        return IcpMotionEstimate(box_motion, False, "icp_too_few_points", **empty_stats)
    if bool(config.translation_only):
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return IcpMotionEstimate(box_motion, False, "icp_scipy_unavailable", **empty_stats)

        source_sample = _sample_points_for_icp(
            source, int(config.max_points), int(config.random_seed)
        )
        target_sample = _sample_points_for_icp(
            target, int(config.max_points), int(config.random_seed) + 1
        )
        tree = cKDTree(target_sample)
        current_motion = box_motion.copy()
        correspondence_distance = float(config.max_correspondence_distance_m)
        inlier_distances = np.zeros(0, dtype=np.float64)
        inlier_count = 0
        for _ in range(int(config.max_iterations)):
            transformed = transform_points(source_sample, current_motion).astype(np.float64)
            distances, indices = tree.query(transformed, k=1, workers=-1)
            inliers = np.asarray(distances <= correspondence_distance, dtype=bool)
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < int(config.min_points):
                break
            residual = target_sample[indices[inliers]] - transformed[inliers]
            delta_t = np.median(residual, axis=0)
            if not np.all(np.isfinite(delta_t)):
                break
            current_motion[:3, 3] += delta_t
            inlier_distances = np.asarray(distances[inliers], dtype=np.float64)
            if float(np.linalg.norm(delta_t)) < 1.0e-4:
                break
        fitness = float(inlier_count / max(len(source_sample), 1))
        rmse = float(np.sqrt(np.mean(inlier_distances**2))) if inlier_distances.size else float("nan")
        delta = current_motion @ np.linalg.inv(box_motion)
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        rotation_delta = _rotation_angle_deg(delta)
        stats = {
            "fitness": fitness,
            "rmse_m": rmse,
            "inlier_count": inlier_count,
            "translation_delta_from_box_m": translation_delta,
            "rotation_delta_from_box_deg": rotation_delta,
        }
        if inlier_count < int(config.min_points):
            return IcpMotionEstimate(box_motion, False, "icp_too_few_inliers", **stats)
        if fitness < float(config.fitness_min):
            return IcpMotionEstimate(box_motion, False, "icp_low_fitness", **stats)
        if not np.isfinite(rmse) or rmse > float(config.rmse_max_m):
            return IcpMotionEstimate(box_motion, False, "icp_high_rmse", **stats)
        if translation_delta > float(config.max_icp_translation_delta_m):
            return IcpMotionEstimate(box_motion, False, "icp_translation_delta_too_large", **stats)
        if rotation_delta > float(config.max_icp_rotation_delta_deg) + 1.0e-9:
            return IcpMotionEstimate(box_motion, False, "icp_rotation_delta_too_large", **stats)
        return IcpMotionEstimate(current_motion, True, "icp_translation_only", **stats)
    try:
        import open3d as o3d
    except Exception:
        return IcpMotionEstimate(box_motion, False, "icp_open3d_unavailable", **empty_stats)

    source_sample = _sample_points_for_icp(source, int(config.max_points), int(config.random_seed))
    target_sample = _sample_points_for_icp(target, int(config.max_points), int(config.random_seed) + 1)
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_sample)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_sample)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(config.max_iterations)
    )
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        float(config.max_correspondence_distance_m),
        box_motion,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    icp_motion = np.asarray(result.transformation, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(icp_motion)):
        return IcpMotionEstimate(box_motion, False, "icp_non_finite", **empty_stats)
    delta = icp_motion @ np.linalg.inv(box_motion)
    translation_delta = float(np.linalg.norm(delta[:3, 3]))
    rotation_delta = _rotation_angle_deg(delta)
    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)
    inlier_count = int(round(fitness * len(source_sample)))
    stats = {
        "fitness": fitness,
        "rmse_m": rmse,
        "inlier_count": inlier_count,
        "translation_delta_from_box_m": translation_delta,
        "rotation_delta_from_box_deg": rotation_delta,
    }
    if fitness < float(config.fitness_min):
        return IcpMotionEstimate(box_motion, False, "icp_low_fitness", **stats)
    if not np.isfinite(rmse) or rmse > float(config.rmse_max_m):
        return IcpMotionEstimate(box_motion, False, "icp_high_rmse", **stats)
    if translation_delta > float(config.max_icp_translation_delta_m):
        return IcpMotionEstimate(box_motion, False, "icp_translation_delta_too_large", **stats)
    if rotation_delta > float(config.max_icp_rotation_delta_deg):
        return IcpMotionEstimate(box_motion, False, "icp_rotation_delta_too_large", **stats)
    return IcpMotionEstimate(icp_motion, True, "icp", **stats)


def build_object_tracks(
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    frame_point_clouds: Iterable[tuple[np.ndarray, np.ndarray | None]],
    *,
    min_object_points: int = 5,
    box_margin_m: float = 0.0,
) -> dict[str, list[ObjectTrackSample]]:
    """Build object tracks using each object's box-internal point mean timestamp.

    The timestamp of a labeled 3D box is not the frame timestamp here.  For each
    object observation, points inside the current-frame annotated box are selected
    from the perception PCD, and their mean per-point timestamp is used as the
    object observation time.  Objects without enough timestamped points are not
    inserted into the track, so downstream compensation fails explicitly instead of
    falling back to the incorrect frame-timestamp model.
    """

    clouds = list(frame_point_clouds)
    if len(clouds) != len(infos):
        raise ValueError("frame_point_clouds must have the same length as infos")
    tracks: dict[str, list[ObjectTrackSample]] = {}
    for frame_index, (info, cloud_payload) in enumerate(zip(infos, clouds, strict=True)):
        cloud_xyz, point_timestamps = cloud_payload
        if point_timestamps is None:
            continue
        xyz = np.asarray(cloud_xyz, dtype=np.float64).reshape(-1, 3)
        timestamps = np.asarray(point_timestamps, dtype=np.float64).reshape(-1)
        if len(xyz) != len(timestamps):
            raise ValueError(f"point/timestamp length mismatch at frame {frame_index}")
        claimed_points = np.zeros(len(xyz), dtype=bool)
        for obj in dynamic_objects_for_info(info):
            oid = object_id(obj)
            if not oid:
                continue
            mask = points_in_box_mask(xyz, obj, margin_m=box_margin_m) & ~claimed_points
            if int(np.count_nonzero(mask)) < int(min_object_points):
                continue
            finite_times = timestamps[mask]
            finite_times = finite_times[np.isfinite(finite_times)]
            if len(finite_times) < int(min_object_points):
                continue
            time_s = float(np.mean(finite_times))
            claimed_points[mask] = True
            tracks.setdefault(oid, []).append(
                ObjectTrackSample(
                    frame_index=int(frame_index),
                    time_s=float(time_s),
                    transform_world_from_box=box_transform_world(info, obj),
                    obj=obj,
                )
            )
    for samples in tracks.values():
        samples.sort(key=lambda item: item.time_s)
    return tracks


def _interpolate_track_transform(
    samples: list[ObjectTrackSample],
    time_s: float,
) -> np.ndarray | None:
    if len(samples) < 2:
        return None
    t = float(time_s)
    times = [item.time_s for item in samples]
    index = int(np.searchsorted(times, t, side="right"))
    left = index - 1
    right = index
    if left < 0 or right >= len(samples):
        return None
    t0 = samples[left].time_s
    t1 = samples[right].time_s
    if t1 <= t0:
        return None
    alpha = (t - t0) / (t1 - t0)
    return interpolate_transform(
        samples[left].transform_world_from_box,
        samples[right].transform_world_from_box,
        alpha,
    )


def _interpolate_ego_transform(
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    time_s: float,
) -> np.ndarray | None:
    if len(infos) < 2:
        return None
    times = [timestamp_to_seconds(info["timestamp"]) for info in infos]
    index = int(np.searchsorted(times, float(time_s), side="right"))
    left = index - 1
    right = index
    if left < 0 or right >= len(infos):
        return None
    t0 = times[left]
    t1 = times[right]
    if t1 <= t0:
        return None
    return interpolate_transform(
        ego_world_transform(infos[left]),
        ego_world_transform(infos[right]),
        (float(time_s) - t0) / (t1 - t0),
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return values.astype(np.float32)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64).reshape(4, 4) @ homogeneous.T).T[
        :, :3
    ].astype(np.float32)


def compensate_points_between_box_poses(
    points_ego: np.ndarray,
    *,
    source_box_to_world: np.ndarray,
    target_box_to_world: np.ndarray,
    world_to_output_ego: np.ndarray,
    source_ego_to_world: np.ndarray,
) -> np.ndarray:
    source_points_world = transform_points(points_ego, source_ego_to_world).astype(np.float64)
    source_box_to_world = np.asarray(source_box_to_world, dtype=np.float64).reshape(4, 4)
    target_box_to_world = np.asarray(target_box_to_world, dtype=np.float64).reshape(4, 4)
    local_in_box = transform_points(source_points_world, np.linalg.inv(source_box_to_world))
    target_world = transform_points(local_in_box, target_box_to_world)
    return transform_points(target_world, world_to_output_ego)


def compensate_dynamic_objects_for_frame(
    *,
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    frame_index: int,
    cloud: PcdCloud,
    object_tracks: dict[str, list[ObjectTrackSample]],
    reference_cameras: Iterable[str] = DEFAULT_REFERENCE_CAMERAS_6,
    min_object_points: int = 5,
    box_margin_m: float = 0.0,
    max_motion_translation_m: float | None = None,
) -> FrameCompensationResult:
    info = infos[int(frame_index)]
    reference_time_s = reference_time_seconds_from_cameras(info, reference_cameras)
    target_ego_to_world = _interpolate_ego_transform(infos, reference_time_s)
    if target_ego_to_world is None:
        target_ego_to_world = ego_world_transform(info)
    world_to_output_ego = np.linalg.inv(target_ego_to_world)
    source_ego_to_world = ego_world_transform(info)
    tracks = object_tracks

    xyz = cloud.xyz.copy()
    timestamps = cloud.timestamps
    moved = np.zeros(len(xyz), dtype=bool)
    reports: list[CompensationObjectReport] = []
    for obj in dynamic_objects_for_info(info):
        oid = object_id(obj)
        mask = points_in_box_mask(xyz, obj, margin_m=box_margin_m) & ~moved
        point_count = int(np.count_nonzero(mask))
        if point_count < int(min_object_points):
            reports.append(
                CompensationObjectReport(
                    frame_index=int(frame_index),
                    object_id=oid,
                    object_type=str(obj.get("type", "")),
                    point_count=point_count,
                    source_time_s=float("nan"),
                    reference_time_s=reference_time_s,
                    motion_translation_m=float("nan"),
                    status="too_few_points",
                )
            )
            continue
        if timestamps is None:
            reports.append(
                CompensationObjectReport(
                    frame_index=int(frame_index),
                    object_id=oid,
                    object_type=str(obj.get("type", "")),
                    point_count=point_count,
                    source_time_s=float("nan"),
                    reference_time_s=reference_time_s,
                    motion_translation_m=float("nan"),
                    status="missing_point_timestamps",
                )
            )
            continue
        source_time_s = float(np.mean(timestamps[mask]))
        samples = tracks.get(oid, [])
        source_box_to_world = _interpolate_track_transform(samples, source_time_s)
        target_box_to_world = _interpolate_track_transform(samples, reference_time_s)
        if source_box_to_world is None or target_box_to_world is None:
            reports.append(
                CompensationObjectReport(
                    frame_index=int(frame_index),
                    object_id=oid,
                    object_type=str(obj.get("type", "")),
                    point_count=point_count,
                    source_time_s=source_time_s,
                    reference_time_s=reference_time_s,
                    motion_translation_m=float("nan"),
                    status="no_bracketing_object_track",
                )
            )
            continue
        moved_xyz = compensate_points_between_box_poses(
            xyz[mask],
            source_box_to_world=source_box_to_world,
            target_box_to_world=target_box_to_world,
            world_to_output_ego=world_to_output_ego,
            source_ego_to_world=source_ego_to_world,
        )
        motion = np.linalg.norm(moved_xyz.astype(np.float64) - xyz[mask].astype(np.float64), axis=1)
        motion_m = float(np.median(motion)) if motion.size else 0.0
        motion_limit = (
            float(max_motion_translation_m)
            if max_motion_translation_m is not None
            else float("inf")
        )
        if np.isfinite(motion_limit) and motion_m > motion_limit:
            reports.append(
                CompensationObjectReport(
                    frame_index=int(frame_index),
                    object_id=oid,
                    object_type=str(obj.get("type", "")),
                    point_count=point_count,
                    source_time_s=source_time_s,
                    reference_time_s=reference_time_s,
                    motion_translation_m=motion_m,
                    status="motion_translation_too_large",
                )
            )
            continue
        xyz[mask] = moved_xyz
        moved[mask] = True
        reports.append(
            CompensationObjectReport(
                frame_index=int(frame_index),
                object_id=oid,
                object_type=str(obj.get("type", "")),
                point_count=point_count,
                source_time_s=source_time_s,
                reference_time_s=reference_time_s,
                motion_translation_m=motion_m,
                status="compensated",
            )
        )

    return FrameCompensationResult(
        cloud=cloud.with_xyz(xyz),
        reports=tuple(reports),
        moved_point_count=int(np.count_nonzero(moved)),
        reference_time_s=reference_time_s,
    )
