from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dongfeng_registration.dynamic_object_compensation import pcd_path_is_ego_frame


CAMERA_TO_IMAGE_FOLDER = {
    "center_camera_fov120": "front_camera_fov120",
    "center_camera_fov30": "front_camera_fov30",
    "left_front_camera": "left_front_camera",
    "right_front_camera": "right_front_camera",
    "left_rear_camera": "left_rear_camera",
    "right_rear_camera": "right_rear_camera",
    "rear_camera": "rear_camera",
}


def read_point_cloud_xyz(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        array = read_pcd_xyz(path)
    else:
        pcd = o3d.t.io.read_point_cloud(str(path))
        items = dict(pcd.point.items())
        points = items.get("positions", items.get("points"))
        if points is None:
            raise ValueError(f"No positions/points in {path}")
        array = points.numpy().astype(np.float32)
    return array[np.isfinite(array).all(axis=1)]


def read_pcd_xyz(path: Path) -> np.ndarray:
    header, data_offset = read_pcd_header(path)
    fields = header["FIELDS"]
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"{path} does not contain x/y/z fields")

    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])
    data_kind = header["DATA"][0].lower()

    with path.open("rb") as handle:
        handle.seek(data_offset)
        if data_kind == "ascii":
            text = handle.read().decode("utf-8", errors="replace")
            numeric = np.loadtxt(text.splitlines(), dtype=np.float32)
            if numeric.ndim == 1:
                numeric = numeric.reshape(1, -1)
            return numeric[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(
                np.float32
            )
        if data_kind == "binary":
            raw = handle.read()
            structured = np.frombuffer(raw, dtype=pcd_struct_dtype(fields, sizes, types, counts), count=points)
            return np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(
                np.float32
            )
        if data_kind == "binary_compressed":
            compressed_size = int(np.frombuffer(handle.read(4), dtype="<u4")[0])
            uncompressed_size = int(np.frombuffer(handle.read(4), dtype="<u4")[0])
            compressed = handle.read(compressed_size)
            decompressed = lzf_decompress(compressed, uncompressed_size)
            return xyz_from_binary_compressed(
                decompressed,
                fields=fields,
                sizes=sizes,
                types=types,
                counts=counts,
                points=points,
            )
    raise ValueError(f"unsupported PCD DATA kind in {path}: {data_kind}")


def read_pcd_header(path: Path) -> tuple[dict[str, list[str]], int]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"missing DATA line in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                return header, handle.tell()


def pcd_numpy_dtype(size: int, type_code: str) -> str:
    code = type_code.upper()
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


def xyz_from_binary_compressed(
    decompressed: bytes,
    *,
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
    points: int,
) -> np.ndarray:
    offsets: dict[str, tuple[int, int, str]] = {}
    offset = 0
    for field, size, type_code, count in zip(fields, sizes, types, counts, strict=True):
        byte_count = points * size * count
        offsets[field] = (offset, byte_count, pcd_numpy_dtype(size, type_code))
        offset += byte_count

    columns = []
    for field in ("x", "y", "z"):
        start, byte_count, dtype = offsets[field]
        values = np.frombuffer(decompressed[start : start + byte_count], dtype=dtype, count=points)
        columns.append(values.astype(np.float32, copy=False))
    return np.column_stack(columns).astype(np.float32, copy=False)


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
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


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload.get("infos")
    if not isinstance(infos, list):
        raise ValueError(f"{path} does not contain list field 'infos'")
    return infos


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if values.size == 0:
        return values.copy()
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_h = np.concatenate(
        [values.astype(np.float64), np.ones((len(values), 1), dtype=np.float64)],
        axis=1,
    )
    return (matrix @ points_h.T).T[:, :3].astype(np.float32)


def lidar_points_for_info(
    info: dict[str, Any],
    lidar_dir: Path,
    *,
    lidar_name: str = "perception",
    ego_shift_m: np.ndarray | None = None,
) -> tuple[Path, np.ndarray] | None:
    lidar_data = info.get("sensors", {}).get("lidar", {}).get(lidar_name)
    if not isinstance(lidar_data, dict):
        return None
    data_path = lidar_data.get("data_path")
    if not data_path:
        return None
    pcd_path = lidar_dir / Path(str(data_path)).name
    if not pcd_path.exists():
        return None
    points = read_point_cloud_xyz(pcd_path)
    if lidar_name != "perception" and not pcd_path_is_ego_frame(pcd_path):
        points = transform_points(points, np.asarray(lidar_data["extrinsic"], dtype=np.float64))
    if ego_shift_m is not None:
        points = points.copy()
        points += np.asarray(ego_shift_m, dtype=np.float32).reshape(1, 3)
    return pcd_path, points


def image_for_camera(
    cam_data: dict[str, Any],
    *,
    image_root: Path,
    camera: str,
) -> tuple[Path, np.ndarray] | None:
    data_path = cam_data.get("data_path")
    if not data_path:
        return None
    folder = CAMERA_TO_IMAGE_FOLDER.get(camera, camera)
    image_name = Path(str(data_path)).name
    for candidate in (image_root / folder / image_name, image_root / camera / image_name):
        if not candidate.exists():
            continue
        image = cv2.imread(str(candidate))
        if image is not None:
            return candidate, image
    return None


def project_lidar_to_image(
    *,
    points_ego: np.ndarray,
    ego_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image: np.ndarray,
    max_depth_m: float,
    use_filter: bool,
    max_depth_difference_m: float,
    point_stride: int,
    max_points: int | None,
    point_radius: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    if points_ego.size == 0:
        return image.copy()

    points_h = np.concatenate(
        [points_ego, np.ones((points_ego.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_camera = (ego_to_camera @ points_h.T)[:3]
    depth = points_camera[2]

    projected = intrinsic @ points_camera
    uv = projected[:2] / np.clip(projected[2:3], 1e-8, None)

    valid = (
        (depth > 0.01)
        & (depth < float(max_depth_m))
        & np.isfinite(depth)
        & (uv[0] >= 0)
        & (uv[0] < width)
        & (uv[1] >= 0)
        & (uv[1] < height)
    )
    if not valid.any():
        return image.copy()

    xs = np.clip(np.rint(uv[0, valid]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.rint(uv[1, valid]).astype(np.int32), 0, height - 1)
    depths = depth[valid].astype(np.float32)

    order = np.argsort(depths)
    xs, ys, depths = xs[order], ys[order], depths[order]

    _, unique_indices = np.unique(
        np.stack([xs, ys], axis=1),
        axis=0,
        return_index=True,
    )
    xs, ys, depths = xs[unique_indices], ys[unique_indices], depths[unique_indices]

    if use_filter:
        xs, ys, depths = filter_depth_outliers(
            xs=xs,
            ys=ys,
            depths=depths,
            image_size=(height, width),
            max_depth_difference_m=max_depth_difference_m,
        )

    if len(xs) == 0:
        return image.copy()

    point_stride = max(1, int(point_stride))
    if point_stride > 1:
        xs, ys, depths = xs[::point_stride], ys[::point_stride], depths[::point_stride]

    if max_points is not None and len(xs) > int(max_points):
        sample_indices = np.linspace(0, len(xs) - 1, int(max_points), dtype=np.int32)
        xs, ys, depths = xs[sample_indices], ys[sample_indices], depths[sample_indices]

    near = float(np.percentile(depths, 1))
    far = float(np.percentile(depths, 99))
    if far <= near:
        far = near + 1.0
    normalized = np.clip((depths - near) / (far - near), 0.0, 1.0)
    colors = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_PLASMA,
    ).reshape(-1, 3)

    output = image.copy()
    point_radius = max(1, int(point_radius))
    for x, y, color in zip(xs, ys, colors, strict=False):
        cv2.circle(output, (int(x), int(y)), point_radius, color.tolist(), -1)
    return output


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    target_width = max(1, int(width))
    height, current_width = image.shape[:2]
    if current_width == target_width:
        return image.copy()
    scale = target_width / float(current_width)
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def put_panel_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(output.shape[1] / 1200.0, 1.0))
    thickness = max(1, int(round(font_scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 12, 12 + text_height
    cv2.rectangle(
        output,
        (x - 6, y - text_height - 6),
        (x + text_width + 6, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def build_multicamera_shift_compare_canvas(
    *,
    before_by_camera: dict[str, np.ndarray],
    after_by_camera: dict[str, np.ndarray],
    camera_order: list[str],
    tile_width: int,
    title: str = "",
) -> np.ndarray:
    rows = []
    for camera in camera_order:
        before = before_by_camera.get(camera)
        after = after_by_camera.get(camera)
        if before is None or after is None:
            continue
        before_resized = resize_to_width(before, tile_width)
        after_resized = resize_to_width(after, tile_width)
        if before_resized.shape[0] != after_resized.shape[0]:
            after_resized = cv2.resize(
                after_resized,
                (before_resized.shape[1], before_resized.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        before_labeled = put_panel_label(before_resized, f"{camera} | original")
        after_labeled = put_panel_label(after_resized, f"{camera} | shifted {title}".strip())
        rows.append(np.hstack([before_labeled, after_labeled]))
    if not rows:
        raise ValueError("no camera panels available for comparison")
    max_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] == max_width:
            padded_rows.append(row)
            continue
        canvas = np.zeros((row.shape[0], max_width, 3), dtype=row.dtype)
        canvas[:, : row.shape[1]] = row
        padded_rows.append(canvas)
    return np.vstack(padded_rows)


def filter_depth_outliers(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    image_size: tuple[int, int],
    max_depth_difference_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_size
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    depth_map[ys, xs] = depths

    keep = np.ones(len(xs), dtype=bool)
    half_window = 2
    for index, (x, y, depth) in enumerate(zip(xs, ys, depths, strict=False)):
        y0, y1 = max(0, y - half_window), min(height, y + half_window + 1)
        x0, x1 = max(0, x - half_window), min(width, x + half_window + 1)
        if depth > np.min(depth_map[y0:y1, x0:x1]) + max_depth_difference_m:
            keep[index] = False
    return xs[keep], ys[keep], depths[keep]


def visualize_one_pkl(
    *,
    pkl_path: Path,
    label: str,
    lidar_dir: Path,
    image_root: Path,
    output_root: Path,
    camera: str,
    start: int,
    end: int | None,
    frame_stride: int,
    max_depth_m: float,
    use_filter: bool,
    max_depth_difference_m: float,
    point_stride: int,
    max_points: int | None,
    point_radius: int,
    lidar_name: str,
    ego_shift_m: np.ndarray,
    save_individual: bool,
) -> dict[int, Path]:
    infos = load_infos(pkl_path)
    frame_end = len(infos) if end is None else min(int(end), len(infos))
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[int, Path] = {}
    processed = 0
    frame_stride = max(1, int(frame_stride))
    for frame_index in range(int(start), frame_end, frame_stride):
        info = infos[frame_index]
        cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
        if not isinstance(cam_data, dict):
            continue

        lidar_payload = lidar_points_for_info(
            info,
            lidar_dir,
            lidar_name=lidar_name,
            ego_shift_m=ego_shift_m,
        )
        image_payload = image_for_camera(cam_data, image_root=image_root, camera=camera)
        if lidar_payload is None or image_payload is None:
            continue

        _pcd_path, points_ego = lidar_payload
        image_path, image = image_payload
        intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
        ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)

        result = project_lidar_to_image(
            points_ego=points_ego,
            ego_to_camera=ego_to_camera,
            intrinsic=intrinsic,
            image=image,
            max_depth_m=max_depth_m,
            use_filter=use_filter,
            max_depth_difference_m=max_depth_difference_m,
            point_stride=point_stride,
            max_points=max_points,
            point_radius=point_radius,
        )
        output_path = output_dir / f"{camera}_{frame_index:06d}_{image_path.name}"
        if save_individual:
            cv2.imwrite(str(output_path), result)
        rendered[frame_index] = output_path
        processed += 1
        if processed % 20 == 0:
            print(f"[{label}] rendered {processed} frames")

    print(f"[{label}] done: {processed} images -> {output_dir}")
    return rendered


def write_comparisons(
    *,
    before: dict[int, Path],
    after: dict[int, Path],
    output_root: Path,
) -> None:
    compare_dir = output_root / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    common_frames = sorted(set(before) & set(after))
    for frame_index in common_frames:
        before_img = cv2.imread(str(before[frame_index]))
        after_img = cv2.imread(str(after[frame_index]))
        if before_img is None or after_img is None:
            continue
        if before_img.shape[:2] != after_img.shape[:2]:
            after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
        joined = np.concatenate([before_img, after_img], axis=1)
        cv2.imwrite(str(compare_dir / f"compare_{frame_index:06d}.jpg"), joined)
    print(f"[compare] done: {len(common_frames)} images -> {compare_dir}")


def write_compare_direct(
    *,
    before_pkl: Path,
    after_pkl: Path,
    lidar_dir: Path,
    image_root: Path,
    output_root: Path,
    camera: str,
    start: int,
    end: int | None,
    frame_stride: int,
    max_depth_m: float,
    use_filter: bool,
    max_depth_difference_m: float,
    point_stride: int,
    max_points: int | None,
    point_radius: int,
    lidar_name: str,
    before_ego_shift_m: np.ndarray,
    after_ego_shift_m: np.ndarray,
) -> None:
    before_infos = load_infos(before_pkl)
    after_infos = load_infos(after_pkl)
    frame_end = min(len(before_infos), len(after_infos)) if end is None else min(
        int(end),
        len(before_infos),
        len(after_infos),
    )
    compare_dir = output_root / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    frame_stride = max(1, int(frame_stride))
    for frame_index in range(int(start), frame_end, frame_stride):
        before_cam = before_infos[frame_index].get("sensors", {}).get("cams", {}).get(camera)
        after_cam = after_infos[frame_index].get("sensors", {}).get("cams", {}).get(camera)
        if not isinstance(before_cam, dict) or not isinstance(after_cam, dict):
            continue

        lidar_payload = lidar_points_for_info(
            before_infos[frame_index],
            lidar_dir,
            lidar_name=lidar_name,
            ego_shift_m=before_ego_shift_m,
        )
        after_lidar_payload = lidar_points_for_info(
            before_infos[frame_index],
            lidar_dir,
            lidar_name=lidar_name,
            ego_shift_m=after_ego_shift_m,
        )
        image_payload = image_for_camera(before_cam, image_root=image_root, camera=camera)
        if lidar_payload is None or after_lidar_payload is None or image_payload is None:
            continue

        _pcd_path, points_ego = lidar_payload
        _after_pcd_path, after_points_ego = after_lidar_payload
        image_path, image = image_payload

        before_image = project_lidar_to_image(
            points_ego=points_ego,
            ego_to_camera=np.asarray(before_cam["extrinsic"], dtype=np.float64),
            intrinsic=np.asarray(before_cam["cam_intrinsic"], dtype=np.float64),
            image=image,
            max_depth_m=max_depth_m,
            use_filter=use_filter,
            max_depth_difference_m=max_depth_difference_m,
            point_stride=point_stride,
            max_points=max_points,
            point_radius=point_radius,
        )
        after_image = project_lidar_to_image(
            points_ego=after_points_ego,
            ego_to_camera=np.asarray(after_cam["extrinsic"], dtype=np.float64),
            intrinsic=np.asarray(after_cam["cam_intrinsic"], dtype=np.float64),
            image=image,
            max_depth_m=max_depth_m,
            use_filter=use_filter,
            max_depth_difference_m=max_depth_difference_m,
            point_stride=point_stride,
            max_points=max_points,
            point_radius=point_radius,
        )
        joined = np.concatenate([before_image, after_image], axis=1)
        cv2.imwrite(str(compare_dir / f"compare_{frame_index:06d}_{image_path.name}"), joined)
        processed += 1
        if processed % 20 == 0:
            print(f"[compare] rendered {processed} frames")

    print(f"[compare] done: {processed} images -> {compare_dir}")


def write_multicamera_shift_compare_direct(
    *,
    pkl_path: Path,
    lidar_dir: Path,
    image_root: Path,
    output_root: Path,
    cameras: list[str],
    start: int,
    end: int | None,
    frame_stride: int,
    sample_ratio: float,
    max_depth_m: float,
    use_filter: bool,
    max_depth_difference_m: float,
    point_stride: int,
    max_points: int | None,
    point_radius: int,
    lidar_name: str,
    before_ego_shift_m: np.ndarray,
    after_ego_shift_m: np.ndarray,
    tile_width: int,
) -> None:
    infos = load_infos(pkl_path)
    frame_end = len(infos) if end is None else min(int(end), len(infos))
    frame_indices = list(range(int(start), frame_end, max(1, int(frame_stride))))
    if 0.0 < float(sample_ratio) < 1.0:
        sample_step = max(1, int(round(1.0 / float(sample_ratio))))
        frame_indices = frame_indices[::sample_step]

    compare_dir = output_root / "multicamera_shift_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"dx={after_ego_shift_m[0] - before_ego_shift_m[0]:+.2f}m "
        f"dy={after_ego_shift_m[1] - before_ego_shift_m[1]:+.2f}m "
        f"dz={after_ego_shift_m[2] - before_ego_shift_m[2]:+.2f}m"
    )

    processed = 0
    for frame_index in frame_indices:
        info = infos[frame_index]
        lidar_payload = lidar_points_for_info(
            info,
            lidar_dir,
            lidar_name=lidar_name,
            ego_shift_m=before_ego_shift_m,
        )
        shifted_payload = lidar_points_for_info(
            info,
            lidar_dir,
            lidar_name=lidar_name,
            ego_shift_m=after_ego_shift_m,
        )
        if lidar_payload is None or shifted_payload is None:
            continue
        _pcd_path, points_ego = lidar_payload
        _shifted_pcd_path, shifted_points_ego = shifted_payload

        before_by_camera: dict[str, np.ndarray] = {}
        after_by_camera: dict[str, np.ndarray] = {}
        for camera in cameras:
            cam_data = info.get("sensors", {}).get("cams", {}).get(camera)
            if not isinstance(cam_data, dict):
                continue
            image_payload = image_for_camera(cam_data, image_root=image_root, camera=camera)
            if image_payload is None:
                continue
            _image_path, image = image_payload
            intrinsic = np.asarray(cam_data["cam_intrinsic"], dtype=np.float64)
            ego_to_camera = np.asarray(cam_data["extrinsic"], dtype=np.float64)
            before_by_camera[camera] = project_lidar_to_image(
                points_ego=points_ego,
                ego_to_camera=ego_to_camera,
                intrinsic=intrinsic,
                image=image,
                max_depth_m=max_depth_m,
                use_filter=use_filter,
                max_depth_difference_m=max_depth_difference_m,
                point_stride=point_stride,
                max_points=max_points,
                point_radius=point_radius,
            )
            after_by_camera[camera] = project_lidar_to_image(
                points_ego=shifted_points_ego,
                ego_to_camera=ego_to_camera,
                intrinsic=intrinsic,
                image=image,
                max_depth_m=max_depth_m,
                use_filter=use_filter,
                max_depth_difference_m=max_depth_difference_m,
                point_stride=point_stride,
                max_points=max_points,
                point_radius=point_radius,
            )
        if not before_by_camera:
            continue
        canvas = build_multicamera_shift_compare_canvas(
            before_by_camera=before_by_camera,
            after_by_camera=after_by_camera,
            camera_order=cameras,
            tile_width=tile_width,
            title=title,
        )
        cv2.imwrite(str(compare_dir / f"shift_compare_{frame_index:06d}.jpg"), canvas)
        processed += 1
        if processed % 10 == 0:
            print(f"[multicamera_shift_compare] rendered {processed} frames")

    print(f"[multicamera_shift_compare] done: {processed} images -> {compare_dir}")


def parse_vector3(value: str | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("shift must contain exactly 3 comma-separated values, e.g. 0,0.8,0")
    return np.asarray([float(item) for item in parts], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize L29 perception LiDAR projection before/after optimized PKL "
            "for center_camera_fov30 by default."
        )
    )
    parser.add_argument("--before-pkl", type=Path, required=True)
    parser.add_argument("--after-pkl", type=Path, required=True)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--lidar-name", default="perception")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="center_camera_fov30")
    parser.add_argument(
        "--cameras",
        default=(
            "center_camera_fov120,center_camera_fov30,left_front_camera,right_front_camera,"
            "left_rear_camera,right_rear_camera,rear_camera"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--max-dif", type=float, default=1.0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--ego-shift", default="0,0,0")
    parser.add_argument("--before-ego-shift")
    parser.add_argument("--after-ego-shift")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--multi-camera-shift-compare", action="store_true")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--tile-width", type=int, default=960)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    ego_shift = parse_vector3(args.ego_shift)
    before_ego_shift = (
        parse_vector3(args.before_ego_shift) if args.before_ego_shift else ego_shift
    )
    after_ego_shift = parse_vector3(args.after_ego_shift) if args.after_ego_shift else ego_shift

    if args.multi_camera_shift_compare:
        write_multicamera_shift_compare_direct(
            pkl_path=args.before_pkl,
            lidar_dir=args.lidar_dir,
            image_root=args.image_root,
            output_root=args.output_dir,
            cameras=[item.strip() for item in str(args.cameras).split(",") if item.strip()],
            start=args.start,
            end=args.end,
            frame_stride=args.frame_stride,
            sample_ratio=args.sample_ratio,
            max_depth_m=args.max_depth,
            use_filter=args.use_filter,
            max_depth_difference_m=args.max_dif,
            point_stride=args.point_stride,
            max_points=args.max_points,
            point_radius=args.point_radius,
            lidar_name=args.lidar_name,
            before_ego_shift_m=before_ego_shift,
            after_ego_shift_m=after_ego_shift,
            tile_width=args.tile_width,
        )
        return

    if args.compare_only:
        write_compare_direct(
            before_pkl=args.before_pkl,
            after_pkl=args.after_pkl,
            lidar_dir=args.lidar_dir,
            image_root=args.image_root,
            output_root=args.output_dir,
            camera=args.camera,
            start=args.start,
            end=args.end,
            frame_stride=args.frame_stride,
            max_depth_m=args.max_depth,
            use_filter=args.use_filter,
            max_depth_difference_m=args.max_dif,
            point_stride=args.point_stride,
            max_points=args.max_points,
            point_radius=args.point_radius,
            lidar_name=args.lidar_name,
            before_ego_shift_m=before_ego_shift,
            after_ego_shift_m=after_ego_shift,
        )
        return

    before = visualize_one_pkl(
        pkl_path=args.before_pkl,
        label="before",
        lidar_dir=args.lidar_dir,
        image_root=args.image_root,
        output_root=args.output_dir,
        camera=args.camera,
        start=args.start,
        end=args.end,
        frame_stride=args.frame_stride,
        max_depth_m=args.max_depth,
        use_filter=args.use_filter,
        max_depth_difference_m=args.max_dif,
        point_stride=args.point_stride,
        max_points=args.max_points,
        point_radius=args.point_radius,
        lidar_name=args.lidar_name,
        ego_shift_m=before_ego_shift,
        save_individual=True,
    )
    after = visualize_one_pkl(
        pkl_path=args.after_pkl,
        label="after",
        lidar_dir=args.lidar_dir,
        image_root=args.image_root,
        output_root=args.output_dir,
        camera=args.camera,
        start=args.start,
        end=args.end,
        frame_stride=args.frame_stride,
        max_depth_m=args.max_depth,
        use_filter=args.use_filter,
        max_depth_difference_m=args.max_dif,
        point_stride=args.point_stride,
        max_points=args.max_points,
        point_radius=args.point_radius,
        lidar_name=args.lidar_name,
        ego_shift_m=after_ego_shift,
        save_individual=True,
    )
    if not args.no_compare:
        write_comparisons(before=before, after=after, output_root=args.output_dir)


if __name__ == "__main__":
    main()
