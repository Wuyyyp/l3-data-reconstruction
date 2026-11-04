from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

import dongfeng_registration.dynamic_object_compensation as dynamic_compensation_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compensate_dynamic_raw_lidars as raw_compensation_module  # noqa: E402
from export_dynamic_box_shapefiles import boxes_for_frame  # noqa: E402
from compensate_dynamic_raw_lidars import (  # noqa: E402
    _box_payload_for_lidar,
    output_lidar_data,
    _reference_time_seconds_with_offset,
)

from dongfeng_registration.dynamic_object_compensation import (
    IcpMotionConfig,
    ObjectIcpMotionCandidate,
    PcdCloud,
    select_object_motion_velocity,
    box_corners_ego,
    build_object_tracks,
    box_corners_from_transform,
    box_correction_delta_ego,
    box_edges_polylinez,
    box_transform_ego,
    compensate_dynamic_objects_for_frame,
    compensate_points_between_box_poses,
    dynamic_objects_for_info,
    estimate_icp_corrected_motion,
    interpolate_transform,
    interpolate_relative_transform,
    pcd_path_is_ego_frame,
    points_in_box_mask,
    read_pcd_cloud,
    reference_time_seconds_from_cameras,
    split_indices_by_timestamp_clusters,
    timestamp_to_seconds,
    write_polylinez_shapefile,
    write_pcd_cloud,
)


def test_output_lidar_data_uses_configured_single_lidar_when_perception_is_absent():
    front_lidar = {"data_path": "lidar/front_lidar_compensation/000001.pcd"}

    assert output_lidar_data(
        {"front_lidar": front_lidar},
        lidar_names=("front_lidar",),
    ) == ("front_lidar", front_lidar)


def _write_tiny_binary_pcd(path: Path, points: np.ndarray, timestamps: np.ndarray) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("timestamp", "<f8"),
            ("intensity", "<f4"),
        ]
    )
    data = np.zeros(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["timestamp"] = timestamps
    data["intensity"] = np.arange(len(points), dtype=np.float32)
    header = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS x y z timestamp intensity",
            "SIZE 4 4 4 8 4",
            "TYPE F F F F F",
            "COUNT 1 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    )
    path.write_bytes(header.encode("ascii") + data.tobytes())


def _lzf_literal_compress(payload: bytes) -> bytes:
    chunks = []
    for start in range(0, len(payload), 32):
        chunk = payload[start : start + 32]
        chunks.append(bytes([len(chunk) - 1]) + chunk)
    return b"".join(chunks)


def _write_tiny_binary_compressed_pcd_with_split_timestamp(
    path: Path, points: np.ndarray, timestamps_s: np.ndarray
) -> None:
    fields = ["x", "y", "z", "timestamp_nsec", "timestamp_sec", "ring", "intensity"]
    sizes = [4, 4, 4, 4, 4, 4, 4]
    types = ["F", "F", "F", "U4", "U4", "U4", "F"]
    counts = [1, 1, 1, 1, 1, 1, 1]
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("timestamp_nsec", "<u4"),
            ("timestamp_sec", "<u4"),
            ("ring", "<u4"),
            ("intensity", "<f4"),
        ]
    )
    data = np.zeros(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    seconds = np.floor(timestamps_s).astype(np.uint32)
    data["timestamp_sec"] = seconds
    data["timestamp_nsec"] = np.rint((timestamps_s - seconds) * 1.0e9).astype(np.uint32)
    data["ring"] = np.arange(len(points), dtype=np.uint32)
    data["intensity"] = np.arange(len(points), dtype=np.float32)

    soa_columns = []
    for field in fields:
        soa_columns.append(np.asarray(data[field]).tobytes())
    uncompressed = b"".join(soa_columns)
    compressed = _lzf_literal_compress(uncompressed)
    header = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS " + " ".join(fields),
            "SIZE " + " ".join(str(value) for value in sizes),
            "TYPE " + " ".join(types),
            "COUNT " + " ".join(str(value) for value in counts),
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary_compressed",
            "",
        ]
    )
    path.write_bytes(
        header.encode("ascii")
        + np.asarray([len(compressed)], dtype="<u4").tobytes()
        + np.asarray([len(uncompressed)], dtype="<u4").tobytes()
        + compressed
    )


def _raw_compensation_infos(dataset_root: Path) -> list[dict]:
    base_time_s = 1_766_546_000.0
    base_time_ns = int(base_time_s * 1.0e9)
    infos = []
    for frame_index, x in enumerate((0.0, 10.0, 20.0)):
        timestamp_ns = base_time_ns + frame_index * 1_000_000_000
        pcd_name = f"{timestamp_ns}.pcd"
        pcd_path = dataset_root / "lidar" / "front_lidar_compensation" / pcd_name
        pcd_path.parent.mkdir(parents=True, exist_ok=True)
        points = np.asarray(
            [[x, 0.0, 0.0], [x + 0.2, 0.0, 0.0], [x + 0.4, 0.0, 0.0]],
            dtype=np.float32,
        )
        timestamps = np.full(3, base_time_s + frame_index + 0.1, dtype=np.float64)
        _write_tiny_binary_pcd(pcd_path, points, timestamps)
        infos.append(
            {
                "timestamp": timestamp_ns,
                "ego2global_transformation_matrix": np.eye(4).tolist(),
                "sensors": {
                    "cams": {"cam_a": {"timestamp": timestamp_ns + 200_000_000}},
                    "lidar": {
                        "front_lidar": {
                            "data_path": f"lidar/front_lidar_compensation/{pcd_name}",
                            "extrinsic": np.eye(4).tolist(),
                        }
                    },
                },
                "objects": [
                    {
                        "id": "car-1",
                        "type": "Car",
                        "location": [x, 0.0, 0.0],
                        "size": [4.0, 2.0, 2.0],
                        "rotation": [0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    return infos


def test_read_write_binary_pcd_preserves_non_xyz_fields(tmp_path: Path):
    path = tmp_path / "in.pcd"
    out = tmp_path / "out.pcd"
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    timestamps = np.asarray([10.0, 11.0], dtype=np.float64)
    _write_tiny_binary_pcd(path, points, timestamps)

    cloud = read_pcd_cloud(path)
    updated = cloud.with_xyz(points + 1.0)
    write_pcd_cloud(out, updated)

    reread = read_pcd_cloud(out)
    assert np.allclose(reread.xyz, points + 1.0)
    assert np.allclose(reread.data["timestamp"], timestamps)
    assert np.allclose(reread.data["intensity"], [0.0, 1.0])


def test_read_binary_compressed_pcd_and_split_timestamps(tmp_path: Path):
    path = tmp_path / "compressed.pcd"
    points = np.asarray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    timestamps = np.asarray([100.25, 101.75], dtype=np.float64)
    _write_tiny_binary_compressed_pcd_with_split_timestamp(path, points, timestamps)

    cloud = read_pcd_cloud(path)

    assert np.allclose(cloud.xyz, points)
    assert np.allclose(cloud.timestamps, timestamps)
    assert cloud.data["ring"].tolist() == [0, 1]


def test_lzf_decompress_prefers_valid_native_decoder(monkeypatch):
    calls = []

    class NativeLzf:
        @staticmethod
        def decompress(data: bytes, expected_size: int) -> bytes:
            calls.append((data, expected_size))
            return b"native"

    monkeypatch.setattr(
        dynamic_compensation_module,
        "_native_lzf",
        NativeLzf(),
        raising=False,
    )

    assert dynamic_compensation_module.lzf_decompress(b"compressed", 6) == b"native"
    assert calls == [(b"compressed", 6)]


def test_lzf_decompress_falls_back_when_native_decoder_declines(monkeypatch):
    monkeypatch.setattr(
        dynamic_compensation_module,
        "_native_lzf",
        type("NativeLzf", (), {"decompress": staticmethod(lambda _data, _size: None)})(),
        raising=False,
    )
    payload = b"python-fallback"

    assert dynamic_compensation_module.lzf_decompress(
        _lzf_literal_compress(payload), len(payload)
    ) == payload


def test_write_pcd_cloud_streams_without_path_write_bytes(monkeypatch, tmp_path: Path):
    path = tmp_path / "streamed.pcd"
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    timestamps = np.asarray([10.0, 11.0], dtype=np.float64)
    source = tmp_path / "source.pcd"
    _write_tiny_binary_pcd(source, points, timestamps)
    cloud = read_pcd_cloud(source)

    monkeypatch.setattr(
        Path,
        "write_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("write_pcd_cloud must stream instead of building one bytes payload")
        ),
    )

    write_pcd_cloud(path, cloud)
    reread = read_pcd_cloud(path)
    np.testing.assert_array_equal(reread.data, cloud.data)


def test_interpolate_transform_returns_exact_endpoint_without_rotation_normalization():
    transform0 = np.eye(4, dtype=np.float64)
    transform0[0, 0] = 1.0001
    transform0[:3, 3] = [1.0, 2.0, 3.0]
    transform1 = np.eye(4, dtype=np.float64)
    transform1[:3, 3] = [4.0, 5.0, 6.0]

    result0 = interpolate_transform(transform0, transform1, 0.0)
    result1 = interpolate_transform(transform0, transform1, 1.0)

    np.testing.assert_array_equal(result0, transform0)
    np.testing.assert_array_equal(result1, transform1)


def test_pcd_cloud_cache_discards_partial_cache_when_budget_is_exceeded():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    data = np.zeros(4, dtype=dtype)
    cloud = PcdCloud(header_lines=(), header={}, data=data, dtype=dtype)
    cache = raw_compensation_module.PcdCloudCache(max_bytes=data.nbytes + 1)

    assert cache.put((0, "front_lidar"), cloud)
    assert cache.get((0, "front_lidar")) is cloud
    assert not cache.put((1, "front_lidar"), cloud)

    assert not cache.usable
    assert cache.stored_bytes == 0
    assert cache.get((0, "front_lidar")) is None


def test_parallel_track_build_matches_serial_and_populates_reusable_selections(tmp_path: Path):
    infos = _raw_compensation_infos(tmp_path)
    kwargs = {
        "infos": infos,
        "dataset_root": tmp_path,
        "frame_start": 0,
        "frame_end": len(infos),
        "lidar_names": ("front_lidar",),
        "min_object_points": 1,
        "box_margin_m": 0.0,
        "timestamp_cluster_max_gap_s": 0.030,
        "timestamp_cluster_max_gap_fraction": 0.30,
    }
    serial_cache = raw_compensation_module.PcdCloudCache(max_bytes=1_000_000)
    parallel_cache = raw_compensation_module.PcdCloudCache(max_bytes=1_000_000)

    serial = raw_compensation_module._build_raw_lidar_object_tracks(
        **kwargs,
        track_workers=1,
        cloud_cache=serial_cache,
    )
    parallel = raw_compensation_module._build_raw_lidar_object_tracks(
        **kwargs,
        track_workers=2,
        cloud_cache=parallel_cache,
    )

    key = ("car-1", "front_lidar")
    assert [sample.time_s for sample in serial.tracks[key]] == [
        sample.time_s for sample in parallel.tracks[key]
    ]
    assert sorted(serial.selections) == sorted(parallel.selections)
    for selection_key in serial.selections:
        serial_selection = serial.selections[selection_key]
        parallel_selection = parallel.selections[selection_key]
        assert serial_selection.total_point_count == parallel_selection.total_point_count
        assert [cluster.tolist() for cluster in serial_selection.clusters] == [
            cluster.tolist() for cluster in parallel_selection.clusters
        ]
    assert serial_cache.usable
    assert parallel_cache.usable


def test_compensation_reuses_cached_cloud_and_track_selection(monkeypatch, tmp_path: Path):
    infos = _raw_compensation_infos(tmp_path)
    cache = raw_compensation_module.PcdCloudCache(max_bytes=1_000_000)
    build = raw_compensation_module._build_raw_lidar_object_tracks(
        infos=infos,
        dataset_root=tmp_path,
        frame_start=0,
        frame_end=len(infos),
        lidar_names=("front_lidar",),
        min_object_points=1,
        box_margin_m=0.0,
        timestamp_cluster_max_gap_s=0.030,
        timestamp_cluster_max_gap_fraction=0.30,
        track_workers=1,
        cloud_cache=cache,
    )

    monkeypatch.setattr(
        raw_compensation_module,
        "read_pcd_cloud",
        lambda _path: (_ for _ in ()).throw(AssertionError("cached cloud was reread")),
    )
    monkeypatch.setattr(
        raw_compensation_module,
        "points_in_box_mask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached object selection was recomputed")
        ),
    )

    cloud, boxes, per_lidar = raw_compensation_module.compensate_frame_from_raw_lidars(
        infos=infos,
        frame_index=0,
        dataset_root=tmp_path,
        lidar_names=("front_lidar",),
        object_tracks=build.tracks,
        object_selections=build.selections,
        cloud_cache=cache,
        reference_cameras=("cam_a",),
        min_object_points=1,
        box_margin_m=0.0,
        icp_config=IcpMotionConfig(enabled=False),
        timestamp_cluster_max_gap_s=0.030,
        timestamp_cluster_max_gap_fraction=0.30,
        max_motion_translation_m=10.0,
    )

    assert len(cloud.data) == 3
    assert set(per_lidar) == {"front_lidar"}
    assert any(row["status"] == "compensated" for row in boxes["lidar_reports"])


def test_single_lidar_output_uses_hardlink_when_supported(tmp_path: Path):
    source = tmp_path / "fused.pcd"
    destination = tmp_path / "raw" / "front_lidar" / "fused.pcd"
    source.write_bytes(b"identical point cloud")

    method = raw_compensation_module._materialize_single_lidar_output(
        source,
        destination,
    )

    assert method == "hardlink"
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino == source.stat().st_ino


def test_single_lidar_output_falls_back_to_copy(monkeypatch, tmp_path: Path):
    source = tmp_path / "fused.pcd"
    destination = tmp_path / "raw" / "front_lidar" / "fused.pcd"
    source.write_bytes(b"identical point cloud")
    monkeypatch.setattr(
        raw_compensation_module.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    method = raw_compensation_module._materialize_single_lidar_output(
        source,
        destination,
    )

    assert method == "copy"
    assert destination.read_bytes() == source.read_bytes()


def test_points_in_box_mask_respects_yaw_rotation():
    obj = {
        "location": [0.0, 0.0, 0.0],
        "size": [4.0, 2.0, 2.0],
        "rotation": [0.0, 0.0, np.pi / 2.0],
    }
    points = np.asarray(
        [
            [0.0, 1.9, 0.0],  # local x after inverse yaw, inside length
            [1.1, 0.0, 0.0],  # local y after inverse yaw, outside width
            [0.0, 0.0, 1.1],  # outside height
        ],
        dtype=np.float64,
    )

    assert points_in_box_mask(points, obj).tolist() == [True, False, False]


def test_dynamic_objects_for_info_falls_back_to_gt_boxes_when_objects_empty():
    info = {
        "objects": [],
        "gt_boxes": np.asarray([[1.0, 2.0, 3.0, 2.0, 4.0, 1.5, 0.25]], dtype=np.float64),
        "gt_names": ["Car"],
        "ID": [42],
        "vels": [np.asarray([0.1, 0.2, 0.3], dtype=np.float64)],
        "gt_attrs": [{"motion": 1, "ignore": 0}],
        "num_points": [{"top_center_lidar": 12}],
    }

    objects = dynamic_objects_for_info(info)

    assert len(objects) == 1
    assert objects[0]["id"] == 42
    assert objects[0]["type"] == "Car"
    assert objects[0]["location"] == [1.0, 2.0, 3.0]
    assert objects[0]["size"] == [2.0, 4.0, 1.5]
    assert objects[0]["rotation"] == [0.0, 0.0, 0.25]
    assert objects[0]["velocity"] == [0.1, 0.2, 0.3]


def test_dynamic_objects_for_info_accepts_t3_vehicle_labels():
    info = {
        "objects": [],
        "gt_boxes": np.asarray([[1.0, 2.0, 3.0, 2.0, 4.0, 1.5, 0.25]], dtype=np.float64),
        "gt_names": ["VEHICLE_CAR"],
        "ID": [42],
    }

    objects = dynamic_objects_for_info(info)

    assert len(objects) == 1
    assert objects[0]["type"] == "VEHICLE_CAR"


def test_reference_time_uses_mean_of_available_configured_cameras():
    info = {
        "sensors": {
            "cams": {
                "a": {"timestamp": 1_766_546_037_101_000},
                "b": {"timestamp": 1_766_546_037_103_000},
                "c": {"timestamp": 1_766_546_037_100_000},
            }
        }
    }

    assert reference_time_seconds_from_cameras(info, ["a", "b"]) == 1_766_546_037.102


def test_reference_time_offset_is_added_in_seconds():
    info = {
        "sensors": {
            "cams": {
                "a": {"timestamp": 1_766_546_037_101_000},
                "b": {"timestamp": 1_766_546_037_103_000},
            }
        }
    }

    assert _reference_time_seconds_with_offset(info, ["a", "b"], 0.010) == pytest.approx(
        1_766_546_037.112
    )


def test_timestamp_to_seconds_handles_ns_us_and_seconds():
    assert timestamp_to_seconds(1_766_546_037_101_262_000) == 1_766_546_037.101262
    assert timestamp_to_seconds(1_766_546_037_101_098) == 1_766_546_037.101098
    assert timestamp_to_seconds(1_766_546_037.101098) == 1_766_546_037.101098


def test_pcd_path_is_ego_frame_for_compensation_and_perception_dirs():
    assert pcd_path_is_ego_frame(Path("lidar/top_center_lidar_compensation/1.pcd"))
    assert pcd_path_is_ego_frame(Path("perception/1.pcd"))
    assert pcd_path_is_ego_frame(Path("compensated_perception/1.pcd"))
    assert not pcd_path_is_ego_frame(Path("lidar/top_center_lidar/1.pcd"))


def test_split_indices_by_timestamp_clusters_splits_scan_seam_double_hits():
    indices = np.asarray([10, 11, 12, 13, 14, 15])
    timestamps = np.zeros(20, dtype=np.float64)
    timestamps[indices] = [100.000, 100.002, 100.004, 100.096, 100.098, 100.100]

    clusters = split_indices_by_timestamp_clusters(
        indices,
        timestamps,
        min_points=2,
        max_gap_s=0.030,
        max_gap_fraction=0.30,
    )

    assert [cluster.tolist() for cluster in clusters] == [[10, 11, 12], [13, 14, 15]]


def test_split_indices_by_timestamp_clusters_discards_tiny_seam_fragments():
    indices = np.asarray([0, 1, 2, 3])
    timestamps = np.asarray([10.000, 10.001, 10.100, 10.101], dtype=np.float64)

    clusters = split_indices_by_timestamp_clusters(
        indices,
        timestamps,
        min_points=3,
        max_gap_s=0.030,
        max_gap_fraction=0.30,
    )

    assert [cluster.tolist() for cluster in clusters] == []


def test_compensate_points_between_box_poses_moves_points_with_rigid_box_motion():
    source = box_transform_ego(
        {
            "location": [0.0, 0.0, 0.0],
            "size": [4.0, 2.0, 2.0],
            "rotation": [0.0, 0.0, 0.0],
        }
    )
    target = box_transform_ego(
        {
            "location": [2.0, 0.0, 0.0],
            "size": [4.0, 2.0, 2.0],
            "rotation": [0.0, 0.0, 0.0],
        }
    )
    points = np.asarray([[0.5, 0.0, 0.0], [1.0, 0.2, 0.0]], dtype=np.float64)

    moved = compensate_points_between_box_poses(
        points,
        source_box_to_world=source,
        target_box_to_world=target,
        world_to_output_ego=np.eye(4),
        source_ego_to_world=np.eye(4),
    )

    assert np.allclose(moved, points + np.asarray([2.0, 0.0, 0.0]))


def test_interpolate_relative_transform_scales_translation_and_rotation():
    transform = box_transform_ego(
        {
            "location": [4.0, 0.0, 0.0],
            "size": [1.0, 1.0, 1.0],
            "rotation": [0.0, 0.0, np.pi / 2.0],
        }
    )

    halfway = interpolate_relative_transform(transform, 0.5)

    assert np.allclose(halfway[:3, 3], [2.0, 0.0, 0.0])
    moved = (halfway @ np.asarray([1.0, 0.0, 0.0, 1.0]))[:3]
    assert np.allclose(moved[:2], [2.0 + np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None, reason="open3d unavailable")
def test_icp_corrected_motion_refines_box_translation_initialization():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(120, 3)) * np.asarray([1.0, 0.4, 0.3])
    true_motion = np.eye(4)
    true_motion[:3, 3] = [1.0, -0.2, 0.1]
    target = (true_motion @ np.column_stack([source, np.ones(len(source))]).T).T[:, :3]
    box_motion = np.eye(4)
    box_motion[:3, 3] = [0.7, -0.2, 0.1]

    estimate = estimate_icp_corrected_motion(
        source_points_world=source,
        target_points_world=target,
        box_motion_target_from_source=box_motion,
        config=IcpMotionConfig(
            enabled=True,
            min_points=10,
            max_correspondence_distance_m=0.6,
            fitness_min=0.5,
            rmse_max_m=0.05,
            max_iterations=80,
        ),
    )

    assert estimate.used_icp
    assert estimate.status == "icp"
    assert np.allclose(estimate.transform_target_from_source[:3, 3], true_motion[:3, 3], atol=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None, reason="open3d unavailable")
def test_icp_corrected_motion_falls_back_when_fitness_is_low():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(80, 3))
    target = rng.normal(size=(80, 3)) + np.asarray([100.0, 0.0, 0.0])
    box_motion = np.eye(4)

    estimate = estimate_icp_corrected_motion(
        source_points_world=source,
        target_points_world=target,
        box_motion_target_from_source=box_motion,
        config=IcpMotionConfig(
            enabled=True,
            min_points=10,
            max_correspondence_distance_m=0.1,
            fitness_min=0.5,
            rmse_max_m=0.05,
            max_iterations=10,
        ),
    )

    assert not estimate.used_icp
    assert estimate.status == "icp_low_fitness"
    assert np.allclose(estimate.transform_target_from_source, box_motion)


def test_translation_only_icp_refines_translation_without_rotation_delta():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(160, 3)) * np.asarray([1.0, 0.3, 0.2])
    box_motion = np.eye(4)
    box_motion[:3, :3] = box_transform_ego(
        {
            "location": [0.0, 0.0, 0.0],
            "size": [1.0, 1.0, 1.0],
            "rotation": [0.0, 0.0, 0.1],
        }
    )[:3, :3]
    box_motion[:3, 3] = [1.0, 0.0, 0.0]
    true_motion = box_motion.copy()
    true_motion[:3, 3] += [0.25, -0.10, 0.05]
    target = (true_motion @ np.column_stack([source, np.ones(len(source))]).T).T[:, :3]

    estimate = estimate_icp_corrected_motion(
        source_points_world=source,
        target_points_world=target,
        box_motion_target_from_source=box_motion,
        config=IcpMotionConfig(
            enabled=True,
            translation_only=True,
            min_points=10,
            max_correspondence_distance_m=1.0,
            fitness_min=0.1,
            rmse_max_m=0.3,
            max_icp_translation_delta_m=2.0,
            max_icp_rotation_delta_deg=0.0,
            max_iterations=80,
        ),
    )

    assert estimate.used_icp
    assert estimate.status == "icp_translation_only"
    assert np.allclose(estimate.transform_target_from_source[:3, :3], box_motion[:3, :3])
    assert abs(estimate.rotation_delta_from_box_deg) < 1e-9
    assert np.allclose(estimate.transform_target_from_source[:3, 3], true_motion[:3, 3], atol=1e-3)


def test_select_object_motion_velocity_rejects_outlier_icp_candidate():
    candidates = [
        ObjectIcpMotionCandidate(
            object_id="car-1",
            lidar_name="front_lidar",
            transform_target_from_source=np.diag([1.0, 1.0, 1.0, 1.0]),
            source_time_s=10.0,
            target_time_s=11.0,
            fitness=0.9,
            rmse_m=0.1,
            inlier_count=100,
        ),
        ObjectIcpMotionCandidate(
            object_id="car-1",
            lidar_name="right_lidar",
            transform_target_from_source=np.asarray(
                [[1.0, 0.0, 0.0, 1.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            source_time_s=10.0,
            target_time_s=11.0,
            fitness=0.8,
            rmse_m=0.1,
            inlier_count=90,
        ),
        ObjectIcpMotionCandidate(
            object_id="car-1",
            lidar_name="top_center_lidar",
            transform_target_from_source=np.asarray(
                [[1.0, 0.0, 0.0, 9.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            source_time_s=10.0,
            target_time_s=11.0,
            fitness=0.9,
            rmse_m=0.1,
            inlier_count=100,
        ),
    ]
    candidates[0].transform_target_from_source[:3, 3] = [1.0, 0.0, 0.0]

    velocity = select_object_motion_velocity(
        candidates,
        min_candidates=2,
        max_velocity_deviation_mps=2.0,
    )

    assert velocity is not None
    assert velocity.candidate_count == 2
    assert velocity.source == "object_icp_voted"
    assert np.allclose(velocity.velocity_world_mps, [1.05, 0.0, 0.0], atol=0.05)


def test_select_object_motion_velocity_requires_unique_lidar_candidates():
    transform0 = np.eye(4, dtype=np.float64)
    transform0[:3, 3] = [1.0, 0.0, 0.0]
    transform1 = np.eye(4, dtype=np.float64)
    transform1[:3, 3] = [1.1, 0.0, 0.0]

    velocity = select_object_motion_velocity(
        [
            ObjectIcpMotionCandidate(
                object_id="car-1",
                lidar_name="front_lidar",
                transform_target_from_source=transform0,
                source_time_s=10.0,
                target_time_s=11.0,
                fitness=0.9,
                rmse_m=0.1,
                inlier_count=100,
            ),
            ObjectIcpMotionCandidate(
                object_id="car-1",
                lidar_name="front_lidar",
                transform_target_from_source=transform1,
                source_time_s=10.0,
                target_time_s=11.0,
                fitness=0.8,
                rmse_m=0.1,
                inlier_count=90,
            ),
        ],
        min_candidates=2,
        max_velocity_deviation_mps=2.0,
    )

    assert velocity is None


def test_select_object_motion_velocity_best_supported_icp_selects_best_supported_candidate():
    front = np.eye(4, dtype=np.float64)
    front[:3, 3] = [1.0, 0.0, 0.0]
    top = np.eye(4, dtype=np.float64)
    top[:3, 3] = [1.2, 0.0, 0.0]
    outlier = np.eye(4, dtype=np.float64)
    outlier[:3, 3] = [8.0, 0.0, 0.0]

    velocity = select_object_motion_velocity(
        [
            ObjectIcpMotionCandidate(
                object_id="car-1",
                lidar_name="front_lidar",
                transform_target_from_source=front,
                source_time_s=10.0,
                target_time_s=11.0,
                fitness=0.6,
                rmse_m=0.2,
                inlier_count=50,
            ),
            ObjectIcpMotionCandidate(
                object_id="car-1",
                lidar_name="top_center_lidar",
                transform_target_from_source=top,
                source_time_s=10.0,
                target_time_s=11.0,
                fitness=0.9,
                rmse_m=0.1,
                inlier_count=100,
            ),
            ObjectIcpMotionCandidate(
                object_id="car-1",
                lidar_name="right_lidar",
                transform_target_from_source=outlier,
                source_time_s=10.0,
                target_time_s=11.0,
                fitness=0.99,
                rmse_m=0.01,
                inlier_count=1000,
            ),
        ],
        min_candidates=2,
        max_velocity_deviation_mps=2.0,
        mode="best_supported_icp",
    )

    assert velocity is not None
    assert velocity.source == "object_icp_best_supported"
    assert velocity.candidate_count == 2
    assert velocity.candidate_lidars == ("front_lidar", "top_center_lidar")
    assert np.allclose(velocity.velocity_world_mps, [1.2, 0.0, 0.0])


def test_box_corners_ego_use_location_size_and_rotation():
    obj = {
        "location": [10.0, 0.0, 1.0],
        "size": [4.0, 2.0, 2.0],
        "rotation": [0.0, 0.0, 0.0],
    }

    corners = box_corners_ego(obj)

    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [8.0, -1.0, 0.0])
    assert np.allclose(corners.max(axis=0), [12.0, 1.0, 2.0])


def test_box_corners_from_transform_applies_transform_translation():
    transform = np.eye(4)
    transform[:3, 3] = [3.0, 4.0, 5.0]

    corners = box_corners_from_transform([2.0, 2.0, 2.0], transform)

    assert np.allclose(corners.mean(axis=0), [3.0, 4.0, 5.0])


def test_box_polylinez_shapefile_writes_one_record_per_box(tmp_path: Path):
    boxes = [
        {
            "frame": 7,
            "object_id": "a",
            "object_type": "Car",
            "timestamp_s": 12.5,
            "corners": box_corners_ego(
                {
                    "location": [0.0, 0.0, 1.0],
                    "size": [4.0, 2.0, 2.0],
                    "rotation": [0.0, 0.0, 0.0],
                }
            ),
        },
        {
            "frame": 7,
            "object_id": "b",
            "object_type": "Truck",
            "timestamp_s": 12.5,
            "corners": box_corners_ego(
                {
                    "location": [10.0, 0.0, 1.0],
                    "size": [6.0, 2.5, 3.0],
                    "rotation": [0.0, 0.0, 0.2],
                }
            ),
        },
    ]

    shp_path = write_polylinez_shapefile(tmp_path / "boxes.shp", boxes)

    assert shp_path.exists()
    assert shp_path.with_suffix(".shx").exists()
    assert shp_path.with_suffix(".dbf").exists()
    assert int.from_bytes(shp_path.read_bytes()[32:36], byteorder="little", signed=True) == 13
    assert int.from_bytes(shp_path.with_suffix(".shx").read_bytes()[24:28], byteorder="big") == 58
    assert int.from_bytes(shp_path.with_suffix(".dbf").read_bytes()[4:8], byteorder="little") == 2
    assert len(box_edges_polylinez(boxes[0]["corners"])) == 12


def test_box_correction_delta_ego_uses_box_centers():
    before = box_corners_ego(
        {
            "location": [1.0, 2.0, 3.0],
            "size": [4.0, 2.0, 2.0],
            "rotation": [0.0, 0.0, 0.0],
        }
    )
    after = before + np.asarray([0.25, -0.5, 0.1])

    delta = box_correction_delta_ego(before, after)

    assert np.allclose(delta, [0.25, -0.5, 0.1])


def test_box_payload_for_lidar_keeps_lidar_specific_after_box():
    payload = {
        "frame_index": 3,
        "before": [
            {"object_id": "a", "corners": np.zeros((8, 3)).tolist()},
            {"object_id": "b", "corners": np.ones((8, 3)).tolist()},
        ],
        "after": [
            {
                "object_id": "a",
                "lidar_name": "front_lidar",
                "point_count": 5,
                "corners": (np.zeros((8, 3)) + 1.0).tolist(),
            },
            {
                "object_id": "a",
                "lidar_name": "rear_lidar",
                "point_count": 50,
                "corners": (np.zeros((8, 3)) + 2.0).tolist(),
            },
            {
                "object_id": "b",
                "lidar_name": "front_lidar",
                "point_count": 10,
                "corners": (np.zeros((8, 3)) + 3.0).tolist(),
            },
        ],
        "lidar_reports": [
            {"object_id": "a", "lidar_name": "front_lidar"},
            {"object_id": "b", "lidar_name": "front_lidar"},
            {"object_id": "a", "lidar_name": "rear_lidar"},
        ],
    }

    front = _box_payload_for_lidar(payload, "front_lidar")

    assert {row["object_id"] for row in front["lidar_reports"]} == {"a", "b"}
    assert {row["object_id"] for row in front["before"]} == {"a", "b"}
    assert len(front["after"]) == 2
    assert all(row["lidar_name"] == "front_lidar" for row in front["after"])
    assert np.allclose(np.asarray(front["after_by_object_id"]["a"]["corners"]), 1.0)


def test_exported_before_box_is_original_annotation_and_after_uses_reference_time():
    base_time_s = 1_766_546_000.0
    base_time_ns = int(base_time_s * 1_000_000_000)
    obj0 = {
        "id": "car-1",
        "type": "Car",
        "location": [0.0, 0.0, 0.0],
        "size": [4.0, 2.0, 2.0],
        "rotation": [0.0, 0.0, 0.0],
    }
    obj1 = {
        **obj0,
        "location": [10.0, 0.0, 0.0],
    }
    infos = [
        {
            "timestamp": base_time_ns,
            "ego2global_transformation_matrix": np.eye(4).tolist(),
            "sensors": {"cams": {"cam_a": {"timestamp": base_time_ns + 200_000_000}}},
            "objects": [obj0],
        },
        {
            "timestamp": base_time_ns + 1_000_000_000,
            "ego2global_transformation_matrix": np.eye(4).tolist(),
            "sensors": {"cams": {"cam_a": {"timestamp": base_time_ns + 1_200_000_000}}},
            "objects": [obj1],
        },
    ]
    current_cloud_xyz = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    current_point_timestamps = np.asarray(
        [base_time_s + 0.1, base_time_s + 0.1], dtype=np.float64
    )
    next_cloud_xyz = np.asarray([[10.0, 0.0, 0.0], [10.2, 0.0, 0.0]], dtype=np.float32)
    next_point_timestamps = np.asarray(
        [base_time_s + 1.1, base_time_s + 1.1], dtype=np.float64
    )
    tracks = build_object_tracks(
        infos,
        [
            (current_cloud_xyz, current_point_timestamps),
            (next_cloud_xyz, next_point_timestamps),
        ],
        min_object_points=1,
        box_margin_m=0.0,
    )

    before, after, reference_time_s = boxes_for_frame(
        infos=infos,
        tracks=tracks,
        cloud_xyz=current_cloud_xyz,
        point_timestamps=current_point_timestamps,
        frame_index=0,
        reference_cameras=("cam_a",),
        dynamic_only=True,
        min_object_points=1,
        box_margin_m=0.0,
    )

    assert reference_time_s == base_time_s + 0.2
    assert np.allclose(before[0]["corners"], box_corners_ego(obj0))
    assert before[0]["timestamp_s"] == base_time_s
    assert before[0]["status"] == "before"
    assert np.allclose(np.asarray(after[0]["corners"]).mean(axis=0), [1.0, 0.0, 0.0])
    assert after[0]["timestamp_s"] == base_time_s + 0.2
    assert after[0]["status"] == "after_comp"


def test_dynamic_compensation_skips_unphysical_large_box_motion():
    base_time_s = 1_766_546_000.0
    base_time_ns = int(base_time_s * 1_000_000_000)
    obj0 = {
        "id": "car-1",
        "type": "Car",
        "location": [0.0, 0.0, 0.0],
        "size": [4.0, 2.0, 2.0],
        "rotation": [0.0, 0.0, 0.0],
    }
    obj1 = {
        **obj0,
        "location": [100.0, 0.0, 0.0],
    }
    infos = [
        {
            "timestamp": base_time_ns,
            "ego2global_transformation_matrix": np.eye(4).tolist(),
            "sensors": {"cams": {"cam_a": {"timestamp": base_time_ns + 200_000_000}}},
            "objects": [obj0],
        },
        {
            "timestamp": base_time_ns + 1_000_000_000,
            "ego2global_transformation_matrix": np.eye(4).tolist(),
            "sensors": {"cams": {"cam_a": {"timestamp": base_time_ns + 1_200_000_000}}},
            "objects": [obj1],
        },
    ]
    points = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    timestamps = np.asarray([base_time_s + 0.1, base_time_s + 0.1], dtype=np.float64)
    next_points = np.asarray([[100.0, 0.0, 0.0], [100.2, 0.0, 0.0]], dtype=np.float32)
    next_timestamps = np.asarray([base_time_s + 1.1, base_time_s + 1.1], dtype=np.float64)
    tracks = build_object_tracks(
        infos,
        [(points, timestamps), (next_points, next_timestamps)],
        min_object_points=1,
        box_margin_m=0.0,
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("timestamp", "<f8")]
    )
    data = np.zeros(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["timestamp"] = timestamps
    cloud = PcdCloud(header_lines=(), header={}, data=data, dtype=dtype)

    result = compensate_dynamic_objects_for_frame(
        infos=infos,
        frame_index=0,
        cloud=cloud,
        object_tracks=tracks,
        reference_cameras=("cam_a",),
        min_object_points=1,
        box_margin_m=0.0,
        max_motion_translation_m=10.0,
    )

    assert result.reports[0].status == "motion_translation_too_large"
    assert result.reports[0].motion_translation_m > 10.0
    assert result.moved_point_count == 0
    assert np.allclose(result.cloud.xyz, points)
