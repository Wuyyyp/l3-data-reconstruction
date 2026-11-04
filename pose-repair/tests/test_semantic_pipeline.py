from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dongfeng_registration.semantic_masks import DynamicMaskConfig
from dongfeng_registration.semantic_pipeline import segment_camera


def test_segment_camera_samples_preview_frames_by_ratio(monkeypatch, tmp_path: Path):
    frames = [
        type(
            "Frame",
            (),
            {
                "index": index,
                "timestamp_ns": 1000 + index,
                "path": tmp_path / f"{1000 + index}.jpg",
            },
        )()
        for index in range(10)
    ]
    for frame in frames:
        cv2.imwrite(str(frame.path), np.zeros((4, 4, 3), dtype=np.uint8))

    seen_flags = []

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return frames

    class StubCache:
        def __init__(self, _config):
            pass

        def load_if_valid(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.MaskCache", StubCache)
    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.create_predictor",
        lambda _config: object(),
    )

    def fake_generate_cached_masks_batch(
        items,
        _predictor,
        _cache,
        force=False,
        write_preview_flags=None,
        image_store=None,
        write_workers=None,
    ):
        assert write_workers == 1
        seen_flags.extend(write_preview_flags or [])
        assert image_store is not None
        (tmp_path / "previews").mkdir(exist_ok=True)
        for _, _, timestamp_ns in items:
            cv2.imwrite(
                str(tmp_path / "previews" / f"{timestamp_ns}.jpg"),
                np.zeros((4, 4, 3), dtype=np.uint8),
            )
        return [
            type(
                "Result",
                (),
                {
                    "mask_path": tmp_path / f"{timestamp_ns}.png",
                    "class_mask_path": tmp_path / f"{timestamp_ns}.classes.png",
                    "metadata_path": tmp_path / f"{timestamp_ns}.json",
                    "preview_path": tmp_path / "previews" / f"{timestamp_ns}.jpg",
                    "cache_hit": False,
                    "instance_count": 1,
                    "dynamic_pixel_ratio": 0.1,
                },
            )()
            for _, _, timestamp_ns in items
        ]

    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.generate_cached_masks_batch",
        fake_generate_cached_masks_batch,
    )

    report = segment_camera(
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        config=DynamicMaskConfig(
            cache_root=tmp_path / "cache",
            batch_size=4,
            preview_ratio=0.1,
        ),
        start_frame=0,
        frame_count=10,
        force=False,
    )

    assert isinstance(report, pd.DataFrame)
    assert len(report) == 10
    assert sum(bool(flag) for flag in seen_flags) == 1
    assert len(list((tmp_path / "previews").glob("*.jpg"))) == 1


def test_segment_camera_writes_reports_under_per_camera_directory(monkeypatch, tmp_path: Path):
    frame = type(
        "Frame",
        (),
        {
            "index": 0,
            "timestamp_ns": 1000,
            "path": tmp_path / "1000.jpg",
        },
    )()
    cv2.imwrite(str(frame.path), np.zeros((4, 4, 3), dtype=np.uint8))

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return [frame]

    class StubCache:
        def __init__(self, _config):
            pass

        def load_if_valid(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.MaskCache", StubCache)
    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.create_predictor",
        lambda _config: object(),
    )

    def fake_generate_cached_masks_batch(
        items,
        _predictor,
        _cache,
        force=False,
        write_preview_flags=None,
        image_store=None,
        write_workers=None,
    ):
        assert write_workers == 1
        assert image_store is not None
        return [
            type(
                "Result",
                (),
                {
                    "mask_path": tmp_path / "1000.png",
                    "class_mask_path": tmp_path / "1000.classes.png",
                    "metadata_path": tmp_path / "1000.json",
                    "preview_path": tmp_path / "1000.jpg",
                    "cache_hit": False,
                    "instance_count": 1,
                    "dynamic_pixel_ratio": 0.1,
                },
            )()
            for _ in items
        ]

    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.generate_cached_masks_batch",
        fake_generate_cached_masks_batch,
    )

    config = DynamicMaskConfig(cache_root=tmp_path / "cache", batch_size=1, preview_ratio=1.0)
    segment_camera(
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        config=config,
        start_frame=0,
        frame_count=1,
        force=False,
    )

    assert (config.cache_root / "reports" / "center_camera_fov30" / "segment_report.csv").exists()
    assert (config.cache_root / "reports" / "center_camera_fov30" / "config.json").exists()


def test_cached_frames_do_not_split_uncached_inference_batches(monkeypatch, tmp_path: Path):
    frames = [
        type(
            "Frame",
            (),
            {
                "index": index,
                "timestamp_ns": 1000 + index,
                "path": tmp_path / f"{1000 + index}.jpg",
            },
        )()
        for index in range(6)
    ]
    for frame in frames:
        cv2.imwrite(str(frame.path), np.zeros((4, 4, 3), dtype=np.uint8))

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return frames

    cached_result = type(
        "Result",
        (),
        {
            "mask_path": tmp_path / "cached-mask.png",
            "class_mask_path": tmp_path / "cached-class-mask.png",
            "metadata_path": tmp_path / "cached.json",
            "preview_path": tmp_path / "cached.jpg",
            "cache_hit": True,
            "instance_count": 1,
            "dynamic_pixel_ratio": 0.1,
        },
    )()

    class StubCache:
        def __init__(self, _config):
            pass

        def load_if_valid(self, _path, _camera, timestamp_ns, _model, **_kwargs):
            return cached_result if timestamp_ns == 1002 else None

    batch_sizes = []

    def fake_generate(items, *_args, **_kwargs):
        batch_sizes.append(len(items))
        return [
            type(
                "Result",
                (),
                {
                    "mask_path": tmp_path / f"{timestamp_ns}.dynamic.png",
                    "class_mask_path": tmp_path / f"{timestamp_ns}.png",
                    "metadata_path": tmp_path / f"{timestamp_ns}.json",
                    "preview_path": tmp_path / f"{timestamp_ns}.jpg",
                    "cache_hit": False,
                    "instance_count": 1,
                    "dynamic_pixel_ratio": 0.1,
                },
            )()
            for _path, _camera, timestamp_ns in items
        ]

    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.MaskCache", StubCache)
    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.create_predictor", lambda _config: object()
    )
    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.generate_cached_masks_batch", fake_generate
    )

    report = segment_camera(
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        config=DynamicMaskConfig(
            cache_root=tmp_path / "cache", batch_size=4, preview_ratio=0.0
        ),
    )

    assert len(report) == 6
    assert batch_sizes == [4, 1]


def test_all_cache_hits_do_not_decode_source_images(monkeypatch, tmp_path: Path):
    frames = [
        type(
            "Frame",
            (),
            {
                "index": index,
                "timestamp_ns": 1000 + index,
                "path": tmp_path / f"{1000 + index}.jpg",
            },
        )()
        for index in range(3)
    ]

    class StubDataset:
        def __init__(self, _root):
            pass

        def frames(self, _camera):
            return frames

    class StubCache:
        def __init__(self, _config):
            pass

        def load_if_valid(self, _path, _camera, timestamp_ns, _model, **_kwargs):
            return type(
                "Result",
                (),
                {
                    "mask_path": tmp_path / f"{timestamp_ns}.dynamic.png",
                    "class_mask_path": tmp_path / f"{timestamp_ns}.png",
                    "metadata_path": tmp_path / f"{timestamp_ns}.json",
                    "preview_path": tmp_path / f"{timestamp_ns}.jpg",
                    "cache_hit": True,
                    "instance_count": 1,
                    "dynamic_pixel_ratio": 0.1,
                },
            )()

    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.Dataset", StubDataset)
    monkeypatch.setattr("dongfeng_registration.semantic_pipeline.MaskCache", StubCache)
    monkeypatch.setattr(
        "dongfeng_registration.semantic_pipeline.ImageStore.from_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source images must not be decoded on a full cache hit")
        ),
    )

    report = segment_camera(
        dataset_root=tmp_path,
        camera="center_camera_fov30",
        config=DynamicMaskConfig(
            cache_root=tmp_path / "cache", preview_ratio=0.0
        ),
    )

    assert len(report) == 3
    assert report["cache_hit"].all()
