from pathlib import Path

import numpy as np

from dongfeng_registration.image_store import ImageStore


def test_image_store_preload_reads_each_image_once(monkeypatch, tmp_path: Path):
    paths = [tmp_path / f"{idx}.jpg" for idx in range(3)]
    for path in paths:
        path.write_bytes(b"stub")

    calls: list[str] = []

    def fake_imread(path, flags=None):
        calls.append(str(path))
        return np.full((2, 3, 3), len(calls), dtype=np.uint8)

    monkeypatch.setattr("dongfeng_registration.image_store.cv2.imread", fake_imread)

    store = ImageStore.from_paths(paths, mode="preload")

    assert len(calls) == 3
    assert store.get(paths[0])[0, 0, 0] == 1
    assert store.get(paths[0])[0, 0, 0] == 1
    assert len(calls) == 3


def test_image_store_lazy_caches_after_first_read(monkeypatch, tmp_path: Path):
    path = tmp_path / "0.jpg"
    path.write_bytes(b"stub")
    calls: list[str] = []

    def fake_imread(path_arg, flags=None):
        calls.append(str(path_arg))
        return np.ones((2, 3, 3), dtype=np.uint8)

    monkeypatch.setattr("dongfeng_registration.image_store.cv2.imread", fake_imread)

    store = ImageStore.from_paths([path], mode="lazy")

    assert calls == []
    assert store.get(path).shape == (2, 3, 3)
    assert store.get(path).shape == (2, 3, 3)
    assert calls == [str(path)]


def test_global_cpu_cache_reuses_resized_image_across_stores(monkeypatch, tmp_path: Path):
    path = tmp_path / "large.jpg"
    path.write_bytes(b"stub")
    calls: list[str] = []

    def fake_imread(path_arg, flags=None):
        calls.append(str(path_arg))
        return np.zeros((2000, 3000, 3), dtype=np.uint8)

    monkeypatch.setattr("dongfeng_registration.image_store.cv2.imread", fake_imread)
    ImageStore.clear_global_cache()

    first = ImageStore.from_paths(
        [path], mode="preload", target_long_side=1600
    )
    second = ImageStore.from_paths(
        [path], mode="lazy", target_long_side=1600
    )

    assert first.get(path).shape[:2] == (1067, 1600)
    assert second.get(path).shape[:2] == (1067, 1600)
    assert first.original_shape(path) == (2000, 3000)
    assert second.original_shape(path) == (2000, 3000)
    assert calls == [str(path)]
    assert ImageStore.global_cache_stats()["disk_reads"] == 1
    assert ImageStore.global_cache_stats()["global_hits"] >= 1


def test_global_cpu_cache_is_shared_by_preflight_semantic_and_matching_stores(
    monkeypatch, tmp_path: Path
):
    paths = [tmp_path / f"{idx}.jpg" for idx in range(2)]
    for path in paths:
        path.write_bytes(b"stub")
    calls: list[str] = []

    def fake_imread(path_arg, flags=None):
        calls.append(str(path_arg))
        return np.zeros((1800, 2400, 3), dtype=np.uint8)

    monkeypatch.setattr("dongfeng_registration.image_store.cv2.imread", fake_imread)
    ImageStore.clear_global_cache()

    preflight_store = ImageStore.from_paths(
        paths, mode="preload", target_long_side=1600
    )
    semantic_store = ImageStore.from_paths(
        paths, mode="lazy", target_long_side=1600
    )
    matching_store = ImageStore.from_paths(
        paths, mode="lazy", target_long_side=1600
    )

    for store in (preflight_store, semantic_store, matching_store):
        for path in paths:
            assert store.get(path).shape[:2] == (1200, 1600)

    assert len(calls) == len(paths)
    stats = ImageStore.global_cache_stats()
    assert stats["disk_reads"] == len(paths)
    assert stats["global_hits"] >= len(paths) * 2


def test_global_cpu_cache_evicts_old_images_at_configured_limit(
    monkeypatch, tmp_path: Path
):
    paths = [tmp_path / f"{idx}.jpg" for idx in range(3)]
    for path in paths:
        path.write_bytes(b"stub")
    calls: list[str] = []

    def fake_imread(path_arg, flags=None):
        calls.append(str(path_arg))
        return np.full((2, 3, 3), len(calls), dtype=np.uint8)

    monkeypatch.setattr("dongfeng_registration.image_store.cv2.imread", fake_imread)
    ImageStore.clear_global_cache()

    store = ImageStore.from_paths(
        paths,
        mode="preload",
        global_cache=True,
        global_max_items=2,
    )

    stats = ImageStore.global_cache_stats()
    assert stats["cached_items"] == 2
    assert stats["evictions"] == 1

    store.get(paths[0])

    assert calls == [str(path) for path in paths] + [str(paths[0])]
    assert ImageStore.global_cache_stats()["cached_items"] == 2
