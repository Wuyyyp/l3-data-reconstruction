from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


_GLOBAL_IMAGE_CACHE: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
_GLOBAL_IMAGE_CACHE_SOURCE_SHAPES: dict[tuple[object, ...], tuple[int, int]] = {}
_GLOBAL_IMAGE_CACHE_STATS = {
    "disk_reads": 0,
    "global_hits": 0,
    "cached_bytes": 0,
    "evictions": 0,
}


def image_store_options(config: object) -> dict[str, object]:
    """Resolve the shared image-cache settings from a pipeline or stage config."""
    nested = getattr(config, "image_cache", None)
    if nested is not None:
        enabled = bool(getattr(nested, "enabled", True))
        mode = str(getattr(nested, "mode", "preload"))
        target_long_side = int(getattr(nested, "max_long_side", 1600))
        global_max_items = getattr(nested, "max_items", 1024)
    else:
        enabled = bool(getattr(config, "image_cache_enabled", True))
        mode = str(getattr(config, "image_cache_mode", "preload"))
        value = getattr(config, "image_cache_max_long_side", 1600)
        target_long_side = None if value is None else int(value)
        global_max_items = getattr(config, "image_cache_max_items", 1024)
    if not enabled:
        return {
            "mode": "stream",
            "target_long_side": None,
            "global_cache": False,
            "global_max_items": None,
        }
    return {
        "mode": mode,
        "target_long_side": target_long_side,
        "global_cache": True,
        "global_max_items": (
            None if global_max_items is None else int(global_max_items)
        ),
    }


class ImageStore:
    """Shared CPU image cache for repeated segmentation and matching access."""

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        mode: str = "lazy",
        flags: int = cv2.IMREAD_COLOR,
        target_long_side: int | None = None,
        global_cache: bool = True,
        global_max_items: int | None = 1024,
    ):
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"lazy", "preload", "stream"}:
            raise ValueError("image store mode must be 'lazy', 'preload', or 'stream'")
        if target_long_side is not None and int(target_long_side) < 1:
            raise ValueError("target_long_side must be positive when provided")
        self.mode = normalized_mode
        self.flags = flags
        self.paths = [Path(path) for path in paths]
        self.target_long_side = (
            None if target_long_side is None else int(target_long_side)
        )
        self.global_cache = bool(global_cache)
        self.global_max_items = (
            None if global_max_items is None else int(global_max_items)
        )
        if self.global_max_items is not None and self.global_max_items < 1:
            raise ValueError("global_max_items must be positive when provided")
        self._cache: dict[tuple[object, ...], np.ndarray] = {}
        self._source_shapes: dict[tuple[object, ...], tuple[int, int]] = {}
        if self.mode == "preload":
            for path in self.paths:
                self.get(path)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[Path],
        *,
        mode: str = "lazy",
        flags: int = cv2.IMREAD_COLOR,
        target_long_side: int | None = None,
        global_cache: bool = True,
        global_max_items: int | None = 1024,
    ) -> "ImageStore":
        return cls(
            paths,
            mode=mode,
            flags=flags,
            target_long_side=target_long_side,
            global_cache=global_cache,
            global_max_items=global_max_items,
        )

    @classmethod
    def clear_global_cache(cls) -> None:
        _GLOBAL_IMAGE_CACHE.clear()
        _GLOBAL_IMAGE_CACHE_SOURCE_SHAPES.clear()
        _GLOBAL_IMAGE_CACHE_STATS.update(
            {
                "disk_reads": 0,
                "global_hits": 0,
                "cached_bytes": 0,
                "evictions": 0,
            }
        )

    @classmethod
    def global_cache_stats(cls) -> dict[str, int]:
        return {
            **{key: int(value) for key, value in _GLOBAL_IMAGE_CACHE_STATS.items()},
            "cached_items": len(_GLOBAL_IMAGE_CACHE),
        }

    def _store_global(
        self,
        key: tuple[object, ...],
        image: np.ndarray,
        source_shape: tuple[int, int],
    ) -> None:
        _GLOBAL_IMAGE_CACHE[key] = image
        _GLOBAL_IMAGE_CACHE.move_to_end(key)
        _GLOBAL_IMAGE_CACHE_SOURCE_SHAPES[key] = source_shape
        _GLOBAL_IMAGE_CACHE_STATS["cached_bytes"] += int(image.nbytes)
        if self.global_max_items is None:
            return
        while len(_GLOBAL_IMAGE_CACHE) > self.global_max_items:
            evicted_key, evicted_image = _GLOBAL_IMAGE_CACHE.popitem(last=False)
            _GLOBAL_IMAGE_CACHE_SOURCE_SHAPES.pop(evicted_key, None)
            _GLOBAL_IMAGE_CACHE_STATS["cached_bytes"] -= int(evicted_image.nbytes)
            _GLOBAL_IMAGE_CACHE_STATS["evictions"] += 1

    def _cache_key(self, path: Path) -> tuple[object, ...]:
        stat = path.stat()
        return (
            str(path.resolve()),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(self.flags),
            self.target_long_side,
        )

    def _read(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        image = cv2.imread(str(path), self.flags)
        if image is None:
            raise ValueError(f"image could not be read: {path}")
        source_shape = (int(image.shape[0]), int(image.shape[1]))
        _GLOBAL_IMAGE_CACHE_STATS["disk_reads"] += 1
        if self.target_long_side is not None:
            height, width = image.shape[:2]
            long_side = max(height, width)
            if long_side > self.target_long_side:
                scale = self.target_long_side / float(long_side)
                resized_width = max(1, int(round(width * scale)))
                resized_height = max(1, int(round(height * scale)))
                image = cv2.resize(
                    image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
        return image, source_shape

    def get(self, path: Path) -> np.ndarray:
        normalized = Path(path)
        key = self._cache_key(normalized)
        if self.mode == "stream":
            image, source_shape = self._read(normalized)
            self._source_shapes[key] = source_shape
            return image
        if self.global_cache:
            image = _GLOBAL_IMAGE_CACHE.get(key)
            if image is not None:
                _GLOBAL_IMAGE_CACHE.move_to_end(key)
                _GLOBAL_IMAGE_CACHE_STATS["global_hits"] += 1
                source_shape = _GLOBAL_IMAGE_CACHE_SOURCE_SHAPES.get(
                    key, (int(image.shape[0]), int(image.shape[1]))
                )
                self._source_shapes[key] = source_shape
                return image
            image, source_shape = self._read(normalized)
            self._source_shapes[key] = source_shape
            self._store_global(key, image, source_shape)
            return image
        image = self._cache.get(key)
        if image is None:
            image, source_shape = self._read(normalized)
            self._source_shapes[key] = source_shape
            self._cache[key] = image
        return image

    def original_shape(self, path: Path) -> tuple[int, int]:
        """Return the source image shape before optional cache resizing."""
        key = self._cache_key(Path(path))
        source_shape = self._source_shapes.get(key)
        if source_shape is None and self.global_cache:
            source_shape = _GLOBAL_IMAGE_CACHE_SOURCE_SHAPES.get(key)
            if source_shape is not None:
                self._source_shapes[key] = source_shape
        if source_shape is None:
            self.get(Path(path))
            source_shape = self._source_shapes.get(key)
        if source_shape is None:
            raise RuntimeError(f"source image shape is unavailable for {path}")
        return source_shape
