from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .semantic_masks import build_dynamic_mask_from_class_labels, canonicalize_label


DEFAULT_LIDAR_DYNAMIC_CLASSES = frozenset(
    {
        "animal",
        "bicycle",
        "bus",
        "car",
        "caravan",
        "motorcycle",
        "other_vehicle",
        "person",
        "rider",
        "trailer",
        "train",
        "truck",
    }
)


@dataclass(frozen=True)
class SemanticClassMask:
    labels: np.ndarray
    dynamic_label_ids: frozenset[int]

    def dynamic_pixels(self, *, dilation_pixels: int = 0) -> np.ndarray:
        class_id_to_name = {
            int(class_id): "dynamic" for class_id in self.dynamic_label_ids
        }
        return build_dynamic_mask_from_class_labels(
            self.labels,
            class_id_to_name,
            dynamic_classes={"dynamic"},
            dilation_pixels=dilation_pixels,
        ) > 0


def camera_timestamp_ns_candidates(value: int | float) -> tuple[int, ...]:
    """Return likely nanosecond representations for camera timestamps."""

    timestamp = int(round(float(value)))
    candidates = [timestamp]
    magnitude = abs(timestamp)
    if 1_000_000_000_000 <= magnitude < 1_000_000_000_000_000_000:
        candidates.append(timestamp * 1_000)
    elif 1_000_000_000 <= magnitude < 1_000_000_000_000:
        candidates.append(timestamp * 1_000_000_000)
    return tuple(dict.fromkeys(candidates))


class SemanticClassMaskIndex:
    """Read semantic class masks for a single cached model without image IO."""

    def __init__(
        self,
        cache_root: Path | str,
        *,
        backend: str = "mask2former",
        cache_key: str | None = None,
        dynamic_classes: frozenset[str] = DEFAULT_LIDAR_DYNAMIC_CLASSES,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.backend = str(backend)
        self.dynamic_classes = frozenset(
            canonicalize_label(value) for value in dynamic_classes
        )
        backend_root = self.cache_root / "backends" / self.backend
        if cache_key is None:
            candidates = sorted(path.name for path in backend_root.iterdir() if path.is_dir())
            if len(candidates) != 1:
                raise ValueError(
                    "semantic cache key is ambiguous; pass --semantic-cache-key explicitly"
                )
            cache_key = candidates[0]
        self.cache_key = str(cache_key)
        self.root = backend_root / self.cache_key
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def load(self, camera: str, timestamp: int | float) -> SemanticClassMask | None:
        camera_root = self.root / str(camera)
        for timestamp_ns in camera_timestamp_ns_candidates(timestamp):
            metadata_path = camera_root / "metadata" / f"{timestamp_ns}.json"
            class_mask_path = camera_root / "class_masks" / f"{timestamp_ns}.png"
            if not metadata_path.exists() or not class_mask_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = cv2.imread(str(class_mask_path), cv2.IMREAD_UNCHANGED)
            if labels is None or labels.ndim != 2:
                raise ValueError(f"cannot read semantic class mask: {class_mask_path}")
            label_names = metadata.get("class_id_to_name", {})
            dynamic_ids = frozenset(
                int(class_id)
                for class_id, name in label_names.items()
                if canonicalize_label(str(name)) in self.dynamic_classes
            )
            return SemanticClassMask(
                labels=np.asarray(labels, dtype=np.uint16),
                dynamic_label_ids=dynamic_ids,
            )
        return None


def dynamic_point_mask_from_semantic_views(
    points_ego: np.ndarray,
    *,
    camera_views: Mapping[str, tuple[np.ndarray, np.ndarray, SemanticClassMask]],
    dilation_pixels: int = 0,
    min_depth_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Mark ego-frame LiDAR points that fall in a dynamic semantic pixel.

    A point is removed when any visible camera view classifies its projected pixel
    as dynamic. Points outside every camera image remain untouched.
    """

    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    dynamic = np.zeros(len(points), dtype=bool)
    details: dict[str, dict[str, int]] = {}
    if len(points) == 0:
        return dynamic, details
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    for camera, (ego_to_camera, intrinsic, semantic) in camera_views.items():
        points_camera = (
            np.asarray(ego_to_camera, dtype=np.float64).reshape(4, 4) @ homogeneous.T
        ).T[:, :3]
        depth = points_camera[:, 2]
        projected = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3) @ points_camera.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = projected[0] / projected[2]
            v = projected[1] / projected[2]
        height, width = semantic.labels.shape
        visible = (
            np.isfinite(depth)
            & (depth > float(min_depth_m))
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u < width)
            & (v >= 0.0)
            & (v < height)
        )
        visible_indices = np.flatnonzero(visible)
        hit = np.zeros(len(points), dtype=bool)
        if len(visible_indices):
            xs = np.rint(u[visible_indices]).astype(np.int32)
            ys = np.rint(v[visible_indices]).astype(np.int32)
            xs = np.clip(xs, 0, width - 1)
            ys = np.clip(ys, 0, height - 1)
            dynamic_pixels = semantic.dynamic_pixels(dilation_pixels=dilation_pixels)
            hit[visible_indices] = dynamic_pixels[ys, xs]
        dynamic |= hit
        details[str(camera)] = {
            "projected_points": int(len(visible_indices)),
            "dynamic_points": int(np.count_nonzero(hit)),
        }
    return dynamic, details
