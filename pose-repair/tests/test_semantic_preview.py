from pathlib import Path

import cv2
import numpy as np

from dongfeng_registration.semantic_masks import (
    DynamicMaskConfig,
    InstancePrediction,
    MaskCache,
    generate_cached_mask,
)


class PreviewPredictor:
    model_id = "preview-model"

    def predict(self, image):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[3:12, 5:18] = True
        return [InstancePrediction("car", 0.91, mask)]


def test_preview_is_side_by_side_and_preserves_inspection_metadata(tmp_path: Path):
    image_path = tmp_path / "1.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    result = generate_cached_mask(
        image_path,
        "camera",
        1,
        PreviewPredictor(),
        MaskCache(DynamicMaskConfig(cache_root=tmp_path / "cache", dilation_pixels=0)),
    )
    preview = cv2.imread(str(result.preview_path))
    assert preview.shape[:2] == (20, 60)
