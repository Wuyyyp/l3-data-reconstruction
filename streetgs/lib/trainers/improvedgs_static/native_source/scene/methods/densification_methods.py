"""
Small preprocessing helpers for densification.

ImprovedGS needs edge-aware scores during densification. This module computes
simple edge maps from training images and normalizes rasterizer statistics before
they are used as Gaussian importance scores.
"""
from typing import Any

import torch
import torch.nn.functional as F


FIND_EDGES_KERNEL = torch.tensor(
    [[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]],
    dtype=torch.float32,
).unsqueeze(0)


def normalize_to_unit_range(value_tensor: torch.Tensor) -> torch.Tensor:
    """
        Normalize a tensor into `[0, 1]` while handling NaN and infinite values.
    """
    sanitized = torch.nan_to_num(value_tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if sanitized.numel() == 0:
        return sanitized
    min_value = sanitized.amin()
    max_value = sanitized.amax()
    scale = max_value - min_value
    if float(scale.item()) <= 0.0:
        return torch.zeros_like(sanitized)
    return (sanitized - min_value) / scale


def compute_edge_map(image: torch.Tensor) -> torch.Tensor:
    """
        Compute a grayscale edge map from a training image.

        The image is converted to an 8-bit grayscale approximation and filtered
        with a small Laplacian-style kernel. The result becomes pixel weights for
        ImprovedGS edge-aware scoring.
    """
    image = image[:3].detach().unsqueeze(0).to(torch.float32)
    rgb_255 = torch.round(image * 255.0)
    grayscale_uint8 = torch.round(
        (
            299.0 * rgb_255[:, 0:1]
            + 587.0 * rgb_255[:, 1:2]
            + 114.0 * rgb_255[:, 2:3]
        ) / 1000.0
    )
    edge_kernel = FIND_EDGES_KERNEL.to(device=image.device, dtype=torch.float32)
    edge_response = F.conv2d(grayscale_uint8, edge_kernel, padding=1)
    edge_map = torch.clamp(edge_response, min=0.0, max=255.0) / 255.0
    return normalize_to_unit_range(edge_map).squeeze(0).squeeze(0)


def prepare_edge_maps(train_cameras: list[Any], opt: Any) -> list[torch.Tensor]:
    """
        Precompute one edge map per training camera for later scoring calls.
    """
    del opt
    return [compute_edge_map(camera.original_image) for camera in train_cameras]
