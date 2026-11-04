"""Per-training-view bilateral grids for the L1 reconstruction branch."""
from __future__ import annotations

from typing import Any
import math

import torch

from lib_bilagrid import BilateralGrid, slice as slice_grid


class BilateralGridTraining:
    def __init__(self, cameras: list[Any], opt: Any, device: torch.device) -> None:
        self.image_names = [camera.image_name for camera in cameras]
        if not self.image_names or len(set(self.image_names)) != len(self.image_names):
            raise ValueError("Bilateral grids require unique, nonempty training camera names.")
        self.view_indices = {name: index for index, name in enumerate(self.image_names)}
        dimensions = (int(opt.bilateral_grid_x), int(opt.bilateral_grid_y), int(opt.bilateral_grid_w))
        if min(dimensions) < 2:
            raise ValueError("Bilateral grid dimensions must be at least 2.")
        if float(opt.bilateral_grid_lr) <= 0 or float(opt.bilateral_grid_tv_weight) < 0:
            raise ValueError("Bilateral grid learning rate must be positive and TV weight nonnegative.")
        self.model = BilateralGrid(len(cameras), *dimensions).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(opt.bilateral_grid_lr),
            eps=float(getattr(opt, "bilateral_grid_adam_eps", 1e-15)),
        )
        self.scheduler = None
        if bool(getattr(opt, "bilateral_grid_lr_schedule", True)):
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=1000),
                torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.01 ** (1.0 / int(opt.iterations))),
            ])
        self.tv_weight = float(opt.bilateral_grid_tv_weight)
        self.detached_training = bool(getattr(opt, "bilateral_grid_detached_training", False))
        self.sample_fraction = float(getattr(opt, "bilateral_grid_sample_fraction", 1.0))
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("Bilateral grid sample fraction must be in (0, 1].")
        if self.sample_fraction != 1.0 and not self.detached_training:
            raise ValueError("Pixel sampling requires bilateral_grid_detached_training.")
        # Keep pixel sampling independent of the camera-sampling RNG.
        self.sampler = torch.Generator(device=device).manual_seed(0)
        self._xy: torch.Tensor | None = None

    def _coordinates(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        if (
            self._xy is None
            or self._xy.shape[:2] != (height, width)
            or self._xy.device != image.device
            or self._xy.dtype != image.dtype
        ):
            # Pixel centers in normalized image coordinates.
            y = (torch.arange(height, device=image.device, dtype=image.dtype) + 0.5) / height
            x = (torch.arange(width, device=image.device, dtype=image.dtype) + 0.5) / width
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self._xy = torch.stack((xx, yy), dim=-1)
        return self._xy

    def correct(self, image: torch.Tensor, image_name: str) -> torch.Tensor:
        """Transform CHW RGB without detaching either RGB gradient path."""
        xy = self._coordinates(image)
        view_index = torch.tensor([self.view_indices[image_name]], device=image.device, dtype=torch.long)
        corrected = slice_grid(self.model, xy, image.permute(1, 2, 0), view_index)["rgb"]
        return corrected.permute(2, 0, 1).contiguous()

    def sampled_l1(
        self, image: torch.Tensor, gt: torch.Tensor, image_name: str,
        alpha_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train only the grid from detached RGB; uniform samples are unbiased for full-image L1."""
        xy = self._coordinates(image).reshape(-1, 2)
        rgb = image.detach().permute(1, 2, 0).reshape(-1, 3)
        target = gt.detach().permute(1, 2, 0).reshape(-1, 3)
        if self.sample_fraction == 1.0:
            indices = torch.arange(rgb.shape[0], device=image.device)
        else:
            count = max(1, math.ceil(rgb.shape[0] * self.sample_fraction))
            indices = torch.randint(rgb.shape[0], (count,), device=image.device, generator=self.sampler)
        view_index = torch.tensor([self.view_indices[image_name]], device=image.device, dtype=torch.long)
        corrected = slice_grid(self.model, xy[indices], rgb[indices], view_index)["rgb"]
        if alpha_mask is not None:
            mask = alpha_mask.detach().permute(1, 2, 0).reshape(-1, 1)
            corrected = corrected * mask[indices]
        return (corrected - target[indices]).abs().mean()

    def capture(self) -> dict[str, Any]:
        return {
            "image_names": self.image_names,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "tv_weight": self.tv_weight,
            "detached_training": self.detached_training,
            "sample_fraction": self.sample_fraction,
            "sampler_state": self.sampler.get_state(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def restore(self, state: dict[str, Any]) -> None:
        if state["image_names"] != self.image_names:
            raise ValueError("Bilateral grid checkpoint camera order does not match this training scene.")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is None:
            # Older checkpoints used a constant learning rate.
            self.scheduler = None
        elif self.scheduler is None:
            raise ValueError("This bilateral grid checkpoint requires the learning-rate schedule.")
        else:
            self.scheduler.load_state_dict(state["scheduler"])
        self.tv_weight = float(state["tv_weight"])
        if state.get("detached_training", False) != self.detached_training or state.get("sample_fraction", 1.0) != self.sample_fraction:
            raise ValueError("Bilateral grid checkpoint sampling settings do not match this run.")
        if "sampler_state" in state:
            self.sampler.set_state(state["sampler_state"].cpu())
