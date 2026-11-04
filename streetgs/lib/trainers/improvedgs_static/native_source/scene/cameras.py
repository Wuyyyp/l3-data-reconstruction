"""
Camera objects used by rendering and training.

Camera loads image tensors, optional inverse-depth maps, alpha masks, and camera
matrices. MiniCam is a smaller camera view for code paths that only need
projection data.
"""
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2


def _resolve_camera_device(data_device: str) -> torch.device:
    """
        Convert a device string to torch.device, falling back to CUDA on failure.
    """
    try:
        return torch.device(data_device)
    except Exception as e:
        print(e)
        print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        return torch.device("cuda")


def _build_alpha_mask(
    resized_image_rgb: torch.Tensor,
    data_device: torch.device,
    train_test_exp: bool,
    is_test_dataset: bool,
    is_test_view: bool,
) -> torch.Tensor:
    """
        Build the alpha mask used to ignore invalid image regions.

        RGBA images provide the base mask. In train/test exposure experiments,
        one half of the image is masked depending on whether the view belongs to
        the train or test side.
    """
    if resized_image_rgb.shape[0] == 4:
        alpha_mask = resized_image_rgb[3:4, ...].to(data_device)
    else:
        alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(data_device))

    if train_test_exp and is_test_view:
        if is_test_dataset:
            alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
        else:
            alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0
    return alpha_mask


class Camera(nn.Module):
    """
        Full camera record with image data and projection matrices.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        colmap_id: int,
        R: NDArray[np.floating],
        T: NDArray[np.floating],
        FoVx: float,
        FoVy: float,
        depth_params: dict[str, float] | None,
        image: PILImage.Image,
        invdepthmap: NDArray[np.floating] | None,
        image_name: str,
        uid: int,
        trans: NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
        train_test_exp: bool = False,
        is_test_dataset: bool = False,
        is_test_view: bool = False,
    ) -> None:
        """
            Load image/depth tensors and precompute transforms for rendering.
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.data_device = _resolve_camera_device(data_device)

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = _build_alpha_mask(
            resized_image_rgb,
            self.data_device,
            train_test_exp,
            is_test_dataset,
            is_test_view,
        )

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    """
        Lightweight camera view that stores only projection-related fields.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ) -> None:
        """
            Store projection fields and derive the camera center from the view matrix.
        """
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
