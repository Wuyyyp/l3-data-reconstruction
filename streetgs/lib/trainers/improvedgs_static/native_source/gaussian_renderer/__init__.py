"""
Renderer wrapper around the CUDA Gaussian rasterizer.

This module hides the low-level rasterizer argument layout from training code.
The main path renders normal 3DGS images and keeps gradients when training needs
them. The MiniGS paths call special rasterizer entry points that return extra
statistics for blur-aware splitting and depth back-sampling.
"""
from __future__ import annotations

from typing import Any

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from arguments import GroupParams
from scene.cameras import Camera, MiniCam
from scene.gaussian_model import GaussianModel as GaussianModel3DGS

RenderCamera = Camera | MiniCam


def _apply_trained_exposure(rendered_image: torch.Tensor, pc: Any, viewpoint_camera: RenderCamera) -> torch.Tensor:
    """
        Apply the per-image exposure matrix learned during training.

        Exposure is stored by image name. The helper multiplies RGB by the
        3x3 color transform and adds the learned bias term.
    """
    exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    return (
        torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
        + exposure[:3, 3, None, None]
    )


def _build_screenspace_points(pc: GaussianModel3DGS, track_gradients: bool) -> torch.Tensor:
    """
        Create the 2D placeholder tensor passed to the rasterizer.

        The CUDA backward pass writes gradients into this tensor. Channels 0-1
        are the standard screen-space gradients used by 3DGS; channels 2-3 store
        absolute gradients used by ABSGS and ImprovedGS.
    """
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=track_gradients,
        device=pc.get_xyz.device,
    )
    if track_gradients:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
    return screenspace_points


def _build_minigs_full_proj_transform(viewpoint_camera: RenderCamera) -> torch.Tensor:
    """
        Build the projection matrix expected by the MiniGS reference kernels.

        MiniGS auxiliary and depth outputs depend on the exact projection
        convention. The main renderer keeps the repository's normal projection,
        while this helper recreates the MiniGS reference convention only for
        MiniGS-specific calls.
    """
    znear = float(viewpoint_camera.znear)
    zfar = float(viewpoint_camera.zfar)
    tan_half_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)
    tan_half_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    projection_matrix = torch.zeros(
        (4, 4),
        dtype=viewpoint_camera.world_view_transform.dtype,
        device=viewpoint_camera.world_view_transform.device,
    )
    projection_matrix[0, 0] = 2.0 * znear / (right - left)
    projection_matrix[1, 1] = 2.0 * znear / (top - bottom)
    projection_matrix[0, 2] = (right + left) / (right - left)
    projection_matrix[1, 2] = (top + bottom) / (top - bottom)
    projection_matrix[3, 2] = 1.0
    projection_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    projection_matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    projection_matrix = projection_matrix.transpose(0, 1)

    return (
        viewpoint_camera.world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )


def _build_minigs_raster_settings(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float,
) -> GaussianRasterizationSettings:
    """
        Build rasterizer settings shared by the MiniGS auxiliary and depth paths.

        The settings mirror the main renderer, but use the MiniGS projection
        matrix so returned auxiliary statistics match the reference method.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=_build_minigs_full_proj_transform(viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        inference_only=False,
        pixel_weights=None,
    )


def _build_3dgs_raster_settings(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float,
    inference_only: bool,
    pixel_weights: torch.Tensor | None,
) -> GaussianRasterizationSettings:
    """
        Build rasterizer settings for the main 3DGS render call.

        These settings choose image size, camera transforms, active SH degree,
        debug flags, and optional pixel weights used for edge-aware scoring.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        inference_only=inference_only,
        pixel_weights=pixel_weights,
    )


def render_3dgs(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    use_trained_exp: bool = False,
    track_gradients: bool = True,
    inference_only: bool = False,
    pixel_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
        Render one camera view with the standard 3DGS parameter tensors.

        The function always passes DC color, higher-order SH, scale, rotation,
        and opacity explicitly. During training it returns gradients and
        visibility data for densification; during inference it returns only the
        rendered image to reduce work.
    """
    if inference_only:
        track_gradients = False
    if inference_only and pixel_weights is not None:
        raise ValueError("pixel_weights is not supported when inference_only=True.")

    screenspace_points = _build_screenspace_points(pc, track_gradients)
    if pixel_weights is not None:
        pixel_weights = pixel_weights.to(device=pc.get_xyz.device, dtype=torch.float32)

    raster_settings = _build_3dgs_raster_settings(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
        inference_only,
        pixel_weights,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image, accum_weights = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        dc=pc.get_features_dc,
        shs=pc.get_features_rest,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )

    if use_trained_exp:
        rendered_image = _apply_trained_exposure(rendered_image, pc, viewpoint_camera)

    if inference_only:
        return {
            "render": rendered_image,
        }

    render_pkg = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,
    }
    if accum_weights is not None and accum_weights.numel() > 0:
        render_pkg["accum_weights"] = accum_weights
    return render_pkg


def render(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    use_trained_exp: bool = False,
    track_gradients: bool = True,
    inference_only: bool = False,
    pixel_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
        Common render entry point used by training and evaluation code.

        It currently forwards directly to `render_3dgs`, which keeps call sites
        stable if more render variants are added later.
    """
    return render_3dgs(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        use_trained_exp=use_trained_exp,
        track_gradients=track_gradients,
        inference_only=inference_only,
        pixel_weights=pixel_weights,
    )


def render_minigs_aux(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    use_trained_exp: bool = False,
) -> dict[str, torch.Tensor]:
    """
        Render MiniGS auxiliary statistics for the current view.

        The CUDA path returns projected area and accumulated weights. MiniGS uses
        those values to find blurred or oversized Gaussians before splitting.
    """
    screenspace_points = _build_screenspace_points(pc, track_gradients=False)
    raster_settings = _build_minigs_raster_settings(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii, accum_weights, area_proj, area_max = rasterizer.minigs_forward_aux(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        dc=pc.get_features_dc,
        shs=pc.get_features_rest,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )

    if use_trained_exp:
        rendered_image = _apply_trained_exposure(rendered_image, pc, viewpoint_camera)

    return {
        "render": rendered_image,
        "visibility_filter": radii > 0,
        "radii": radii,
        "accum_weights": accum_weights,
        "area_proj": area_proj,
        "area_max": area_max,
    }


def render_minigs_depth(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
        Render MiniGS depth back-sampling data for one view.

        The returned 3D points, alpha map, and Gaussian indices are used when
        MiniGS periodically rebuilds the Gaussian set from image-space samples.
    """
    screenspace_points = _build_screenspace_points(pc, track_gradients=False)
    raster_settings = _build_minigs_raster_settings(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, out_pts, rendered_depth, accum_alpha, gidx, discriminants, radii = rasterizer.minigs_render_depth(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        dc=pc.get_features_dc,
        shs=pc.get_features_rest,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )
    return {
        "render": rendered_image,
        "out_pts": out_pts,
        "rendered_depth": rendered_depth,
        "accum_alpha": accum_alpha,
        "gidx": gidx,
        "discriminants": discriminants,
        "radii": radii,
    }
