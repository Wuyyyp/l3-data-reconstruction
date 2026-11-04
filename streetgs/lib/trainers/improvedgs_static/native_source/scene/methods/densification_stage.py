"""
Densification stage for all supported training methods.

The training loop calls this module after backpropagation and parameter updates.
It first collects the gradient or auxiliary statistics required by the selected
method, then runs the matching structure update: clone/split, MiniGS blur split,
MCMC relocation, or ImprovedGS budgeted long-axis split.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gaussian_renderer import render, render_minigs_aux, render_minigs_depth
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from scene.methods.densification_methods import normalize_to_unit_range
from scene.training_context import TrainingContext


def _new_gaussian_cuda_bool_mask(gaussians: GaussianModel3DGS) -> torch.Tensor:
    """
        Create an all-false CUDA mask with one entry per current Gaussian.
    """
    return torch.zeros((gaussians.get_xyz.shape[0],), device="cuda", dtype=torch.bool)


def _should_reset_opacity(iteration: int, opacity_reset_interval: int) -> bool:
    """
        Check whether the current iteration lands on an opacity-reset interval.
    """
    return iteration % opacity_reset_interval == 0


def _sample_improvedgs_views(scene: Scene, runtime_state: dict[str, Any], opt: Any) -> tuple[list[Any], list[torch.Tensor]]:
    """
        Pick the training views used for ImprovedGS edge-aware scoring.

        The function keeps a small rotating pool in runtime state so repeated
        scoring calls cover different cameras without rebuilding edge maps.
    """
    edge_maps = runtime_state.get("edge_maps", [])
    if not edge_maps:
        return [], []

    edge_sample_cams = int(opt.edge_sample_cams)
    train_cameras = list(runtime_state.get("train_cameras") or scene.getTrainCameras())
    if edge_sample_cams == -1 or edge_sample_cams >= len(train_cameras):
        return train_cameras, edge_maps.copy()

    if not runtime_state.get("edge_camera_pool"):
        runtime_state["edge_camera_pool"] = train_cameras
        runtime_state["edge_map_pool"] = edge_maps.copy()

    sampled_cameras: list[Any] = []
    sampled_edges: list[torch.Tensor] = []
    sample_count = min(edge_sample_cams, len(runtime_state["edge_camera_pool"]))
    for _ in range(sample_count):
        sampled_cameras.append(runtime_state["edge_camera_pool"].pop())
        sampled_edges.append(runtime_state["edge_map_pool"].pop())
        if not runtime_state["edge_camera_pool"] and len(sampled_cameras) < edge_sample_cams:
            runtime_state["edge_camera_pool"] = train_cameras.copy()
            runtime_state["edge_map_pool"] = edge_maps.copy()
    return sampled_cameras, sampled_edges


def _compute_improvedgs_scores(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    render_state: dict[str, Any],
    iteration: int,
) -> torch.Tensor | None:
    """
        Compute edge-aware Gaussian importance scores for ImprovedGS.

        It renders sampled training views with edge maps as pixel weights, then
        accumulates normalized rasterizer weights for visible Gaussians. Returning
        None lets densification fall back to gradient-only scoring.
    """
    scene = context.scene
    runtime_state = context.runtime_state
    if scene is None or not bool(context.method_config.get("use_eas", True)):
        return None

    edge_maps = runtime_state.get("edge_maps", [])
    if not edge_maps:
        return None

    edge_sample_cams = int(getattr(context.opt, "edge_sample_cams", 10))
    if edge_sample_cams == -1 or (
        iteration % 3000 == 400
        and iteration < 9000
        and edge_sample_cams < len(runtime_state.get("train_cameras") or scene.getTrainCameras())
    ):
        sampled_cameras = list(runtime_state.get("train_cameras") or scene.getTrainCameras())
        sampled_edges = edge_maps.copy()
    else:
        sampled_cameras, sampled_edges = _sample_improvedgs_views(scene, runtime_state, context.opt)

    gaussian_importance = torch.zeros((gaussians.get_xyz.shape[0],), device=gaussians.get_xyz.device, dtype=torch.float32)
    visibility_filter_all = torch.zeros_like(gaussian_importance, dtype=torch.bool)
    for camera, edge_map in zip(sampled_cameras, sampled_edges):
        render_pkg = render(
            camera,
            gaussians,
            context.pipe,
            render_state["bg"],
            track_gradients=False,
            pixel_weights=edge_map,
        )
        accum_weights = render_pkg.get("accum_weights")
        if accum_weights is None or accum_weights.numel() == 0:
            continue
        visibility_filter = render_pkg["visibility_filter"].detach()
        gaussian_importance[visibility_filter] += normalize_to_unit_range(accum_weights)[visibility_filter] / max(
            len(sampled_cameras),
            1,
        )
        visibility_filter_all[visibility_filter] = True
    if not torch.any(visibility_filter_all):
        return None
    return gaussian_importance


def _compute_current_budget(iteration: int, opt: Any) -> int:
    """
        Compute the active Gaussian budget for the current densification step.

        Early iterations use a square-root warmup toward the configured budget,
        which avoids adding too many Gaussians before the model has useful
        gradients.
    """
    budget = int(opt.budget)
    if budget > 0:
        max_budget = budget
    else:
        final_budget = int(opt.final_budget)
        budget_multiplier = float(opt.budget_multiplier)
        max_budget = max(int(final_budget * budget_multiplier), final_budget)
    densify_start = int(opt.densify_from_iter)
    densify_end = int(opt.densify_until_iter) - int(opt.budget_warmup_until_offset)
    if densify_end <= densify_start:
        return max_budget
    progress = (iteration - densify_start) / float(densify_end - densify_start)
    progress = min(max(progress, 0.0), 1.0)
    if progress >= 1.0:
        return max_budget
    return max(int(progress ** 0.5 * max_budget), 1)


def _collect_minigs_densification_stats_3dgs(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    render_state: dict[str, Any],
    bg: torch.Tensor,
) -> torch.Tensor:
    """
        Update MiniGS gradient stats and return the blur mask for split decisions.

        The normal 3DGS gradient accumulator is updated first. A MiniGS auxiliary
        render then marks Gaussians whose projected area is large enough to be
        treated as blurred.
    """
    opt = context.opt
    dataset = context.dataset
    runtime_state = context.runtime_state
    viewpoint_camera = render_state["viewpoint_camera"]
    visibility_filter = render_state["visibility_filter"]
    gaussians.add_densification_stats(render_state["viewspace_point_tensor"], visibility_filter)
    mask_blur = runtime_state.get("minigs_mask_blur")
    if mask_blur is None or mask_blur.shape[0] != gaussians.get_xyz.shape[0]:
        mask_blur = _new_gaussian_cuda_bool_mask(gaussians)
    aux_render_pkg = render_minigs_aux(
        viewpoint_camera,
        gaussians,
        context.pipe,
        runtime_state.get("minigs_background", bg),
        use_trained_exp=dataset.train_test_exp,
    )
    area_max = aux_render_pkg["area_max"]
    blur_divisor = max(float(opt.minigs_blur_screen_coverage_divisor), 1.0)
    blur_threshold = float(viewpoint_camera.image_height) * float(viewpoint_camera.image_width) / blur_divisor
    return torch.logical_or(mask_blur, area_max > blur_threshold)


def _run_minigs_reinitialization_3dgs(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    render_state: dict[str, Any],
) -> torch.Tensor:
    """
        Rebuild MiniGS Gaussians from depth back-sampled training pixels.

        Each view contributes samples from pixels with low accumulated alpha.
        Those samples provide new 3D positions and colors, then the optimizer is
        rebuilt for the new Gaussian tensors.
    """
    scene = context.scene
    opt = context.opt
    runtime_state = context.runtime_state
    views = runtime_state.get("train_cameras") or scene.getTrainCameras()
    if not views:
        return _new_gaussian_cuda_bool_mask(gaussians)

    image_height = int(views[0].image_height)
    image_width = int(views[0].image_width)
    factor = 1.0 / (image_height * image_width * len(views) / float(opt.minigs_num_depth))
    out_pts_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    for view in views:
        gt = view.original_image[0:3, :, :]
        render_depth_pkg = render_minigs_depth(
            view,
            gaussians,
            context.pipe,
            runtime_state.get("minigs_background", render_state["bg"]),
        )
        out_pts = render_depth_pkg["out_pts"]
        accum_alpha = render_depth_pkg["accum_alpha"]
        prob = 1.0 - accum_alpha
        prob = prob.reshape(-1).cpu().numpy()
        num_total = prob.shape[0]
        prob_sum = float(prob.sum())
        if num_total == 0:
            continue
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            prob = np.full((num_total,), 1.0 / float(num_total), dtype=np.float64)
        else:
            prob = prob / prob_sum
        num_sampled = min(max(int(num_total * factor), 1), num_total)
        indices = np.random.choice(num_total, size=num_sampled, p=prob, replace=False)
        out_pts_list.append(out_pts.permute(1, 2, 0).reshape(-1, 3)[indices])
        gt_list.append(gt.permute(1, 2, 0).reshape(-1, 3)[indices])
    if not out_pts_list:
        return _new_gaussian_cuda_bool_mask(gaussians)
    gaussians.reinitial_pts(torch.cat(out_pts_list), torch.cat(gt_list))
    gaussians.training_setup(opt)
    torch.cuda.empty_cache()
    return _new_gaussian_cuda_bool_mask(gaussians)


def _collect_3dgs_densification_stats(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    render_state: dict[str, Any],
) -> torch.Tensor | None:
    """
        Collect the statistics needed by the selected densification method.

        Standard 3DGS uses normal screen-space gradients, ABSGS/ImprovedGS use
        absolute gradients, MiniGS also returns a blur mask, and MCMC skips
        gradient-based densification stats.
    """
    method = str(context.method_config["training_method"]).lower()
    visibility_filter = render_state["visibility_filter"]

    if method == "mcmc":
        return None
    if method == "minigs":
        gaussians.add_densification_stats(render_state["viewspace_point_tensor"], visibility_filter)
        return _collect_minigs_densification_stats_3dgs(context, gaussians, render_state, render_state["bg"])
    if method == "absgs":
        gaussians.add_densification_stats_abs(
            render_state["viewspace_point_tensor"], visibility_filter, sync_default_accumulator=True
        )
        return None
    if method in ("improvedgs", "gns"):
        gaussians.add_densification_stats_abs(render_state["viewspace_point_tensor"], visibility_filter)
        return None

    gaussians.add_densification_stats(render_state["viewspace_point_tensor"], visibility_filter)
    return None


def _run_3dgs_original_interval_densification(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    densify_from_iter: int,
) -> None:
    """
        Run the original 3DGS clone/split/prune rule at configured intervals.
    """
    opt = context.opt
    scene = context.scene
    should_apply = iteration > densify_from_iter and iteration % int(opt.densification_interval) == 0
    if should_apply:
        gaussians.densify_and_prune(float(opt.densify_grad_threshold), float(opt.min_opacity), scene.cameras_extent)
    if _should_reset_opacity(iteration, int(opt.opacity_reset_interval)):
        gaussians.reset_opacity()


def _run_3dgs_improvedgs_budget_densification(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    render_state: dict[str, Any],
) -> None:
    """
        Run ImprovedGS budgeted densification on densification intervals.

        Edge-aware scores are computed when enabled. The Gaussian model then
        decides how many candidates fit within the current budget.
    """
    opt = context.opt
    if iteration % int(opt.densification_interval) != 0:
        return
    scores = _compute_improvedgs_scores(context, gaussians, render_state, iteration)
    gaussians.densify_and_prune_improved(
        scores,
        float(opt.min_opacity),
        _compute_current_budget(iteration, opt),
        opt,
        iteration,
        float(context.scene.cameras_extent),
    )


def _run_3dgs_minigs_rule_densification(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    render_state: dict[str, Any],
    mask_blur: torch.Tensor | None,
    densify_from_iter: int,
) -> None:
    """
        Apply MiniGS structure updates.

        Most intervals split blurred Gaussians with the MiniGS mask. At the
        reinitialization interval, the whole Gaussian set is rebuilt from depth
        back-sampling instead.
    """
    opt = context.opt
    scene = context.scene
    runtime_state = context.runtime_state
    reinit_interval = max(int(opt.minigs_reinit_interval), 1)
    densification_interval = int(opt.densification_interval)
    if mask_blur is None:
        raise ValueError("MiniGS densification requires blur statistics before decision.")
    if (
        iteration > densify_from_iter
        and iteration % densification_interval == 0
        and iteration % reinit_interval != 0
        and gaussians.get_xyz.shape[0] < int(opt.minigs_num_max)
    ):
        gaussians.densify_and_prune_split(
            float(opt.densify_grad_threshold),
            float(opt.min_opacity),
            scene.cameras_extent,
            mask_blur,
        )
        mask_blur = _new_gaussian_cuda_bool_mask(gaussians)
    if iteration % reinit_interval == 0:
        mask_blur = _run_minigs_reinitialization_3dgs(context, gaussians, render_state)
    runtime_state["minigs_mask_blur"] = mask_blur


def _run_3dgs_mcmc_interval_budget_densification(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
) -> None:
    """
        Apply the MCMC update by relocating weak Gaussians and adding new ones.
    """
    opt = context.opt
    if iteration % int(opt.densification_interval) != 0:
        return
    budget = int(opt.budget)
    if budget <= 0:
        raise ValueError("MCMC densification requires budget > 0.")
    dead_mask = (gaussians.get_opacity <= float(opt.min_opacity)).squeeze(-1)
    gaussians.relocate_gs(dead_mask=dead_mask)
    gaussians.add_new_gs(cap_max=budget)


def run_3dgs_densification_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    render_state: dict[str, Any],
) -> None:
    """
        Dispatch one iteration's densification work for the active method.

        The caller already produced `render_state` during optimization. This
        function consumes that state to update accumulators, then calls exactly
        one method-specific structure update path.
    """
    opt = context.opt
    method = str(context.method_config["training_method"]).lower()
    densify_from_iter = int(opt.densify_from_iter)
    densify_until_iter = int(opt.densify_until_iter)
    if not (densify_from_iter < iteration < densify_until_iter):
        return

    mask_blur = _collect_3dgs_densification_stats(context, gaussians, render_state)
    if method in ("3dgs", "absgs"):
        _run_3dgs_original_interval_densification(context, gaussians, iteration, densify_from_iter)
    elif method in ("improvedgs", "gns"):
        _run_3dgs_improvedgs_budget_densification(context, gaussians, iteration, render_state)
    elif method == "minigs":
        _run_3dgs_minigs_rule_densification(context, gaussians, iteration, render_state, mask_blur, densify_from_iter)
    elif method == "mcmc":
        _run_3dgs_mcmc_interval_budget_densification(context, gaussians, iteration)
