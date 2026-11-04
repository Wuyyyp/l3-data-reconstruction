"""
MCMC structure and noise operations.

These helpers implement the MCMC-style update outside GaussianModel so the model
class stays focused on tensor storage. Low-opacity Gaussians are replaced by
samples from stronger Gaussians, new Gaussians are added up to a budget, and
SGLD-like position noise is injected after optimizer steps.
"""
from __future__ import annotations

import math

import torch

from utils.reloc_utils import compute_relocation_cuda


def _sample_alives(
    total_count: int,
    probs: torch.Tensor,
    num: int,
    alive_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Sample alive Gaussian indices by opacity and count how often each one is used.
    """
    normalized_probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    sampled_idxs = torch.multinomial(normalized_probs, num, replacement=True)
    if alive_indices is not None:
        sampled_idxs = alive_indices[sampled_idxs]
    ratio = torch.bincount(sampled_idxs, minlength=total_count).unsqueeze(-1)
    return sampled_idxs, ratio


def _rotate_vector_by_quaternion(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
        Rotate vectors by normalized quaternions stored as `[w, x, y, z]`.
    """
    quaternion_xyz = quaternion[:, 1:]
    twice_cross = 2.0 * torch.cross(quaternion_xyz, vector, dim=1)
    return vector + quaternion[:, :1] * twice_cross + torch.cross(quaternion_xyz, twice_cross, dim=1)


def _apply_mcmc_covariance_to_noise(
    rotation: torch.Tensor,
    scaling: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
        Shape isotropic noise by each Gaussian's local covariance.
    """
    inverse_rotation = rotation.clone()
    inverse_rotation[:, 1:] = -inverse_rotation[:, 1:]
    local_noise = _rotate_vector_by_quaternion(inverse_rotation, noise)
    local_noise = local_noise * scaling.square()
    return _rotate_vector_by_quaternion(rotation, local_noise)


def _update_params(model, idxs: torch.Tensor, ratio: torch.Tensor):
    """
        Build relocated child parameters from selected source Gaussians.
    """
    new_opacity, new_scaling = compute_relocation_cuda(
        opacity_old=model.get_opacity[idxs, 0],
        scale_old=model.get_scaling[idxs],
        N=ratio[idxs, 0] + 1,
    )
    new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
    new_opacity = model.inverse_opacity_activation(new_opacity)
    new_scaling = model.scaling_inverse_activation(new_scaling.reshape(-1, 3))
    return (
        model._xyz[idxs],
        model._features_dc[idxs],
        model._features_rest[idxs],
        new_opacity,
        new_scaling,
        model._rotation[idxs],
    )


def relocate_gs(model, dead_mask: torch.Tensor | None = None) -> None:
    """
        Replace dead Gaussians with opacity-weighted samples from alive ones.
    """
    if dead_mask is None or dead_mask.numel() == 0 or int(dead_mask.sum().item()) == 0:
        return
    alive_mask = ~dead_mask
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]
    if alive_indices.shape[0] <= 0:
        return

    sampled_indices, ratio = _sample_alives(
        total_count=model.get_xyz.shape[0],
        probs=model.get_opacity[alive_indices, 0],
        num=int(dead_indices.shape[0]),
        alive_indices=alive_indices,
    )
    (
        model._xyz[dead_indices],
        model._features_dc[dead_indices],
        model._features_rest[dead_indices],
        model._opacity[dead_indices],
        model._scaling[dead_indices],
        model._rotation[dead_indices],
    ) = _update_params(model, sampled_indices, ratio=ratio)

    model._opacity[sampled_indices] = model._opacity[dead_indices]
    model._scaling[sampled_indices] = model._scaling[dead_indices]
    model.replace_tensors_to_optimizer(inds=sampled_indices)


def add_new_gs(model, cap_max: int) -> int:
    """
        Add new Gaussians by sampling existing ones until the budget cap is reached.
    """
    current_num_points = int(model._opacity.shape[0])
    target_num = min(int(cap_max), int(1.05 * current_num_points))
    num_gs = max(0, target_num - current_num_points)
    if num_gs <= 0:
        return 0

    add_idx, ratio = _sample_alives(
        total_count=model.get_xyz.shape[0],
        probs=model.get_opacity.squeeze(-1),
        num=num_gs,
    )
    new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation = _update_params(
        model, add_idx, ratio=ratio
    )
    model._opacity[add_idx] = new_opacity
    model._scaling[add_idx] = new_scaling
    model.densification_postfix(
        new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False,
    )
    model.replace_tensors_to_optimizer(inds=add_idx)
    return num_gs


def apply_mcmc_noise(model, xyz_lr: float) -> None:
    """
        Add covariance-shaped position noise scaled by opacity and xyz learning rate.
    """
    if model.get_xyz.numel() == 0:
        return

    def opacity_sigmoid(x: torch.Tensor, k: float = 100.0, x0: float = 0.995) -> torch.Tensor:
        """
            Gate noise so low-opacity Gaussians receive stronger perturbations.
        """
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    with torch.no_grad():
        noise = (
            torch.randn_like(model._xyz)
            * opacity_sigmoid(1.0 - model.get_opacity, k=float(model.mcmc_noise_opacity_sharpness))
            * float(xyz_lr)
            * float(getattr(model, "noise_lr", 0.0))
        )
        noise = _apply_mcmc_covariance_to_noise(model.get_rotation, model.get_scaling, noise)
        model._xyz.add_(noise)
