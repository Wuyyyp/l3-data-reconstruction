"""
CUDA wrapper for MCMC relocation math.

The relocation kernel computes how opacity and scale should be shared when one
Gaussian is reused to replace or spawn other Gaussians. This file prepares the
binomial table expected by the CUDA extension.
"""
from __future__ import annotations

import math

import torch

from diff_gaussian_rasterization import compute_relocation


N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX), dtype=torch.float32, device="cuda")
for n_idx in range(N_MAX):
    for k_idx in range(n_idx + 1):
        BINOMS[n_idx, k_idx] = math.comb(n_idx, k_idx)


def compute_relocation_cuda(
    opacity_old: torch.Tensor,
    scale_old: torch.Tensor,
    N: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Clamp sample counts and call the CUDA relocation kernel.
    """
    N = N.clamp(min=1, max=N_MAX - 1).to(dtype=torch.int32)
    return compute_relocation(opacity_old, scale_old, N, BINOMS, N_MAX)
