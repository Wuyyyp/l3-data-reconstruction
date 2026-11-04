"""Adapt gsplat pixel-space statistics to StreetGS's [N,3] gradient protocol.

The third entry is sum_pixels(abs(dx_ndc)+abs(dy_ndc)), not a z derivative
and not the L2 norm used by gsplat's DefaultStrategy. Single-camera only.
"""
import torch

class GsplatScreenStats:
    def __init__(self, means2d, num_gaussians, width, height, gaussian_ids=None):
        self.means2d=means2d
        self.num_gaussians=num_gaussians
        self.scale=means2d.new_tensor([width/2.,height/2.])
        self.gaussian_ids=gaussian_ids

    @property
    def grad(self):
        if self.means2d.grad is None or not hasattr(self.means2d,'absgrad'):
            raise RuntimeError('Read densification statistics only after backward with absgrad=True.')
        signed=self.means2d.grad.detach().reshape(-1,2)*self.scale
        absolute=self.means2d.absgrad.detach().reshape(-1,2)*self.scale
        stats=torch.cat([signed,absolute.sum(dim=-1,keepdim=True)],dim=-1)
        if self.gaussian_ids is None:
            if len(stats)!=self.num_gaussians:raise ValueError('Expected one-camera dense statistics')
            return stats
        full=stats.new_zeros((self.num_gaussians,3))
        full.index_add_(0,self.gaussian_ids,stats)
        return full
