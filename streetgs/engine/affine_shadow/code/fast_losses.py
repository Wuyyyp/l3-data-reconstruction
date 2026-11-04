"""Fast masked losses retaining the original fully-valid 11x11 SSIM support."""
import torch
import torch.nn.functional as F
from fused_ssim import FusedSSIMMap

def masked_mean(values, mask):
    # Avoid CUDA nonzero and CPU synchronization caused by boolean indexing.
    weight=mask.to(values.dtype)
    if weight.shape!=values.shape:weight=weight.expand_as(values)
    return (values*weight).sum()/weight.sum().clamp_min(1)

def masked_ssim(pred,gt,mask):
    score=FusedSSIMMap.apply(.01**2,.03**2,pred.contiguous()[None],gt.contiguous()[None],"valid",torch.is_grad_enabled())[0]
    eligible=(F.max_pool2d((~mask).float()[None],11,1)[0]==0)
    return masked_mean(score,eligible)
