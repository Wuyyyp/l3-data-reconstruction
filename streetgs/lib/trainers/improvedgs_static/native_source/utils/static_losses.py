import torch
import torch.nn.functional as F
def masked_ssim(pred,gt,mask):
 # Only windows fully outside exclusion masks contribute; no masked-pixel gradient.
 x=torch.arange(11,device=pred.device,dtype=pred.dtype)-5
 k=torch.exp(-x*x/(2*1.5**2));k=k/k.sum();window=(k[:,None]*k[None,:])[None,None].expand(3,1,11,11).contiguous()
 def conv(a):return F.conv2d(a[None],window,groups=3)[0]
 mu1,mu2=conv(pred),conv(gt)
 v1,v2,cov=conv(pred*pred)-mu1*mu1,conv(gt*gt)-mu2*mu2,conv(pred*gt)-mu1*mu2
 score=((2*mu1*mu2+.01**2)*(2*cov+.03**2))/((mu1*mu1+mu2*mu2+.01**2)*(v1+v2+.03**2))
 eligible=F.max_pool2d((~mask).float()[None],11,1)[0,0]==0
 if not eligible.any():raise ValueError('No valid static SSIM windows')
 return score[:,eligible].mean()
