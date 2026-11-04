import torch
from gsplat import rasterization
from gaussian_renderer import render as native_render
from shared_sh import eval_sh

class ScreenStats:
 def __init__(self,means2d,w,h):self.means2d=means2d;self.scale=means2d.new_tensor([w/2.,h/2.])
 @property
 def grad(self):
  assert self.means2d.grad is not None and hasattr(self.means2d,'absgrad')
  return torch.cat([self.means2d.grad.reshape(-1,2)*self.scale,self.means2d.absgrad.reshape(-1,2)*self.scale],dim=1)

def gsplat_render(c,m,pipe,bg,scaling_modifier=1.,use_trained_exp=False,track_gradients=True,inference_only=False,pixel_weights=None,metric=False):
 assert not use_trained_exp
 w,h=int(c.image_width),int(c.image_height);means=m.get_xyz
 projection=torch.linalg.solve(c.world_view_transform,c.full_proj_transform).T
 K=means.new_tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
 K[0,0]=projection[0,0]*w/2;K[1,1]=projection[1,1]*h/2;K[0,2]=(projection[0,2]+1)*w/2;K[1,2]=(projection[1,2]+1)*h/2
 kwargs=dict(viewmats=c.world_view_transform.T.contiguous()[None],Ks=K[None],width=w,height=h,near_plane=.2,far_plane=1e10,eps2d=.3,packed=False,sparse_grad=False,rasterize_mode='classic',render_mode='RGB')
 if pixel_weights is not None:
  # d(sum_pixels edge*rendered_constant_color)/d(color_i) == sum_pixels edge*T*alpha_i.
  # Only the dummy colors require gradients: does not change any model gradients.
  with torch.enable_grad():
   colors=means.new_ones((len(means),1),requires_grad=True)
   im,alpha,meta=rasterization(means=means.detach(),quats=None,scales=None,covars=m.covars.detach()*scaling_modifier**2,opacities=m.get_opacity.detach().flatten(),colors=colors,absgrad=False,**kwargs)
   score=torch.autograd.grad((im[0,:,:,0]*pixel_weights.to(im).squeeze()).sum(),colors)[0].flatten().detach()
  radii=meta['radii'].reshape(-1,2).amax(-1)
  return dict(accum_weights=score,visibility_filter=radii>0,radii=radii)
 sh=torch.cat([m.get_features_dc,m.get_features_rest],dim=1).transpose(1,2)
 direction=torch.nn.functional.normalize(means-c.camera_center,dim=-1)
 colors=(eval_sh(m.active_sh_degree,sh,direction)+.5).clamp_min(0)
 z=(means@c.world_view_transform[:3,2]+c.world_view_transform[3,2]).clamp_min(.2)
 features=torch.cat([colors,(z if metric else z.reciprocal())[:,None]],dim=1)
 collect=track_gradients and torch.is_grad_enabled()
 im,alpha,meta=rasterization(means=means,quats=None,scales=None,covars=m.covars*scaling_modifier**2,opacities=m.get_opacity.flatten(),colors=features,absgrad=collect,**kwargs)
 if collect:meta['means2d'].retain_grad()
 radii=meta['radii'].reshape(-1,2).amax(-1);acc=alpha[0].permute(2,0,1)
 rgb=im[0,:,:,:3].permute(2,0,1)+(1-acc)*bg[:,None,None]
 return dict(render=rgb,depth=im[0,:,:,3:4].permute(2,0,1),acc=acc,radii=radii,visibility_filter=radii>0,viewspace_points=ScreenStats(meta['means2d'],w,h))
