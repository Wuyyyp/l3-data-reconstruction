"""Single-camera covariance rasterizer with packed-to-dense training statistics."""
import torch
from gsplat import rasterization
from shared_sh import eval_sh
from lib.config import cfg

class ScreenStats:
    def __init__(self,means2d,w,h,ids=None,count=None):
        self.means2d=means2d
        self.scale=means2d.new_tensor([w/2.,h/2.])
        self.ids=ids
        self.count=count
        self._grad=None
    @property
    def grad(self):
        if self._grad is None:
            assert self.means2d.grad is not None and hasattr(self.means2d,'absgrad')
            g=torch.cat([self.means2d.grad.reshape(-1,2)*self.scale,
                         self.means2d.absgrad.reshape(-1,2)*self.scale],dim=1)
            if self.ids is not None:
                dense=g.new_zeros((self.count,4))
                dense.index_add_(0,self.ids,g)
                g=dense
            self._grad=g
        return self._grad

def camera_matrices(c,means):
    # This pipeline keeps camera calibration and poses fixed.
    key=(int(c.image_width),int(c.image_height),means.device,means.dtype,
         c.world_view_transform.data_ptr(),c.world_view_transform._version,
         c.full_proj_transform.data_ptr(),c.full_proj_transform._version)
    if getattr(c,'_ground_projection_key',None)!=key:
        w,h=key[:2]
        projection=torch.linalg.solve(c.world_view_transform,c.full_proj_transform).T
        K=means.new_zeros((3,3));K[2,2]=1
        K[0,0]=projection[0,0]*w/2;K[1,1]=projection[1,1]*h/2
        K[0,2]=(projection[0,2]+1)*w/2;K[1,2]=(projection[1,2]+1)*h/2
        c._ground_projection=(c.world_view_transform.T.contiguous()[None],K[None])
        c._ground_projection_key=key
    return c._ground_projection

def dense_radii(meta,count,packed):
    r=meta['radii'].reshape(-1,2).amax(-1)
    if packed:
        dense=r.new_zeros(count)
        dense.index_copy_(0,meta['gaussian_ids'].long(),r)
        return dense
    return r

def gsplat_render(c,m,pipe,bg,scaling_modifier=1.,use_trained_exp=False,track_gradients=True,inference_only=False,pixel_weights=None,metric=False):
    assert not use_trained_exp
    w,h=int(c.image_width),int(c.image_height);means=m.get_xyz
    viewmats,Ks=camera_matrices(c,means)
    packed=bool(cfg.render.get('gsplat_packed',True))
    kwargs=dict(viewmats=viewmats,Ks=Ks,width=w,height=h,near_plane=.2,
                far_plane=1e10,eps2d=.3,packed=packed,sparse_grad=False,
                rasterize_mode='classic',render_mode='RGB')
    if pixel_weights is not None:
        with torch.enable_grad():
            colors=means.new_ones((len(means),1),requires_grad=True)
            im,alpha,meta=rasterization(means=means.detach(),quats=None,scales=None,
                covars=m.covars.detach()*scaling_modifier**2,
                opacities=m.get_opacity.detach().flatten(),colors=colors,absgrad=False,**kwargs)
            score=torch.autograd.grad((im[0,:,:,0]*pixel_weights.to(im).squeeze()).sum(),colors)[0].flatten().detach()
        radii=dense_radii(meta,len(means),packed)
        return dict(accum_weights=score,visibility_filter=radii>0,radii=radii)
    sh=torch.cat([m.get_features_dc,m.get_features_rest],dim=1).transpose(1,2)
    direction=torch.nn.functional.normalize(means-c.camera_center,dim=-1)
    colors=(eval_sh(m.active_sh_degree,sh,direction)+.5).clamp_min(0)
    z=(means@c.world_view_transform[:3,2]+c.world_view_transform[3,2]).clamp_min(.2)
    features=torch.cat([colors,(z if metric else z.reciprocal())[:,None]],dim=1)
    if hasattr(m,'road_labels'):features=torch.cat([features,m.road_labels[:,None].to(features),(m.road_labels.to(z)*z)[:,None]],1)
    ground_channel=features.shape[1]
    if hasattr(m,'ground_count'):
        gc=colors.clone();gc[m.ground_count:]=0;features=torch.cat([features,gc],1)
    collect=track_gradients and torch.is_grad_enabled()
    im,alpha,meta=rasterization(means=means,quats=None,scales=None,covars=m.covars*scaling_modifier**2,
        opacities=m.get_opacity.flatten(),colors=features,absgrad=collect,**kwargs)
    if collect:meta['means2d'].retain_grad()
    radii=dense_radii(meta,len(means),packed);acc=alpha[0].permute(2,0,1)
    rgb=im[0,:,:,:3].permute(2,0,1)+(1-acc)*bg[:,None,None]
    result=dict(render=rgb,depth=im[0,:,:,3:4].permute(2,0,1),acc=acc,
        radii=radii,visibility_filter=radii>0,
        viewspace_points=ScreenStats(meta['means2d'],w,h,meta['gaussian_ids'].long() if packed else None,len(means)))
    if hasattr(m,'road_labels'):result['road_background_alpha']=im[0,:,:,4:5].permute(2,0,1);result['road_background_depth']=im[0,:,:,5:6].permute(2,0,1)
    if hasattr(m,'ground_count'):result['ground_contribution']=im[0,:,:,ground_channel:ground_channel+3].permute(2,0,1)
    return result
