"""Independent, sphere-constrained WildGaussians-style sky. Never enters scene depth."""
import math
import numpy as np
import torch
from lib.config import cfg
from lib.models.gaussian_model import GaussianModel
from lib.utils.graphics_utils import BasicPointCloud

class GaussianModelSky(GaussianModel):
    kind = 'gaussian_sphere'
    def __init__(self):
        super().__init__('sky')
        self.register_buffer('center', torch.zeros(3, device='cuda'))
        self.register_buffer('radius', torch.ones((), device='cuda'))
        self.register_buffer('layer_radii', torch.empty(0, device='cuda'))
        self.iteration = 0
        self.last_render = None
        self.frozen_count = None
        self.init_info = {}

    @property
    def get_xyz(self):
        if self._xyz.numel() == 0: return self._xyz
        d=self._xyz-self.center
        d=torch.cat((d[:,:2],d[:,2:3].abs()),1)
        r=self.radius
        if self.layer_radii.numel():
            norm=torch.linalg.vector_norm(d,dim=-1)
            r=self.layer_radii[(norm[:,None]-self.layer_radii[None]).abs().argmin(1)][:,None]
        return self.center + r * torch.nn.functional.normalize(d, dim=-1)

    @property
    def get_scaling(self):
        return self._scaling.exp().clamp(max=float(self.radius)*cfg.model.sky.get('max_scale_ratio', .02))

    @torch.no_grad()
    def initialize(self, points, cameras):
        if self._xyz.numel(): return
        points=points.detach();points=points[torch.isfinite(points).all(1)]
        # Deterministic subsampling for the robust statistic on multi-million point clouds.
        sample=points[::max(1,len(points)//500000)]
        self.center.copy_(sample.mean(0))
        r97=torch.quantile(torch.linalg.vector_norm(sample-self.center,dim=1),.97)
        cam_extent=max(float(torch.linalg.vector_norm(c.camera_center.to(self.center)-self.center)) for c in cameras)
        self.radius.fill_(max(float(r97)*cfg.model.sky.get('radius_multiplier',10.),cam_extent*2.,1.))
        n=int(cfg.model.sky.get('num_points',50000));i=torch.arange(n,device='cuda',dtype=torch.float32)
        z=(i+.5)/n;a=i*(math.pi*(3-math.sqrt(5)));rr=(1-z*z).sqrt()
        xyz=self.center+self.radius*torch.stack((rr*a.cos(),rr*a.sin(),z),1)
        multi_info=None
        if cfg.model.sky.get('multi_layer_init',False):
            from lib.models.multisphere_init import seed_layers
            xyz,self.layer_radii,multi_info=seed_layers(points,cameras,self.center,self.radius)
        rgb=np.tile(np.array([237,247,252],np.float32)/255,(len(xyz),1))
        super().create_from_pcd(BasicPointCloud(xyz.cpu().numpy(),rgb,np.zeros_like(rgb)),float(self.radius))
        self._opacity.fill_(torch.logit(torch.tensor(.9)).item())
        self._scaling.clamp_(max=math.log(float(self.radius)*cfg.model.sky.get('max_scale_ratio',.02)))
        self.init_info=dict(candidates=n,retained=len(xyz),center=self.center.tolist(),radius=float(self.radius),sky_mask_cameras=0,hemisphere=True)
        if multi_info is not None:self.init_info=multi_info
        self.training_setup()

    def training_setup(self):
        if not self._xyz.numel():return
        super().training_setup()
        self.active_sh_degree=self.max_sh_degree
        self.percent_dense=cfg.model.sky.get('percent_dense',.01)

    def update_learning_rate(self, iteration):
        self.iteration=iteration
        if self.optimizer is not None:super().update_learning_rate(iteration)

    def forward(self, camera, acc=None):
        if not self._xyz.numel():raise RuntimeError('Initialize sky sphere with background points and all training cameras first')
        from lib.models.gsplat_renderer import render_gsplat
        if cfg.render.get('backend','legacy')!='gsplat':raise ValueError('Gaussian sky currently requires gsplat')
        self.last_render=render_gsplat(camera,self,white_background=False)
        return self.last_render['rgb']

    def coverage_loss(self, camera):
        m=camera.guidance.get('sky_mask')
        if m is None:return self._xyz.sum()*0
        # Erode supervision near uncertain semantic boundaries, no inference mask.
        m=m.to(device='cuda').float().reshape(1,1,camera.image_height,camera.image_width)
        m=(1-torch.nn.functional.max_pool2d(1-m,7,1,3))[0,0]>.5
        a=self.last_render['acc'][0].clamp(1e-6,1-1e-6)
        return -torch.log(a[m]).mean() if m.any() else a.sum()*0

    @torch.no_grad()
    def topology_step(self):
        stop=int(cfg.model.sky.get('densify_until_iter',15000))
        if self.iteration>=stop:
            if self.frozen_count is None:self.frozen_count=len(self._xyz)
            assert len(self._xyz)==self.frozen_count,'Sky count changed after freeze'
            self.last_render=None
            return
        result=self.last_render
        if result is None:return
        stats=result['viewspace_points'];visible=result['visibility_filter']
        if stats is not None and stats.grad is not None:
            g=stats.grad
            self.xyz_gradient_accum[visible,0]+=torch.linalg.vector_norm(g[visible,:2],dim=-1)
            self.denom[visible]+=1
        if self.iteration>cfg.model.sky.get('densify_from_iter',500) and self.iteration%cfg.model.sky.get('densification_interval',500)==0:
            grads=torch.nan_to_num(self.xyz_gradient_accum[:,:1]/self.denom)
            threshold=cfg.model.sky.get('densify_grad_threshold',.0002)
            # Nonpositive/None means no count budget; positive remains useful for smoke validation.
            limit=cfg.model.sky.get('max_points',0)
            if limit is not None and limit>0:
                room=max(0,int(limit)-len(self._xyz))
                eligible=(grads[:,0]>=threshold).nonzero().flatten()
                if len(eligible)>room:
                    keep=eligible[torch.topk(grads[eligible,0],room).indices] if room else eligible[:0]
                    limited=torch.zeros_like(grads);limited[keep]=grads[keep];grads=limited
            before=len(self._xyz)
            self.scalar_dict.clear();self.densify_and_clone(grads,threshold,self.radius);self.densify_and_split(grads,threshold,self.radius)
            prune=(self.get_opacity[:,0]<.005)
            if (~prune).sum()>=16:self.prune_points(prune)
            import json
            from pathlib import Path
            with (Path(cfg.model_path)/'sky_topology.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=self.iteration,before=before,after=len(self._xyz),limit=limit))+'\n')
            self.xyz_gradient_accum=torch.zeros((len(self._xyz),2),device='cuda');self.denom=torch.zeros((len(self._xyz),1),device='cuda');self.max_radii2D=torch.zeros(len(self._xyz),device='cuda')
        self.last_render=None

    @torch.no_grad()
    def update_optimizer(self):
        if self.optimizer is None:return
        self.topology_step()
        super().update_optimizer()
        self._xyz.copy_(self.get_xyz)
        if self.layer_radii.numel() and self.iteration<=6:
            dist=torch.linalg.vector_norm(self._xyz-self.center,dim=-1)
            assert (dist[:,None]-self.layer_radii[None]).abs().min(1).values.max()<float(self.radius)*1e-5
            assert (self._xyz[:,2]>=self.center[2]-1e-4).all()
        self._scaling.clamp_(min=math.log(float(self.radius)*1e-5),max=math.log(float(self.radius)*cfg.model.sky.get('max_scale_ratio',.02)))

    def save_state_dict(self,is_final=False):
        state=super().state_dict(is_final)
        state.update(kind=self.kind,center=self.center,radius=self.radius,layer_radii=self.layer_radii,sky_iteration=self.iteration,frozen_count=self.frozen_count,init_info=self.init_info)
        return state

    def load_state_dict(self,state):
        if state.get('kind')!=self.kind:raise ValueError('Not a sky-sphere checkpoint; cubemap conversion must be explicit')
        self.center.copy_(state['center']);self.radius.copy_(state['radius']);self.layer_radii=state.get('layer_radii',torch.empty(0,device='cuda')).to(self.center)
        super().load_state_dict(state)
        self.active_sh_degree=self.max_sh_degree;self.iteration=state.get('sky_iteration',0);self.frozen_count=state.get('frozen_count');self.init_info=state.get('init_info',{})
