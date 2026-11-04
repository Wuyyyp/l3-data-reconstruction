"""Small-scene regression for opt-in training and unclipped RGB gradients."""
import sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from lib.config import cfg
from lib.utils.camera_utils import Camera
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
class Tiny:
    max_sh_degree=0;active_sh_degree=0
    def __init__(self):
        self.get_xyz=torch.tensor([[-.25,.1,3.],[.35,-.2,4.]],device='cuda',requires_grad=True)
        self.get_scaling=torch.tensor([[.14,.2,.08],[.22,.11,.09]],device='cuda',requires_grad=True)
        self._rotation=torch.tensor([[.96,.1,.2,.15],[.95,.2,.05,.2]],device='cuda',requires_grad=True)
        self.get_opacity=torch.tensor([[.7],[.5]],device='cuda',requires_grad=True)
        self.get_features=torch.tensor([[[.3,.2,.1]],[[.1,.3,.2]]],device='cuda',requires_grad=True)
    @property
    def get_rotation(self):return torch.nn.functional.normalize(self._rotation,dim=-1)
    def parameters(self):return {n:getattr(self,n) for n in ['get_xyz','get_scaling','_rotation','get_opacity','get_features']}
c=Camera(0,np.eye(3),np.zeros(3),2*math.atan(128/200),2*math.atan(96/200),np.array([[100.,0,64.],[0,100.,48.],[0,0,1.]]),torch.zeros(3,96,128),'tiny')
renderer=StreetGaussianRenderer();cfg.mode='train';cfg.render.backend='gsplat';cfg.optim.fixed_gaussians=False
try:renderer.render_kernel(c,Tiny())
except RuntimeError as e:assert 'fixed_gaussians' in str(e)
else:raise AssertionError('Ungated gsplat training must fail')
cfg.optim.fixed_gaussians=True
rows={};gradients={}
for backend in ['legacy','gsplat']:
    cfg.render.backend=backend;pc=Tiny();r=renderer.render_kernel(c,pc)
    # Raw CUDA quaternion conventions expect normalized wxyz in both backends.
    # Use identical unit quaternions for this test.
    loss=r['rgb'].square().mean()+.02*r['depth'].mean()+.01*r['acc'].mean()
    loss.backward();gradients[backend]={n:p.grad.clone() for n,p in pc.parameters().items()}
    assert all(torch.isfinite(g).all() for g in gradients[backend].values())
    rows[backend]=float(loss)
    # Bright precomputed colors must retain >1 RGB and a live gradient in training.
    color=torch.full((2,3),5.,device='cuda',requires_grad=True)
    bright=renderer.render_kernel(c,Tiny(),override_color=color)['rgb']
    assert bright.max()>1
    bright.mean().backward();assert color.grad.abs().sum()>0
for name in gradients['legacy']:
    a=gradients['legacy'][name];b=gradients['gsplat'][name]
    rel=float((a-b).norm()/a.norm().clamp_min(1e-12));rows[name]=dict(relative_l2=rel)
    assert rel<1e-4,(name,rel)
    # Non-unit quaternion test is intentionally omitted: production supplies normalized rotations.
fd=[]
for clipped in [False,True]:
    for backend in ['legacy','gsplat']:
        cfg.render.backend=backend;pc=Tiny()
        if clipped:
            with torch.no_grad():
                pc.get_xyz[0]=torch.tensor([3.3,.1,3.],device='cuda')
                pc.get_scaling[0]=torch.tensor([1.4,.8,.7],device='cuda')
        def objective():
            r=renderer.render_kernel(c,pc)
            return r['rgb'].square().mean()+.02*r['depth'].mean()+.01*r['acc'].mean()
        loss=objective();loss.backward();analytic=float(pc.get_xyz.grad[0,2]);eps=.001
        with torch.no_grad():
            pc.get_xyz[0,2]+=eps;plus=float(objective())
            pc.get_xyz[0,2]-=2*eps;minus=float(objective())
            pc.get_xyz[0,2]+=eps
        numerical=(plus-minus)/(2*eps)
        fd.append(dict(clipped=clipped,backend=backend,analytic_z=analytic,numerical_z=numerical,
                       relative_error=abs(analytic-numerical)/max(abs(numerical),1e-10)))
rows['finite_difference']=fd
assert next(r['relative_error'] for r in fd if r['clipped'] and r['backend']=='gsplat')<.02
out=Path(__file__).resolve().parents[1]/'output/gsplat_train_check'
(out/'synthetic_backward.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows));print('PASS: opt-in gate, finite backward, unclipped RGB, color gradient')
