import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from types import SimpleNamespace
from lib.config import cfg
from lib.models.gsplat_renderer import render_gsplat
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
c=SimpleNamespace(world_view_transform=torch.eye(4,device='cuda'),image_height=6,image_width=8)
class Empty:
    num_gaussians=0
    @property
    def get_xyz(self):
        raise AssertionError('empty scene must not concatenate xyz')
cfg.mode='evaluate'
for white in [True,False]:
    r=render_gsplat(c,Empty(),white_background=white)
    assert r['rgb'].shape==(3,6,8) and (r['rgb']==int(white)).all()
    assert not r['acc'].any() and not r['depth'].any()
    assert r['radii'].numel()==0 and r['visibility_filter'].dtype==torch.bool
cfg.mode='train';cfg.optim.fixed_gaussians=False
try: render_gsplat(c,Empty())
except RuntimeError as e: assert 'fixed_gaussians' in str(e)
else: raise AssertionError('training must be blocked')
cfg.mode='evaluate';cfg.render.backend='invalid'
try: StreetGaussianRenderer().render_kernel(c,Empty())
except ValueError as e: assert 'Unknown render backend' in str(e)
else: raise AssertionError('unknown backend must fail')
pc=SimpleNamespace(get_xyz=torch.tensor([[0.,0.,3.]],device='cuda'),
    get_rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda'),
    get_scaling=torch.full((1,3),.1,device='cuda'),get_opacity=torch.tensor([[.7]],device='cuda'))
proj=torch.zeros(4,4,device='cuda');proj[0,0]=1.25;proj[1,1]=5/3;proj[2,2]=1;proj[3,2]=1
c.projection_matrix=proj.T.contiguous();c.full_proj_transform=c.projection_matrix
with torch.no_grad():
    a=render_gsplat(c,pc,override_color=torch.ones(1,3,device='cuda'))
    del c.projection_matrix
    b=render_gsplat(c,pc,override_color=torch.ones(1,3,device='cuda'))
    for key in ['rgb','acc','depth']:
        assert torch.equal(a[key],b[key]),key
print('PASS: empty black/white scenes, training guard, invalid backend, MiniCam projection fallback')
