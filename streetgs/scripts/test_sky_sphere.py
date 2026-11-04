import sys
from pathlib import Path
import torch,numpy as np
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.config import cfg
from lib.models.gaussian_model_sky import GaussianModelSky
from lib.utils.graphics_utils import BasicPointCloud
cfg.model.sky.densify_until_iter=15000;cfg.model.sky.max_points=256
sky=GaussianModelSky();sky.radius.fill_(100)
d=torch.randn(64,3,device='cuda');d=torch.nn.functional.normalize(d,dim=-1)*100
sky.create_from_pcd(BasicPointCloud(d.cpu().numpy(),np.ones((64,3),np.float32)*.8,np.zeros((64,3))),100)
sky.training_setup()
# Force both topology paths on separate parent sets and check new points remain on sphere.
with torch.no_grad():
 sky._scaling[:32].fill_(np.log(.5));sky._scaling[32:].fill_(np.log(2.))
 sky.xyz_gradient_accum[:,0]=1;sky.denom[:]=1
 sky.iteration=1000;sky.last_render=dict(viewspace_points=None,visibility_filter=torch.ones(64,dtype=torch.bool,device='cuda'))
 sky.topology_step();assert len(sky.get_xyz)==128,len(sky.get_xyz)
 assert torch.allclose(torch.linalg.vector_norm(sky.get_xyz-sky.center,dim=1),sky.radius.expand(128),atol=1e-4)
 # Force opacity pruning before freeze.
 sky._opacity[:4].fill_(-20);sky.iteration=1500;sky.last_render=dict(viewspace_points=None,visibility_filter=torch.ones(128,dtype=torch.bool,device='cuda'));sky.topology_step();assert len(sky.get_xyz)==124
 # Freeze guards must prevent deletion even with low opacity and large gradients.
 sky._opacity[:4].fill_(-20);sky.xyz_gradient_accum[:]=1;sky.denom[:]=1;sky.iteration=15000
 sky.topology_step();assert len(sky.get_xyz)==124
loss=sky.get_xyz[:,0].mean()+sky.get_scaling.mean()+sky.get_opacity.mean()+sky.get_features.mean();loss.backward()
assert torch.isfinite(sky._xyz.grad).all();before=sky._features_dc.detach().clone();sky.update_optimizer();assert not torch.equal(before,sky._features_dc)
state=sky.save_state_dict(False);other=GaussianModelSky();other.load_state_dict(state);assert len(other.get_xyz)==124 and other.frozen_count==124
print('PASS clone, split, prune, sphere constraint, 15000 freeze, continuing parameter updates, checkpoint reload',flush=True)
