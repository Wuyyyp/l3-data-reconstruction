import sys,os,torch,random,numpy as np
from pathlib import Path
from types import SimpleNamespace
from runtime_setup import REPO, SETTINGS
STREET=REPO
sys.path.insert(1,str(STREET));
os.environ['PWD']=str(STREET)
argv=sys.argv;sys.argv=[argv[0],'--config',str(Path(__file__).resolve().parent/'sky_config.yaml')]
from lib.config import cfg
sys.argv=argv
cfg.mode='train';cfg.render.backend='gsplat';cfg.model.gaussian.sh_degree=1
from lib.models.gaussian_model_sky import GaussianModelSky
from gaussian_renderer import render
class NativeSky:
 def __init__(self,cams,points=None,state=None):
  self.model=GaussianModelSky();self.cameras={}
  geo=torch.load(SETTINGS['scene_geometry'],map_location='cpu');ks={c['name']:c['K'] for c in geo['cameras']};del geo
  for c in cams:
   self.cameras[c.image_name]=SimpleNamespace(image_name=c.image_name,image_width=c.image_width,image_height=c.image_height,world_view_transform=c.world_view_transform,projection_matrix=torch.linalg.solve(c.world_view_transform,c.full_proj_transform),camera_center=c.camera_center,K=ks[c.image_name].cuda(),guidance={'sky_mask':c.sky})
  if state is not None:self.model.load_state_dict(state)
  else:
   rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all());self.model.initialize(points,list(self.cameras.values()));random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  if os.environ.get('SMOKE')=='1':
   cfg.model.sky.densify_from_iter=0;cfg.model.sky.densification_interval=2;cfg.model.sky.densify_grad_threshold=1e-12;cfg.model.sky.max_points=len(self.model.get_xyz)+32
 def capture(self):return self.model.save_state_dict(is_final=False)
 def render(self,c):return self.model(self.cameras[c.image_name])
class WhiteProxy:
 def __init__(self,m):self.m=m;self.active_sh_degree=0
 def __getattr__(self,k):return getattr(self.m,k)
 @property
 def get_features_dc(self):return torch.full_like(self.m.get_features_dc,.5/.28209479177387814)
 @property
 def get_features_rest(self):return torch.zeros_like(self.m.get_features_rest)
def compose(c,m,pipe,bg,sky):
 pkg=render(c,m,pipe,bg);ap=render(c,WhiteProxy(m),pipe,bg);alpha=ap['render'][:1];sky_rgb=sky.render(c)
 pkg['render']=pkg['render']+(1-alpha)*sky_rgb;pkg['scene_alpha']=alpha;pkg['alpha_pkg']=ap
 return pkg
def sky_loss(c,pkg,sky):
 mask=c.alpha_mask.cuda().bool();sm=c.sky.cuda().bool();a=pkg['scene_alpha'].clamp(1e-6,1-1e-6)
 bce=torch.where(sm,-torch.log1p(-a),-torch.log(a))[mask].mean()
 valid=(1-torch.nn.functional.max_pool2d(1-sm.float()[None],7,1,3))[0].bool() & mask
 acc=sky.model.last_render['acc'].clamp(1e-6,1-1e-6);coverage=-acc[valid].log().mean() if valid.any() else acc.sum()*0
 return .05*bce+.01*coverage
