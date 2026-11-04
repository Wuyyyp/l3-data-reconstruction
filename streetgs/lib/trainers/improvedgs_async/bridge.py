import os,sys,json,random,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import importlib
for _name in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+_name)
for _name,_module in list(sys.modules.items()):
 if _name=='numpy.core' or _name.startswith('numpy.core.'):sys.modules.setdefault(_name.replace('numpy.core','numpy._core',1),_module)
REPO=Path(__file__).resolve().parents[3]
STATIC=REPO/'lib/trainers/improvedgs_static'
sys.path.insert(0,str(STATIC/'native_source'));sys.path.insert(0,str(STATIC));sys.path.insert(0,str(REPO))
argv=sys.argv;sys.argv=[argv[0],'--config',os.environ['DYNAMIC_CONFIG']]
from lib.config import cfg
sys.argv=argv
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import loadguidance
from async_pipeline import Prefetch,cpu_payload,masks_for_sky
from scene.methods.densification_methods import compute_edge_map
from backend_bridge import gsplat_render
PARAMS=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']

class BackgroundAdapter(torch.nn.Module):
 def __init__(self,model):
  super().__init__();object.__setattr__(self,'native',model)
 def __getattr__(self,k):
  try:return super().__getattr__(k)
  except AttributeError:return getattr(object.__getattribute__(self,'native'),k)
 def set_background_mask(self,c):pass
 @property
 def get_features(self):return torch.cat([self.native.get_features_dc,self.native.get_features_rest],1)

class RenderModel:
 def __init__(self,sg):
  self.get_xyz=sg.get_xyz;self.get_rotation=sg.get_rotation;self.get_scaling=sg.get_scaling;self.get_opacity=sg.get_opacity
  f=sg.get_features;self.get_features_dc=f[:,:1];self.get_features_rest=f[:,1:];self.active_sh_degree=sg.active_sh_degree

class SlicedStats:
 def __init__(self,stats,n):self.stats=stats;self.n=n
 @property
 def grad(self):return self.stats.grad[:self.n]

def valid_mask(c):
 h,w=c.image_height,c.image_width;mask=torch.ones((1,h,w),dtype=torch.bool);cam=int(c.image_name.rsplit('_',1)[1])
 if cam==3:mask[:,:int(250/1280*h),:int(550/1920*w)]=False;mask[:,:,:int(80/1920*w)]=False
 if cam==4:mask[:,:int(250/1280*h),int((1920-550)/1920*w):]=False;mask[:,:,int((1920-80)/1920*w):]=False
 if cam==10:mask[:,int(1500/2160*h):]=False
 return mask

def dataset():
 cfg.data.write_input_snapshot=False
 ds=Dataset();cams=sorted(ds.train_cameras[1],key=lambda c:c.image_name)
 for i,c in enumerate(cams):c.uid=i;c.alpha_mask=valid_mask(c)
 return ds,cams

def sky_cameras(ds,cams):
 infos={c.image_name:c for c in ds.scene_info.train_cameras};result=[]
 for c in cams:
  info=infos[c.image_name];guide={'sky_mask':masks_for_sky(c)['sky_mask']}
  result.append(SimpleNamespace(guidance=guide,camera_center=c.camera_center,K=c.K,world_view_transform=c.world_view_transform,image_width=c.image_width,image_height=c.image_height))
 return result

class DynamicScene:
 def __init__(self,ds,cams,bg,out,initialize=True):
  self.cameras=cams;self.gaussians=bg;self.model_path=str(out);self.cameras_extent=20.;self.sg=StreetGaussianModel(ds.scene_info.metadata)
  self.sg.background=BackgroundAdapter(bg);self.sg.set_visibility(list(self.sg.model_name_id));self.edges={}
  if initialize:
   for name in self.sg.obj_list:
    m=getattr(self.sg,name);m.create_from_pcd(20.);m.training_setup()
   self.sg.actor_pose.training_setup() if self.sg.actor_pose is not None else None
   rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
   self.sg.sky_cubemap.initialize(bg.get_xyz,sky_cameras(ds,cams))
   random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  if self.sg.color_correction is not None:self.sg.color_correction.training_setup()
  self.sg.active_sh_degree=0
 def getTrainCameras(self):return self.cameras
 def getTestCameras(self):return []
 def edge(self,c):
  if c.uid not in self.edges:
   im,guide=cpu_payload(c);edge=compute_edge_map(im);valid=c.alpha_mask & ~guide['obj_bound'].bool()
   invalid=torch.nn.functional.max_pool2d((~valid).float()[None],3,1,1)[0,0].bool();self.edges[c.uid]=edge*(~invalid)
  return self.edges[c.uid]
 def render(self,c,bg,pipe,background,**kw):
  self.sg.parse_camera(c);model=RenderModel(self.sg);n=len(bg.get_xyz)
  if 'pixel_weights' in kw:
   kw['pixel_weights']=self.edge(c);pkg=gsplat_render(c,model,pipe,background,**kw)
   return {k:v[:n] for k,v in pkg.items()}
  pkg=gsplat_render(c,model,pipe,background,**kw);pkg['actor_ranges']=dict(self.sg.graph_gaussian_range)
  pkg['joint_radii']=pkg['radii'];pkg['joint_visible']=pkg['visibility_filter'];pkg['joint_stats']=pkg['viewspace_points']
  pkg['radii']=pkg['radii'][:n];pkg['visibility_filter']=pkg['visibility_filter'][:n];pkg['viewspace_points']=SlicedStats(pkg['viewspace_points'],n)
  return pkg
 def actor_count(self):return sum(len(getattr(self.sg,n).get_xyz) for n in self.sg.obj_list)
 def update_lr(self,it):
  if self.sg.color_correction is not None:self.sg.color_correction.update_learning_rate(it)
  for n in self.sg.obj_list:getattr(self.sg,n).update_learning_rate(it)
  self.sg.actor_pose.update_learning_rate(it) if self.sg.actor_pose is not None else None;self.sg.sky_cubemap.update_learning_rate(it)
  if it%1000==0:
   for n in self.sg.obj_list:getattr(self.sg,n).oneupSHdegree()
  self.sg.active_sh_degree=self.gaussians.active_sh_degree
 @torch.no_grad()
 def update_actors(self,it,pkg,smoke=False):
  sg=self.sg
  if it<15000:
   g=pkg['joint_stats'].grad
   for n,(a,b) in pkg['actor_ranges'].items():
    if n=='background':continue
    m=getattr(sg,n);v=pkg['joint_visible'][a:b];m.max_radii2D[v]=torch.maximum(m.max_radii2D[v],pkg['joint_radii'][a:b][v].float())
    m.xyz_gradient_accum[v,0:1]+=g[a:b][v,:2].norm(dim=-1,keepdim=True);m.xyz_gradient_accum[v,1:2]+=g[a:b][v,2:].norm(dim=-1,keepdim=True);m.denom[v]+=1
   if smoke:
    if it==2:
     # Force an actual optimizer/topology change on one visible actor, bounded to one clone.
     names=[n for n in pkg['actor_ranges'] if n!='background']
     if names:
      m=getattr(sg,names[0]);grads=torch.zeros_like(m.denom);ids=(m.get_scaling.max(1).values<=m.percent_dense*m.extent).nonzero().flatten()
      if len(ids):grads[ids[0]]=1.;m.densify_and_clone(grads,.5,m.extent)
      else:
       grads[0]=1.;m.densify_and_split(grads,.5,m.extent)
   elif it>500 and it%100==0:
    for n in sg.obj_list:getattr(sg,n).densify_and_prune(.0002,.0075,it>3000)
   if not smoke and it%3000==0:
    for n in sg.obj_list:getattr(sg,n).reset_opacity()
  if it<25000:
   if sg.color_correction is not None:sg.color_correction.update_optimizer()
   for n in sg.obj_list:getattr(sg,n).update_optimizer()
   sg.actor_pose.update_optimizer() if sg.actor_pose is not None else None;sg.sky_cubemap.update_optimizer()
 def capture(self):return self.sg.save_state_dict(is_final=False,exclude_list=['background'])
 def restore(self,state):self.sg.load_state_dict(state,exclude_list=['background'])
 def hashes(self):
  models=[('background',self.gaussians)]+[(n,getattr(self.sg,n)) for n in self.sg.obj_list]+[('sky',self.sg.sky_cubemap)]
  return {name:{k:hashlib.sha256(getattr(m,k).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in PARAMS} for name,m in models}

def sky_loss(c,pkg,sky):
 mask=c.alpha_mask.cuda();mask=mask & ~c.guidance['obj_bound'].cuda().bool() if not cfg.model.nsg.include_obj else mask;sm=c.guidance['sky_mask'].cuda().bool();a=pkg['acc'].clamp(1e-6,1-1e-6)
 bce=torch.where(sm,-torch.log1p(-a),-torch.log(a))[mask].mean()
 valid=(1-torch.nn.functional.max_pool2d(1-sm.float()[None],7,1,3))[0].bool()&mask
 acc=sky.last_render['acc'].clamp(1e-6,1-1e-6);coverage=-acc[valid].log().mean() if valid.any() else acc.sum()*0
 return .05*bce+.01*coverage
