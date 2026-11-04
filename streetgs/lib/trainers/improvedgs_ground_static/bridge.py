import os,sys,json,random,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import importlib
for _name in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+_name)
for _name,_module in list(sys.modules.items()):
 if _name=='numpy.core' or _name.startswith('numpy.core.'):sys.modules.setdefault(_name.replace('numpy.core','numpy._core',1),_module)
REPO=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians')
STATIC=REPO/'lib/trainers/improvedgs_static'
sys.path.insert(0,str(STATIC/'native_source'));sys.path.insert(0,str(STATIC));sys.path.insert(0,str(REPO))
argv=sys.argv;sys.argv=[argv[0],'--config',os.environ['DYNAMIC_CONFIG']]
from lib.config import cfg
sys.argv=argv
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import loadguidance
from async_pipeline import Prefetch,cpu_payload,masks_for_sky
from lib.utils.batched_actor_adam import BatchedActorAdam
from scene.methods.densification_methods import compute_edge_map
from mesh_backend import gsplat_render
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

from lib.datasets.dataset import sceneLoadTypeCallbacks
from mesh_reader import read_t3
sceneLoadTypeCallbacks['T3Prepared']=read_t3

def dataset():
 cfg.data.write_input_snapshot=False
 cfg.data.write_obj_bounds=False
 ds=Dataset();cams=sorted(ds.train_cameras[1],key=lambda c:c.image_name)
 for i,c in enumerate(cams):c.uid=i;c.alpha_mask=masks_for_sky(c)['mask']
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
   self.sg.sky_cubemap.training_setup()
   random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  if self.sg.color_correction is not None:self.sg.color_correction.training_setup()
  self.sg.active_sh_degree=0
  self.merged_actor_adam=BatchedActorAdam() if cfg.optim.get("merge_actor_adam",False) else None
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
   if self.merged_actor_adam is not None and sg.obj_list:
    self.merged_actor_adam.step([getattr(sg,n).optimizer for n in sg.obj_list])
   else:
    for n in sg.obj_list:getattr(sg,n).update_optimizer()
   sg.actor_pose.update_optimizer() if sg.actor_pose is not None else None;sg.sky_cubemap.update_optimizer()
 def capture(self):return self.sg.save_state_dict(is_final=False,exclude_list=['background'])
 def restore(self,state):self.sg.load_state_dict(state,exclude_list=['background'])
 def hashes(self):
  models=[('background',self.gaussians)]+[(n,getattr(self.sg,n)) for n in self.sg.obj_list]
  result={name:{k:hashlib.sha256(getattr(m,k).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in PARAMS} for name,m in models}
  result['sky_cubemap']={'texture':hashlib.sha256(self.sg.sky_cubemap.sky_cube_map.detach().cpu().numpy().tobytes()).hexdigest()};return result

def sky_loss(c,pkg,sky):
 mask=c.alpha_mask.cuda();mask=mask & ~c.guidance['obj_bound'].cuda().bool() if not cfg.model.nsg.include_obj else mask;sm=c.guidance['sky_mask'].cuda().bool();a=pkg['acc'].clamp(1e-6,1-1e-6)
 bce=torch.where(sm,-torch.log1p(-a),-torch.log(a))[mask].mean()
 return .05*bce

# Fixed supplied affine poses, including per-frame nonuniform scale.
from lib.utils.general_utils import quaternion_to_matrix
from lib.models.actor_pose import ActorPose
_ACTORS=json.loads((Path(cfg.source_path).parent/'actors.json').read_text())
_AFFINE={}
for a in _ACTORS:
 for j,f in enumerate(a['frames']):
  M=torch.tensor(a['transforms'][j],dtype=torch.float32,device='cuda');A=M[:3,:3]/torch.tensor(a['size'],device='cuda')[None,:];_AFFINE[a['id'],f]=(A,M[:3,3])
_old_visibility=StreetGaussianModel.get_visibility
def _visibility(self,name):
 if name.startswith('obj_') and hasattr(self,'viewpoint_camera'):
  if (getattr(self,name).track_id,self.viewpoint_camera.meta['frame']) not in _AFFINE:return False
 return _old_visibility(self,name)
StreetGaussianModel.get_visibility=_visibility
ActorPose.get_tracking_translation=lambda self,tid,c:self.input_trans[c.meta['frame'],self.obj_info[tid]['column']]
ActorPose.get_tracking_rotation=lambda self,tid,c:self.input_rots[c.meta['frame'],self.obj_info[tid]['column']]
ActorPose.save_state_dict=lambda self,is_final:{}
ActorPose.load_state_dict=lambda self,state:None
from gsplat.cuda._wrapper import quat_scale_to_covar_preci
# A is fixed. Vectorized covariance transform is a 9x9 linear map, with one GEMM backward.
_AFFINE_K={key:torch.einsum('ai,bj->abij',A,A).reshape(9,9).contiguous() for key,(A,t) in _AFFINE.items()}
class MeshRenderModel:
 def __init__(self,sg):
  frame=sg.viewpoint_camera.meta['frame'];names=['background']+sg.graph_obj_list;models=[getattr(sg,n) for n in names]
  q=torch.cat([m.get_rotation for m in models]);scales=torch.cat([m.get_scaling for m in models])
  covars,_=quat_scale_to_covar_preci(q,scales,compute_covar=True,compute_preci=False,triu=False)
  xyz=[];cov=[];offset=0
  for name,m in zip(names,models):
   p=m.get_xyz;n=len(p);v=covars[offset:offset+n];offset+=n
   if name!='background':
    key=(m.track_id,frame);A,t=_AFFINE[key];p=torch.nn.functional.linear(p,A,t)
    v=torch.nn.functional.linear(v.reshape(n,9),_AFFINE_K[key]).reshape(n,3,3)
   xyz.append(p);cov.append(v)
  self.get_xyz=torch.cat(xyz);self.covars=torch.cat(cov);self.get_rotation=None;self.get_scaling=None;self.get_opacity=sg.get_opacity
  f=sg.get_features;self.get_features_dc=f[:,:1];self.get_features_rest=f[:,1:];self.active_sh_degree=sg.active_sh_degree
RenderModel=MeshRenderModel

from lib.models.sky_cubemap import SkyCubeMap
def _cube_capture(self,is_final):
 state={'params':self.state_dict()}
 if not is_final:state['optimizer']=self.optimizer.state_dict()
 return state
SkyCubeMap.save_state_dict=_cube_capture
