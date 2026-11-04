import os,sys,json,random,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import importlib

# Reuse the compatible compiled extension; do not trigger JIT compilation.
import importlib.util
_ext_path=os.environ.get('GSPLAT_CUDA_SO',str(Path(__file__).resolve().parents[3]/'cache/improvedgs_static/torch_extensions/gsplat_cuda/gsplat_cuda.so'))
if not Path(_ext_path).is_file():
 raise FileNotFoundError(f'Set GSPLAT_CUDA_SO to a compatible built gsplat extension: {_ext_path}')
_spec=importlib.util.spec_from_file_location('gsplat_cuda',_ext_path)
_ext=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_ext)
sys.modules['gsplat.csrc']=_ext
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
from fast_actor_adam import BatchedActorAdam
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

# Prepared source PLYs are authoritative even for sparse native targets.
import lib.models.gaussian_model_actor as _actor_module
_namespace=dict(vars(_actor_module))
exec(compile((Path(__file__).parent/'prepared_actor_create.py.txt').read_text(),str(Path(__file__).parent/'prepared_actor_create.py.txt'),'exec'),_namespace)
_actor_module.GaussianModelActor.create_from_pcd=_namespace['create_from_pcd']

class DynamicScene:
 def __init__(self,ds,cams,bg,out,initialize=True):
  self.test_cameras=ds.test_cameras[1];self.cameras=cams;self.gaussians=bg;self.model_path=str(out);self.cameras_extent=20.;self.sg=StreetGaussianModel(ds.scene_info.metadata)
  if cfg.model.sky.get('geometry')=='hemisphere':
   from lib.models.gaussian_model_hemisphere import GaussianModelSky
   self.sg.sky_cubemap=GaussianModelSky()
  self.sg.background=BackgroundAdapter(bg);self.sg.set_visibility(list(self.sg.model_name_id));self.edges={}
  if initialize:
   for name in self.sg.obj_list:
    m=getattr(self.sg,name);m.create_from_pcd(20.);assert not m.random_initialization, 'Prepared actor unexpectedly randomized: '+name;m.training_setup()
   self.sg.actor_pose.training_setup() if self.sg.actor_pose is not None else None
   rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
   if cfg.model.sky.get('geometry')=='hemisphere':
    _saved_lr_steps=cfg.optim.position_lr_max_steps;cfg.optim.position_lr_max_steps=25000
    try:self.sg.sky_cubemap.initialize(bg.get_xyz,cams)
    finally:cfg.optim.position_lr_max_steps=_saved_lr_steps
   else:self.sg.sky_cubemap.training_setup()
   random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
  if self.sg.color_correction is not None:self.sg.color_correction.training_setup()
  self.sg.active_sh_degree=0
  self.merged_actor_adam=BatchedActorAdam() if cfg.optim.get("merge_actor_adam",False) else None
 def getTrainCameras(self):return self.cameras
 def getTestCameras(self):return self.test_cameras
 def edge(self,c):
  if c.uid not in self.edges:
   im,guide=cpu_payload(c);edge=compute_edge_map(im);valid=c.alpha_mask & ~guide['obj_bound'].bool()
   invalid=torch.nn.functional.max_pool2d((~valid).float()[None],3,1,1)[0,0].bool();self.edges[c.uid]=edge*(~invalid)
  return self.edges[c.uid]
 def render(self,c,bg,pipe,background,**kw):
  parse_mesh_camera(self.sg,c);model=RenderModel(self.sg);n=len(bg.get_xyz)
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
  self.sg.sky_cubemap.update_learning_rate(it)
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
    m=getattr(sg,n);v=pkg['joint_visible'][a:b];m.max_radii2D.copy_(torch.where(v,torch.maximum(m.max_radii2D,pkg['joint_radii'][a:b].float()),m.max_radii2D))
    m.xyz_gradient_accum[:,0:1]+=g[a:b,:2].norm(dim=-1,keepdim=True)*v[:,None];m.xyz_gradient_accum[:,1:2]+=g[a:b,2:].norm(dim=-1,keepdim=True)*v[:,None];m.denom+=v[:,None]
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
  if it<=cfg.train.iterations:
   if sg.color_correction is not None:sg.color_correction.update_optimizer()
   if self.merged_actor_adam is not None and sg.obj_list:
    self.merged_actor_adam.step([getattr(sg,n).optimizer for n in sg.obj_list])
   else:
    for n in sg.obj_list:getattr(sg,n).update_optimizer()
   sky=sg.sky_cubemap
   if smoke and it==2 and hasattr(sky,'_xyz'):
    sky.last_render=None;g=torch.zeros_like(sky.denom);g[:min(32,len(g))]=1.
    before=len(sky._xyz);sky.densify_and_clone(g,.5,sky.radius);sky.densify_and_split(g,.5,sky.radius)
    assert len(sky._xyz)>before
    (Path(self.model_path)/'sky_smoke_topology.json').write_text(json.dumps(dict(before=before,after=len(sky._xyz),max_points=cfg.model.sky.get('max_points'))))
   sky.update_optimizer()
 def capture(self):return self.sg.save_state_dict(is_final=False,exclude_list=['background'])
 def restore(self,state):self.sg.load_state_dict(state,exclude_list=['background'])
 def hashes(self):
  models=[('background',self.gaussians)]+[(n,getattr(self.sg,n)) for n in self.sg.obj_list]
  result={name:{k:hashlib.sha256(getattr(m,k).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in PARAMS} for name,m in models}
  sky=self.sg.sky_cubemap
  result['sky_cubemap']={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in sky.named_parameters()};return result

def sky_loss(c,pkg,sky):
 if not cfg.data.get('use_sky_mask',True):return pkg['render'].sum()*0
 mask=c.guidance['mask'].cuda().bool();mask=mask & ~c.guidance['obj_bound'].cuda().bool() if not cfg.model.nsg.include_obj else mask;sm=c.guidance['sky_mask'].cuda().bool();a=pkg['acc'].clamp(1e-6,1-1e-6)
 from fast_losses import masked_mean
 bce=masked_mean(torch.where(sm,-torch.log1p(-a),-torch.log(a)),mask)
 return .05*bce

# Fixed supplied affine poses, including per-frame nonuniform scale.
from lib.utils.general_utils import quaternion_to_matrix
from lib.models.actor_pose import ActorPose
_ACTORS=json.loads((Path(cfg.source_path).parent/'actors.json').read_text())
_AFFINE={}
for a in _ACTORS:
 for j,f in enumerate(a['frames']):
  M=torch.tensor(a['transforms'][j],dtype=torch.float32,device='cuda');A=M[:3,:3]/torch.tensor(a['size'],device='cuda')[None,:];_AFFINE[a['id'],f]=(A,M[:3,3])
_VIEW_IDS={a['id'] for a in _ACTORS if a.get('view_transforms')}
for a in _ACTORS:
 for name,matrix in a.get('view_transforms',{}).items():
  M=torch.tensor(matrix,dtype=torch.float32,device='cuda');_AFFINE[a['id'],name]=(M[:3,:3]/torch.tensor(a['size'],device='cuda')[None,:],M[:3,3])
def affine_key(tid,c):
 return (tid,c.image_name) if tid in _VIEW_IDS else (tid,c.meta['frame'])
_old_visibility=StreetGaussianModel.get_visibility
def _visibility(self,name):
 if name.startswith('obj_') and hasattr(self,'viewpoint_camera'):
  if affine_key(getattr(self,name).track_id,self.viewpoint_camera) not in _AFFINE:return False
 return _old_visibility(self,name)
StreetGaussianModel.get_visibility=_visibility
ActorPose.get_tracking_translation=lambda self,tid,c:self.input_trans[c.meta['frame'],self.obj_info[tid]['column']]
ActorPose.get_tracking_rotation=lambda self,tid,c:self.input_rots[c.meta['frame'],self.obj_info[tid]['column']]
ActorPose.save_state_dict=lambda self,is_final:{}
ActorPose.load_state_dict=lambda self,state:None
from gsplat.cuda._wrapper import quat_scale_to_covar_preci
# A is fixed. Vectorized covariance transform is a 9x9 linear map, with one GEMM backward.
_AFFINE_K={key:torch.einsum('ai,bj->abij',A,A).reshape(9,9).contiguous() for key,(A,t) in _AFFINE.items()}
def parse_mesh_camera(sg,c):
 sg.viewpoint_camera=c;sg.background.set_background_mask(c)
 sg.frame=c.meta['frame'];sg.frame_idx=c.meta['frame_idx'];sg.frame_is_val=c.meta['is_val']
 names=['background'] if sg.get_visibility('background') else []
 sg.graph_obj_list=[]
 if sg.include_obj:
  for name in sg.obj_list:
   m=getattr(sg,name)
   if m.start_frame<=sg.frame<=m.end_frame and sg.get_visibility(name):sg.graph_obj_list.append(name)
 names+=sg.graph_obj_list;sg.graph_gaussian_range={};offset=0
 for name in names:
  n=len(getattr(sg,name).get_xyz);sg.graph_gaussian_range[name]=[offset,offset+n];offset+=n
 sg.num_gaussians=offset

def actor_affines(sg,names,counts):
 from collections import OrderedDict
 key=(sg.frame,tuple(names),tuple(counts))
 if not hasattr(sg,'_mesh_affine_cache'):sg._mesh_affine_cache=OrderedDict()
 cache=sg._mesh_affine_cache
 if key not in cache:
  transforms=[_AFFINE[getattr(sg,name).track_id,sg.frame] for name in names]
  sizes=torch.tensor(counts,device='cuda',dtype=torch.long)
  ids=torch.repeat_interleave(torch.arange(len(names),device='cuda'),sizes,output_size=sum(counts))
  A=torch.stack([a for a,t in transforms]).index_select(0,ids)
  t=torch.stack([t for a,t in transforms]).index_select(0,ids)
  cache[key]=(A,t)
  if len(cache)>2:cache.popitem(last=False)
 cache.move_to_end(key)
 return cache[key]

class MeshRenderModel:
 def __init__(self,sg):
  names=['background']+sg.graph_obj_list;models=[getattr(sg,n) for n in names]
  q=torch.cat([m.get_rotation for m in models]);scales=torch.cat([m.get_scaling for m in models])
  covars,_=quat_scale_to_covar_preci(q,scales,compute_covar=True,compute_preci=False,triu=False)
  xyz=torch.cat([m.get_xyz for m in models]);n=len(models[0].get_xyz)
  if sg.graph_obj_list:
   worlds=[];covariances=[];offset=n
   for name,m in zip(sg.graph_obj_list,models[1:]):
    count=len(m.get_xyz);key=affine_key(m.track_id,sg.viewpoint_camera);A,t=_AFFINE[key];K=_AFFINE_K[key]
    worlds.append(xyz[offset:offset+count]@A.T+t)
    covariances.append((covars[offset:offset+count].reshape(-1,9)@K.T).reshape(-1,3,3));offset+=count
   world=torch.cat(worlds);covariance=torch.cat(covariances)
   xyz=torch.cat([xyz[:n],world]);covars=torch.cat([covars[:n],covariance])
  self.get_xyz=xyz;self.covars=covars;self.get_rotation=None;self.get_scaling=None
  self.get_opacity=sg.get_opacity
  f=sg.get_features;self.get_features_dc=f[:,:1];self.get_features_rest=f[:,1:];self.active_sh_degree=sg.active_sh_degree
RenderModel=MeshRenderModel

from lib.models.sky_cubemap import SkyCubeMap
def _cube_capture(self,is_final):
 state={'params':self.state_dict()}
 if not is_final:state['optimizer']=self.optimizer.state_dict()
 return state
SkyCubeMap.save_state_dict=_cube_capture
