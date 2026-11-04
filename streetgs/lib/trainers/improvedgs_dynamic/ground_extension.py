"""Opt-in ground extension for ImprovedGS. Existing trainer remains unchanged."""
from bridge import *
from PIL import Image
from plyfile import PlyData
from lib.utils.general_utils import quaternion_to_matrix,matrix_to_quaternion
from prepared_ground_reader import read_prepared
import lib.datasets.dataset as dataset_module
dataset_module.sceneLoadTypeCallbacks['PreparedGround']=read_prepared
MODE=os.environ.get('GROUND_MODE','baseline')
# Identical sky initialization for all variants, independent of ground/background partition.
from lib.models.gaussian_model_sky import GaussianModelSky
_sky_initialize=GaussianModelSky.initialize
def _common_sky_initialize(self,points,cameras):
 a=PlyData.read(Path(cfg.source_path).parent/'model/input_ply/points3D_bkgd.ply')['vertex']
 common=torch.tensor(np.column_stack([a[k] for k in 'xyz']),device=points.device,dtype=points.dtype)
 return _sky_initialize(self,common,cameras)
GaussianModelSky.initialize=_common_sky_initialize
class GroundModel(torch.nn.Module):
 def __init__(self,path,anchored=False):
  super().__init__();a=np.load(path);self.register_buffer('initial',torch.tensor(a['xyz'],device='cuda'));self.register_buffer('normals',torch.tensor(a['normals'],device='cuda'));self.register_buffer('rotation',torch.tensor(a['rotation'],device='cuda'));self.offset=torch.nn.Parameter(torch.zeros((len(a['xyz']),1),device='cuda'),requires_grad=anchored)
  f=torch.zeros((len(a['xyz']),4,3),device='cuda');f[:,0]=(torch.tensor(a['rgb'],device='cuda')-.5)/.28209479177387814;self.features=torch.nn.Parameter(f);self.scale=torch.nn.Parameter(torch.full((len(f),2),np.log(.10),device='cuda'));self.opacity=torch.nn.Parameter(torch.full((len(f),1),2.1972246,device='cuda'));self.active_sh_degree=1
  self.optimizer=torch.optim.Adam([{'params':[self.features],'lr':.0025},{'params':[self.scale],'lr':.001},{'params':[self.opacity],'lr':.01}]+([{'params':[self.offset],'lr':.0001}] if anchored else []))
 @property
 def get_xyz(self):return self.initial+self.normals*(.10*torch.tanh(self.offset))
 @property
 def get_rotation(self):return self.rotation
 @property
 def get_scaling(self):return torch.cat([self.scale.clamp(np.log(.005),np.log(.5)).exp(),torch.full_like(self.scale[:,:1],.001)],1)
 @property
 def get_opacity(self):return self.opacity.sigmoid()
 @property
 def get_features_dc(self):return self.features[:,:1]
 @property
 def get_features_rest(self):return self.features[:,1:]
 def capture(self):return dict(model=self.state_dict(),optimizer=self.optimizer.state_dict())
 def restore(self,s):self.load_state_dict(s['model']);self.optimizer.load_state_dict(s['optimizer'])
 def freeze_geometry(self):
  if not hasattr(self,'_frozen_geometry'):self._frozen_geometry=[p.detach().clone() for p in [self.offset,self.scale,self.opacity]]
  for p in [self.offset,self.scale,self.opacity]:p.requires_grad_(False);p.grad=None

_BaseScene=DynamicScene
class DynamicScene(_BaseScene):
 def __init__(self,ds,cams,bg,out,initialize=True):
  super().__init__(ds,cams,bg,out,initialize)
  self.sg.actor_pose=None  # Exact supplied affine tracks; no pose optimizer/state required.
  self.ground=None if MODE=='baseline' else GroundModel(os.environ['GROUND_INIT'],MODE=='anchored' or MODE=='two_stage')
  self.iteration=0;self.masks={};self.pose_by={}
  for a in json.loads((Path(cfg.source_path).parent/'actors.json').read_text()):
   for fid,M in zip(a['frames'],a['transforms']):
    M=torch.tensor(M,device='cuda',dtype=torch.float32);self.pose_by[a['id'],fid]=(M[:3,:3]/torch.tensor(a['size'],device='cuda')[None,:],M[:3,3])
 def ground_mask(self,c):
  if c.image_name not in self.masks:
   fid,cam=c.image_name.split('_');p=Path(os.environ['GROUND_MASKS'])/('cam_'+cam)/(fid+'.npy');m=np.load(p);m=np.asarray(Image.fromarray(m).resize((c.image_width,c.image_height),Image.Resampling.NEAREST)).copy();self.masks[c.image_name]=torch.from_numpy(m)[None]
  return self.masks[c.image_name].to(c.alpha_mask.device)&c.alpha_mask
 def warmup(self):return MODE=='two_stage' and self.iteration<=int(os.environ.get('GROUND_WARMUP','3000'))
 def update_lr(self,it):
  self.iteration=it;super().update_lr(it)
  if MODE=='two_stage' and not self.warmup():self.ground.freeze_geometry()
 def render(self,c,bg,pipe,background,**kw):
  self.sg.viewpoint_camera=c;self.sg.frame=c.meta['frame'];self.sg.frame_idx=c.meta['frame_idx'];self.sg.frame_is_val=c.meta['is_val'];self.sg.active_sh_degree=bg.active_sh_degree
  self.sg.graph_obj_list=[name for name in self.sg.obj_list if (getattr(self.sg,name).track_id,c.meta['frame']) in self.pose_by]
  # Exact actor-presence gating, including gaps in tracks.
  self.sg.graph_obj_list=[n for n in self.sg.graph_obj_list if (getattr(self.sg,n).track_id,c.meta['frame']) in self.pose_by]
  pieces=[SimpleNamespace(get_xyz=bg.get_xyz,get_scaling=bg.get_scaling,get_rotation=bg.get_rotation,get_opacity=bg.get_opacity,get_features_dc=bg.get_features_dc,get_features_rest=bg.get_features_rest)]
  n=len(bg.get_xyz);ranges={'background':(0,n)};cursor=n
  for name in self.sg.graph_obj_list:
   obj=getattr(self.sg,name);A,t=self.pose_by[obj.track_id,c.meta['frame']];L=A@quaternion_to_matrix(obj.get_rotation)@torch.diag_embed(obj.get_scaling);val,axes=torch.linalg.eigh(L@L.transpose(-1,-2));axes=torch.cat([axes[:,:,:2],axes[:,:,2:]*torch.linalg.det(axes)[:,None,None]],-1)
   features=obj.get_features_fourier(c.meta['frame']);pieces.append(SimpleNamespace(get_xyz=obj.get_xyz@A.T+t,get_scaling=val.clamp_min(1e-12).sqrt(),get_rotation=matrix_to_quaternion(axes),get_opacity=obj.get_opacity,get_features_dc=features[:,:1],get_features_rest=features[:,1:]));ranges[name]=(cursor,cursor+len(obj.get_xyz));cursor+=len(obj.get_xyz)
  ground_start=cursor
  if self.ground is not None:pieces.append(self.ground)
  keys=['get_xyz','get_scaling','get_rotation','get_opacity','get_features_dc','get_features_rest'];model=SimpleNamespace(**{k:torch.cat([getattr(p,k) for p in pieces],0) for k in keys},active_sh_degree=bg.active_sh_degree)
  if 'pixel_weights' in kw:
   kw['pixel_weights']=self.edge(c);pkg=gsplat_render(c,model,pipe,background,**kw);return {k:v[:n] for k,v in pkg.items()}
  pkg=gsplat_render(c,model,pipe,background,**kw);pkg['actor_ranges']=ranges;pkg['ground_range']=(ground_start,len(model.get_xyz));pkg['joint_radii']=pkg['radii'];pkg['joint_visible']=pkg['visibility_filter'];pkg['joint_stats']=pkg['viewspace_points'];pkg['radii']=pkg['radii'][:n];pkg['visibility_filter']=pkg['visibility_filter'][:n];pkg['viewspace_points']=SlicedStats(pkg['viewspace_points'],n)
  assert n==ranges['background'][1] and ground_start==max(b for a,b in ranges.values())
  return pkg
 def ground_loss(self,c,pipe,background,gt):
  if self.ground is None:return gt.sum()*0,{}
  mask=self.ground_mask(c).cuda().bool();pkg=gsplat_render(c,self.ground,pipe,background,track_gradients=False);pred=pkg['render'];l1=(pred-gt).abs()[mask.expand_as(pred)].mean() if mask.any() else pred.sum()*0
  invalid=torch.nn.functional.pad((~mask).float()[None],(5,5,5,5),value=1);centers=(torch.nn.functional.max_pool2d(invalid,11,1)==0)[0]
  from utils.static_losses import masked_ssim
  rgb=.8*l1+.2*(1-masked_ssim(pred,gt,mask)) if centers.any() else l1
  anchor=(.10*torch.tanh(self.ground.offset)).square().mean();loss=.2*rgb+10*anchor
  return loss,dict(ground_l1=float(l1.detach()),ground_anchor=float(anchor.detach()),ground_pixels=int(mask.sum()),ground_points=len(self.ground.initial))
 def step_ground(self):
  if self.ground is not None:
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.ground.parameters())
   self.ground.optimizer.step();self.ground.optimizer.zero_grad(set_to_none=True)
   if hasattr(self.ground,'_frozen_geometry'):
    assert all(torch.equal(p,q) for p,q in zip([self.ground.offset,self.ground.scale,self.ground.opacity],self.ground._frozen_geometry)), 'Frozen ground geometry changed'
 def capture(self):return dict(street=super().capture(),ground=None if self.ground is None else self.ground.capture(),mode=MODE,iteration=self.iteration)
 def restore(self,s):
  assert s['mode']==MODE;super().restore(s['street']);self.iteration=s['iteration']
  if self.ground is not None:self.ground.restore(s['ground'])
  if MODE=='two_stage' and not self.warmup():self.ground.freeze_geometry()





