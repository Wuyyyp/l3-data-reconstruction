from bridge import *
from PIL import Image
MODE=os.environ.get('GROUND_MODE','anchored')
class GroundModel(torch.nn.Module):
 @property
 def initial_sh_degree(self):return int(os.environ.get('GROUND_SH_DEGREE','0'))
 def __init__(self,path,anchored=False):
  super().__init__();a=np.load(path);self.register_buffer('initial',torch.tensor(a['xyz'],device='cuda'));self.register_buffer('normals',torch.tensor(a['normals'],device='cuda'));self.register_buffer('rotation',torch.tensor(a['rotation'],device='cuda'));self.offset=torch.nn.Parameter(torch.zeros((len(a['xyz']),1),device='cuda'),requires_grad=anchored)
  f=torch.zeros((len(a['xyz']),(self.initial_sh_degree+1)**2,3),device='cuda');f[:,0]=(torch.tensor(a['rgb'],device='cuda')-.5)/.28209479177387814;self.features=torch.nn.Parameter(f);self.scale=torch.nn.Parameter(torch.full((len(f),2),np.log(.10),device='cuda'));self.opacity=torch.nn.Parameter(torch.full((len(f),1),2.1972246,device='cuda'))
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
 def active_sh_degree(self):
  # Restored ground tensors carry the degree, including legacy four-coefficient checkpoints.
  return {1: 0, 4: 1}[self.features.shape[1]]
 @property
 def joint_features(self):
  # Objects and background use SH1. Only pad for their shared rasterization call.
  if self.active_sh_degree==0:
   return torch.nn.functional.pad(self.features,(0,0,0,3))
  return self.features
 @property
 def get_features_dc(self):return self.features[:,:1]
 @property
 def get_features_rest(self):return self.features[:,1:]
 @property
 def covars(self):return quat_scale_to_covar_preci(self.get_rotation,self.get_scaling,compute_covar=True,compute_preci=False,triu=False)[0]
 def capture(self):return dict(model=self.state_dict(),optimizer=self.optimizer.state_dict())
 def restore(self,s):self.load_state_dict(s['model']);self.optimizer.load_state_dict(s['optimizer'])
 def freeze_geometry(self):
  if not hasattr(self,'_frozen_geometry'):self._frozen_geometry=[p.detach().clone() for p in [self.offset,self.scale,self.opacity]]
  for p in [self.offset,self.scale,self.opacity]:p.requires_grad_(False);p.grad=None


from adaptive_ground import install
GroundModel=install(GroundModel)
if os.environ.get('GROUND_SHARED_SURFACE'):
 from shared_surface import install as install_shared_surface
 GroundModel=install_shared_surface(GroundModel)
_BaseScene=DynamicScene
class DynamicScene(_BaseScene):
 def __init__(self,ds,cams,bg,out,initialize=True):
  super().__init__(ds,cams,bg,out,initialize)
  self.ground=None if MODE=='baseline' else GroundModel(os.environ['GROUND_INIT'],True)
  self.iteration=0;self.masks={};self.audit_out=Path(out)
  self.ground_pkg=None
  (self.audit_out/'ground_split_policy.json').write_text(json.dumps(dict(enabled=True,from_step=500,until_step=15000,interval=100,threshold=float(os.environ.get('GROUND_SPLIT_THRESHOLD','0.0002')),point_cap=int(os.environ.get('GROUND_POINT_CAP','1200000')),warmup_freezes_geometry=False)))
 def ground_mask(self,c):
  if c.image_name not in self.masks:
   fid,cam=c.image_name.split('_');p=Path(os.environ['GROUND_MASKS'])/('cam_'+cam)/(fid+'.npy');m=np.load(p);m=np.asarray(Image.fromarray(m).resize((c.image_width,c.image_height),Image.Resampling.NEAREST)).copy();self.masks[c.image_name]=torch.from_numpy(m)[None].bool()
  from collections import OrderedDict
  if not hasattr(self,'_gpu_ground_masks'):self._gpu_ground_masks=OrderedDict()
  key=(c.image_name,c.image_width,c.image_height)
  if key not in self._gpu_ground_masks:
   self._gpu_ground_masks[key]=(self.masks[c.image_name].cuda()&c.guidance['mask'].cuda().bool()&~c.guidance['obj_bound'].cuda().bool())
   if len(self._gpu_ground_masks)>2:self._gpu_ground_masks.popitem(last=False)
  self._gpu_ground_masks.move_to_end(key)
  return self._gpu_ground_masks[key]
 def warmup(self):return self.iteration<=int(os.environ.get('GROUND_WARMUP','3000'))
 def update_lr(self,it):
  self.iteration=it;super().update_lr(it)

 def render(self,c,bg,pipe,background,**kw):
  parse_mesh_camera(self.sg,c);model=MeshRenderModel(self.sg);n=len(bg.get_xyz);ground_start=len(model.get_xyz)
  if self.ground is not None:
   g=self.ground
   for k in ['get_xyz','get_opacity','get_features_dc']:setattr(model,k,torch.cat([getattr(model,k),getattr(g,k)],0))
   model.get_features_rest=torch.cat([model.get_features_rest,g.joint_features[:,1:]],0)
   model.covars=torch.cat([model.covars,g.covars],0)
  if 'pixel_weights' in kw:
   kw['pixel_weights']=self.edge(c);pkg=gsplat_render(c,model,pipe,background,**kw);return {k:v[:n] for k,v in pkg.items()}
  pkg=gsplat_render(c,model,pipe,background,**kw);pkg['actor_ranges']=dict(self.sg.graph_gaussian_range)
  pkg['ground_range']=(ground_start,len(model.get_xyz));pkg['joint_radii']=pkg['radii'];pkg['joint_visible']=pkg['visibility_filter'];pkg['joint_stats']=pkg['viewspace_points']
  pkg['radii']=pkg['radii'][:n];pkg['visibility_filter']=pkg['visibility_filter'][:n];pkg['viewspace_points']=SlicedStats(pkg['viewspace_points'],n)
  assert max(b for a,b in pkg['actor_ranges'].values())==ground_start
  return pkg
 def audit_finite(self,gradients=False,optimizer=False):
  checks=[];labels=[]
  def add(t,label):
   if torch.is_tensor(t) and t.is_floating_point():checks.append(torch.isfinite(t).all());labels.append(label)
  for name in ['background']+self.sg.obj_list:
   m=getattr(self.sg,name)
   for key in PARAMS:
    p=getattr(m,key);add(p,name+key)
    if gradients and p.grad is not None:add(p.grad,name+key+'.grad')
   if optimizer:
    for key,state in m.optimizer.state.items():
     for field,value in state.items():add(value,name+'.adam.'+field)
  for key,p in self.sg.sky_cubemap.named_parameters():
   add(p,'sky.'+key)
   if gradients and p.grad is not None:add(p.grad,'sky.'+key+'.grad')
  if optimizer and getattr(self.sg.sky_cubemap,'optimizer',None) is not None:
   for state in self.sg.sky_cubemap.optimizer.state.values():
    for key,value in state.items():add(value,'sky.adam.'+key)
  if self.ground is not None:
   for key,p in self.ground.named_parameters():
    add(p,'ground.'+key)
    if gradients and p.grad is not None:add(p.grad,'ground.'+key+'.grad')
   if optimizer:
    for state in self.ground.optimizer.state.values():
     for key,v in state.items():add(v,'ground.adam.'+key)
  # Adam step counters may live on CPU; reduce on CUDA without copying model tensors.
  valid=torch.stack([x.to('cuda') for x in checks]).cpu().tolist()
  assert all(valid), 'Nonfinite tensors: '+str([k for k,v in zip(labels,valid) if not v])
  if optimizer:(self.audit_out/'finite_validation.json').write_text(json.dumps(dict(passed=True,iteration=self.iteration,tensors=len(valid),optimizer=True)))
 def update_actors(self,it,pkg,smoke=False):
  super().update_actors(it,pkg,smoke)
 def accumulate_ground(self,pkg):
  if self.ground_pkg is not None:self.ground.accumulate(self.ground_pkg)
  if not self.warmup():self.ground.accumulate(pkg,*pkg['ground_range'])
 def densify_ground(self,it,smoke=False):
  event=self.ground.densify(it,smoke)
  if event:
   with (self.audit_out/'ground_split_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  self.ground_pkg=None
  if smoke and it in [3,8,18]:
   self.audit_finite(optimizer=True)
   (self.audit_out/'ground_topology_validation.json').write_text(json.dumps(dict(iteration=it,split_enabled=True,events=len(self.ground.events),points=len(self.ground.initial),passed=True)))
 def ground_loss(self,c,pipe,background,gt):
  if self.ground is None:return gt.sum()*0,{}
  from fast_losses import masked_mean,masked_ssim
  mask=self.ground_mask(c).cuda().bool();pkg=gsplat_render(c,self.ground,pipe,background,track_gradients=True);self.ground_pkg=pkg;pred=pkg['render'];l1=masked_mean((pred-gt).abs(),mask)
  invalid=torch.nn.functional.pad((~mask).float()[None],(5,5,5,5),value=1);centers=(torch.nn.functional.max_pool2d(invalid,11,1)==0)[0]
  rgb=torch.where(centers.any(),.8*l1+.2*(1-masked_ssim(pred,gt,mask)),l1)
  anchor=((.05 if os.environ.get('GROUND_SHARED_SURFACE') else .10)*torch.tanh(self.ground.offset)).square().mean();loss=.2*rgb+10*anchor
  return loss,dict(ground_l1=l1.detach(),ground_anchor=anchor.detach(),ground_pixels=mask.sum().detach(),ground_points=len(self.ground.initial))
 def step_ground(self):
  if self.ground is not None:
   if os.environ.get('SMOKE')=='1':assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.ground.parameters())
   self.ground.optimizer.step();self.ground.optimizer.zero_grad(set_to_none=True)
   if hasattr(self.ground,'_frozen_geometry'):
    assert all(torch.equal(p,q) for p,q in zip([self.ground.offset,self.ground.scale,self.ground.opacity],self.ground._frozen_geometry)), 'Frozen ground geometry changed'
 def capture(self):return dict(street=super().capture(),ground=None if self.ground is None else self.ground.capture(),mode=MODE,iteration=self.iteration)
 def restore(self,s):
  assert s['mode']==MODE;super().restore(s['street']);self.iteration=s['iteration']
  if self.ground is not None:self.ground.restore(s['ground'])






