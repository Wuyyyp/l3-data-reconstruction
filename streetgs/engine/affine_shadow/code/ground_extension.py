from bridge import *
from PIL import Image
from functools import lru_cache

@lru_cache(maxsize=1)
def wys_full_frame(source):
 manifest=Path(source).parent/'input_manifest.json'
 return manifest.is_file() and json.loads(manifest.read_text()).get('geometry_version')=='l3-wys-full-frame-v1'

MODE=os.environ.get('GROUND_MODE','anchored')
class GroundModel(torch.nn.Module):
 initial_sh_degree=1  # The vehicle shadow layer also derives from this base.
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
from ground_topology import install as install_topology
GroundModel=install_topology(GroundModel)
from shared_surface import install as install_shared_surface
FlatGroundModel=GroundModel
GroundModel=install_shared_surface(GroundModel)
_BaseScene=DynamicScene
class DynamicScene(_BaseScene):
 def __init__(self,ds,cams,bg,out,initialize=True):
  super().__init__(ds,cams,bg,out,initialize)
  self.ground=None if MODE=='baseline' else GroundModel(os.environ['GROUND_INIT'],True)
  self.iteration=0;self.masks={};self.audit_out=Path(out)
  self.ground_pkg=None
  (self.audit_out/'ground_split_policy.json').write_text(json.dumps(dict(enabled=True,from_step=500,until_step=10000,interval=100,threshold=float(os.environ.get('GROUND_SPLIT_THRESHOLD','0.0002')),point_cap=int(os.environ.get('GROUND_POINT_CAP','1200000')),warmup_freezes_geometry=False)))
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
  super().update_lr(max(1,it-int(os.environ.get('GROUND_WARMUP','10000'))));self.iteration=it

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
  event=self.ground.densify(it,smoke) if self.warmup() else None
  if event:
   with (self.audit_out/'ground_split_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  self.ground_pkg=None
  if smoke and it in [3,8,18]:
   self.audit_finite(optimizer=True)
   (self.audit_out/'ground_topology_validation.json').write_text(json.dumps(dict(iteration=it,split_enabled=True,events=len(self.ground.events),points=len(self.ground.initial),passed=True)))
 def ground_loss(self,c,pipe,background,gt):
  if self.ground is None:return gt.sum()*0,{}
  from fast_losses import masked_mean,masked_ssim
  mask=self.ground_mask(c).cuda().bool();pkg=gsplat_render(c,self.ground,pipe,background,track_gradients=True);self.ground_pkg=pkg;pred=self.sg.color_correction(c,pkg['render']);l1=masked_mean((pred-gt).abs(),mask)
  invalid=torch.nn.functional.pad((~mask).float()[None],(5,5,5,5),value=1);centers=(torch.nn.functional.max_pool2d(invalid,11,1)==0)[0]
  rgb=torch.where(centers.any(),.8*l1+.2*(1-masked_ssim(pred,gt,mask)),l1)
  anchor=(.10*torch.tanh(self.ground.offset)).square().mean()
  depth=c.guidance['lidar_depth'].cuda();dm=mask&torch.isfinite(depth)&(depth>=.1)&(depth<=200)
  target=torch.where(dm,depth.clamp_min(.1).reciprocal(),torch.zeros_like(depth))
  invloss=((pkg['depth']-target).abs()*dm).mean()
  dw=1.0*(.01**(self.iteration/30000.))
  loss=.2*rgb+10*anchor+dw*invloss
  if os.environ.get('SMOKE')=='1':
   assert dm.any(), 'No valid ground LiDAR depth'
   assert pkg['depth'].requires_grad

  return loss,dict(ground_l1=l1.detach(),ground_anchor=anchor.detach(),ground_pixels=mask.sum().detach(),ground_points=len(self.ground.initial),ground_invdepth_l1=invloss.detach(),ground_depth_pixels=dm.sum().detach(),ground_depth_weight=dw)
 def step_ground(self):
  if self.ground is not None:
   if os.environ.get('SMOKE')=='1':assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.ground.parameters())
   cc=self.sg.color_correction
   if os.environ.get('SMOKE')=='1':
    assert cc.affine_trans.grad is not None and torch.isfinite(cc.affine_trans.grad).all() and cc.affine_trans.grad.abs().sum()>0
   self.ground.optimizer.step();self.ground.optimizer.zero_grad(set_to_none=True)
   if self.warmup():cc.update_learning_rate(self.iteration);cc.update_optimizer()
   if hasattr(self.ground,'_frozen_geometry'):
    assert all(torch.equal(p,q) for p,q in zip([self.ground.offset,self.ground.scale,self.ground.opacity],self.ground._frozen_geometry)), 'Frozen ground geometry changed'
 def capture(self):return dict(street=super().capture(),ground=None if self.ground is None else self.ground.capture(),mode=MODE,iteration=self.iteration)
 def restore(self,s):
  assert s['mode']==MODE;super().restore(s['street']);self.iteration=s['iteration']
  if self.ground is not None:self.ground.restore(s['ground'])







class OffsetStats:
 def __init__(self,stats,offset):self.stats=stats;self.offset=offset
 @property
 def grad(self):return self.stats.grad[self.offset:]

# Isolated ground + actors experiment: deliberately exclude background and sky.
import bridge as br
import surface_actor as sa
from surface_actor import densify as actor_densify, merge_prune
_BaseGroundScene=DynamicScene
class DynamicScene(_BaseGroundScene):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.frozen_counts=None;self.frozen_opacity=None;self.joint_begin=1 if os.environ.get("SMOKE")=="1" else 5000;self.frozen_background_count=None;self.native_ids={a["id"] for a in br._ACTORS if a.get("source")=="native_lidar"};self.early_masks={}
  if kwargs.get("initialize",True):
   for n in self.sg.obj_list:sa.init(getattr(self.sg,n))
  assert len(self.sg.obj_list)==len(br._ACTORS)
  for n in self.sg.obj_list:getattr(self.sg,n).active_sh_degree=1
  self.sg.active_sh_degree=1
 def warmup(self):return False
 def joint_active(self):return self.iteration>self.joint_begin
 def joint_step(self):return self.iteration-self.joint_begin
 def update_lr(self,it):
  self.iteration=it
  self.sg.color_correction.update_learning_rate(it)
  for n in self.sg.obj_list:
   m=getattr(self.sg,n);m.update_learning_rate(max(1,it-self.joint_begin) if int(m.track_id) in self.native_ids else it)
  self.sg.active_sh_degree=1
  if self.joint_active():self.sg.sky_cubemap.update_learning_rate(self.joint_step())
  if it>(3 if os.environ.get("SMOKE")=="1" else 10000):
   self.ground.opacity.requires_grad_(False);self.ground.opacity.grad=None
   for n in self.sg.obj_list:
    m=getattr(self.sg,n)
    if os.environ.get('SMOKE')=='1' or int(m.track_id) not in self.native_ids or it>20000:m._opacity.requires_grad_(False);m._opacity.grad=None
 def shape_loss(self,pkg):
  vals=[sa.loss(getattr(self.sg,n)) for n in pkg["actor_ranges"]]
  return torch.stack(vals).mean() if vals else self.ground.features.sum()*0
 def render(self,c,bg,pipe,background,**kw):
  br.parse_mesh_camera(self.sg,c);g=self.ground;xyz=[g.get_xyz];cov=[g.covars];feat=[g.joint_features];opacity=[g.get_opacity];offset=len(g.initial);ranges={}
  for name in self.sg.graph_obj_list:
   m=getattr(self.sg,name)
   if not self.joint_active() and int(m.track_id) in self.native_ids:continue
   key=br.affine_key(m.track_id,c);A,t=br._AFFINE[key];K=br._AFFINE_K[key];cc=br.quat_scale_to_covar_preci(m.get_rotation,m.get_scaling,compute_covar=True,compute_preci=False,triu=False)[0]
   xyz.append(m.get_xyz@A.T+t);cov.append((cc.reshape(-1,9)@K.T).reshape(-1,3,3));feat.append(m.get_features_fourier(c.meta['frame']));opacity.append(m.get_opacity);ranges[name]=(offset,offset+len(m.get_xyz));offset+=len(m.get_xyz)
  bg_start=offset
  if self.joint_active():
   xyz.append(bg.get_xyz);cov.append(br.quat_scale_to_covar_preci(bg.get_rotation,bg.get_scaling,compute_covar=True,compute_preci=False,triu=False)[0]);feat.append(torch.cat([bg.get_features_dc,bg.get_features_rest],1));opacity.append(bg.get_opacity)
  features=torch.cat(feat);model=SimpleNamespace(ground_count=len(g.initial),get_xyz=torch.cat(xyz),covars=torch.cat(cov),get_opacity=torch.cat(opacity),get_features_dc=features[:,:1],get_features_rest=features[:,1:],active_sh_degree=1)
  if 'pixel_weights' in kw and not getattr(self,'ground_eas_scoring',False):kw['pixel_weights']=self.edge(c)
  if self.joint_active() and 'pixel_weights' not in kw:
   with torch.no_grad():
    gp=gsplat_render(c,g,pipe,background,track_gradients=False,metric=True)
    reference=gp['depth']/gp['acc'].clamp_min(1e-6);reference_valid=gp['acc']>.9
   model.road_labels=torch.cat([torch.zeros(bg_start,device=bg.get_xyz.device),torch.ones(len(bg.get_xyz),device=bg.get_xyz.device)])
  pkg=gsplat_render(c,model,pipe,background,**kw)
  if 'pixel_weights' in kw:
   if getattr(self,'ground_eas_scoring',False):return pkg
   return {k:v[bg_start:] for k,v in pkg.items()}

  if self.joint_active():pkg['front_road_valid']=reference_valid & ((pkg['road_background_depth']/pkg['road_background_alpha'].clamp_min(1e-6)).detach()<reference)
  pkg.update(actor_ranges=ranges,ground_range=(0,len(g.initial)),joint_stats=pkg['viewspace_points'],joint_radii=pkg['radii'],joint_visible=pkg['visibility_filter']);self.ground_pkg=None
  pkg['background_start']=bg_start
  if self.joint_active():
   pkg['radii']=pkg['joint_radii'][bg_start:];pkg['visibility_filter']=pkg['joint_visible'][bg_start:];pkg['viewspace_points']=OffsetStats(pkg['joint_stats'],bg_start)
   pkg['render']=pkg['render']+(1-pkg['acc'])*self.sg.sky_cubemap(c)
  return pkg
 def supervision_masks(self,c):
  valid=c.guidance['mask'].cuda().bool();ground=valid&self.ground_mask(c).cuda().bool();dynamic=valid&c.guidance['obj_bound'].cuda().bool()
  cam_id=int(c.image_name.split('_')[-1])
  if cam_id<9 and not wys_full_frame(cfg.source_path) and cam_id not in getattr(self,'_checked_padding',set()):
   if valid.shape[-2:]!=(900,1600) or valid[...,:125].any() or valid[...,1475:].any():
    raise ValueError(f'Padded columns incorrectly supervised for camera {cam_id}')
   if not hasattr(self,'_checked_padding'):self._checked_padding=set()
   self._checked_padding.add(cam_id)
  if not self.joint_active():
   from deferred_native import exclusion
   keep=~exclusion(self,c);ground=ground&keep;dynamic=dynamic&keep
  return ground,dynamic,(valid if self.joint_active() else ground|dynamic)
 def audit_gradients(self,pkg):
  fg=self.ground.features.grad;cc=self.sg.color_correction.affine_trans.grad
  assert fg is not None and torch.isfinite(fg).all() and fg.abs().sum()>0
  assert cc is not None and torch.isfinite(cc).all() and cc.abs().sum()>0
  grads={}
  for n in pkg['actor_ranges']:
   m=getattr(self.sg,n);g=m._xyz.grad;f=m._features_dc.grad
   assert g is not None and f is not None and torch.isfinite(g).all() and torch.isfinite(f).all()
   grads[n]=float(g.abs().sum()+f.abs().sum())
  # A sampled view may contain only occluded/excluded mesh pixels.
  self.actor_gradient_seen=getattr(self,'actor_gradient_seen',False) or sum(grads.values())>0
  if self.joint_active():
   gg=self.gaussians._features_dc.grad;ss=self.sg.sky_cubemap._features_dc.grad
   assert gg is not None and ss is not None and torch.isfinite(gg).all() and torch.isfinite(ss).all()
   assert gg.abs().sum()>0 and ss.abs().sum()>0
  else:
   assert all(getattr(self.gaussians,k).grad is None for k in PARAMS)
   assert all(p.grad is None for _,p in self.sg.sky_cubemap.named_parameters())
  (self.audit_out/'gradient_validation.json').write_text(json.dumps(dict(passed=True,iteration=self.iteration,actors=grads,ground=float(fg.abs().sum()),affine=float(cc.abs().sum()),background_sky_no_grads=not self.joint_active())))
 def optimize(self,it,pkg,smoke=False):
  limit=3 if smoke else 10000
  if it<=limit:
   self.ground.accumulate(pkg,0,len(self.ground.initial));grad=pkg['joint_stats'].grad
   for n,(a,b) in pkg['actor_ranges'].items():
    m=getattr(self.sg,n);v=pkg['joint_visible'][a:b];m.xyz_gradient_accum[:,0:1]+=grad[a:b,:2].norm(dim=1,keepdim=True)*v[:,None];m.xyz_gradient_accum[:,1:2]+=grad[a:b,2:].norm(dim=1,keepdim=True)*v[:,None];m.denom+=v[:,None]
  self.ground.optimizer.step();self.ground.optimizer.zero_grad(set_to_none=True)
  self.merged_actor_adam.step([getattr(self.sg,n).optimizer for n in self.sg.obj_list if self.joint_active() or int(getattr(self.sg,n).track_id) not in self.native_ids]);self.sg.color_correction.update_optimizer()
  for n in self.sg.obj_list:sa.project(getattr(self.sg,n))
  if it<=limit:
   event=self.ground.densify(it,smoke)
   if event:
    with (self.audit_out/'ground_split_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
   due=it==2 if smoke else it>=500 and it%100==0
   if due:
    for n in (list(pkg['actor_ranges'])[:1] if smoke else self.sg.obj_list):
     if not self.joint_active() and int(getattr(self.sg,n).track_id) in self.native_ids:continue
     m=getattr(self.sg,n);e=actor_densify(m,smoke);e.update(actor=n,iteration=it,type='densify')
     with (self.audit_out/'actor_topology.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
   if (it==3 if smoke else it>=1000 and it%500==0):
    for n in self.sg.obj_list:
     if not self.joint_active() and int(getattr(self.sg,n).track_id) in self.native_ids:continue
     e=merge_prune(getattr(self.sg,n))
     if e:
      e.update(actor=n,iteration=it,type='merge_prune')
      with (self.audit_out/'actor_topology.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
  if not smoke and 10000<it<=20000:
   grad=pkg['joint_stats'].grad
   for n,(a,b) in pkg['actor_ranges'].items():
    m=getattr(self.sg,n)
    if int(m.track_id) not in self.native_ids:continue
    v=pkg['joint_visible'][a:b];m.xyz_gradient_accum[:,0:1]+=grad[a:b,:2].norm(dim=1,keepdim=True)*v[:,None];m.xyz_gradient_accum[:,1:2]+=grad[a:b,2:].norm(dim=1,keepdim=True)*v[:,None];m.denom+=v[:,None]
   for n in self.sg.obj_list:
    m=getattr(self.sg,n)
    if int(m.track_id) not in self.native_ids:continue
    if it%100==0:
     e=actor_densify(m,False);e.update(actor=n,iteration=it,type='native_densify')
     with (self.audit_out/'actor_topology.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
    if it%500==0:merge_prune(m)
   for j,n in enumerate(self.sg.obj_list,1):
    if int(getattr(self.sg,n).track_id) in self.native_ids:
     self.frozen_counts[j]=len(getattr(self.sg,n).get_xyz);self.frozen_opacity[j]=getattr(self.sg,n)._opacity.detach().clone()
  counts=[len(self.ground.initial)]+[len(getattr(self.sg,n).get_xyz) for n in self.sg.obj_list]
  if it==limit:
   self.frozen_counts=counts
   self.frozen_opacity=[self.ground.opacity.detach().clone()]+[getattr(self.sg,n)._opacity.detach().clone() for n in self.sg.obj_list]
  if it>limit:
   assert self.frozen_counts==counts,'Topology changed in fixed stage'
   if smoke or it%1000==0:assert all(torch.equal(a,b) for a,b in zip(self.frozen_opacity,[self.ground.opacity]+[getattr(self.sg,n)._opacity for n in self.sg.obj_list])), 'Fixed-stage opacity changed'
  if smoke or it%1000==0:
   audit={n:sa.audit(getattr(self.sg,n)) for n in self.sg.obj_list}
   (self.audit_out/'surface_validation.json').write_text(json.dumps(dict(iteration=it,actors=audit,ground_thickness=.001,actor_thickness=.003,passed=True)))
  if it>=limit and (smoke or it%100==0):(self.audit_out/'topology_freeze_validation.json').write_text(json.dumps(dict(iteration=it,limit=limit,counts=counts,passed=True)))
 def capture(self):
  s=super().capture();s['joint_begin']=self.joint_begin;s['frozen_background_count']=self.frozen_background_count;s['frozen_counts']=self.frozen_counts;s['frozen_opacity']=self.frozen_opacity;s['surface']={n:getattr(self.sg,n).surface for n in self.sg.obj_list};return s
 def restore(self,s):
  self.joint_begin=s['joint_begin'];self.frozen_background_count=s['frozen_background_count'];self.frozen_counts=s['frozen_counts'];self.frozen_opacity=s['frozen_opacity'];super().restore(s)
  for n in self.sg.obj_list:
   m=getattr(self.sg,n);m.surface=s['surface'][n];m._rotation.requires_grad_(False)
  if self.iteration>(3 if os.environ.get('SMOKE')=='1' else 10000):
   self.ground.opacity.requires_grad_(False)
   for n in self.sg.obj_list:
    if os.environ.get('SMOKE')=='1' or int(getattr(self.sg,n).track_id) not in self.native_ids or self.iteration>20000:getattr(self.sg,n)._opacity.requires_grad_(False)

from shadow_layer import install as install_shadow
DynamicScene=install_shadow(DynamicScene,FlatGroundModel)
