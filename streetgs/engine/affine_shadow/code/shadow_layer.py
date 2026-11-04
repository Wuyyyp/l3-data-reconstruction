"""Shared vehicle-local attenuation splats reprojected to a fixed road surface.
Only subtract their attenuation from the *visible ground RGB contribution*.
They never enter inverse depth or actor RGB, and are not physical light transport.
"""
import os,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import bridge as br
from mesh_backend import gsplat_render

class RoadGrid:
 def __init__(self,path):
  a=np.load(path);x=a['xyz'];n=a['normals'];cell=np.floor(x[:,:2]/.15).astype('int64');self.lo=torch.tensor(cell.min(0)-2,device='cuda');hi=cell.max(0)+2;self.width=int(hi[1]-self.lo[1].item()+1)
  key=(cell[:,0]-int(self.lo[0]))*self.width+cell[:,1]-int(self.lo[1]);keys,ix=np.unique(key,return_index=True)
  self.keys=torch.tensor(keys,device='cuda');self.x=torch.tensor(x[ix],device='cuda');self.n=torch.tensor(n[ix],device='cuda');self.shifts=torch.tensor([(i,j) for i in [-1,0,1] for j in [-1,0,1]],device='cuda')
 def project(self,xy):
  cell=torch.floor(xy.detach()/.15).long()-self.lo;cells=cell[:,None,:]+self.shifts
  keys=cells[:,:,0]*self.width+cells[:,:,1];idx=torch.searchsorted(self.keys,keys).clamp_max(len(self.keys)-1)
  found=self.keys[idx]==keys;centers=self.x[idx];normals=self.n[idx];d=(centers[:,:,:2]-xy[:,None,:]).square().sum(-1)
  w=torch.exp(-d/(2*.15**2))*found*(d<.3**2);total=w.sum(1);supported=total>1e-8;w=w/total.clamp_min(1e-8)[:,None]
  z=(centers[:,:,2]-(normals[:,:,:2]*(xy[:,None,:]-centers[:,:,:2])).sum(-1)/normals[:,:,2].clamp_min(.5))
  z=(z*w).sum(1);n=(normals*w[:,:,None]).sum(1);n=torch.where(supported[:,None],torch.nn.functional.normalize(n,dim=-1),torch.tensor([0.,0.,1.],device='cuda'))
  return torch.cat([xy,z[:,None]],1),n,supported

def install(Base,Flat):
 class ShadowGround(Flat):
  @property
  def get_xyz(self):return self.initial+self.normals*(.05*torch.tanh(self.offset))
 class Scene(Base):
  def __init__(self,*args,**kw):
   super().__init__(*args,**kw);self.shadow_grid=RoadGrid(os.environ['GROUND_SHARED_SURFACE']);self.shadows={};self.shadow_freeze=None;self.shadow_events=[];self.shadow_pkg=None;self.shadow_gradient_seen=False
   root=Path(os.environ['SHADOW_TEMPLATES'])
   for name in self.sg.obj_list:
    tid=getattr(self.sg,name).track_id;file=root/('%d.npz'%tid)
    if not file.exists():continue
    m=ShadowGround(str(file),True)
    with torch.no_grad():m.opacity.fill_(float(np.log(.1/.9)));m.scale.fill_(np.log(.12))
    self.shadows[name]=m
   assert len(self.shadows)==len(list(root.glob('*.npz'))) and len(self.shadows)>0
   self._shadow_transforms={}
  def shadow_count(self):return sum(len(m.initial) for m in self.shadows.values())
  def shadow_model(self,c):
   xyz=[];cov=[];feat=[];alpha=[];ranges={};offset=0;support=0
   for name in self.sg.graph_obj_list:
    if name not in self.shadows:continue
    if not self.joint_active() and int(getattr(self.sg,name).track_id) in self.native_ids:continue
    m=self.shadows[name];actor=getattr(self.sg,name);key=br.affine_key(actor.track_id,c)
    if key not in self._shadow_transforms:
     A,t=br._AFFINE[key];x=torch.nn.functional.normalize(A[:2,0],dim=0);yaw=torch.stack([x,torch.stack([-x[1],x[0]])],1);self._shadow_transforms[key]=(yaw,t)
    yaw,t=self._shadow_transforms[key];xy=m.initial[:,:2]@yaw.T+t[:2];pos,n,valid=self.shadow_grid.project(xy)
    pos=pos+n*(.003+.05*torch.tanh(m.offset));u=torch.stack([yaw[0,0].expand(len(n)),yaw[1,0].expand(len(n)),torch.zeros_like(n[:,0])],1);u=torch.nn.functional.normalize(u-(u*n).sum(1)[:,None]*n,dim=1);v=torch.cross(n,u,dim=1);basis=torch.stack([u,v,n],2)
    scale=m.get_scaling;C=(basis*scale[:,None,:])@(basis*scale[:,None,:]).transpose(1,2)
    xyz.append(pos);cov.append(C);feat.append(m.features);alpha.append(m.get_opacity*valid[:,None]);ranges[name]=(offset,offset+len(pos));offset+=len(pos);support+=valid.sum()
   if not xyz:return None,{},0
   f=torch.cat(feat);model=SimpleNamespace(get_xyz=torch.cat(xyz),covars=torch.cat(cov),get_opacity=torch.cat(alpha),get_features_dc=f[:,:1],get_features_rest=f[:,1:],active_sh_degree=1)
   return model,ranges,support
  def render(self,c,bg,pipe,background,**kw):
   pkg=super().render(c,bg,pipe,background,**kw)
   if 'pixel_weights' in kw:return pkg
   model,ranges,support=self.shadow_model(c)
   if model is not None:
    sh=gsplat_render(c,model,pipe,background,track_gradients=kw.get('track_gradients',True))
    # shadow_rgb <= shadow_acc; the layer can darken but never brighten road.
    attenuation=(sh['acc']-torch.minimum(sh['render'],sh['acc'])).clamp(0,1)
    pkg['unshadowed_render']=pkg['render']
    pkg['shadow_attenuation']=attenuation
    pkg['shadow_visible_effect']=(pkg['ground_contribution']*attenuation*self.supervision_masks(c)[0]).detach()
    pkg['render']=pkg['render']-pkg['ground_contribution']*attenuation
    pkg['shadow_pkg']=sh;pkg['shadow_ranges']=ranges;pkg['shadow_supported']=support
   else:pkg['shadow_ranges']={};pkg['shadow_supported']=0
   return pkg
  def shape_loss(self,pkg):
   loss=super().shape_loss(pkg)
   for name in pkg.get('shadow_ranges',{}):loss=loss+10*(.05*torch.tanh(self.shadows[name].offset)).square().mean()
   return loss
  def optimize(self,it,pkg,smoke=False):
   limit=3 if smoke else 5000
   if it<=limit and pkg.get('shadow_pkg') is not None:
    for name,(a,b) in pkg['shadow_ranges'].items():self.shadows[name].accumulate(pkg['shadow_pkg'],a,b)
   report = it==1 or it%100==0
   if report or (not self.shadow_gradient_seen and pkg.get('shadow_ranges')):
    vals={n:float(m.opacity.grad.abs().sum()) for n,m in self.shadows.items() if m.opacity.grad is not None}
    self.shadow_gradient_seen=self.shadow_gradient_seen or any(np.isfinite(v) and v>0 for v in vals.values())
   if report:
    (self.audit_out/'shadow_gradient_progress.json').write_text(json.dumps(dict(iteration=it,nonzero_seen=self.shadow_gradient_seen,opacity_grad_sum=sum(vals.values()),visible_ground_effect=float(pkg.get('shadow_visible_effect',torch.zeros(1,device='cuda')).abs().sum()))))
    if it>=500:assert self.shadow_gradient_seen,'No supervised shadow gradient in first 500 steps'
   super().optimize(it,pkg,smoke)
   for name,m in self.shadows.items():
    m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
    if it<=limit:
     e=m.densify(it,smoke)
     if e:self.shadow_events.append(dict(actor=name,**e))
   counts={n:len(m.initial) for n,m in self.shadows.items()}
   if it==limit:self.shadow_freeze=counts
   if it>limit:assert self.shadow_freeze==counts,'Shadow topology changed after freeze'
   if it in [1,limit,10000,20000,30000]:
    (self.audit_out/'shadow_validation.json').write_text(json.dumps(dict(iteration=it,templates=len(self.shadows),points=self.shadow_count(),supported_current_view=int(pkg['shadow_supported']),rgb_only=True,shared_across_views=True,topology_frozen=it>=limit)))
  def audit_gradients(self,pkg):
   super().audit_gradients(pkg)
   vals={n:float(self.shadows[n].opacity.grad.abs().sum()) for n in pkg['shadow_ranges'] if self.shadows[n].opacity.grad is not None}
   assert all(np.isfinite(v) for v in vals.values())
   effect=float(pkg.get('shadow_visible_effect',torch.zeros(1,device='cuda')).abs().sum())
   if effect>1e-6:assert vals and sum(vals.values())>0,'Visible supervised shadow has no gradient'
   self.shadow_gradient_seen=self.shadow_gradient_seen or sum(vals.values())>0
   (self.audit_out/'shadow_gradients.json').write_text(json.dumps(vals))
  def audit_finite(self,gradients=False,optimizer=False):
   super().audit_finite(gradients,optimizer)
   if not hasattr(self,'shadows'):return
   for m in self.shadows.values():
    for p in m.parameters():
     assert torch.isfinite(p).all()
     if gradients and p.grad is not None:assert torch.isfinite(p.grad).all()
  def capture(self):
   s=super().capture();s['shadow_models']={n:m.capture() for n,m in self.shadows.items()};s['shadow_freeze']=self.shadow_freeze;s['shadow_events']=self.shadow_events;s['shadow_gradient_seen']=self.shadow_gradient_seen;return s
  def restore(self,s):
   super().restore(s);assert set(self.shadows)==set(s['shadow_models'])
   for n,m in self.shadows.items():m.restore(s['shadow_models'][n])
   self.shadow_freeze=s['shadow_freeze'];self.shadow_events=s['shadow_events'];self.shadow_gradient_seen=s['shadow_gradient_seen']
 return Scene
