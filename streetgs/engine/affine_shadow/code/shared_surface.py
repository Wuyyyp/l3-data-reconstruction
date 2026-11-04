"""Opt-in shared fixed surface. Reproject split anchors; no added merge/smoothing."""
import os,hashlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
class Surface:
 def __init__(self,path):
  self.digest=hashlib.sha256(open(path,'rb').read()).hexdigest();a=np.load(path);self.xyz=a['xyz'].astype(float);self.norm=a['normals'].astype(float);self.tree=cKDTree(self.xyz[:,:2]);assert len(self.xyz)>=8
 def project(self,p):
  d,ix=self.tree.query(p[:,:2],k=8,workers=4);n=self.norm[ix];x=self.xyz[ix];w=np.exp(-.5*(d/.15)**2)*(d<.3);valid=w.sum(1)>1e-10;w/=w.sum(1,keepdims=True).clip(1e-10)
  z=x[:,:,2]-(n[:,:,:2]*(p[:,None,:2]-x[:,:,:2])).sum(2)/n[:,:,2].clip(.5);z=(z*w).sum(1);normal=(n*w[:,:,None]).sum(1);normal/=np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-10)
  q=p.copy();q[:,2]=z
  return q,normal,valid
 def rotations(self,n):
  x=np.cross(np.broadcast_to([0.,1.,0.],n.shape),n);x/=np.linalg.norm(x,axis=1,keepdims=True);y=np.cross(n,x);return Rotation.from_matrix(np.stack([x,y,n],2)).as_quat()[:,[3,0,1,2]]
def install(Base):
 class SharedGround(Base):
  @property
  def initial_sh_degree(self):
   return int(os.environ.get('GROUND_SH_DEGREE','1'))
  def __init__(self,path,anchored=False):
   super().__init__(path,anchored);self.shared_surface=Surface(os.environ['GROUND_SHARED_SURFACE']);q,n,ok=self.shared_surface.project(self.initial.detach().cpu().numpy());assert ok.all(),'Unsupported surface anchors'
   self.initial.copy_(torch.as_tensor(q,device=self.initial.device,dtype=self.initial.dtype));self.normals.copy_(torch.as_tensor(n,device=self.normals.device,dtype=self.normals.dtype));self.rotation.copy_(torch.as_tensor(self.shared_surface.rotations(n),device=self.rotation.device,dtype=self.rotation.dtype))
  @property
  def get_xyz(self):return self.initial+self.normals*(.05*torch.tanh(self.offset))
  @torch.no_grad()
  def split(self,ids,iteration):
   # Reject parents whose children leave the fitted support, before mutating topology.
   from lib.utils.general_utils import quaternion_to_matrix
   ids=ids.long();axes=self.get_scaling[ids,:2].argmax(-1);rot=quaternion_to_matrix(self.get_rotation[ids]);direction=rot[torch.arange(len(ids),device=ids.device),:,axes];delta=direction*self.get_scaling[ids,:2].gather(1,axes[:,None])*.5
   candidates=torch.stack([self.initial[ids]-delta,self.initial[ids]+delta],1).reshape(-1,3).cpu().numpy();_,_,ok=self.shared_surface.project(candidates);ids=ids[torch.as_tensor(ok.reshape(-1,2).all(1),device=ids.device)]
   if not len(ids):return None
   before=len(self.initial);k=len(ids);event=super().split(ids,iteration);start=before-k;q,n,ok=self.shared_surface.project(self.initial[start:].cpu().numpy());assert ok.all()
   self.initial[start:]=torch.as_tensor(q,device=self.initial.device,dtype=self.initial.dtype);self.normals[start:]=torch.as_tensor(n,device=self.normals.device,dtype=self.normals.dtype);self.rotation[start:]=torch.as_tensor(self.shared_surface.rotations(n),device=self.rotation.device,dtype=self.rotation.dtype);self.offset[start:]=0
   event['shared_surface_reprojected']=2*k;return event
  @torch.no_grad()
  def compact(self,drop,pairs,it):
   # Reproject surviving merged anchors onto the same fixed reference surface.
   merged=pairs[:,0].clone();keep=~drop.clone();keep[pairs[:,1]]=False;dest=keep.long().cumsum(0)[merged]-1
   event=super().compact(drop,pairs,it)
   if len(dest):
    q,n,ok=self.shared_surface.project(self.initial[dest].cpu().numpy());assert ok.all()
    self.initial[dest]=torch.as_tensor(q,device=self.initial.device,dtype=self.initial.dtype)
    self.normals[dest]=torch.as_tensor(n,device=self.normals.device,dtype=self.normals.dtype)
    self.rotation[dest]=torch.as_tensor(self.shared_surface.rotations(n),device=self.rotation.device,dtype=self.rotation.dtype)
   return event
  def capture(self):
   s=super().capture();s['shared_surface_sha256']=self.shared_surface.digest;s['normal_limit_m']=.05;return s
  def restore(self,s):
   assert s.get('shared_surface_sha256')==self.shared_surface.digest and s.get('normal_limit_m')==.05,'Incompatible surface checkpoint'
   super().restore(s)
 return SharedGround
