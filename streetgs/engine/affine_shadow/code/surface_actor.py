import torch,numpy as np
from pathlib import Path
from actor_topology import compact,ATTR
import os
ROOT=Path(os.environ['ACTOR_ANCHORS'])
@torch.no_grad()
def init(m):
 a=np.load(ROOT/(m.model_name+'.npz'));xyz=torch.tensor(a['xyz'],device='cuda');assert torch.allclose(m._xyz,xyz,atol=1e-6)
 m.surface=dict(initial=xyz,basis=torch.tensor(a['basis'],device='cuda'),rotation=torch.tensor(a['rotation'],device='cuda'),ids=torch.arange(len(xyz),device='cuda'))
 m._rotation.copy_(m.surface['rotation']);m._rotation.requires_grad_(False);project(m)
def coords(m):
 s=m.surface;ids=s['ids'];b=s['basis'][ids];d=m._xyz-s['initial'][ids];local=torch.einsum('ni,nij->nj',d,b);return local,b
@torch.no_grad()
def project(m):
 local,b=coords(m);norm=local[:,:2].norm(dim=1,keepdim=True);local[:,:2]*=(.05/norm.clamp_min(.05));local[:,2].clamp_(-.05,.05)
 s=m.surface;m._xyz.copy_(s['initial'][s['ids']]+torch.einsum('nj,nij->ni',local,b));m._rotation.copy_(s['rotation'][s['ids']]);m._rotation.requires_grad_(False)
 m._scaling[:,:2].clamp_(np.log(.002),np.log(.05));m._scaling[:,2]=np.log(.003)
 for group in m.optimizer.param_groups:
  if group['name']=='scaling':
   for k in ['exp_avg','exp_avg_sq']:
    v=m.optimizer.state.get(m._scaling,{}).get(k)
    if v is not None:v[:,2]=0
@torch.no_grad()
def audit(m):
 local,b=coords(m);assert local[:,2].abs().max()<.05001 and local[:,:2].norm(dim=1).max()<.05001
 assert torch.allclose(m._rotation,m.surface['rotation'][m.surface['ids']]);assert len(m.surface['ids'])==len(m._xyz)
 return dict(points=len(m._xyz),max_normal=float(local[:,2].abs().max()),max_tangent=float(local[:,:2].norm(dim=1).max()),surface_sources=int(m.surface['ids'].unique().numel()))
def loss(m):
 local,_=coords(m);return 10*local[:,2].square().mean()+.1*local[:,:2].square().mean()
@torch.no_grad()
def densify(m,smoke=False):
 n=len(m._xyz);g=torch.nan_to_num(m.xyz_gradient_accum[:,0]/m.denom[:,0].clamp_min(1));ids=(g>=.0002).nonzero().flatten()
 if smoke:ids=torch.tensor([0],device='cuda')
 if len(ids)>1024:ids=ids[g[ids].topk(1024).indices]
 if not len(ids):return dict(before=n,cloned=0,split_parents=0,after=n)
 split=ids[m.get_scaling[ids,:2].max(1).values>.01];clone=ids[m.get_scaling[ids,:2].max(1).values<=.01];parents=torch.cat([clone,split,split]);s=m.surface;newids=s['ids'][parents];d={key:getattr(m,attr)[parents].detach().clone() for key,attr in ATTR.items()}
 b=s['basis'][newids];jitter=torch.randn((len(parents),2),device='cuda')*m.get_scaling[parents,:2]*.4;d['xyz']+=torch.einsum('nj,nij->ni',jitter,b[:,:,:2]);d['scaling'][len(clone):,:2]-=np.log(1.6)
 m.densification_postfix(d);s['ids']=torch.cat([s['ids'],newids]);drop=torch.zeros(len(m._xyz),device='cuda',dtype=torch.bool);drop[split]=True;m.prune_points(drop);s['ids']=s['ids'][~drop]
 project(m);m.xyz_gradient_accum.zero_();m.denom.zero_();m.max_radii2D.zero_()
 return dict(before=n,cloned=len(clone),split_parents=len(split),after=len(m._xyz))
@torch.no_grad()
def merge_prune(m,force=False):
 n=len(m._xyz);s=m.surface;drop=m.get_opacity.flatten()<.0075;ix=drop.nonzero().flatten();limit=min(max(1,n//100),max(0,n-16))
 if len(ix)>limit:
  ix=ix[m.get_opacity.flatten()[ix].topk(limit,largest=False).indices];drop.zero_();drop[ix]=True
 order=torch.argsort(s['ids'],stable=True);a=order[:n//2*2:2];b=order[1:n//2*2:2]
 ok=(s['ids'][a]==s['ids'][b])&~drop[a]&~drop[b]&((m.get_xyz[a]-m.get_xyz[b]).norm(dim=1)<.03)
 ok&=(m._scaling[a]-m._scaling[b]).abs().amax(1)<.2
 ok&=(m._features_dc[a]-m._features_dc[b]).abs().flatten(1).amax(1)<.05
 ok&=(m._features_rest[a]-m._features_rest[b]).abs().flatten(1).amax(1)<.05
 pairs=torch.stack([a[ok],b[ok]],1)[:4096]
 if not drop.any() and not len(pairs):return None
 keep=~drop;keep[pairs[:,1]]=False;e=compact(m,drop,pairs);s['ids']=s['ids'][keep];project(m);return e
