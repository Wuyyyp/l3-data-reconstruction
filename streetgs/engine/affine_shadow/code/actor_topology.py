import torch
from lib.utils.general_utils import quaternion_to_matrix, matrix_to_quaternion
ATTR={'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity','scaling':'_scaling','rotation':'_rotation','semantic':'_semantic'}
@torch.no_grad()
def compact(m,drop,pairs):
 n=len(m.get_xyz);pairs=pairs.reshape(-1,2);a,b=pairs.T;assert not drop[a].any() and not drop[b].any();assert len(torch.cat([a,b]).unique())==len(a)*2
 keep=~drop.clone();keep[b]=False;idx=keep.nonzero().flatten();assert len(idx)>0
 vals={k:getattr(m,k).detach().clone() for k in ATTR.values()}
 if len(a):
  scale=m.get_scaling;mass=m.get_opacity.flatten()*scale.prod(1);w=(mass[a]/(mass[a]+mass[b]).clamp_min(1e-15))[:,None]
  for k in ['_xyz','_features_dc','_features_rest','_semantic']:
   ww=w.reshape((-1,)+(1,)*(vals[k].ndim-1));vals[k][a]=ww*vals[k][a]+(1-ww)*vals[k][b]
  # Keep the nearly identical orientation; match covariance diagonal in that frame.
  rot=quaternion_to_matrix(m.get_rotation[a]);rb=quaternion_to_matrix(m.get_rotation[b]);rel=rot.transpose(1,2)@rb
  var_b=(rel.square()*scale[b,None,:].square()).sum(-1);delta=((m.get_xyz[b]-m.get_xyz[a])[:,None,:]@rot).squeeze(1)
  var=w*scale[a].square()+(1-w)*var_b+w*(1-w)*delta.square();newscale=var.clamp_min(1e-12).sqrt();vals['_scaling'][a]=newscale.log()
  op=((mass[a]+mass[b])/newscale.prod(1)).clamp(.001,.99);vals['_opacity'][a]=torch.logit(op)[:,None]
 merged=torch.zeros(n,dtype=torch.bool,device=drop.device);merged[a]=True
 for group in m.optimizer.param_groups:
  attr=ATTR[group['name']];old=group['params'][0];new=torch.nn.Parameter(vals[attr][idx],requires_grad=old.requires_grad);state=m.optimizer.state.pop(old,{})
  for k,v in list(state.items()):
   if torch.is_tensor(v) and v.shape==old.shape and v.ndim>0:
    q=v[idx].clone();q[merged[idx]]=0;state[k]=q
  group['params'][0]=new;setattr(m,attr,new)
  if state:m.optimizer.state[new]=state
 for k in ['xyz_gradient_accum','denom','max_radii2D']:setattr(m,k,getattr(m,k)[idx].clone())
 return dict(before=n,pruned=int(drop.sum()),merged_pairs=len(a),after=len(idx))
@torch.no_grad()
def merge_prune(m,allow_merge=True):
 n=len(m.get_xyz);drop=m.get_opacity.flatten()<.0075;ids=drop.nonzero().flatten();limit=min(max(1,n//100),max(0,n-16))
 if len(ids)>limit:
  selected=ids[m.get_opacity.flatten()[ids].topk(limit,largest=False).indices];drop.zero_();drop[selected]=True
 pairs=torch.empty((0,2),device=drop.device,dtype=torch.long)
 if allow_merge and n>1:
  # Deterministic same-actor 3cm canonical grid, no cross-actor merging.
  cell=torch.floor(m.get_xyz/.03).long();_,inv=torch.unique(cell,dim=0,return_inverse=True);order=torch.argsort(inv,stable=True);a=order[:n//2*2:2];b=order[1:n//2*2:2]
  ok=(inv[a]==inv[b])&~drop[a]&~drop[b]&((m.get_xyz[a]-m.get_xyz[b]).norm(dim=1)<.03)
  ok&=(m.get_rotation[a]*m.get_rotation[b]).sum(1).abs()>.999
  ok&=(m._scaling[a]-m._scaling[b]).abs().amax(1)<.2
  ok&=(m._features_dc[a]-m._features_dc[b]).abs().flatten(1).amax(1)<.05
  if m._features_rest.numel():ok&=(m._features_rest[a]-m._features_rest[b]).abs().flatten(1).amax(1)<.05
  pairs=torch.stack([a[ok],b[ok]],1)[:4096]
 if not drop.any() and not len(pairs):return None
 return compact(m,drop,pairs)
@torch.no_grad()
def densify(m,smoke=False):
 n=len(m.get_xyz);g=torch.nan_to_num(m.xyz_gradient_accum[:,:1]/m.denom.clamp_min(1));selected=g[:,0]>=.0002
 if smoke:selected.zero_();selected[0]=True;g.zero_();g[0]=1
 # Bound each topology event to 1024 parents per actor, same grad selection otherwise.
 ids=selected.nonzero().flatten()
 if len(ids)>1024:ids=ids[g[ids,0].topk(1024).indices]
 limited=torch.zeros_like(g);limited[ids]=g[ids]
 m.densify_and_clone(limited,.0002,m.extent);cloned=len(m.get_xyz)-n
 m.densify_and_split(limited,.0002,m.extent);split=len(m.get_xyz)-n-cloned
 m.xyz_gradient_accum.zero_();m.denom.zero_();m.max_radii2D.zero_()
 return dict(before=n,cloned=cloned,split_parents=split,after=len(m.get_xyz))
