import torch

def install(Base):
 class TopologyGround(Base):
  @torch.no_grad()
  def compact(self,drop,pairs,it):
   n=len(self.initial);pairs=pairs.reshape(-1,2);a,b=pairs[:,0],pairs[:,1]
   assert not drop[a].any() and not drop[b].any()
   keep=~drop.clone();keep[b]=False;idx=keep.nonzero().flatten()
   old_count=n;alpha=self.get_opacity.flatten();area=self.get_scaling[:,:2].prod(1)
   wa=alpha[a]*area[a];wb=alpha[b]*area[b];w=(wa/(wa+wb).clamp_min(1e-12))[:,None]
   # Restrict merges to siblings sharing the same fixed surface orientation.
   values={k:getattr(self,k).detach().clone() for k in ['features','scale','opacity','offset']}
   for k in ['features','offset']:
    ww=w.reshape((-1,)+(1,)*(values[k].ndim-1));values[k][a]=values[k][a]*ww+values[k][b]*(1-ww)
   initial=self.initial.clone();initial[a]=initial[a]*w+initial[b]*(1-w)
   if len(a):
    from lib.utils.general_utils import quaternion_to_matrix
    rot=quaternion_to_matrix(self.rotation[a]);delta=self.initial[b]-self.initial[a]
    local=torch.bmm(delta[:,None,:],rot).squeeze(1)[:,:2]
    v=w*self.get_scaling[a,:2].square()+(1-w)*self.get_scaling[b,:2].square()+w*(1-w)*local.square()
    scales=v.sqrt().clamp(.005,.5);values['scale'][a]=scales.log()
    opacity=((wa+wb)/scales.prod(1)).clamp(.001,.99);values['opacity'][a]=torch.logit(opacity)[:,None]
   merged=torch.zeros(n,dtype=torch.bool,device=idx.device);merged[a]=True
   for name in ['features','scale','opacity','offset']:
    old=getattr(self,name);new=torch.nn.Parameter(values[name][idx],requires_grad=old.requires_grad)
    for group in self.optimizer.param_groups:
     for j,p in enumerate(group['params']):
      if p is old:
       group['params'][j]=new;state=self.optimizer.state.pop(old,{})
       for field,v in list(state.items()):
        if torch.is_tensor(v) and v.shape==old.shape:
         q=v[idx].clone();q[merged[idx]]=0;state[field]=q
       if state:self.optimizer.state[new]=state
    setattr(self,name,new)
   self.initial=initial[idx]
   for k in ['normals','rotation','source_id']:setattr(self,k,getattr(self,k)[idx].clone())
   self.grad_sum=torch.zeros(len(idx),device=idx.device);self.grad_count=torch.zeros_like(self.grad_sum)
   e=dict(iteration=it,type='prune_merge',before=old_count,pruned=int(drop.sum()),merged_pairs=len(a),after=len(idx))
   self.events.append(e);return e

  @torch.no_grad()
  def densify(self,it,smoke=False):
   split=super().densify(it,smoke)
   due=(it in [3,6]) if smoke else (1000<=it<=10000 and it%500==0)
   if not due:return split
   n=len(self.initial)
   drop=self.get_opacity.flatten()<.0075
   # Never discard most of the surface in one event.
   ids=drop.nonzero().flatten();limit=max(1,n//100)
   if len(ids)>limit:
    selected=ids[self.get_opacity.flatten()[ids].topk(limit,largest=False).indices];drop.zero_();drop[selected]=True
   # Adjacent siblings only; no cross-surface nearest-neighbour merging.
   order=torch.argsort(self.source_id,stable=True);a=order[0:(len(order)//2)*2:2];b=order[1:(len(order)//2)*2:2]
   same=self.source_id[a]==self.source_id[b]
   ok=same&~drop[a]&~drop[b]
   ok&=(self.get_xyz[a]-self.get_xyz[b]).norm(dim=1)<.03
   ok&=(self.normals[a]*self.normals[b]).sum(1)>.999
   ok&=(self.features[a]-self.features[b]).abs().flatten(1).amax(1)<.05
   ok&=(self.scale[a]-self.scale[b]).abs().amax(1)<.2
   pairs=torch.stack([a[ok],b[ok]],1)[:8192]
   if not drop.any() and not len(pairs):return split
   return self.compact(drop,pairs,it)
 return TopologyGround
