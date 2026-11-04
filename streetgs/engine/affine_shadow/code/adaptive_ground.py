"""Ground-only tangent splitting with topology-safe Adam state migration."""
import os
import torch
from lib.utils.general_utils import quaternion_to_matrix

def install(Base):
 class AdaptiveGround(Base):
  def __init__(self,path,anchored=False):
   super().__init__(path,anchored)
   if os.environ.get('GROUND_SPLIT','1')!='1':raise ValueError('Ground splitting is required; remove GROUND_SPLIT=0')
   n=len(self.initial)
   if int(os.environ.get('GROUND_POINT_CAP','1200000'))<=n:raise ValueError('GROUND_POINT_CAP must exceed initial ground count to permit splitting')
   self.register_buffer('grad_sum',torch.zeros(n,device='cuda'))
   self.register_buffer('grad_count',torch.zeros(n,device='cuda'))
   self.register_buffer('source_id',torch.arange(n,device='cuda'))
   self.events=[]

  @torch.no_grad()
  def accumulate(self,pkg,start=0,end=None):
   stats=pkg.get('joint_stats',pkg.get('viewspace_points'))
   if stats.means2d.grad is None:return
   g=stats.grad[start:end,2:].norm(dim=-1)
   visible=pkg.get('joint_visible',pkg['visibility_filter'])[start:end]
   assert len(g)==len(self.initial)
   if os.environ.get('SMOKE')=='1':assert torch.isfinite(g).all()
   self.grad_sum+=g*visible;self.grad_count+=visible

  @torch.no_grad()
  def split(self,ids,iteration):
   ids=ids.long();n=len(self.initial);k=len(ids)
   assert k>0 and len(ids.unique())==k
   keep=torch.ones(n,dtype=torch.bool,device='cuda');keep[ids]=False;retained=keep.nonzero().flatten()
   parent=ids.repeat_interleave(2);index=torch.cat([retained,parent]);nkeep=len(retained)
   axes=self.get_scaling[ids,:2].argmax(-1);scales=self.get_scaling[ids,:2]
   rotations=quaternion_to_matrix(self.get_rotation[ids])
   direction=rotations[torch.arange(k,device='cuda'),:,axes]
   magnitude=scales.gather(1,axes[:,None]).flatten()*.5
   delta=(direction*magnitude[:,None]).repeat_interleave(2,0)
   delta[::2]*=-1
   plane_error=float((delta*self.normals[parent]).sum(-1).abs().max())
   assert plane_error<1e-5,plane_error
   # Preserve unaffected rows and optimizer moments; children start with zero moments.
   for name in ['features','scale','opacity','offset']:
    old=getattr(self,name);value=old.detach()[index].clone()
    if name=='scale':
     rows=torch.arange(2*k,device='cuda')+nkeep
     value[rows,axes.repeat_interleave(2)]-=torch.log(value.new_tensor(1.6))
    new=torch.nn.Parameter(value,requires_grad=old.requires_grad)
    for group in self.optimizer.param_groups:
     for j,p in enumerate(group['params']):
      if p is old:
       group['params'][j]=new;state=self.optimizer.state.pop(old,{})
       migrated={}
       for field,v in state.items():
        if torch.is_tensor(v) and v.ndim>0 and v.shape==old.shape:
         q=v[index].clone();q[nkeep:]=0;migrated[field]=q
        else:migrated[field]=v
       if migrated:self.optimizer.state[new]=migrated
    setattr(self,name,new)
   for name in ['initial','normals','rotation','source_id']:setattr(self,name,getattr(self,name)[index].clone())
   self.initial[nkeep:]+=delta
   self.grad_sum=torch.zeros(len(index),device='cuda');self.grad_count=torch.zeros_like(self.grad_sum)
   assert torch.allclose(self.get_scaling[:,2],torch.full_like(self.get_scaling[:,2],.001))
   event=dict(iteration=iteration,before=n,parents=k,children=2*k,after=len(index),max_tangent_normal_error=plane_error)
   self.events.append(event);return event

  @torch.no_grad()
  def densify(self,it,smoke=False):
   # Splitting is mandatory in the supported ground trainer.
   if os.environ.get('GROUND_SPLIT','1')!='1':raise ValueError('Ground splitting is required; remove GROUND_SPLIT=0')
   due=it in [3,8,18] if smoke else 500<=it<=10000 and it%100==0
   if not due:return None
   score=self.grad_sum/self.grad_count.clamp_min(1)
   eligible=(self.grad_count>0)&(self.get_scaling[:,:2].max(-1).values>.025)
   if not smoke:eligible&=score>=float(os.environ.get('GROUND_SPLIT_THRESHOLD','0.0002'))
   ids=eligible.nonzero().flatten();capacity=int(os.environ.get('GROUND_POINT_CAP','1200000'))-len(self.initial)
   k=min(len(ids),max(0,capacity),1 if smoke else 8192)
   event=None
   if k:
    importance=getattr(self,'_eas_score',None)
    if importance is not None:
     from ground_eas import select_candidates
     gradient_ids=ids[score[ids].topk(k).indices]
     ids,weighted=select_candidates(ids,score,importance,k,it)
     self._eas_audit.update(overlap_with_gradient_topk=int(torch.isin(ids,gradient_ids).sum()),selected_mean_edge=float(importance[ids].mean()),gradient_topk_mean_edge=float(importance[gradient_ids].mean()))
     audit=dict(self._eas_audit,selected=k,edge_weighted_selected=weighted,gradient_fallback=k-weighted)
    else:
     ids=ids[score[ids].topk(k).indices];audit=None
    event=self.split(ids,it)
    if event is not None and audit is not None:
     event['eas']=audit
     with (self._eas_out/'ground_eas_events.jsonl').open('a') as f:
      import json
      f.write(json.dumps(audit)+'\n')
   self._eas_score=None
   self.grad_sum.zero_();self.grad_count.zero_()
   return event

  def capture(self):
   s=super().capture();s['events']=self.events;return s

  def restore(self,s):
   # Recreate all variable-length tensors and Adam parameter references before load.
   model=s['model']
   for name in ['initial','normals','rotation','source_id','grad_sum','grad_count']:setattr(self,name,model[name].clone())
   for name in ['features','scale','opacity','offset']:
    old=getattr(self,name);new=torch.nn.Parameter(model[name].clone(),requires_grad=old.requires_grad)
    for group in self.optimizer.param_groups:
     group['params']=[new if p is old else p for p in group['params']]
    setattr(self,name,new)
   super().restore(s);self.events=s['events']
 return AdaptiveGround
