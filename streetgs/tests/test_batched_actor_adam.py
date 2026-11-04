"""Standalone CPU test: lazy Adam state, topology replacement and checkpoint restore."""
import importlib.util
from pathlib import Path
import copy
import torch
spec=importlib.util.spec_from_file_location('batched_actor_adam',Path(__file__).resolve().parents[1]/'lib/utils/batched_actor_adam.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_equivalence():
 torch.manual_seed(7)
 a=[torch.nn.Parameter(torch.randn(4,3)) for _ in range(3)]
 b=[torch.nn.Parameter(x.detach().clone()) for x in a]
 oa=[torch.optim.Adam([x],lr=.01) for x in a];ob=[torch.optim.Adam([x],lr=.01) for x in b]
 merged=mod.BatchedActorAdam()
 for step in range(12):
  if step==5:
   # Same state/parameter replacement as a clone operation after densification.
   for params,opts in [(a,oa),(b,ob)]:
    old=params[0];new=torch.nn.Parameter(torch.cat([old.detach(),old[:1].detach()]))
    state=opts[0].state.pop(old)
    for k in ['exp_avg','exp_avg_sq']:state[k]=torch.cat([state[k],torch.zeros_like(state[k][:1])])
    opts[0].param_groups[0]['params']=[new];opts[0].state[new]=state;params[0]=new
  if step==8:
   for j in range(3):
    restored=torch.optim.Adam([b[j]],lr=.01);restored.load_state_dict(copy.deepcopy(ob[j].state_dict()));ob[j]=restored
   merged=mod.BatchedActorAdam()
  for j in range(3):
   g=None if (j==2 and step<4) or (step+j)%4==0 else torch.randn_like(a[j])
   a[j].grad=g;b[j].grad=None if g is None else g.clone()
   oa[j].param_groups[0]['lr']=ob[j].param_groups[0]['lr']=.01/(step+1+j)
  for opt in oa:opt.step();opt.zero_grad(set_to_none=True)
  merged.step(ob)
  for j in range(3):
   assert torch.equal(a[j],b[j])
   assert (a[j] in oa[j].state)==(b[j] in ob[j].state)
   if a[j] in oa[j].state:
    for k,v in oa[j].state[a[j]].items():assert torch.equal(v,ob[j].state[b[j]][k]),(step,j,k)
if __name__=='__main__':
 test_equivalence();print('PASS: lazy state, absent gradients, learning rates, clone, checkpoint restore')
