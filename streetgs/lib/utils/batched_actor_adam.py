import torch
from collections import defaultdict
class BatchedActorAdam:
 """Group compatible Adam parameter groups; original optimizers own all state."""
 def __init__(self):self.optimizer=None
 def step(self,optimizers):
  groups={};states=defaultdict(dict);seen=set();owners={}
  for opt in optimizers:
   assert isinstance(opt,torch.optim.Adam)
   for g in opt.param_groups:
    options={k:v for k,v in g.items() if k not in ('params','name')}
    assert not options.get('differentiable',False)
    assert all(not torch.is_tensor(v) for v in options.values())
    key=tuple(sorted(options.items()))
    if key not in groups:groups[key]=dict(options,params=[])
    for p in g['params']:
     assert id(p) not in seen;seen.add(id(p));groups[key]['params'].append(p);owners[p]=opt
     if p in opt.state:states[p]=opt.state[p]
  gs=list(groups.values())
  if self.optimizer is None:self.optimizer=torch.optim.Adam(gs)
  else:self.optimizer.param_groups=gs
  self.optimizer.state=states
  self.optimizer.step()
  for p,opt in owners.items():
   if p in states and states[p]:opt.state[p]=states[p]
   p.grad=None
  return len(gs)
