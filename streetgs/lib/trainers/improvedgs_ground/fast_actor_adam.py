"""Topology-aware merged foreach Adam; original optimizers retain checkpoint ownership."""
from collections import defaultdict
import torch

class BatchedActorAdam:
    def __init__(self):
        self.optimizer=None
        self.signature=None
        self.members=[]
        self.owners={}
    def step(self,optimizers):
        groups=[(opt,g) for opt in optimizers for g in opt.param_groups]
        signature=tuple((id(opt),id(opt.state),tuple(id(p) for p in g['params']),
            tuple(sorted((k,v) for k,v in g.items() if k not in ('params','name','lr','foreach'))))
            for opt,g in groups)
        rebuild=signature!=self.signature
        if not rebuild:
            # Learning rates change every step; only regroup if a formerly equal group diverges.
            rebuild=any(any(g['lr']!=member[0]['lr'] for g in member) for member in self.members)
        if rebuild:
            buckets={};self.owners={};state=defaultdict(dict)
            for opt,g in groups:
                options={k:v for k,v in g.items() if k not in ('params','name','foreach')}
                assert not options.get('differentiable',False)
                assert not options.get('fused',False)
                options['foreach']=True
                key=tuple(sorted(options.items()))
                if key not in buckets:buckets[key]=(dict(options,params=[]),[])
                merged,members=buckets[key];members.append(g)
                for p in g['params']:
                    assert p not in self.owners
                    merged['params'].append(p);self.owners[p]=opt
                    if p in opt.state:state[p]=opt.state[p]
            self.optimizer=torch.optim.Adam([v[0] for v in buckets.values()])
            self.optimizer.state=state;self.members=[v[1] for v in buckets.values()]
            self.signature=signature
        else:
            for merged,members in zip(self.optimizer.param_groups,self.members):
                merged['lr']=members[0]['lr']
        self.optimizer.step()
        for p,opt in self.owners.items():
            if p in self.optimizer.state:opt.state[p]=self.optimizer.state[p]
        self.optimizer.zero_grad(set_to_none=True)
        return len(self.optimizer.param_groups)
