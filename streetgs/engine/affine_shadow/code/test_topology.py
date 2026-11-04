from ground_extension import *
g=GroundModel(os.environ['GROUND_INIT'],True)
loss=sum(p.square().mean() for p in g.parameters());loss.backward();g.optimizer.step();g.optimizer.zero_grad()
n=len(g.initial);g.initial[1]=g.initial[0];g.normals[1]=g.normals[0];g.rotation[1]=g.rotation[0];g.source_id[1]=g.source_id[0]
for k in ['features','scale','opacity','offset']:getattr(g,k).data[1]=getattr(g,k).data[0]
drop=torch.zeros(n,dtype=torch.bool,device='cuda');drop[2]=True
old=g.features.detach()[3].clone();e=g.compact(drop,torch.tensor([[0,1]],device='cuda'),0)
assert len(g.initial)==n-2 and torch.equal(g.features[1],old)
s=g.capture();h=GroundModel(os.environ['GROUND_INIT'],True);h.restore(s)
for k,v in g.state_dict().items():assert torch.equal(v,h.state_dict()[k]),k
for p,q in zip(g.optimizer.param_groups,h.optimizer.param_groups):
 for a,b in zip(p['params'],q['params']):
  for k,v in g.optimizer.state[a].items():
   assert torch.equal(v,h.optimizer.state[b][k]) if torch.is_tensor(v) else v==h.optimizer.state[b][k]
loss=sum(p.square().mean() for p in h.parameters());loss.backward();h.optimizer.step()
assert all(torch.isfinite(p).all() for p in h.parameters())
print('TOPOLOGY_TEST_PASS',e,flush=True)
