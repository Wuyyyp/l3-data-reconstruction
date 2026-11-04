import copy,torch,json
from actor_topology import compact,densify,ATTR

def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b

def run(scene,out):
 m=getattr(scene.sg,scene.sg.obj_list[0]);original=copy.deepcopy(m.state_dict(False));rng=torch.cuda.get_rng_state_all();cpu=torch.get_rng_state()
 loss=sum(getattr(m,k).square().mean() for k in ATTR.values() if getattr(m,k).numel());loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
 e=densify(m,True);assert e['after']==e['before']+1
 with torch.no_grad():
  for k in ATTR.values():getattr(m,k)[1].copy_(getattr(m,k)[0])
 n=len(m.get_xyz);drop=torch.zeros(n,dtype=torch.bool,device='cuda');drop[2]=True;before=m._xyz.detach()[3].clone();mom=m.optimizer.state[m._xyz]['exp_avg'][3].clone();e2=compact(m,drop,torch.tensor([[0,1]],device='cuda'))
 assert len(m.get_xyz)==n-2 and torch.equal(m._xyz[1],before)
 assert torch.equal(m.optimizer.state[m._xyz]['exp_avg'][1],mom) and not m.optimizer.state[m._xyz]['exp_avg'][0].any()
 saved=copy.deepcopy(m.state_dict(False));m.load_state_dict(saved);exact(saved,m.state_dict(False))
 loss=sum(getattr(m,k).square().mean() for k in ATTR.values() if getattr(m,k).numel());loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
 assert all(torch.isfinite(getattr(m,k)).all() for k in ATTR.values())
 m.load_state_dict(original);exact(original,m.state_dict(False));torch.cuda.set_rng_state_all(rng);torch.set_rng_state(cpu)
 (out/'actor_topology_test.json').write_text(json.dumps(dict(passed=True,densify=e,compact=e2,optimizer_restored_exact=True,unaffected_moments_preserved=True,merged_moments_zero=True)))
