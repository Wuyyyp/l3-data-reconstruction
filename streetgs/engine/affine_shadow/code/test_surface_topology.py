import copy,torch,json
import surface_actor as sa
from actor_topology import compact,ATTR
from test_actor_topology import exact
@torch.no_grad()
def add_duplicate(m):
 old=len(m._xyz);m.densification_postfix({k:getattr(m,v)[:1].detach().clone() for k,v in ATTR.items()});m.surface['ids']=torch.cat([m.surface['ids'],m.surface['ids'][:1]]);sa.project(m);return old
def run(scene,out):
 m=getattr(scene.sg,scene.sg.obj_list[0]);original=copy.deepcopy(m.state_dict(False));surf=copy.deepcopy(m.surface);rng=torch.cuda.get_rng_state_all();cpu=torch.get_rng_state()
 loss=sum(getattr(m,k).square().mean() for k in ATTR.values() if getattr(m,k).numel() and getattr(m,k).requires_grad);loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);sa.project(m)
 e=sa.densify(m,True);assert e['after']==e['before']+1;sa.audit(m)
 n=add_duplicate(m);drop=torch.zeros(n+1,dtype=torch.bool,device='cuda');drop[2]=True;keep=~drop;keep[n]=False
 before=m._xyz.detach()[3].clone();mom=m.optimizer.state[m._xyz]['exp_avg'][3].clone();e2=compact(m,drop,torch.tensor([[0,n]],device='cuda'));m.surface['ids']=m.surface['ids'][keep];sa.project(m)
 assert torch.allclose(m._xyz[2],before,atol=1e-7);assert torch.equal(m.optimizer.state[m._xyz]['exp_avg'][2],mom);assert not m.optimizer.state[m._xyz]['exp_avg'][0].any();sa.audit(m)
 saved=copy.deepcopy(m.state_dict(False));saved_surface=copy.deepcopy(m.surface);m.load_state_dict(saved);m.surface=saved_surface;m._rotation.requires_grad_(False);exact(saved,m.state_dict(False));sa.audit(m)
 loss=sum(getattr(m,k).square().mean() for k in ATTR.values() if getattr(m,k).numel() and getattr(m,k).requires_grad);loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);sa.project(m);sa.audit(m)
 m.load_state_dict(original);m.surface=surf;m._rotation.requires_grad_(False);exact(original,m.state_dict(False));torch.cuda.set_rng_state_all(rng);torch.set_rng_state(cpu)
 (out/'surface_topology_test.json').write_text(json.dumps(dict(passed=True,densify=e,merge_prune=e2,optimizer_restored_exact=True,anchors_restored_exact=True,unaffected_moments_preserved=True,merged_moments_zero=True)))
