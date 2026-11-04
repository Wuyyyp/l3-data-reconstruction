import tempfile,io
from ground_extension import *

def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

with tempfile.TemporaryDirectory() as temp:
 path=Path(temp)/'small.npz';a=np.load(os.environ['GROUND_INIT']);np.savez(path,**{k:a[k][:64] for k in a.files})
 g=GroundModel(path,True)
 def update(model):
  loss=sum(p.square().sum() for p in model.parameters());loss.backward();model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
 update(g);xyz=g.get_xyz.detach().clone();mom=g.optimizer.state[g.features]['exp_avg'].clone()
 ids=torch.tensor([1,5,9],device='cuda');e=g.split(ids,3);assert len(g.initial)==67
 retained=torch.tensor([i for i in range(64) if i not in [1,5,9]],device='cuda')
 assert torch.equal(g.get_xyz[:61],xyz[retained])
 assert torch.equal(g.optimizer.state[g.features]['exp_avg'][:61],mom[retained])
 assert (g.optimizer.state[g.features]['exp_avg'][61:]==0).all()
 centers=g.get_xyz[61:].reshape(3,2,3).mean(1);assert torch.allclose(centers,xyz[ids],atol=2e-5,rtol=0)
 update(g);buffer=io.BytesIO();torch.save(g.capture(),buffer);buffer.seek(0);saved=torch.load(buffer);restored=GroundModel(path,True);restored.restore(saved);equal(saved,restored.capture())
 assert torch.isfinite(g.covars).all()
 update(g);update(restored);equal(g.capture(),restored.capture())
 report=dict(passed=True,points=67,split_event=e,unchanged_rows_preserved=True,adam_moments_preserved=True,child_moments_zero=True,restore_exact=True,next_update_exact=True)
 Path(os.environ['UNIT_REPORT']).write_text(json.dumps(report,indent=2));print(report,flush=True)
