import runpy,torch,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts/check_fixed_training.py'))
cfg=ns['cfg'];renderer=ns['renderer'];c=ns['c'];Tiny=ns['Tiny']
cfg.mode='train';cfg.optim.fixed_gaussians=False;cfg.render.gsplat_densification=True
results={}
for backend,packed in [('legacy',False),('gsplat',False),('gsplat',True)]:
    cfg.render.backend=backend;cfg.render.gsplat_packed=packed
    pc=Tiny();r=renderer.render_kernel(c,pc)
    loss=r['rgb'].square().mean()+.02*r['depth'].mean()+.01*r['acc'].mean()
    loss.backward();grad=r['viewspace_points'].grad
    assert grad.shape==(2,3) and torch.isfinite(grad).all() and (grad[:,2]>0).all()
    results[backend+('_packed' if packed else '')]=grad
reference=results['legacy'];report={}
for name,g in results.items():
    relative=float((g-reference).norm()/reference.norm())
    assert relative<1e-4,(name,relative,g,reference)
    report[name]=dict(relative_l2=relative,statistics=g.tolist())
# Packed remapping with an absent Gaussian.
from lib.models.gsplat_screen_stats import GsplatScreenStats
x=torch.zeros(2,2,device='cuda',requires_grad=True);(x.sum()).backward();x.absgrad=torch.full_like(x,2.)
g=GsplatScreenStats(x,4,128,96,torch.tensor([1,3],device='cuda')).grad
assert torch.equal(g[[0,2]],torch.zeros(2,3,device='cuda'))
assert torch.equal(g[1],torch.tensor([64.,48.,224.],device='cuda'))
out=ROOT/'output/gsplat_stage67';out.mkdir(exist_ok=True)
(out/'screen_stats_test.json').write_text(json.dumps(report,indent=2))
print('SCREEN_STATS_PASS',json.dumps(report))
white={}
for backend,packed in [('legacy',False),('gsplat',False),('gsplat',True)]:
    cfg.render.backend=backend;cfg.render.gsplat_packed=packed
    pc=Tiny();r=renderer.render_kernel(c,pc,white_background=True)
    r['rgb'].square().mean().backward()
    white[backend+str(packed)]=(r['rgb'].detach(),r['viewspace_points'].grad)
for name,(rgb,g) in white.items():
    ref,rg=white['legacyFalse']
    assert (rgb-ref).abs().max()<1e-5,(name,'white RGB')
    assert (g-rg).norm()/rg.norm()<1e-4,(name,'white gradients')
print('WHITE_PACKED_PASS')
