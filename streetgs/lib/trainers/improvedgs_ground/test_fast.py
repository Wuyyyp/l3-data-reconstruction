"""GPU regression tests for the optimized ground pipeline."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from ground_extension import *
from fast_losses import masked_ssim as fast_ssim
from utils.static_losses import masked_ssim as reference_ssim
from fast_actor_adam import BatchedActorAdam
from mesh_backend import gsplat_render
from bridge import actor_affines,_AFFINE
torch.manual_seed(123);torch.set_num_threads(4)
report={}
def close(a,b,atol,rtol,label):
 d=(a-b).abs();m=float(d.max()) if d.numel() else 0.
 assert torch.allclose(a,b,atol=atol,rtol=rtol),(label,m)
 report[label+'_max_abs']=m
# SSIM: same window support, no gradients in excluded windows, nontrivial masks.
x=torch.rand(3,96,128,device='cuda',requires_grad=True);y=torch.rand_like(x)
mask=torch.ones(1,96,128,dtype=torch.bool,device='cuda');mask[:,20:55,10:35]=False
a=reference_ssim(x,y,mask);ga=torch.autograd.grad(a,x)[0]
b=fast_ssim(x,y,mask);gb=torch.autograd.grad(b,x)[0]
close(a,b,3e-5,1e-4,'ssim_value');close(ga,gb,2e-6,2e-3,'ssim_gradient')
assert torch.equal(gb[:,~mask[0]],torch.zeros_like(gb[:,~mask[0]]))
# Packed rasterization: RGB/depth/alpha and all model/screen-space gradients, EAS.
n=6000
c=SimpleNamespace(image_width=320,image_height=192,world_view_transform=torch.eye(4,device='cuda'),
 full_proj_transform=torch.eye(4,device='cuda'),camera_center=torch.zeros(3,device='cuda'))
xyz=torch.randn(n,3,device='cuda');xyz[:,2]=xyz[:,2]*2+4;xyz.requires_grad_()
sc=torch.full((n,3),.035,device='cuda',requires_grad=True)
q=torch.zeros(n,4,device='cuda');q[:,0]=1
cov=quat_scale_to_covar_preci(q,sc,compute_covar=True,compute_preci=False,triu=False)[0]
opacity=torch.full((n,1),.4,device='cuda',requires_grad=True)
dc=torch.randn(n,1,3,device='cuda',requires_grad=True)*.1
rest=torch.randn(n,3,3,device='cuda',requires_grad=True)*.03
m=SimpleNamespace(get_xyz=xyz,covars=cov,get_opacity=opacity,get_features_dc=dc,get_features_rest=rest,active_sh_degree=1)
pipe=SimpleNamespace(debug=False);bg=torch.zeros(3,device='cuda');weight=torch.rand(3,192,320,device='cuda')
outs=[];grads=[];stats=[];scores=[]
for packed in [False,True]:
 cfg.render.gsplat_packed=packed
 out=gsplat_render(c,m,pipe,bg,metric=True)
 loss=(out['render']*weight).mean()+out['depth'].mean()*.01
 gradients=torch.autograd.grad(loss,[xyz,sc,opacity,dc,rest],retain_graph=True)
 outs.append(out);grads.append(gradients);stats.append(out['viewspace_points'].grad.clone())
 scores.append(gsplat_render(c,m,pipe,bg,pixel_weights=weight[0])['accum_weights'])
for key in ['render','depth','acc','radii']:close(outs[0][key],outs[1][key],3e-4,2e-3,'packed_'+key)
assert torch.equal(outs[0]['visibility_filter'],outs[1]['visibility_filter'])
for i,(a,b) in enumerate(zip(*grads)):close(a,b,2e-5,3e-3,'packed_param_grad_'+str(i))
close(stats[0],stats[1],2e-5,3e-3,'packed_absgrad')
close(scores[0],scores[1],1e-3,2e-3,'packed_eas')
# Batch affine covariance math and changing topology/frame.
counts=[19,37,5];names=['obj_%03d'%i for i in range(3)]
sg=SimpleNamespace(frame=0)
for i,name in enumerate(names):
 setattr(sg,name,SimpleNamespace(track_id=i))
 for frame in [0,1]:
  A=torch.randn(3,3,device='cuda')*.2+torch.diag(torch.tensor([1.,1.4,.7],device='cuda'))
  _AFFINE[i,frame]=(A,torch.randn(3,device='cuda'))
for frame,counts in [(0,counts),(1,counts),(1,[20,37,5])]:
 sg.frame=frame;p=torch.randn(sum(counts),3,device='cuda',requires_grad=True)
 cov=torch.randn(sum(counts),3,3,device='cuda',requires_grad=True)
 A,t=actor_affines(sg,names,counts)
 gotp=torch.bmm(A,p[:,:,None]).squeeze(-1)+t;gotc=torch.bmm(torch.bmm(A,cov),A.transpose(1,2))
 xp=[];xc=[];start=0
 for i,n in enumerate(counts):
  a,t=_AFFINE[i,frame];k=torch.einsum('ai,bj->abij',a,a).reshape(9,9)
  xp.append(torch.nn.functional.linear(p[start:start+n],a,t))
  xc.append(torch.nn.functional.linear(cov[start:start+n].reshape(n,9),k).reshape(n,3,3));start+=n
 xp=torch.cat(xp);xc=torch.cat(xc)
 close(gotp,xp,1e-6,1e-5,'affine_xyz');close(gotc,xc,2e-6,1e-5,'affine_cov')
 g1=torch.autograd.grad(gotp.square().sum()+gotc.square().sum(),[p,cov],retain_graph=True)
 g2=torch.autograd.grad(xp.square().sum()+xc.square().sum(),[p,cov])
 for i,(a,b) in enumerate(zip(g1,g2)):close(a,b,2e-5,2e-5,'affine_grad_'+str(i))
# Adam owners retain complete state even when an actor first becomes visible later.
params=[torch.nn.Parameter(torch.randn(17,3,device='cuda')) for _ in range(4)]
refs=[torch.nn.Parameter(p.detach().clone()) for p in params]
opts=[torch.optim.Adam([p],lr=.001) for p in params];refopts=[torch.optim.Adam([p],lr=.001) for p in refs]
merged=BatchedActorAdam()
for step in range(5):
 for i,(p,r,o,ro) in enumerate(zip(params,refs,opts,refopts)):
  o.param_groups[0]['lr']=ro.param_groups[0]['lr']=.001*(.9**step)*(2 if i==2 and step>=3 else 1)
  if i==3 and step<2:continue
  g=torch.randn_like(p);p.grad=g.clone();r.grad=g.clone()
 merged.step(opts)
 for o in refopts:o.step();o.zero_grad(set_to_none=True)
 for i,(p,r,o,ro) in enumerate(zip(params,refs,opts,refopts)):
  close(p,r,1e-7,1e-6,'adam_param')
  if p in o.state:
   for key in ['step','exp_avg','exp_avg_sq']:close(o.state[p][key],ro.state[r][key],1e-7,1e-6,'adam_'+key)
  else:assert r not in ro.state
# Replacement after densification, with an initialized replacement state.
params[0]=torch.nn.Parameter(params[0].detach()[:8].clone());refs[0]=torch.nn.Parameter(refs[0].detach()[:8].clone())
opts[0]=torch.optim.Adam([params[0]],lr=.001);refopts[0]=torch.optim.Adam([refs[0]],lr=.001)
for p,r in zip(params,refs):g=torch.randn_like(p);p.grad=g.clone();r.grad=g.clone()
merged.step(opts)
for o in refopts:o.step();o.zero_grad(set_to_none=True)
for p,r in zip(params,refs):close(p,r,1e-7,1e-6,'adam_topology')
report.update(passed=True,masked_gradients_excluded=True,late_actor_state_owned=True,topology_and_frame_cache_invalidated=True)
Path(os.environ['PERF_TEST_REPORT']).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
