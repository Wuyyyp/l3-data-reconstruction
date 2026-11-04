import os,sys,json,importlib,time,ast
from pathlib import Path
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');sys.path.insert(0,str(ROOT))
import numpy as np,torch
for n in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+n)
for n,m in list(sys.modules.items()):
 if n=='numpy.core' or n.startswith('numpy.core.'):sys.modules[n.replace('numpy.core','numpy._core',1)]=m
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.loss_utils import l1_loss,ssim,psnr
from gsplat.optimizers import SelectiveAdam
base=ROOT/'output/gsplat_stage67_fixed15k_20261006';out=base/'optimizer_appearance_probe';out.mkdir(exist_ok=True)
cfg.mode='train';cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False;cfg.optim.fixed_gaussians=True;cfg.optim.actor_reg_start_iter=25000
cfg.model_path=str(out/'dataset');Path(cfg.model_path).mkdir(exist_ok=True);p=Path(cfg.model_path)/'input_ply'
if not p.exists():p.symlink_to(ROOT/'output/gsplat_stage123/initialization_baseline/input_ply',target_is_directory=True)
dataset=Dataset();cams={c.image_name:c for c in dataset.train_cameras[1]};model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup();state=torch.load(base/'workers/gsplat/gsplat/iteration_25000.pth');model.load_state_dict(state);del state
model.active_sh_degree=model.max_sh_degree;cfg.render.backend='gsplat';cfg.render.gsplat_packed=True;cfg.render.gsplat_radius_clip=0.
renderer=StreetGaussianRenderer();src=ast.parse((ROOT/'train.py').read_text());fn=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='training');loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gt_image' for t in n.targets));end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward');loss_code=compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'production_loss','exec')
opts=[]
for name in list(model.model_name_id)+['actor_pose','sky_cubemap','color_correction','pose_correction']:
 obj=getattr(model,name,None)
 if obj is not None and hasattr(obj,'optimizer'):
  opts.append((name,obj.optimizer))
  for group in obj.optimizer.param_groups:group['lr']=0.
bg=model.background.optimizer
sel=SelectiveAdam([dict(params=g['params'],lr=0.,name=g.get('name','')) for g in bg.param_groups],eps=bg.param_groups[0]['eps'],betas=bg.param_groups[0]['betas'])
for group in bg.param_groups:
 for par in group['params']:
  old=bg.state[par]
  if old:sel.state[par]={k:v.clone() if torch.is_tensor(v) else v for k,v in old.items()}
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return (time.perf_counter()-t)*1000,v
orig=renderer.render;render_ms=[]
def wrapped(*a,**kw):
 t,v=timed(lambda:orig(*a,**kw));render_ms.append(t);return v
renderer.render=wrapped
rows=[];names=[f'{f:06d}_{c:02d}' for f in [0,40,99] for c in [0,1,2,3,4,9,10]]
for i,name in enumerate(names[:7]+names*2):
 c=cams[name];render_ms.clear();ns=dict(torch=torch,l1_loss=l1_loss,ssim=ssim,psnr=psnr,gaussians=model,gaussians_renderer=renderer,training_args=cfg.train,optim_args=cfg.optim,data_args=cfg.data,viewpoint_cam=c,iteration=20000,sky_mask=None,lidar_depth=None,obj_bound=None,tensor_dict={})
 fwd,_=timed(lambda:exec(loss_code,ns));back,_=timed(lambda:ns['loss'].backward())
 def make_mask():
  start,end=model.graph_gaussian_range['background'];vis=ns['render_pkg']['visibility_filter'][start:end].contiguous()
  assert model.background.background_mask is None,'Need full background indexing'
  assert len(vis)==len(model.background._xyz)
  return vis
 maskms,vis=timed(make_mask)
 funcs={'adam':lambda:bg.step(),'selective':lambda:sel.step(vis)};tm={}
 for key in (['adam','selective'] if i%2 else ['selective','adam']):tm[key],_=timed(funcs[key])
 other,_=timed(lambda:[o.step() for n,o in opts if n!='background']);zero,_=timed(lambda:[o.zero_grad(set_to_none=True) for n,o in opts])
 row=dict(image=name,forward_ms=fwd,render_ms=sum(render_ms),loss_other_ms=fwd-sum(render_ms),backward_ms=back,bg_adam_ms=tm['adam'],bg_selective_ms=tm['selective'],mask_ms=maskms,other_optimizer_ms=other,zero_ms=zero,visible_fraction=float(vis.float().mean()),loss=float(ns['loss']))
 row['baseline_ms']=fwd+back+tm['adam']+other+zero;row['candidate_estimated_ms']=fwd+back+tm['selective']+maskms+other+zero
 if i>=7:rows.append(row)
 assert np.isfinite(row['loss']);del ns
 print('STEP',i,name,tm,flush=True)
summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='image'};summary['estimated_time_reduction']=1-summary['candidate_estimated_ms']/summary['baseline_ms'];summary['optimizer_fraction']=(summary['bg_adam_ms']+summary['other_optimizer_ms'])/summary['baseline_ms']
(out/'optimizer.json').write_text(json.dumps(dict(summary=summary,rows=rows,protocol='GPU1 gsplat25k checkpoint packed=True. 7 warmup +42 measurements,21 images x2,rotating Adam/Selective order. CUDA synchronize per segment. Production forward/loss/backward at iteration20000 fixed topology. LR=0 for all optimizers: model unchanged; moment states update in memory. SelectiveAdam tested ONLY background, real gradients and projected visibility; other optimizers unchanged. Both kernels run each iteration, candidate whole-step time is a sum estimate not a separate training run. No quality/convergence validation, no checkpoint saved.'),indent=2));print('COMPLETE',json.dumps(summary),flush=True)
