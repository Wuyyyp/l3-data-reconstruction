"""Isolated paired 10000->11000 continuation; no production source changes."""
import os,sys,ast,json,time,random,importlib,hashlib,gc,math,types,traceback
from pathlib import Path
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
for n in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+n)
for n,m in list(sys.modules.items()):
 if n=='numpy.core' or n.startswith('numpy.core.'):sys.modules[n.replace('numpy.core','numpy._core',1)]=m
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.gaussian_model import GaussianModel
from lib.utils.loss_utils import l1_loss,ssim,psnr
base=ROOT/'output/gsplat_stage67_fixed15k_20261006';out=base/'bilateral_mcmc_probe/mcmc';out.mkdir(exist_ok=True)
assert not (out/'started.json').exists(),'Do not duplicate experiment'
(out/'started.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid())))
def status(**kw):(out/'status.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
def failed(k,v,t):status(state='failed',error=''.join(traceback.format_exception(k,v,t)));sys.__excepthook__(k,v,t)
sys.excepthook=failed;status(state='loading')
cfg.mode='train';cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
cfg.optim.fixed_gaussians=True;cfg.render.gsplat_densification=True;cfg.render.backend='gsplat';cfg.render.gsplat_packed=True;cfg.render.gsplat_radius_clip=0.
cfg.train.iterations=25000;cfg.optim.densify_until_iter=15000;cfg.optim.actor_reg_start_iter=25000
cfg.model_path=str(out/'dataset');Path(cfg.model_path).mkdir(exist_ok=True);p=Path(cfg.model_path)/'input_ply'
if not p.exists():p.symlink_to(ROOT/'output/gsplat_stage123/initialization_baseline/input_ply',target_is_directory=True)
dataset=Dataset();cameras={c.image_name:c for c in dataset.train_cameras[1]};renderer=StreetGaussianRenderer()
worker=ast.parse((ROOT/'scripts/run_stage67_fixed.py').read_text());ns0={}
# Compile production loss/mask/topology exactly as the earlier worker did.
src=(ROOT/'train.py').read_text();tree=ast.parse(src);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training');loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gt_image' for t in n.targets));end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward')
loss_code=compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'production_loss','exec');mask_end=next(i for i,n in enumerate(loop.body[start:end]) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='render_pkg' for t in n.targets));mask_code=compile(ast.Module(body=loop.body[start:start+mask_end],type_ignores=[]),'production_mask','exec')
post=next(n for n in loop.body if isinstance(n,ast.With) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='no_grad' for x in ast.walk(n.items[0].context_expr)));nodes=[n for n in post.body if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['add_densification_stats','reset_opacity'] for x in ast.walk(n))];assert len(nodes)==2;topology_code=compile(ast.Module(body=nodes,type_ignores=[]),'production_topology','exec')
def ctx(model,c,step):return dict(torch=torch,l1_loss=l1_loss,ssim=ssim,psnr=psnr,gaussians=model,gaussians_renderer=renderer,training_args=cfg.train,optim_args=cfg.optim,data_args=cfg.data,viewpoint_cam=c,iteration=step,sky_mask=None,lidar_depth=None,obj_bound=None,tensor_dict={})
def count(model):return sum(len(getattr(model,n)._xyz) for n in model.model_name_id)
def seed(i,top=False):
 v=628031+i+(1000000 if top else 0);torch.manual_seed(v);torch.cuda.manual_seed_all(v);np.random.seed(v);random.seed(v)
def saveimg(t,path):Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype('uint8')).save(path)
@torch.no_grad()
def evaluate(model,label,full):
 cfg.mode='evaluate';rows=[];names=sorted(cameras) if full else [f'{f:06d}_{c:02d}' for f in [0,40,99] for c in [0,1,2,3,4,9,10]]
 directory=out/label;directory.mkdir(exist_ok=True)
 for idx,name in enumerate(names):
  c=cameras[name];ns=ctx(model,c,11500);exec(mask_code,ns);r=renderer.render(c,model);row=dict(image=name,psnr=float(psnr(r['rgb'],ns['gt_image'],ns['mask'])),ssim=float(ssim(r['rgb'],ns['gt_image'],mask=ns['mask'])))
  if name=='000040_02':
   h,w=r['rgb'].shape[1:];row['regions']={}
   for labelr,box in {'trees':(0,0,.4,.55),'road':(0,.6,1,1)}.items():
    x0,y0,x1,y1=box;mask=ns['mask'].clone();region=torch.zeros_like(mask);region[:,int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True;mask&=region;row['regions'][labelr]=float(psnr(r['rgb'],ns['gt_image'],mask))
  rows.append(row)
  if full and name.startswith('000040_'):
   saveimg(r['rgb'],directory/(name+'.png'))
   if label=='baseline':saveimg(ns['gt_image'],directory/(name+'_gt.png'))
  if idx%100==0:status(state='evaluating',variant=label,images=idx,total=len(names))
 result=dict(psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),points=count(model),rows=rows);assert math.isfinite(result['psnr']);(directory/('eval.json' if full else 'initial_eval.json')).write_text(json.dumps(result,indent=2));cfg.mode='train';return result

from types import SimpleNamespace
from gsplat.strategy import MCMCStrategy
from gsplat.strategy.ops import relocate,inject_noise_to_position
sequence=json.loads((base/'workers/gsplat/camera_sequence.json').read_text())[10000:11500];ckpt=base/'workers/gsplat/gsplat/iteration_10000.pth'
mapping={'xyz':('means','_xyz'),'f_dc':('f_dc','_features_dc'),'f_rest':('f_rest','_features_rest'),'opacity':('opacities','_opacity'),'scaling':('scales','_scaling'),'rotation':('quats','_rotation'),'semantic':('semantic','_semantic')}
def adapter(obj):
 params={};opts={};attrs={}
 for group in obj.optimizer.param_groups:
  key,attr=mapping[group['name']];assert len(group['params'])==1 and group['params'][0] is getattr(obj,attr)
  params[key]=getattr(obj,attr);opts[key]=SimpleNamespace(state=obj.optimizer.state,param_groups=[group]);attrs[key]=attr
 assert {'means','opacities','scales','quats'}<=set(params)
 return params,opts,attrs
def apply_mcmc(obj,iteration,binoms):
 params,opts,attrs=adapter(obj);n=len(params['means']);dead=0;skipped=0
 if not n:return 0,0
 if iteration%100==0:
  dead_mask=torch.sigmoid(params['opacities'].flatten())<=.005;dead=int(dead_mask.sum())
  if 0<dead<n:relocate(params,opts,{},dead_mask,binoms,min_opacity=.005)
  elif dead==n:skipped=dead;dead=0
  for k,attr in attrs.items():setattr(obj,attr,params[k])
  for group in obj.optimizer.param_groups:
   key,attr=mapping[group['name']];assert group['params'][0] is getattr(obj,attr)
 inject_noise_to_position(params,opts,{},scaler=obj.optimizer.param_groups[0]['lr']*5e5)
 assert len(obj._xyz)==n
 return dead,skipped
# Small GPU adapter smoke test exercises relocation, moment remapping and all attributes.
fake=SimpleNamespace();groups=[]
for name,(key,attr) in mapping.items():
 shape=(8,3) if name in ['xyz','scaling'] else (8,4) if name=='rotation' else (8,1) if name=='opacity' else (8,1,3)
 value=torch.zeros(shape,device='cuda')
 if name=='opacity':value[:2]=-10
 if name=='rotation':value[:,0]=1
 if name=='scaling':value.fill_(-3)
 par=torch.nn.Parameter(value);setattr(fake,attr,par);groups.append(dict(params=[par],lr=1e-6,name=name))
fake.optimizer=torch.optim.Adam(groups);sum(g['params'][0].sum() for g in groups).backward();fake.optimizer.step();fake.optimizer.zero_grad(set_to_none=True)
binoms=MCMCStrategy().initialize_state()['binoms'].cuda();a,b=apply_mcmc(fake,100,binoms);assert a==2 and b==0
for g in fake.optimizer.param_groups:assert torch.isfinite(g['params'][0]).all()
del fake;torch.cuda.empty_cache()
protocol=dict(start=10000,stop=11500,mcmc_until=11000,recovery_steps=500,packed=True,regularizer_opacity=.01,regularizer_scale=.01,noise_lr=5e5,relocate_every=100,min_opacity=.005,point_cap='initial count in each model; growth disabled',checkpoint=str(ckpt),camera_sequence=sequence,note='Both arms share added .01 opacity and .01 activated-scale point-weighted regularization. MCMC relocation/noise only first1000steps, then500 identical fixed optimization. No original densification/prune/reset. Actor local coords separate. All-dead models skip relocation, never borrow another model.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));summary={};events=[]
for variant in ['baseline','mcmc']:
 cfg.mode='train';status(state='loading_checkpoint',variant=variant);model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup();state=torch.load(ckpt);model.load_state_dict(state)
 flags=state['_stage67']['random_initialization']
 for n,flag in flags.items():getattr(model,n).random_initialization=flag
 del state;model.active_sh_degree=model.max_sh_degree;initial=evaluate(model,variant,False);fixed_count=count(model)
 if variant=='mcmc':assert abs(initial['psnr']-summary['baseline']['initial_psnr'])<1e-5
 curves=[];since=time.perf_counter()
 for step,name in enumerate(sequence,10001):
  cfg.mode='train';seed(step);model.update_learning_rate(step);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ns=ctx(model,cameras[name],step);exec(loss_code,ns)
  regularizer=0.
  for n in model.model_name_id:
   obj=getattr(model,n);regularizer=regularizer+.01*torch.sigmoid(obj._opacity).sum()/fixed_count+.01*torch.exp(obj._scaling).sum()/(fixed_count*3)
  loss=ns['loss']+regularizer;assert torch.isfinite(loss);loss.backward()
  with torch.no_grad():
   model.update_optimizer()
   if variant=='mcmc' and step<=11000:
    seed(step,True)
    for n in model.model_name_id:
     obj=getattr(model,n);dead,skipped=apply_mcmc(obj,step,binoms)
     if step%100==0:events.append(dict(step=step,model=n,relocated=dead,all_dead_skipped=skipped))
  torch.cuda.synchronize();assert count(model)==fixed_count;curves.append(dict(step=step,ms=(time.perf_counter()-t)*1000,points=fixed_count,loss=float(loss),peak_mb=torch.cuda.max_memory_allocated()/2**20));del ns,loss,regularizer
  if step%100==0:
   for n in model.model_name_id:assert torch.isfinite(getattr(model,n)._xyz).all(),n
   status(state='training',variant=variant,step=step,target=11500,points=fixed_count);print(variant,step,curves[-1],flush=True);(out/variant/'curve.json').write_text(json.dumps(curves));(out/'events.json').write_text(json.dumps(events,indent=2))
 trainwall=time.perf_counter()-since
 for n in model.model_name_id:
  for g in getattr(model,n).optimizer.param_groups:
   for par in g['params']:assert torch.isfinite(par).all(),n
 final=evaluate(model,variant,True);summary[variant]=dict(initial_psnr=initial['psnr'],final_psnr=final['psnr'],final_ssim=final['ssim'],points=final['points'],training_wall_seconds=trainwall,step_seconds=sum(r['ms'] for r in curves)/1000,peak_mb=max(r['peak_mb'] for r in curves))
 cfg.model_path=str(out/variant);state=model.save_state_dict(is_final=False);state['iter']=11500;state['_stage67']=dict(random_initialization=flags);torch.save(state,out/variant/'iteration_11500.pth');del state,model;gc.collect();torch.cuda.empty_cache();(out/'summary.json').write_text(json.dumps(summary,indent=2))
status(state='complete',summary=summary);print('COMPLETE',json.dumps(summary),flush=True)
