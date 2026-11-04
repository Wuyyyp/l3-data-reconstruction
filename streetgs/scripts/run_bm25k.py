"""Fresh 25k worker: topology stops at 15k; actor loss starts independently at 25k. No Git writes.

4k gate -> isolated gsplat optimizations -> 25k continuation for accepted runs.
Uses production loss and topology AST blocks from train.py.
"""
import sys,os,ast,json,time,random,importlib,hashlib,gc,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
for n in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+n)
for n,m in list(sys.modules.items()):
    if n=='numpy.core' or n.startswith('numpy.core.'):sys.modules[n.replace('numpy.core','numpy._core',1)]=m
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.general_utils import safe_state
from lib.utils.loss_utils import l1_loss,ssim,psnr

OUT=Path(os.environ.get('STAGE67_OUT',str(ROOT/'output/gsplat_stage67')));OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'output/gsplat_stage123/initialization_baseline'
PILOT=int(os.environ.get('STAGE67_PILOT','4000'));FINAL=int(os.environ.get('STAGE67_FINAL','25000'))
RUN_NAME=os.environ['STAGE67_VARIANT']
BENCH=os.environ.get('STAGE67_BENCH','0')=='1'
JOB_START=time.time()
MATRIX={n:dict(backend='gsplat',packed=True,radius_clip=0.) for n in ['baseline','bilateral','mcmc']}
status=dict(state='loading',pilot=PILOT,final=FINAL,results={},updated=time.time())
def atomic(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def report_status(**kw):
    status.update(kw);status['updated']=time.time();atomic(OUT/'status.json',status)
def uncaught(kind,value,tb):
    report_status(state='failed',error=''.join(traceback.format_exception(kind,value,tb)))
    sys.__excepthook__(kind,value,tb)
sys.excepthook=uncaught
assert os.environ.get('BM_RESUME','0')=='1' or not any(OUT.glob('*/curve.jsonl')),'Existing training output: select a fresh STAGE67_OUT'
report_status()
cfg.mode='train';cfg.optim.fixed_gaussians=False;cfg.render.gsplat_densification=True
cfg.train.iterations=25000;cfg.optim.densify_until_iter=15000;cfg.optim.actor_reg_start_iter=25000
cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
assert not cfg.optim.get('static_prune_enabled',False),'Static-prune extension requires separate adaptation'
cfg.model_path=str(OUT/'dataset');Path(cfg.model_path).mkdir(exist_ok=True)
p=Path(cfg.model_path)/'input_ply'
if not p.exists():p.symlink_to(BASE/'input_ply',target_is_directory=True)
safe_state(False)
dataset=Dataset();ordered_cameras=dataset.train_cameras[1]
cameras={c.image_name:c for c in ordered_cameras};assert len(cameras)==707
baseline_rng=torch.load(BASE/'rng_state.pth',map_location='cpu')
sampler=random.Random();sampler.setstate(baseline_rng['python']);stack=[];sequence=[]
for _ in range(FINAL):
    if not stack:stack=[c.image_name for c in ordered_cameras]
    sequence.append(stack.pop(sampler.randint(0,len(stack)-1)))
oldsequence=json.loads((BASE/'camera_sequence_1000.json').read_text())
assert sequence[:min(1000,FINAL)]==oldsequence[:min(1000,FINAL)],'Initial camera ordering changed'
atomic(OUT/'camera_sequence.json',sequence)
renderer=StreetGaussianRenderer()
src=(ROOT/'train.py').read_text();tree=ast.parse(src)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training')
loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gt_image' for t in n.targets))
end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward')
loss_code=compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'production_loss','exec')
mask_end=next(i for i,n in enumerate(loop.body[start:end]) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='render_pkg' for t in n.targets))
mask_code=compile(ast.Module(body=loop.body[start:start+mask_end],type_ignores=[]),'production_mask','exec')
post=next(n for n in loop.body if isinstance(n,ast.With) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='no_grad' for x in ast.walk(n.items[0].context_expr)))
topology=[]
for n in post.body:
    if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['add_densification_stats','reset_opacity'] for x in ast.walk(n)):
        topology.append(n)
assert len(topology)==2
topology_code=compile(ast.Module(body=topology,type_ignores=[]),'production_topology','exec')
atomic(OUT/'protocol.json',dict(pilot=PILOT,final=FINAL,matrix=MATRIX,
    init_sha256=hashlib.sha256((BASE/'initial_state.pth').read_bytes()).hexdigest(),
    train_sha256=hashlib.sha256(src.encode()).hexdigest(),
    seed_policy='Per-iteration separate seeds for rendering and topology; independent of variant order and Gaussian count',
    densify_until=15000,actor_reg_start=25000,learning_rate_horizon=cfg.train.iterations,eval_frames=[0,40,99],gpu=os.environ['CUDA_VISIBLE_DEVICES']))

def setup_variant(name):
    c=MATRIX[name];cfg.render.backend=c['backend'];cfg.render.gsplat_packed=c['packed'];cfg.render.gsplat_radius_clip=c['radius_clip']
def parameters(model):
    for name in list(model.model_name_id)+['actor_pose','sky_cubemap','color_correction','pose_correction']:
        obj=getattr(model,name,None)
        if obj is not None and hasattr(obj,'optimizer'):
            for g in obj.optimizer.param_groups:
                yield from g['params']
def counts(model):return {n:len(getattr(model,n).get_xyz) for n in model.model_name_id}
def load(name,step=0):
    cfg.mode='train';setup_variant(name)
    model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup()
    path=BASE/'initial_state.pth' if step==0 else OUT/name/f'iteration_{step}.pth'
    state=torch.load(path);model.load_state_dict(state)
    for obj_name in model.obj_list:
        file=BASE/'input_ply'/('points3D_'+obj_name+'.ply');size=0
        if file.exists():
            with file.open('rb') as stream:
                for line in stream:
                    if line.startswith(b'element vertex '):size=int(line.split()[-1]);break
                    if line.strip()==b'end_header':break
        getattr(model,obj_name).random_initialization=(size<2000)
    model.active_sh_degree=min(step//1000,model.max_sh_degree)
    if '_stage67' in state:
        for n,flag in state['_stage67']['random_initialization'].items():getattr(model,n).random_initialization=flag
    del state
    model.update_learning_rate(max(1,step))
    assert model.background.optimizer.param_groups[0]['lr']>0
    return model
def context(model,c,step):
    return dict(torch=torch,l1_loss=l1_loss,ssim=ssim,psnr=psnr,gaussians=model,gaussians_renderer=renderer,
        training_args=cfg.train,optim_args=cfg.optim,data_args=cfg.data,viewpoint_cam=c,iteration=step,
        sky_mask=None,lidar_depth=None,obj_bound=None,tensor_dict={})
def seed(step,topology=False):
    value=628031+step+(1000000 if topology else 0)
    torch.manual_seed(value);torch.cuda.manual_seed_all(value);random.seed(value);np.random.seed(value)
def image(t,path):Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype('uint8')).save(path)
@torch.no_grad()
def evaluate(name,model,step,save=False):
    cfg.mode='evaluate';setup_variant(name);rows=[]
    directory=OUT/name;directory.mkdir(exist_ok=True)
    for frame in ([None] if step==FINAL else [0,40,99]):
        for cam in (sorted(cameras) if frame is None else [0,1,2,3,4,9,10]):
            c=cameras[cam if frame is None else f'{frame:06d}_{cam:02d}'];ns=context(model,c,step);exec(mask_code,ns)
            r=renderer.render(c,model)
            rows.append(dict(image=c.image_name,psnr=float(psnr(r['rgb'],ns['gt_image'],ns['mask'])),ssim=float(ssim(r['rgb'],ns['gt_image'],mask=ns['mask']))))
            if save:
                folder=directory/f'images_{step}';folder.mkdir(exist_ok=True);image(r['rgb'],folder/(c.image_name+'.png'))
    result=dict(step=step,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),
                points=counts(model),rows=rows)
    assert math.isfinite(result['psnr']) and math.isfinite(result['ssim']),'Nonfinite evaluation'
    result['image_count']=len(rows)
    atomic(directory/f'eval_{step}.json',result);cfg.mode='train'
    print('EVAL',name,step,result['psnr'],result['ssim'],sum(result['points'].values()),flush=True)
    return result
@torch.no_grad()
def checkpoint(name,model,step):
    directory=OUT/name;directory.mkdir(exist_ok=True);cfg.model_path=str(directory)
    state=model.save_state_dict(is_final=False);state['iter']=step
    state['_stage67']=dict(random_initialization={n:getattr(model,n).random_initialization for n in model.obj_list})
    tmp=directory/f'iteration_{step}.pth.tmp';torch.save(state,tmp);tmp.replace(directory/f'iteration_{step}.pth')
    del state


from types import SimpleNamespace
from gsplat.strategy import MCMCStrategy
from gsplat.strategy.ops import relocate,sample_add,inject_noise_to_position
SMOKE=os.environ.get('BM_SMOKE','0')=='1'
caps_reference=json.loads((ROOT/'output/gsplat_stage67_fixed15k_20261006/workers/packed/packed/eval_25000.json').read_text())['points']
mapping={'xyz':('means','_xyz'),'f_dc':('f_dc','_features_dc'),'f_rest':('f_rest','_features_rest'),'opacity':('opacities','_opacity'),'scaling':('scales','_scaling'),'rotation':('quats','_rotation'),'semantic':('semantic','_semantic')}
bil_source=Path('/mnt/zhoukaixuan/code/gsplat-1.5.3/examples/lib_bilagrid.py').read_text();bt=ast.parse(bil_source)
from torch import nn
import torch.nn.functional as F
bnodes=[n for n in bt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'color_affine_transform','_num_tensor_elems','total_variation_loss','slice','BilateralGrid'}]
assert len(bnodes)==5;exec(compile(ast.Module(body=bnodes,type_ignores=[]),'official_bilagrid153','exec'))
cam_ids={c:i for i,c in enumerate([0,1,2,3,4,9,10])};grid_xy={};original_render=renderer.render
def render_with_grid(camera,model,*args,**kwargs):
 result=original_render(camera,model,*args,**kwargs)
 if hasattr(model,'bilagrid'):
  rgb=result['rgb'].permute(1,2,0);h,w=rgb.shape[:2];key=(h,w)
  if key not in grid_xy:
   yy,xx=torch.meshgrid((torch.arange(h,device=rgb.device)+.5)/h,(torch.arange(w,device=rgb.device)+.5)/w,indexing='ij');grid_xy[key]=torch.stack([xx,yy],-1)
  idx=torch.tensor([[cam_ids[int(camera.image_name.split('_')[-1])]]],device=rgb.device)
  result['rgb_uncorrected']=result['rgb'];corrected=slice(model.bilagrid,grid_xy[key],rgb,idx)['rgb'].permute(2,0,1)
  result['rgb']=corrected if cfg.mode=='train' else corrected.clamp(0,1)
 return result
renderer.render=render_with_grid
def setup_extra(model,step):
 if RUN_NAME=='bilateral':
  model.bilagrid=BilateralGrid(7,8,8,8).cuda();model.bilagrid_identity=model.bilagrid.grids.detach().clone();model.bilagrid_optimizer=torch.optim.Adam(model.bilagrid.parameters(),lr=.002)
  if step:
   state=torch.load(OUT/RUN_NAME/f'iteration_{step}.pth');model.bilagrid.load_state_dict(state['_extra']['grid']);model.bilagrid_optimizer.load_state_dict(state['_extra']['optimizer']);del state
 return {n:max(len(getattr(model,n)._xyz),caps_reference.get(n,0)) for n in model.model_name_id}
def adapt(obj):
 params={};opts={};attrs={}
 for g in obj.optimizer.param_groups:
  k,a=mapping[g['name']];assert g['params'][0] is getattr(obj,a);params[k]=getattr(obj,a);attrs[k]=a;opts[k]=SimpleNamespace(param_groups=[g],state=obj.optimizer.state)
 return params,opts,attrs
binoms=MCMCStrategy().initialize_state()['binoms'].cuda()
@torch.no_grad()
def mcmc_step(model,step,caps):
 events=[]
 for n in model.model_name_id:
  obj=getattr(model,n);params,opts,attrs=adapt(obj);size=len(params['means']);dead=added=skipped=0
  if not size:continue
  refine=(step>500 and step%100==0) or (SMOKE and step==1)
  if refine:
   mask=torch.sigmoid(params['opacities'].flatten())<=.005;dead=int(mask.sum())
   if 0<dead<size:relocate(params,opts,{},mask,binoms,min_opacity=.005)
   elif dead==size:skipped=dead;dead=0
   added=max(0,min(caps[n],max(size+1,int(1.05*size)))-size)
   if added:sample_add(params,opts,{},added,binoms,min_opacity=.005)
   for k,a in attrs.items():setattr(obj,a,params[k])
   assert len(obj._xyz)<=caps[n]
   obj.xyz_gradient_accum=torch.zeros((len(obj._xyz),2),device='cuda');obj.denom=torch.zeros((len(obj._xyz),1),device='cuda');obj.max_radii2D=torch.zeros(len(obj._xyz),device='cuda')
   for g in obj.optimizer.param_groups:
    k,a=mapping[g['name']];assert g['params'][0] is getattr(obj,a)
   events.append(dict(model=n,relocated=dead,added=added,all_dead_skipped=skipped,points=len(obj._xyz)))
  inject_noise_to_position(params,opts,{},scaler=obj.optimizer.param_groups[0]['lr']*5e5)
 return events
def extra_loss(model):
 if RUN_NAME=='bilateral':return 10.*model.bilagrid.tv_loss()+.001*(model.bilagrid.grids-model.bilagrid_identity).square().mean()
 if RUN_NAME=='mcmc':
  total=sum(len(getattr(model,n)._xyz) for n in model.model_name_id);loss=0.
  for n in model.model_name_id:
   obj=getattr(model,n);loss=loss+.01*torch.sigmoid(obj._opacity).sum()/total+.01*torch.exp(obj._scaling).sum()/(3*total)
  return loss
 return 0.
@torch.no_grad()
def evaluate_full(model,step):
 cfg.mode='evaluate';directory=OUT/RUN_NAME;directory.mkdir(exist_ok=True);rows=[]
 selected=sorted(cameras) if step==FINAL and not SMOKE else [f'{f:06d}_{c:02d}' for f in [0,40,99] for c in [0,1,2,3,4,9,10]]
 images=directory/f'images_{step}';images.mkdir(exist_ok=True)
 for j,name in enumerate(selected):
  c=cameras[name];ns=context(model,c,step);exec(mask_code,ns);r=renderer.render(c,model);row=dict(image=name,psnr=float(psnr(r['rgb'],ns['gt_image'],ns['mask'])),ssim=float(ssim(r['rgb'],ns['gt_image'],mask=ns['mask'])))
  if 'rgb_uncorrected' in r:
   raw=r['rgb_uncorrected'];row.update(raw_psnr=float(psnr(raw,ns['gt_image'],ns['mask'])),raw_ssim=float(ssim(raw,ns['gt_image'],mask=ns['mask'])))
  rows.append(row)
  if step==FINAL or name.startswith('000040_'):image(r['rgb'],images/(name+'.png'))
  if step==FINAL and 'rgb_uncorrected' in r:
   folder=directory/f'raw_images_{step}';folder.mkdir(exist_ok=True);image(r['rgb_uncorrected'],folder/(name+'.png'))
  if name.startswith('000040_'):
   folder=directory/'gt';folder.mkdir(exist_ok=True);image(ns['gt_image'],folder/(name+'.png'))
  if j%100==0:report_status(state='evaluating',variant=RUN_NAME,step=step,images=j,total=len(selected))
 result=dict(step=step,image_count=len(rows),psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),points=counts(model),rows=rows)
 if RUN_NAME=='bilateral':result.update(raw_psnr=float(np.mean([r['raw_psnr'] for r in rows])),raw_ssim=float(np.mean([r['raw_ssim'] for r in rows])))
 assert all(math.isfinite(r['psnr']) and math.isfinite(r['ssim']) for r in rows)
 atomic(directory/f'eval_{step}.json',result);cfg.mode='train';print('EVAL',RUN_NAME,step,result['psnr'],result['ssim'],flush=True)
 return result
@torch.no_grad()
def save_checkpoint(model,step):
 directory=OUT/RUN_NAME;cfg.model_path=str(directory);state=model.save_state_dict(is_final=False);state['iter']=step;state['_stage67']=dict(random_initialization={n:getattr(model,n).random_initialization for n in model.obj_list})
 if RUN_NAME=='bilateral':state['_extra']=dict(grid=model.bilagrid.state_dict(),optimizer=model.bilagrid_optimizer.state_dict())
 tmp=directory/f'iteration_{step}.pth.tmp';torch.save(state,tmp);tmp.replace(directory/f'iteration_{step}.pth')
def assert_finite(model):
 for p in parameters(model):assert torch.isfinite(p).all()
 if RUN_NAME=='bilateral':assert torch.isfinite(model.bilagrid.grids).all()
start_step=int(os.environ.get('BM_START','0'));directory=OUT/RUN_NAME;directory.mkdir(exist_ok=True)
model=load(RUN_NAME,start_step);caps=setup_extra(model,start_step)
atomic(OUT/'experiment_protocol.json',dict(variant=RUN_NAME,steps=FINAL,smoke=SMOKE,topology_stop=15000,mcmc_noise_stop=15000,actor_reg_start=25000,caps=caps,grid=dict(shared_by_camera=True,resolution=[8,8,8],tv=10.,identity=.001,lr=.002),mcmc=dict(noise_lr=5e5,min_opacity=.005,opacity_regularizer=.01,scale_regularizer=.01,growth=1.05,refine_every=100),note='Algorithm-package comparison: MCMC adds regularizers and changes topology. Baseline and bilateral retain StreetGS topology. All use same initialization/sequence/horizon/production masks.'))
if start_step==0:evaluate_full(model,0)
curve_path=directory/'curve.jsonl'
if start_step:
 old=[json.loads(l) for l in curve_path.read_text().splitlines()];old=[r for r in old if r['step']<=start_step];curve_path.write_text(''.join(json.dumps(r)+'\n' for r in old))
stream=curve_path.open('a');event_stream=(directory/'mcmc_events.jsonl').open('a');since=time.perf_counter();fixed_points=None
for step in range(start_step+1,FINAL+1):
 cfg.mode='train';cfg.optim.fixed_gaussians=(RUN_NAME=='mcmc' or step>=15000);seed(step);model.update_learning_rate(step)
 if step%1000==0:model.oneupSHdegree()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ns=context(model,cameras[sequence[step-1]],step);exec(loss_code,ns);loss=ns['loss']+extra_loss(model);assert torch.isfinite(loss),(step,'loss');loss.backward()
 with torch.no_grad():
  if RUN_NAME!='mcmc':seed(step,True);exec(topology_code,ns)
  model.update_optimizer()
  if RUN_NAME=='bilateral':model.bilagrid_optimizer.step();model.bilagrid_optimizer.zero_grad(set_to_none=True)
  if RUN_NAME=='mcmc' and step<15000:
   seed(step,True);events=mcmc_step(model,step,caps)
   if events:event_stream.write(json.dumps(dict(step=step,models=events))+'\n');event_stream.flush()
 torch.cuda.synchronize();row=dict(step=step,image=sequence[step-1],ms=(time.perf_counter()-t)*1000,points=sum(counts(model).values()),loss=float(loss),peak_mb=torch.cuda.max_memory_allocated()/2**20,fixed=step>=15000,opacity_reset=RUN_NAME!='mcmc' and step<15000 and step%cfg.optim.opacity_reset_interval==0,mcmc_active=RUN_NAME=='mcmc' and step<15000)
 if step>=15000:
  if fixed_points is None:fixed_points=row['points']
  assert row['points']==fixed_points
 stream.write(json.dumps(row)+'\n');stream.flush();del ns,loss
 if step%100==0 or SMOKE:
  assert_finite(model);report_status(state='training',variant=RUN_NAME,step=step,points=row['points']);print('TRAIN',RUN_NAME,step,row,flush=True)
 if step in [1000,4000,10000,15000,20000,FINAL]:evaluate_full(model,step)
 if step%2500==0 or step==FINAL:save_checkpoint(model,step)
stream.close();event_stream.close();assert_finite(model)
rows=[json.loads(l) for l in curve_path.read_text().splitlines()];fixed=[r for r in rows if r['step']>=15000]
if not SMOKE:assert len(rows)==25000 and len(fixed)==10001 and len({r['points'] for r in fixed})==1 and not any(r['opacity_reset'] or r['mcmc_active'] for r in fixed)
atomic(OUT/'completion.json',dict(variant=RUN_NAME,step=FINAL,step_seconds=sum(r['ms'] for r in rows)/1000,loop_wall_seconds=time.perf_counter()-since,job_wall_seconds=time.time()-JOB_START,peak_mb=max(r['peak_mb'] for r in rows),fixed_verified=bool(fixed),resumed=start_step>0))
report_status(state='complete',variant=RUN_NAME,step=FINAL);print('COMPLETE',RUN_NAME,flush=True)
