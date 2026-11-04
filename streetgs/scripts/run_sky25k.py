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
MATRIX={'legacy':dict(backend='legacy',packed=False,radius_clip=0.),
        'gsplat':dict(backend='gsplat',packed=False,radius_clip=0.),
        'packed':dict(backend='gsplat',packed=True,radius_clip=0.),
        'radius1':dict(backend='gsplat',packed=False,radius_clip=1.)}
status=dict(state='loading',pilot=PILOT,final=FINAL,results={},updated=time.time())
def atomic(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def report_status(**kw):
    status.update(kw);status['updated']=time.time();atomic(OUT/'status.json',status)
def uncaught(kind,value,tb):
    report_status(state='failed',error=''.join(traceback.format_exception(kind,value,tb)))
    sys.__excepthook__(kind,value,tb)
sys.excepthook=uncaught
RESUME=int(os.environ.get('SKY_RESUME_STEP','0'))
assert BENCH or RESUME>0 or not any(OUT.glob('*/curve.jsonl')),'Existing training output: select a fresh STAGE67_OUT'
report_status()
cfg.mode='train';cfg.optim.fixed_gaussians=False;cfg.render.gsplat_densification=True
cfg.train.iterations=25000;cfg.optim.densify_until_iter=15000;cfg.optim.actor_reg_start_iter=25000
cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
assert not cfg.optim.get('static_prune_enabled',False),'Static-prune extension requires separate adaptation'
assert cfg.source_path == '/mnt/nuplan/l3data-reconstruction-bingxing/pose_ba_streetgs_runs/sample_00380_clip_M18-2_07_20251230153515_DF_f184_284_right/run_20261001/streetgauss/converted'
cfg.model.sky.type='cubemap'
cfg.model.sky.num_points=50000
cfg.model.sky.densify_until_iter=15000
cfg.model.sky.densification_interval=500
cfg.model.sky.max_points=100000
cfg.model.sky.coverage_weight=0.01
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
def counts(model):return dict({n:len(getattr(model,n).get_xyz) for n in model.model_name_id},sky=len(model.sky_cubemap.get_xyz))
def load(name,step=0):
    cfg.mode='train';setup_variant(name)
    cfg.model.sky.type='gaussian_sphere' if step else 'cubemap'
    model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup()
    path=BASE/'initial_state.pth' if step==0 else OUT/name/f'iteration_{step}.pth'
    state=torch.load(path);model.load_state_dict(state)
    if step==0:
        from lib.models.gaussian_model_sky import GaussianModelSky
        model.sky_cubemap=GaussianModelSky()
        model.sky_cubemap.initialize(model.background.get_xyz,ordered_cameras)
    cfg.model.sky.type='gaussian_sphere'
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
    return dict(cfg=cfg,torch=torch,l1_loss=l1_loss,ssim=ssim,psnr=psnr,gaussians=model,gaussians_renderer=renderer,
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
            if step==FINAL:
                reference_image=ROOT/'output/gsplat_stage67_fixed15k_20261006/workers/gsplat/gsplat/images_25000'/(c.image_name+'.png')
                refimg=torch.from_numpy(np.array(Image.open(reference_image).convert('RGB')).copy()).cuda().float().permute(2,0,1)/255.
                candidate8=(r['rgb'].clamp(0,1)*255+.5).to(torch.uint8).float()/255.
                sky=c.guidance['sky_mask'].to(device='cuda').bool()
                if sky.ndim==2:sky=sky[None]
                dil=torch.nn.functional.max_pool2d(sky.float()[None],7,1,3)[0]>.5
                ero=(1-torch.nn.functional.max_pool2d(1-sky.float()[None],7,1,3))[0]>.5
                regions={}
                for label,region in [('sky',ero),('non_sky',~dil),('boundary',dil & ~ero)]:
                    valid=region & ns['mask'].bool()
                    if valid.any():
                        regions[label]=dict(pixels=int(valid.sum()),cubemap_psnr=float(psnr(refimg,ns['gt_image'],valid)),sphere_psnr=float(psnr(candidate8,ns['gt_image'],valid)))
                rows[-1]['regions_8bit']=regions
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

# Run one candidate; compare to the completed, same-source gsplat cubemap reference.
name='gsplat';model=load(name,RESUME)
atomic(OUT/'sky_initialization.json',model.sky_cubemap.init_info)
directory=OUT/name;directory.mkdir(exist_ok=True)
(directory/'config.yaml').write_text(cfg.dump())
SMOKE=int(os.environ.get('SKY_SMOKE_STEPS','0'));target=SMOKE or FINAL
report_status(state='training',step=RESUME,target=target)
stream=(directory/'curve.jsonl').open('a' if RESUME else 'w')
freeze_count=None
for step in range(RESUME+1,target+1):
    cfg.mode='train';setup_variant(name);seed(step);cfg.optim.fixed_gaussians=step>=15000
    model.update_learning_rate(step)
    if step%1000==0:model.oneupSHdegree()
    t=time.perf_counter();ns=context(model,cameras[sequence[step-1]],step);exec(loss_code,ns)
    loss=ns['loss'];assert torch.isfinite(loss),'Nonfinite loss';loss.backward()
    if step==1 or step%100==0:
        for p in parameters(model):
            if p.grad is not None:assert torch.isfinite(p.grad).all(),'Nonfinite gradient'
    with torch.no_grad():
        seed(step,topology=True);exec(topology_code,ns);model.update_optimizer()
    sky_count=len(model.sky_cubemap.get_xyz)
    if step>=15000:
        if freeze_count is None:freeze_count=sky_count
        assert sky_count==freeze_count
    row=dict(step=step,image=sequence[step-1],loss=float(loss),sky_points=sky_count,scene_points=sum(len(getattr(model,n).get_xyz) for n in model.model_name_id),ms=(time.perf_counter()-t)*1000,sky_frozen=step>=15000)
    stream.write(json.dumps(row)+'\n');stream.flush();del ns,loss
    if step==1 or step%100==0:report_status(step=step,sky_points=sky_count);print('TRAIN',json.dumps(row),flush=True)
    if step in [1000,5000,10000,15000,20000,25000] or step==target:
        evaluate(name,model,step,save=step==target);checkpoint(name,model,step)
stream.close()
if SMOKE:
    # Verify checkpoint restores the independent sky component and its optimizer.
    before=model.sky_cubemap.get_xyz.detach().clone();del model;gc.collect();torch.cuda.empty_cache()
    model=load(name,target);assert torch.allclose(before,model.sky_cubemap.get_xyz)
    sky=model.sky_cubemap;sky.update_learning_rate(15000);n=len(sky.get_xyz);sky.topology_step();assert len(sky.get_xyz)==n
    report_status(state='smoke_complete',step=target,checkpoint_reload=True,freeze_guard=True)
else:
    reference=ROOT/'output/gsplat_stage67_fixed15k_20261006/workers/gsplat/gsplat/eval_25000.json'
    ref=json.loads(reference.read_text());ours=json.loads((directory/'eval_25000.json').read_text())
    atomic(OUT/'comparison.json',dict(reference=str(reference),sky_psnr=ours['psnr'],cubemap_psnr=ref['psnr'],delta_psnr=ours['psnr']-ref['psnr'],sky_ssim=ours['ssim'],cubemap_ssim=ref['ssim'],delta_ssim=ours['ssim']-ref['ssim'],image_count=ours['image_count'],scope='707 training views, production masks, no independent held-out generalization claim'))
    from build_sky25k_report import build
    build(ROOT,OUT)
    report_status(state='complete',step=target,comparison=str(OUT/'comparison.json'))
