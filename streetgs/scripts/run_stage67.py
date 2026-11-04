"""Staged, resumable StreetGS/gsplat training matrix. No Git writes.

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

OUT=Path(os.environ.get('STAGE67_OUT',str(ROOT/'output/gsplat_stage67')));OUT.mkdir(exist_ok=True)
BASE=ROOT/'output/gsplat_stage123/initialization_baseline'
PILOT=int(os.environ.get('STAGE67_PILOT','4000'));FINAL=int(os.environ.get('STAGE67_FINAL','25000'))
SMOKE=os.environ.get('STAGE67_SMOKE','0')=='1'
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
assert not any(OUT.glob('*/curve.jsonl')),'Existing training output: select a fresh STAGE67_OUT'
report_status()
cfg.mode='train';cfg.optim.fixed_gaussians=False;cfg.render.gsplat_densification=True
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
    learning_rate_horizon=cfg.train.iterations,eval_frames=[0,40,99],gpu=os.environ['CUDA_VISIBLE_DEVICES']))

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
    for frame in [0,40,99]:
        for cam in [0,1,2,3,4,9,10]:
            c=cameras[f'{frame:06d}_{cam:02d}'];ns=context(model,c,step);exec(mask_code,ns)
            r=renderer.render(c,model)
            rows.append(dict(image=c.image_name,psnr=float(psnr(r['rgb'],ns['gt_image'],ns['mask'])),ssim=float(ssim(r['rgb'],ns['gt_image'],mask=ns['mask']))))
            if save:
                folder=directory/f'images_{step}';folder.mkdir(exist_ok=True);image(r['rgb'],folder/(c.image_name+'.png'))
    result=dict(step=step,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),
                points=counts(model),rows=rows)
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

def train_group(names,start,stop):
    report_status(state='training',variants=names,step=start,target=stop)
    models={name:load(name,start) for name in names}
    if start==0:
        for name,model in models.items():evaluate(name,model,0)
    streams={}
    for name in names:
        setup_variant(name)
        d=OUT/name;d.mkdir(exist_ok=True);streams[name]=(d/'curve.jsonl').open('a');(d/'config.yaml').write_text(cfg.dump())
    baseline_pose={n:models[n].actor_pose.opt_trans.detach().clone() for n in names}
    since=time.time()
    try:
        for step in range(start+1,stop+1):
            order=names[step%len(names):]+names[:step%len(names)] # rotate first-run/cache advantage
            for name in order:
                model=models[name];setup_variant(name);cfg.mode='train';seed(step)
                model.update_learning_rate(step)
                if step%1000==0:model.oneupSHdegree()
                torch.cuda.synchronize();mem0=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                ns=context(model,cameras[sequence[step-1]],step);exec(loss_code,ns)
                loss=ns['loss'];assert torch.isfinite(loss),(name,step,'nonfinite loss')
                loss.backward();torch.cuda.synchronize();forward_backward_ms=(time.perf_counter()-t)*1000
                if step%100==0 or step==1:
                    for p in parameters(model):
                        if p.grad is not None:assert torch.isfinite(p.grad).all(),(name,step,'nonfinite gradient')
                point_before=sum(counts(model).values());topology_start=time.perf_counter()
                with torch.no_grad():
                    seed(step,topology=True);exec(topology_code,ns)
                    model.update_optimizer()
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-t)*1000
                row=dict(step=step,image=sequence[step-1],loss=float(loss),
                    forward_backward_ms=forward_backward_ms,total_ms=elapsed,
                    topology_optimizer_ms=(time.perf_counter()-topology_start)*1000,
                    points_before=point_before,points=sum(counts(model).values()),
                    peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    peak_increment_mb=(torch.cuda.max_memory_allocated()-mem0)/2**20,
                    position_lr=model.background.optimizer.param_groups[0]['lr'],
                    track_lr=model.actor_pose.optimizer.param_groups[0]['lr'])
                row['opacity_reset_scheduled']=bool(step<cfg.optim.densify_until_iter and step%cfg.optim.opacity_reset_interval==0)
                if step%100==0:
                    row['track_translation_change']=float((model.actor_pose.opt_trans-baseline_pose[name]).abs().max())
                streams[name].write(json.dumps(row)+'\n');streams[name].flush();del ns,loss
            if step%100==0:
                report_status(step=step,elapsed_group_seconds=time.time()-since)
                print('TRAIN',names,step,'elapsed',round(time.time()-since,1),flush=True)
            if step in sorted(set([600,1000,3000,3100,PILOT,FINAL])) or step==stop:
                for name,model in models.items():
                    evaluate(name,model,step,save=step in [PILOT,FINAL])
                    if step in [PILOT,FINAL] or step==stop:checkpoint(name,model,step)
            elif step%5000==0:
                for name,model in models.items():checkpoint(name,model,step)
        for model in models.values():
            for p in parameters(model):assert torch.isfinite(p).all()
    finally:
        for s in streams.values():s.close()
    del model,models,baseline_pose;gc.collect();torch.cuda.empty_cache()

def gate(candidate,reference,step,performance=False):
    a=json.loads((OUT/reference/f'eval_{step}.json').read_text());b=json.loads((OUT/candidate/f'eval_{step}.json').read_text())
    policy=json.loads((OUT/'acceptance.json').read_text())
    pa=a['psnr'];pb=b['psnr'];sa=a['ssim'];sb=b['ssim']
    quality=(pb-pa>=-policy['max_psnr_drop'] and sb-sa>=-policy['max_ssim_drop'])
    def timing(name):
        rows=[json.loads(s) for s in (OUT/name/'curve.jsonl').read_text().splitlines()]
        rows=[r for r in rows if 100<r['step']<=step and r['step']%100!=0]
        return dict(ms=float(np.median([r['total_ms'] for r in rows])),mb=float(np.median([r['peak_increment_mb'] for r in rows])))
    ta=timing(reference);tb=timing(candidate)
    benchmark_path=OUT/f'benchmark_{step}.json'
    controlled=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    benefit=(tb['ms']<ta['ms']*.99 or tb['mb']<ta['mb']*.95)
    if performance:
        assert controlled is not None,'Controlled common-checkpoint benchmark is required'
        benefit=controlled['comparisons'][candidate]['benefit']
    result=dict(candidate=candidate,reference=reference,step=step,delta_psnr=pb-pa,delta_ssim=sb-sa,
        quality_pass=quality,performance_benefit=benefit,reference_performance=ta,candidate_performance=tb,
        controlled_benchmark=controlled['comparisons'].get(candidate) if controlled else None,
        passed=quality and (benefit or not performance))
    atomic(OUT/f'gate_{candidate}_{step}.json',result);status['results'][candidate]=result;report_status()
    print('GATE',json.dumps(result),flush=True);return result['passed']

def benchmark(step):
    """Same model and camera; rotated order, no parameter updates. Isolate backend cost."""
    report_status(state='benchmark',step=step)
    model=load('gsplat',step);rows=[];names=['gsplat','packed','radius1']
    for i in range(39):
        c=cameras[sequence[i]]
        for name in names[i%3:]+names[:i%3]:
            setup_variant(name);cfg.mode='train';seed(50000+i)
            torch.cuda.synchronize();mem=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            ns=context(model,c,min(step,24999));exec(loss_code,ns);ns['loss'].backward();torch.cuda.synchronize()
            ms=(time.perf_counter()-t)*1000;extra=(torch.cuda.max_memory_allocated()-mem)/2**20
            if i>=3:rows.append(dict(round=i,variant=name,image=c.image_name,ms=ms,mb=extra))
            for obj_name in list(model.model_name_id)+['actor_pose','sky_cubemap','color_correction','pose_correction']:
                obj=getattr(model,obj_name,None)
                if obj is not None and hasattr(obj,'optimizer'):obj.optimizer.zero_grad(set_to_none=True)
            del ns
    comparisons={};rng=np.random.RandomState(0)
    base={r['round']:r for r in rows if r['variant']=='gsplat'}
    for name in ['packed','radius1']:
        cand={r['round']:r for r in rows if r['variant']==name};ids=sorted(base)
        ratios=np.array([cand[i]['ms']/base[i]['ms'] for i in ids])
        upper=float(np.percentile([np.median(ratios[rng.randint(0,len(ids),len(ids))]) for _ in range(1000)],97.5))
        memory=float(np.median([cand[i]['mb']/base[i]['mb'] for i in ids]))
        comparisons[name]=dict(time_ratio=float(np.median(ratios)),time_ratio_upper_95=upper,memory_ratio=memory,
            benefit=bool((np.median(ratios)<.99 and upper<1.) or memory<.95))
    atomic(OUT/f'benchmark_{step}.json',dict(rows=rows,comparisons=comparisons,note='Common gsplat checkpoint; forward/loss/backward only; rotated order; no optimizer update'))
    del model;gc.collect();torch.cuda.empty_cache()

try:
    if SMOKE:
        train_group(['legacy','gsplat'],0,PILOT);report_status(state='smoke_complete')
    else:
        train_group(['legacy','gsplat'],0,PILOT)
        if not gate('gsplat','legacy',PILOT):
            report_status(state='needs_review',reason='Stage 6 quality gate failed; no optimizations or long runs started')
        else:
            benchmark(PILOT)
            promoted=[]
            for candidate in ['packed','radius1']:
                # Common-checkpoint rotated-order benchmark controls performance bias.
                train_group([candidate],0,PILOT)
                if gate(candidate,'gsplat',PILOT,performance=True):promoted.append(candidate)
            report_status(state='pilot_complete',promoted=promoted)
            # Long runs continue exact pilot checkpoints with optimizer state.
            train_group(['legacy','gsplat'],PILOT,FINAL)
            if gate('gsplat','legacy',FINAL):
                benchmark(FINAL)
                for candidate in promoted:
                    train_group([candidate],PILOT,FINAL);gate(candidate,'gsplat',FINAL,performance=True)
                report_status(state='complete',promoted=promoted)
            else:
                report_status(state='needs_review',reason='25k baseline quality gate failed; candidate continuation stopped')
except Exception:
    report_status(state='failed',error=traceback.format_exc());raise
