"""Paired 1-update diagnostics and 500-update fixed-point training regression.

Uses the actual mask/render/loss AST block from train.py, not a reimplementation.
Preserves the 25k-step LR/loss schedule while executing exactly 500 updates.
"""
import os,sys,ast,time,json,random,importlib,hashlib,gc
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

out=ROOT/'output/gsplat_train_check';out.mkdir(exist_ok=True)
base=ROOT/'output/gsplat_stage123/initialization_baseline'
phase=os.environ.get('TRAIN_CHECK_PHASE','one')
assert phase in ['one','500','all']
cfg.mode='train';cfg.optim.fixed_gaussians=True;cfg.render.backend='legacy'
cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
cfg.model_path=str(out/'dataset');Path(cfg.model_path).mkdir(exist_ok=True)
p=Path(cfg.model_path)/'input_ply'
if not p.exists():p.symlink_to(base/'input_ply',target_is_directory=True)
safe_state(False)
dataset=Dataset();cameras={c.image_name:c for c in dataset.train_cameras[1]}
sequence=json.loads((base/'camera_sequence_1000.json').read_text())
assert len(cameras)==707 and all(n in cameras for n in sequence)
renderer=StreetGaussianRenderer()

# Extract unchanged production mask, supervision, render and loss operations.
source=(ROOT/'train.py').read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training')
loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gt_image' for t in n.targets))
end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward')
loss_code=compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'train.py:mask_render_loss','exec')
mask_end=next(i for i,n in enumerate(loop.body[start:end]) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='render_pkg' for t in n.targets))
mask_code=compile(ast.Module(body=loop.body[start:start+mask_end],type_ignores=[]),'train.py:mask','exec')
(out/'loss_block.py').write_text('\n'.join(source.splitlines()[loop.body[start].lineno-1:loop.body[end-1].end_lineno])+'\n')
baseline_rng=torch.load(base/'rng_state.pth',map_location='cpu')

def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def new_model():
    model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup()
    model.load_state_dict(torch.load(base/'initial_state.pth'))
    model.update_learning_rate(1)
    for name in model.model_name_id:
        for group in getattr(model,name).optimizer.param_groups:
            if group['name']=='xyz':assert group['lr']>0,('zero position LR after restore',name)
    return model
def named_params(model):
    d={}
    for name in list(model.model_name_id)+['actor_pose','sky_cubemap','color_correction','pose_correction']:
        obj=getattr(model,name,None)
        if obj is None or not hasattr(obj,'optimizer'):continue
        for i,g in enumerate(obj.optimizer.param_groups):
            for j,p in enumerate(g['params']):d[name+'/'+g.get('name','group'+str(i))+'/'+str(j)]=p
    return d
def count(model):return {n:len(getattr(model,n).get_xyz) for n in model.model_name_id}
def context(model,c,iteration):
    return dict(torch=torch,l1_loss=l1_loss,ssim=ssim,psnr=psnr,gaussians=model,gaussians_renderer=renderer,
        training_args=cfg.train,optim_args=cfg.optim,data_args=cfg.data,viewpoint_cam=c,iteration=iteration,
        sky_mask=None,lidar_depth=None,obj_bound=None)
def forward(model,backend,c,iteration):
    cfg.render.backend=backend;model.update_learning_rate(iteration)
    ns=context(model,c,iteration);exec(loss_code,ns)
    assert torch.isfinite(ns['loss']), (backend,iteration,'loss nonfinite')
    return ns
def tensor_compare(a,b):
    if a is None or b is None:return dict(legacy_none=a is None,gsplat_none=b is None)
    assert a.shape==b.shape
    af=a.detach().reshape(-1);bf=b.detach().reshape(-1)
    aa=float(torch.sum(af.double().square()));bb=float(torch.sum(bf.double().square()))
    dd=float(torch.sum((af-bf).double().square()));dot=float(torch.sum(af.double()*bf.double()))
    finite=bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    return dict(elements=af.numel(),finite=finite,legacy_norm=aa**.5,gsplat_norm=bb**.5,
        relative_l2=(dd/aa)**.5 if aa>1e-30 else None,
        cosine=dot/(aa*bb)**.5 if aa*bb>1e-60 else None,
        max_abs_diff=float((af-bf).abs().max()) if af.numel() else 0.,
        legacy_nonzero=int(torch.count_nonzero(af)),gsplat_nonzero=int(torch.count_nonzero(bf)))
def finite_grads(model):
    for n,p in named_params(model).items():
        if p.grad is not None:assert torch.isfinite(p.grad).all(),n
def save_image(t,path):Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)).save(path)

models={b:new_model() for b in ['legacy','gsplat']}
initial_count=count(models['legacy']);assert initial_count==count(models['gsplat'])
params={b:named_params(m) for b,m in models.items()}
assert params['legacy'].keys()==params['gsplat'].keys()
assert all(torch.equal(p,params['gsplat'][n]) for n,p in params['legacy'].items())
restore_rng(baseline_rng)
manifest=dict(phase=phase,initialization_sha256=hashlib.sha256((base/'initial_state.pth').read_bytes()).hexdigest(),
    train_source_sha256=hashlib.sha256(source.encode()).hexdigest(),camera_sequence=sequence[:500],
    fixed_gaussians=True,lr_schedule_horizon=cfg.train.iterations,initial_point_counts=initial_count,
    note='Production train.py loss block; same training masks, sky jitter RNG and schedule; 500 actual optimizer updates, no topology changes.')
(out/('manifest_'+phase+'.json')).write_text(json.dumps(manifest,indent=2));(out/'effective_config.yaml').write_text(cfg.dump())

if phase in ['one','all']:
    c=cameras[sequence[0]];states={};records={};initial={n:p.detach().clone() for n,p in params['legacy'].items()}
    for backend,model in models.items():
        restore_rng(baseline_rng);ns=forward(model,backend,c,1)
        ns['loss'].backward();finite_grads(model)
        records[backend]={k:float(v) for k,v in ns['scalar_dict'].items()}
        records[backend]['psnr']=float(psnr(ns['image'],ns['gt_image'],ns['mask']))
        records[backend]['background_position_lr']=model.background.optimizer.param_groups[0]['lr']
        records[backend]['track_position_lr']=model.actor_pose.optimizer.param_groups[0]['lr']
        states[backend]={k:ns['render_pkg'][k].detach() for k in ['rgb','acc','depth']}
        del ns
    grads={n:tensor_compare(p.grad,params['gsplat'][n].grad) for n,p in params['legacy'].items()}
    # Attribute position-gradient differences to camera covariance clipping.
    pa=params['legacy']['background/xyz/0'];pb=params['gsplat']['background/xyz/0']
    with torch.no_grad():
        view=c.world_view_transform.T
        xyz=pa@view[:3,:3].T+view[:3,3]
        ratios=xyz[:,:2]/xyz[:,2:].clamp_min(1e-10)
        limits=torch.tensor([1.3*np.tan(c.FoVx/2),1.3*np.tan(c.FoVy/2)],device='cuda')
        clipped=(ratios.abs()>limits).any(dim=1)
        energy=(pa.grad-pb.grad).square().sum(dim=1)
        active=(pa.grad.square().sum(dim=1)+pb.grad.square().sum(dim=1))>0
        top=torch.topk(energy,10).indices
        clip_diagnostic=dict(difference_energy_in_clipped_fraction=float(energy[clipped].sum()/energy.sum()),
            clipped_active_points=int((clipped&active).sum()),active_points=int(active.sum()),
            inside=tensor_compare(pa.grad[~clipped],pb.grad[~clipped]),
            clipped=tensor_compare(pa.grad[clipped],pb.grad[clipped]),
            top=[dict(index=int(i),camera_xyz=xyz[i].tolist(),clipped=bool(clipped[i]),
                      legacy_grad=pa.grad[i].tolist(),gsplat_grad=pb.grad[i].tolist()) for i in top])
        (out/'position_gradient_diagnostic.json').write_text(json.dumps(clip_diagnostic,indent=2))
    render={k:tensor_compare(states['legacy'][k],states['gsplat'][k]) for k in ['rgb','acc','depth']}
    for backend,model in models.items():
        cfg.render.backend=backend;model.update_optimizer()
    updates={n:tensor_compare(p.detach()-initial[n],params['gsplat'][n].detach()-initial[n]) for n,p in params['legacy'].items()}
    assert updates['background/xyz/0']['legacy_norm']>0 and updates['background/xyz/0']['gsplat_norm']>0
    result=dict(image=c.image_name,losses=records,gradients=grads,updates=updates,render=render,
                initial_parameters_bitwise_equal=True,point_counts_unchanged=all(count(m)==initial_count for m in models.values()))
    (out/'one_step.json').write_text(json.dumps(result,indent=2))
    for backend in models:save_image(states[backend]['rgb'],out/('one_step_'+backend+'.png'))
    print('ONE_STEP_COMPLETE',json.dumps(records),flush=True)
if phase in ['500','all']:
    if phase=='all':
        assert clip_diagnostic['difference_energy_in_clipped_fraction']>.99
        assert clip_diagnostic['inside']['relative_l2']<.01
        del models,params,initial,states,pa,pb
        gc.collect();torch.cuda.empty_cache()
        models={b:new_model() for b in ['legacy','gsplat']}
        params={b:named_params(m) for b,m in models.items()}
        assert all(torch.equal(p,params['gsplat'][n]) for n,p in params['legacy'].items())
        restore_rng(baseline_rng)
    assert (out/'one_step.json').exists(),'Run and review the one-update diagnostics first.'
    checkpoints=[0,100,250,500];eval_rows=[];train_rows=[]
    eval_names=[f'{frame:06d}_{cam:02d}' for frame in [0,40,99] for cam in [0,1,2,3,4,9,10]]
    def evaluate(step):
        state=rng_state();cfg.mode='evaluate'
        with torch.no_grad():
            for backend,model in models.items():
                cfg.render.backend=backend
                for name in eval_names:
                    c=cameras[name];ns=context(model,c,step);exec(mask_code,ns)
                    r=renderer.render(c,model)
                    row=dict(step=step,backend=backend,image=name,psnr=float(psnr(r['rgb'],ns['gt_image'],ns['mask'])),
                        ssim=float(ssim(r['rgb'],ns['gt_image'],mask=ns['mask'])))
                    eval_rows.append(row)
                    if step in [0,500]:
                        folder=out/'images'/str(step);folder.mkdir(parents=True,exist_ok=True)
                        save_image(r['rgb'],folder/(name+'_'+backend+'.png'))
                        if backend=='legacy':save_image(ns['gt_image'],folder/(name+'_gt.png'))
        cfg.mode='train';restore_rng(state)
        (out/'evaluation.json').write_text(json.dumps(eval_rows,indent=2))
        print('EVAL',step,{b:float(np.mean([r['psnr'] for r in eval_rows if r['step']==step and r['backend']==b])) for b in models},flush=True)
    evaluate(0);start_time=time.time()
    with (out/'train_curve.jsonl').open('w') as stream:
        for step,name in enumerate(sequence[:500],1):
            shared_rng=rng_state()
            for backend,model in models.items():
                restore_rng(shared_rng);torch.cuda.synchronize();t=time.perf_counter()
                ns=forward(model,backend,cameras[name],step);ns['loss'].backward()
                # Check finite losses every update; all gradients every 25 updates.
                if step==1 or step%25==0:finite_grads(model)
                row={k:float(v) for k,v in ns['scalar_dict'].items()}
                row.update(step=step,backend=backend,image=name,psnr=float(psnr(ns['image'],ns['gt_image'],ns['mask'])))
                row['background_position_lr']=model.background.optimizer.param_groups[0]['lr']
                with torch.no_grad():model.update_optimizer()
                torch.cuda.synchronize();row['elapsed_ms']=(time.perf_counter()-t)*1000
                stream.write(json.dumps(row)+'\n');stream.flush();train_rows.append(row)
                del ns
            if step%25==0:print('TRAIN',step,'elapsed_s',round(time.time()-start_time,1),{r['backend']:round(r['loss'],6) for r in train_rows[-2:]},flush=True)
            if step in checkpoints:evaluate(step)
    with torch.no_grad():
        for backend,model in models.items():
            assert count(model)==initial_count
            for n,p in named_params(model).items():assert torch.isfinite(p).all(),n
            cfg.model_path=str(out/backend);Path(cfg.model_path).mkdir(exist_ok=True)
            state=model.save_state_dict(is_final=False);state['iter']=500
            torch.save(state,Path(cfg.model_path)/'iteration_500.pth');del state
    (out/'complete.json').write_text(json.dumps(dict(actual_updates_per_backend=500,elapsed_seconds=time.time()-start_time,
        point_counts_unchanged=True,initial_point_counts=initial_count,finite_parameters=True,checkpoints=checkpoints),indent=2))
    print('TRAINING_COMPLETE',flush=True)
