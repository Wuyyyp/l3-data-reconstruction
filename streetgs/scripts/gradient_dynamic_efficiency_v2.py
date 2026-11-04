import sys,os,importlib,time,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
for n in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+n)
for n,m in list(sys.modules.items()):
 if n=='numpy.core' or n.startswith('numpy.core.'):sys.modules[n.replace('numpy.core','numpy._core',1)]=m
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.loss_utils import ssim
cfg.mode='evaluate';cfg.data_device='cpu';cfg.to_cuda=False;cfg.render.backend='gsplat';cfg.render.gsplat_packed=True;cfg.render.gsplat_densification=True
out=Path(os.environ['EFF_OUT']);out.mkdir(parents=True,exist_ok=True)
cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
dataset=Dataset();model=StreetGaussianModel(dataset.scene_info.metadata)
checkpoint=Path(os.environ['EFF_CHECKPOINT'])
state=torch.load(checkpoint);missing=set(model.model_name_id)-set(state);assert not missing, str(missing);model.load_state_dict(state);del state
renderer=StreetGaussianRenderer();cams=sorted(dataset.train_cameras[1],key=lambda c:c.image_name)
selected=[c for c in cams if int(c.image_name.split('_')[0]) in [0,40,99]]
def flags(pose=False,transform=False,fourier=False,batch_pose=False,batch_xyz=False,ego=False,world=False,skip_flip=False):
 cfg.render.fast_actor_pose=pose;cfg.render.fast_actor_transform=transform;cfg.render.cache_fourier=fourier
 cfg.render.batch_actor_pose=batch_pose;cfg.render.batch_actor_xyz=batch_xyz;cfg.render.reuse_ego_rotation=ego;cfg.render.batch_actor_world=world;cfg.render.skip_empty_actor_flip=skip_flip
def sync():torch.cuda.synchronize()
def call(c,mode):
 if mode=='background':return renderer.render(c,model,exclude_list=model.obj_list)
 if mode=='layers':return renderer.render_all(c,model)
 return renderer.render(c,model)
def timed(c,mode):
 sync();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();t=time.perf_counter();r=call(c,mode);sync()
 return r,dict(ms=(time.perf_counter()-t)*1000,peak_mb=torch.cuda.max_memory_allocated()/2**20,transient_mb=(torch.cuda.max_memory_allocated()-before)/2**20)
def summary(rows):
 a=np.array([r['ms'] for r in rows]);return dict(mean_ms=float(a.mean()),p50_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),fps=1000/float(a.mean()),peak_mb=max(r['peak_mb'] for r in rows))

cfg.mode='train'
model.training_setup();model.active_sh_degree=model.max_sh_degree
model.flip_prob=0.0
selected=[c for c in cams if c.image_name.startswith('000040_')]
records=[]
for c in selected:
 reference=None;reference_grad=None
 for name,f in [('original',(False,False,False)),('previous',(True,True,True)),('batch',(True,False,True,True,False,True,True)),('world_xyz',(True,False,True,True,True,True,True)),('batch_flip',(True,False,True,True,False,True,True,True)),('batch_no_world',(True,False,True,True,False,True,False)),('xyz_no_world',(True,False,True,True,True,True,False)),('flip_no_world',(True,False,True,True,False,True,False,True))]:
  flags(*f);model.zero_grad(set_to_none=True);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
  result=call(c,'full');gt=c.original_image.to('cuda')
  from lib.utils.loss_utils import l1_loss
  loss=.8*l1_loss(result['rgb'],gt)+.2*(1-ssim(result['rgb'],gt))+.001*result['acc'].mean()+.00001*result['depth'].mean()
  loss.backward();sync()
  if name=='original':
   reference={k:result[k].detach().cpu() for k in ['rgb','acc','depth']}
   reference_grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
  else:
   row=dict(image=c.image_name,variant=name,loss=float(loss),max_gradient_error=0.,failed=[],parameters=0)
   for k,ref in reference.items():row[k+'_max']=float((result[k].detach().cpu()-ref).abs().max())
   for n,p in model.named_parameters():
    if n not in reference_grad:
     if p.grad is not None:row['failed'].append(n+':unexpected gradient')
     continue
    if p.grad is None:row['failed'].append(n+':missing gradient');continue
    a=reference_grad[n];b=p.grad.detach().cpu();row['parameters']+=b.numel()
    if b.numel()==0:continue
    row['max_gradient_error']=max(row['max_gradient_error'],float((a-b).abs().max()))
    if not torch.allclose(a,b,atol=1e-5,rtol=1e-3):row['failed'].append(n)
   records.append(row);print(row,flush=True)
  del result,loss
  model.zero_grad(set_to_none=True)
 (out/'gradient.json').write_text(json.dumps(records,indent=2))
gate={n:all(not x['failed'] and x['rgb_max']<1e-4 and x['acc_max']<1e-4 for x in records if x['variant']==n) for n in set(x['variant'] for x in records)}
(out/'gate.json').write_text(json.dumps(gate,indent=2))
root=ROOT/'runs/dynamic_efficiency_20261014'
render=json.loads((root/'comparison_final/compare.json').read_text())
parity=json.loads((root/'comparison_final/parity_gate.json').read_text())
mapping={'batch_no_world':('batch_fourier',False,False),'xyz_no_world':('batch_xyz',False,True),'batch':('optimized',True,False),'world_xyz':('world_xyz',True,True)}
eligible=[n for n,(bn,w,xyz) in mapping.items() if gate[n] and parity[bn]]
assert eligible,'No batch candidate passes unchanged numerical tolerances'
best=min(eligible,key=lambda n:render['summary'][mapping[n][0]]['mean_ms'])
best_time=render['summary'][mapping[best][0]]['mean_ms']
near=[n for n in eligible if render['summary'][mapping[n][0]]['mean_ms']<=best_time*1.02]
best=min(near,key=lambda n:(mapping[n][2],mapping[n][1]))
bn,world,xyz=mapping[best]
# Empty-flip algebra is exact; require its real-gradient test too.
assert gate['batch_flip' if world else 'flip_no_world'],'Empty flip gradient test failed'
import yaml
for n in ['batch','batch_flip']:
 p=root/('short_'+n+'.yaml');c=yaml.safe_load(p.read_text());c['render'].update(batch_actor_world=world,batch_actor_xyz=xyz,skip_empty_actor_flip=n=='batch_flip');p.write_text(yaml.safe_dump(c))
(root/'selected_batch.json').write_text(json.dumps(dict(name=best,benchmark_variant=bn,world=world,xyz=xyz,gradient_gate=gate,render_gate=parity),indent=2))
print('SELECTED',best)

print('GRADIENT_PASS',len(records))
