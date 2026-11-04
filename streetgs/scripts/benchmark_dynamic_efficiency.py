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
cfg.mode='evaluate';cfg.data_device='cpu';cfg.to_cuda=False;cfg.render.backend='gsplat';cfg.render.gsplat_packed=True;cfg.render.gsplat_densification=False
out=Path(os.environ['EFF_OUT']);out.mkdir(parents=True,exist_ok=True)
cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
dataset=Dataset();model=StreetGaussianModel(dataset.scene_info.metadata)
checkpoint=Path(os.environ['EFF_CHECKPOINT'])
state=torch.load(checkpoint);missing=set(model.model_name_id)-set(state);assert not missing, str(missing);model.load_state_dict(state);del state
renderer=StreetGaussianRenderer();cams=sorted(dataset.train_cameras[1],key=lambda c:c.image_name)
selected=[c for c in cams if int(c.image_name.split('_')[0]) in [0,40,99]]
def flags(pose=False,transform=False,fourier=False):
 cfg.render.fast_actor_pose=pose;cfg.render.fast_actor_transform=transform;cfg.render.cache_fourier=fourier
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
phase=os.environ.get('EFF_PHASE','compare');results={};metrics=[];counts=[]
variants={'original':(False,False,False)} if phase=='baseline' else {'original':(False,False,False),'pose':(True,False,False),'transform':(True,True,False),'optimized':(True,True,True)}
with torch.no_grad():
 for c in selected:
  model.set_visibility(list(model.model_name_id));model.parse_camera(c)
  counts.append(dict(image=c.image_name,actors=len(model.graph_obj_list),actor_points=sum(len(getattr(model,n)._xyz) for n in model.graph_obj_list),width=c.image_width,height=c.image_height))
 for name,f in variants.items():
  flags(*f)
  for c in selected[:3]:call(c,'full')
 rows={n:[] for n in variants}
 for round in range(int(os.environ.get('EFF_ROUNDS','5'))):
  order=list(variants);order=order[round%len(order):]+order[:round%len(order)]
  for c in selected:
   for name in order:
    flags(*variants[name]);r,t=timed(c,'full');t.update(image=c.image_name,round=round);rows[name].append(t)
    if round==0 and name=='original':torch.save({k:r[k].cpu() for k in ['rgb','acc','depth']},out/(c.image_name+'_reference.pt'))
    if round==0 and name=='optimized':
     ref=torch.load(out/(c.image_name+'_reference.pt'));m=dict(image=c.image_name)
     for k in ref:
      diff=r[k]-ref[k].cuda();m[k+'_max']=float(diff.abs().max());m[k+'_mae']=float(diff.abs().mean())
      if k=='rgb':m['backend_psnr']=-10*math.log10(max(float(diff.square().mean()),1e-30));m['backend_ssim']=float(ssim(r[k],ref[k].cuda()))
     metrics.append(m)
    del r
 for name in variants:results[name]=summary(rows[name])
 other={}
 for setting in ['original','optimized']:
  if setting not in variants:continue
  flags(*variants[setting]);other[setting]={}
  for mode in ['background','layers']:
   rr=[]
   for c in selected[:3]:call(c,mode)
   for c in selected:
    for j in range(2):r,t=timed(c,mode);rr.append(dict(image=c.image_name,**t));del r
   other[setting][mode]=dict(summary=summary(rr),rows=rr)
 # Separate instrumented pass; synchronization overhead excluded from headline timings.
 import gsplat
 breakdown={}
 original_raster=gsplat.rasterization
 original_parse=model.parse_camera
 for setting in ['original','optimized']:
  flags(*variants[setting]);parts={'parse_camera':[],'rasterization':[],'get_xyz':[],'get_features':[]}
  def instrument(fn,key):
   def wrapped(*a,**kw):
    sync();t=time.perf_counter();value=fn(*a,**kw);sync();parts[key].append((time.perf_counter()-t)*1000);return value
   return wrapped
  gsplat.rasterization=instrument(original_raster,'rasterization')
  model.parse_camera=instrument(original_parse,'parse_camera')
  saved={}
  for prop in ['get_xyz','get_features']:
   saved[prop]=getattr(type(model),prop);setattr(type(model),prop,property(instrument(saved[prop].fget,prop)))
  try:
   for c in selected:call(c,'full')
  finally:
   gsplat.rasterization=original_raster;model.parse_camera=original_parse
   for prop,v in saved.items():setattr(type(model),prop,v)
  breakdown[setting]={k:dict(mean_ms=float(np.mean(v)),samples=len(v)) for k,v in parts.items()}
 from PIL import Image
 for c in selected:
  flags();a=call(c,'full')['rgb'];flags(True,True,True);b=call(c,'full')['rgb']
  gt=c.original_image[:3].to(a.device)
  if int(c.image_name.split('_')[0])==40:
   panel=torch.cat([gt,a,b,(a-b).abs()*100],dim=2).clamp(0,1).permute(1,2,0).cpu().numpy()
   Image.fromarray((panel*255).astype(np.uint8)).save(out/(c.image_name+'_gt_original_optimized_diff100.jpg'))
 data=dict(breakdown=breakdown,phase=phase,summary=results,rows=rows,other=other,metrics=metrics,views=counts,checkpoint=str(checkpoint))
 (out/(phase+'.json')).write_text(json.dumps(data,indent=2));print(json.dumps(results),flush=True)
assert metrics and all(m['rgb_max'] < 1e-4 and m['acc_max'] < 1e-4 for m in metrics), 'Image parity failed; inspect saved compare.json'
print('COMPLETE',phase,flush=True)
