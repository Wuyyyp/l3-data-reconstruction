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
def flags(pose=False,transform=False,fourier=False,batch_pose=False,batch_xyz=False,ego=False,world=False):
 cfg.render.fast_actor_pose=pose;cfg.render.fast_actor_transform=transform;cfg.render.cache_fourier=fourier
 cfg.render.batch_actor_pose=batch_pose;cfg.render.batch_actor_xyz=batch_xyz;cfg.render.reuse_ego_rotation=ego;cfg.render.batch_actor_world=world
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

from lib.utils.loss_utils import psnr
from PIL import Image
# All quality evaluations use the same validated scalar-pose renderer.
flags(True,False,True,False,False,True,False)
metrics=[]
with torch.no_grad():
 for c in cams:
  gt=c.original_image.to('cuda');mask=c.guidance.get('mask',torch.ones_like(gt[0:1]).bool()).to('cuda').clone().bool()
  camid=int(c.image_name.split('_')[-1]);h,w=gt.shape[-2:]
  if camid==3:mask[:,:int(h*250/1280),:int(w*550/1920)]=False;mask[:,:,:int(w*80/1920)]=False
  if camid==4:mask[:,:int(h*250/1280),int(w*(1920-550)/1920):]=False;mask[:,:,int(w*(1920-80)/1920):]=False
  if camid==10:mask[:,int(h*1500/2160):,:]=False
  result=call(c,'full');rgb=result['rgb'].clamp(0,1)
  row=dict(image=c.image_name,cam=camid,psnr=float(psnr(rgb,gt,mask).mean()),ssim=float(ssim(rgb,gt,mask=mask)))
  assert all(math.isfinite(row[k]) for k in ['psnr','ssim']),row
  metrics.append(row)
  if c.image_name.startswith('000040_'):
   panel=torch.cat([gt,rgb],dim=2).clamp(0,1);Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(out/(c.image_name+'.jpg'))
  if len(metrics)%100==0:print('evaluated',len(metrics),flush=True)
summary=dict(count=len(metrics),psnr=float(np.mean([v['psnr'] for v in metrics])),ssim=float(np.mean([v['ssim'] for v in metrics])),per_camera={str(i):dict(psnr=float(np.mean([v['psnr'] for v in metrics if v['cam']==i])),ssim=float(np.mean([v['ssim'] for v in metrics if v['cam']==i]))) for i in sorted(set(v['cam'] for v in metrics))},images=metrics)
(out/'quality.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='images'}))
