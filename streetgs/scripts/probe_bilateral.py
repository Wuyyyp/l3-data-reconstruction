import os,sys,ast,json,time
from pathlib import Path
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
from PIL import Image,ImageDraw,ImageFont
# Reuse exact image loading, masks and affine fitting from the prior probe.
code=(ROOT/'scripts/probe_color.py').read_text();exec(compile(code[:code.index('rows=[];models={}')],'<color_io>','exec'))
out=base/'bilateral_mcmc_probe';out.mkdir(exist_ok=True)
source=Path('/mnt/zhoukaixuan/code/gsplat-1.5.3/examples/lib_bilagrid.py').read_text();tree=ast.parse(source)
keep={'color_affine_transform','_num_tensor_elems','total_variation_loss','slice','BilateralGrid'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in keep];assert len(nodes)==5
exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_gsplat153_bilagrid','exec'))
from lib.utils.loss_utils import ssim
def xy_for(h,w):
 yy,xx=np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing='ij');return np.stack([xx,yy],-1).astype('float32')
def train_grid(samples,frames,tv,steps):
 torch.manual_seed(7401);grid=BilateralGrid(1,8,8,8).cuda();identity=grid.grids.detach().clone();opt=torch.optim.Adam(grid.parameters(),lr=.01)
 x=torch.from_numpy(np.concatenate([samples[f][0] for f in frames])).cuda();y=torch.from_numpy(np.concatenate([samples[f][1] for f in frames])).cuda();xy=torch.from_numpy(np.concatenate([samples[f][2] for f in frames])).cuda();ids=torch.zeros(1,1,dtype=torch.long,device='cuda')
 for step in range(steps):
  ii=torch.randint(len(x),(8192,),device='cuda');pred=slice(grid,xy[ii],x[ii],ids)['rgb'];loss=(pred-y[ii]).square().mean()+tv*grid.tv_loss()+.001*(grid.grids-identity).square().mean();assert torch.isfinite(loss);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
 return grid
@torch.no_grad()
def transform(grid,p):
 h,w=p.shape[:2];x=torch.from_numpy(p.astype('float32').reshape(-1,3)).cuda();xy=torch.from_numpy(xy_for(h,w).reshape(-1,2)).cuda();ids=torch.zeros(1,1,dtype=torch.long,device='cuda');res=[]
 for start in range(0,len(x),131072):res.append(slice(grid,xy[start:start+131072],x[start:start+131072],ids)['rgb'].clamp(0,1).cpu())
 return torch.cat(res).numpy().reshape(h,w,3)
@torch.no_grad()
def ssim_score(p,g,v):
 a=torch.from_numpy(p.astype('float32')).permute(2,0,1).cuda();b=torch.from_numpy(g.astype('float32')).permute(2,0,1).cuda();m=torch.from_numpy(v[None]).cuda();return float(ssim(a,b,mask=m))
rows=[];chosen={};started=time.time();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for cam in [0,1,2,3,4,9,10]:
 samples={}
 for frame in [0,10,20,30,40]:
  name,p,g,v=load(frame,cam);sel=v[::10,::10];samples[frame]=(p[::10,::10][sel].astype('float32'),g[::10,::10][sel].astype('float32'),xy_for(*v.shape)[::10,::10][sel])
 candidates=[]
 for tv in [1.,10.]:
  grid=train_grid(samples,[0,10,20,30],tv,300);name,p,g,v=load(40,cam);cc=transform(grid,p);candidates.append((score(cc,g,v),tv));del grid
 tv=max(candidates)[1];chosen[cam]=dict(tv=tv,validation=candidates);grid=train_grid(samples,[0,10,20,30,40],tv,600);torch.save(grid.state_dict(),out/('bilateral_camera_%02d.pth'%cam))
 x=np.concatenate([samples[f][0] for f in samples]);y=np.concatenate([samples[f][1] for f in samples]);affine=fit(x,y,True)
 for frame in [50,60,70,80,90,100]:
  name,p,g,v=load(frame,cam);t=time.perf_counter();cc=transform(grid,p);elapsed=time.perf_counter()-t;row=dict(image=name,camera=cam,baseline=score(p,g,v),affine=score(np.clip(p@affine[:3]+affine[3],0,1),g,v),bilateral=score(cc,g,v),baseline_ssim=ssim_score(p,g,v),bilateral_ssim=ssim_score(cc,g,v),transform_seconds=elapsed);rows.append(row)
  if frame==70:
   h,w=v.shape;hh=round(h*640/w);im=Image.new('RGB',(1920,hh+60),'white');d=ImageDraw.Draw(im)
   for j,(title,a) in enumerate([('GT '+name,g),('Original %.3f'%row['baseline'],p),('Bilateral %.3f'%row['bilateral'],cc)]):d.text((j*640+8,15),title,font=font,fill='black');im.paste(Image.fromarray((a*255+.5).astype('uint8')).resize((640,hh)),(j*640,60))
   im.save(out/('bilateral_'+name+'.jpg'),quality=94)
 del grid;torch.cuda.empty_cache();print('CAMERA',cam,'selected TV',tv,'mean gain',np.mean([r['bilateral']-r['baseline'] for r in rows if r['camera']==cam]),flush=True)
result=dict(rows=rows,chosen=chosen,seconds=time.time()-started,mean_gain=float(np.mean([r['bilateral']-r['baseline'] for r in rows])),mean_ssim_gain=float(np.mean([r['bilateral_ssim']-r['baseline_ssim'] for r in rows])),positive=sum(r['bilateral']>r['baseline'] for r in rows),protocol='Official gsplat1.5.3 BilateralGrid core AST-extracted without tensorly CP4D dependency. Frozen gsplat25k PNGs, one camera-shared 8x8x8 grid, 3x4 affine, identity init, MSE + TV + 0.001 identity. Choose TV 1 or10 by frame40 (train0..30 every10), refit0..40 for600 steps, test50..100 every10. Same production masks. Held out from color fit only, not reconstruction. Does not test per-image jointly-trained appearance disentanglement.')
(out/'bilateral.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result['mean_gain'],result['mean_ssim_gain'],flush=True)
