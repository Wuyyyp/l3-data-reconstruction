from runtime_setup import ROOT,CONFIG,VARIANT,render
import os,sys,json,csv
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
from PIL import Image,ImageDraw
from train_aligned import load_cameras,apply_grid
from scene.gaussian_model import GaussianModel
from lib_bilagrid import BilateralGrid
from utils.static_losses import masked_ssim

@torch.no_grad()
def main():
 torch.set_num_threads(4)
 ck=torch.load(sys.argv[1],map_location='cpu');out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
 cams=load_cameras(int(os.environ.get('CAMERA_LIMIT','0')));assert ck['camera_names']==[c.image_name for c in cams]
 model=GaussianModel(1,'ours_adam');m=ck['model'];model.active_sh_degree=m[0];model.max_sh_degree=m[1]
 for key,v in zip(['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'],m[2:8]):setattr(model,key,v.cuda())
 grid=BilateralGrid(len(cams),16,16,8).cuda();grid.load_state_dict(ck['grid']);sky=None
 if CONFIG.get('with_sky'):
  from sky_bridge import NativeSky
  
  def device_tree(x):
   if torch.is_tensor(x):return x.cuda()
   if isinstance(x,dict):return {k:device_tree(v) for k,v in x.items()}
   if isinstance(x,list):return [device_tree(v) for v in x]
   if isinstance(x,tuple):return tuple(device_tree(v) for v in x)
   return x
  sky=NativeSky(cams,model.get_xyz,state=device_tree(ck['sky']))
 step=ck['iteration'];del ck,m
 pipe=SimpleNamespace(debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda');xy={};rows=[]
 limit=int(os.environ.get('EVAL_LIMIT',len(cams)))
 for i,c in enumerate(cams[:limit]):
  pkg=render(c,model,pipe,bg,track_gradients=False);raw=pkg['render']
  if sky is not None:raw=raw+(1-pkg['acc'])*sky.render(c)
  pred=apply_grid(grid,c,raw,xy);gt=c.original_image.cuda();mask=c.alpha_mask.cuda().bool();row=dict(image=c.image_name,camera=c.image_name.rsplit('_',1)[1],valid_pixels=int(mask.sum()))
  for name,img in [('raw',raw),('grid',pred)]:
   for quant in ['round','floor','float']:
    x=img.clamp(0,1)
    if quant!='float':x=(x*255+(.5 if quant=='round' else 0)).byte().float()/255
    row[name+'_'+quant+'_psnr']=float(-10*torch.log10(((x-gt)**2)[mask.expand_as(x)].mean().clamp_min(1e-15)))
    if quant=='round':row[name+'_ssim']=float(masked_ssim(x,gt,mask))
   row[name+'_psnr']=row[name+'_round_psnr']
   Image.fromarray((img.clamp(0,1)*255+.5).byte().permute(1,2,0).cpu().numpy()).save(out/'images'/(c.image_name+'_'+name+'.png'))
  rows.append(row)
  if (i+1)%50==0:print('EVAL',i+1,flush=True)
 keys=[k for k in rows[0] if k not in ['image','camera','valid_pixels']]
 def summary(xs):return {k:float(np.mean([x[k] for x in xs])) for k in keys}
 result=dict(step=step,variant=VARIANT,images=len(rows),summary=summary(rows),cameras={c:summary([x for x in rows if x['camera']==c]) for c in sorted(set(x['camera'] for x in rows))},points=len(model.get_xyz),sky_points=len(sky.model.get_xyz) if sky else 0,rows=rows,protocol='Shared static masks; 707 training images, not held-out. Primary rounded uint8 metrics; floor and float PSNR also reported. No posthoc fitting.')
 (out/'results.json').write_text(json.dumps(result,indent=2))
 with (out/'per_image.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 parts=['<meta charset="utf-8"><h1>'+VARIANT+'</h1><pre>'+json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2)+'</pre>'];by={c.image_name:c for c in cams}
 for cam,v in result['cameras'].items():
  rr=sorted([x for x in rows if x['camera']==cam],key=lambda x:x['grid_psnr'])
  for tag,row in [('low',rr[0]),('middle',rr[len(rr)//2]),('high',rr[-1])]:
   c=by[row['image']];ims=[Image.fromarray((c.original_image*255).byte().permute(1,2,0).numpy())]+[Image.open(out/'images'/(c.image_name+'_'+n+'.png')) for n in ['raw','grid']];h=round(480*c.image_height/c.image_width);im=Image.new('RGB',(1440,h+32),'white');d=ImageDraw.Draw(im)
   for j,(p,label) in enumerate(zip(ims,['GT','Raw %.3f'%row['raw_psnr'],'Grid %.3f'%row['grid_psnr']])):im.paste(p.resize((480,h)),(j*480,32));d.text((j*480+8,8),label,fill='black')
   name=cam+'_'+tag+'.jpg';im.save(out/name,quality=93);parts+=['<h2>'+cam+' '+tag+' '+row['image']+'</h2><img style="max-width:100%" src="'+name+'">']
 (out/'report.html').write_text(''.join(parts));print('EVAL_COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
