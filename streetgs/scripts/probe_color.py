import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');base=root/'output/gsplat_stage67_fixed15k_20261006';out=base/'optimizer_appearance_probe';out.mkdir(exist_ok=True)
prev=json.loads((root/'output/sample_00380_pose_ba_25000_20261001_172042/pose_comparison/results.json').read_text())
def load(frame,cam):
 name=f'{frame:06d}_{cam:02d}'
 with Image.open(base/'workers/gsplat/gsplat/images_25000'/(name+'.png')) as im:p=np.asarray(im.convert('RGB'),dtype=np.float64)/255
 h,w=p.shape[:2]
 with Image.open(Path(prev['new_source'])/'images'/(name+'.jpg')) as im:g=np.asarray(im.convert('RGB').resize((w,h),Image.BILINEAR),dtype=np.float64)/255
 v=np.ones((h,w),bool)
 if cam==3:v[:int(250/1280*h),:int(550/1920*w)]=False;v[:,:int(80/1920*w)]=False
 if cam==4:v[:int(250/1280*h),int((1920-550)/1920*w):]=False;v[:,int((1920-80)/1920*w):]=False
 if cam==10:v[int(1500/2160*h):]=False
 return name,p,g,v
def fit(x,y,full=False):
 if full:
  X=np.column_stack([x,np.ones(len(x))]);prior=np.vstack([np.eye(3),np.zeros((1,3))]);return np.linalg.solve(X.T@X+len(x)*1e-5*np.eye(4),X.T@y+len(x)*1e-5*prior)
 out=np.zeros((4,3))
 for c in range(3):
  X=np.column_stack([x[:,c],np.ones(len(x))]);out[[c,3],c]=np.linalg.solve(X.T@X+len(x)*1e-5*np.eye(2),X.T@y[:,c]+len(x)*1e-5*np.array([1,0]))
 return out
def score(p,g,v):return float(-10*np.log10(np.mean((p[v]-g[v])**2)))
rows=[];models={}
for cam in [0,1,2,3,4,9,10]:
 samples={}
 for frame in range(0,101,10):
  _,p,g,v=load(frame,cam);vv=v[::12,::12];samples[frame]=(p[::12,::12][vv],g[::12,::12][vv])
 models[str(cam)]={}
 for split,train,test in [('interleaved',list(range(0,101,20)),list(range(10,100,20))),('future',list(range(0,41,10)),list(range(50,101,10)))]:
  x=np.concatenate([samples[f][0] for f in train]);y=np.concatenate([samples[f][1] for f in train]);fits={k:fit(x,y,k=='affine') for k in ['diagonal','affine']};models[str(cam)][split]={k:m.tolist() for k,m in fits.items()}
  for frame in test:
   name,p,g,v=load(frame,cam);baseps=score(p,g,v);rr=dict(image=name,camera=cam,frame=frame,split=split,baseline=baseps)
   for k,m in fits.items():rr[k]=score(np.clip(p@m[:3]+m[3],0,1),g,v)
   oracle=fit(*samples[frame],full=True);rr['oracle_same_frame']=score(np.clip(p@oracle[:3]+oracle[3],0,1),g,v);rows.append(rr)
 print('CAMERA',cam,flush=True)
summary={}
for split in ['interleaved','future']:
 summary[split]={}
 for cam in ['all',0,1,2,3,4,9,10]:
  rs=[r for r in rows if r['split']==split and (cam=='all' or r['camera']==cam)]
  summary[split][str(cam)]={k:dict(mean_psnr=float(np.mean([r[k] for r in rs])),mean_gain=float(np.mean([r[k]-r['baseline'] for r in rs])),min_gain=float(min(r[k]-r['baseline'] for r in rs)),positive_count=sum(r[k]>r['baseline'] for r in rs),count=len(rs)) for k in ['baseline','diagonal','affine','oracle_same_frame']}
result=dict(summary=summary,rows=rows,models=models,protocol='Existing gsplat25k PNG + same GT bilinear resize + production masks. Fit on 1/12 pixel grid, score all valid pixels at native resolution. Per-camera diagonal or 3x3+bias RGB affine, ridge 1e-5 toward identity, clamp output. Interleaved train frames0,20,...100 test10,30,...90; future train0..40 every10 test50..100 every10. Frames held out from COLOR FIT ONLY, all are reconstruction training views. Oracle fits test frame itself, diagnostic upper reference only. Existing StreetGS color correction remains baked in renders. No bilateral/local correction or retraining. Arithmetic mean per-image PSNR.');(out/'color.json').write_text(json.dumps(result,indent=2))
rs=sorted([r for r in rows if r['split']=='future'],key=lambda r:r['affine']-r['baseline']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for label,r in [('worst',rs[0]),('median',rs[len(rs)//2]),('best',rs[-1])]:
 name,p,g,v=load(r['frame'],r['camera']);m=np.array(models[str(r['camera'])]['future']['affine']);cc=np.clip(p@m[:3]+m[3],0,1);h,w=p.shape[:2];hh=round(h*640/w);im=Image.new('RGB',(1920,hh+65),'white');d=ImageDraw.Draw(im)
 for j,(title,a) in enumerate([('GT '+name,g),('Original %.3f dB'%r['baseline'],p),('Held-out color %.3f dB'%r['affine'],cc)]):d.text((j*640+10,20),title,font=font,fill='black');im.paste(Image.fromarray((a*255+.5).astype('uint8')).resize((640,hh)),(j*640,65))
 im.save(out/('color_'+label+'.jpg'),quality=94)
print(json.dumps(summary),flush=True)
