import os,sys,json,importlib,math
from pathlib import Path
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
for n in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+n)
for n,m in list(sys.modules.items()):
    if n=='numpy.core' or n.startswith('numpy.core.'):sys.modules[n.replace('numpy.core','numpy._core',1)]=m
from lib.config import cfg
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
import gsplat
base=ROOT/'output/gsplat_stage67_fixed15k_20261006'
out=base/'footprint_audit';out.mkdir(exist_ok=True)
cfg.mode='train';cfg.data_device='cpu';cfg.to_cuda=False;cfg.data.regenerate_pcd=False;cfg.data.load_pcd_from=False
cfg.model_path=str(out/'dataset');Path(cfg.model_path).mkdir(exist_ok=True)
p=Path(cfg.model_path)/'input_ply'
if not p.exists():p.symlink_to(ROOT/'output/gsplat_stage123/initialization_baseline/input_ply',target_is_directory=True)
dataset=Dataset();cams={c.image_name:c for c in dataset.train_cameras[1]}
model=StreetGaussianModel(dataset.scene_info.metadata);model.training_setup()
state=torch.load(base/'workers/gsplat/gsplat/iteration_25000.pth');model.load_state_dict(state);del state
model.active_sh_degree=model.max_sh_degree
cfg.mode='evaluate';cfg.render.backend='gsplat';cfg.render.gsplat_packed=False;cfg.render.gsplat_radius_clip=0.
renderer=StreetGaussianRenderer();original=gsplat.rasterization;captured={}
def hook(**kw):
    result=original(**kw);captured['kw']=kw;captured['meta']=result[2];return result
gsplat.rasterization=hook
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
def heat(x,limit):return cv2.cvtColor(cv2.applyColorMap((np.clip(x/limit,0,1)*255).astype('uint8'),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
rows=[]
with torch.no_grad():
 for cam in [2,3,10,0,1,4,9]:
    name='000040_%02d'%cam;c=cams[name];r=renderer.render(c,model)
    pred=r['rgb'].permute(1,2,0).cpu().numpy();gt=c.original_image[:3].permute(1,2,0).cpu().numpy();h,w=gt.shape[:2]
    rad=captured['meta']['radii'].reshape(-1,2).max(-1).values.float()
    feat=torch.stack([rad]+[(rad>t).float() for t in [8,16,32,80]],-1)
    kw=dict(captured['kw']);kw.update(colors=feat,sh_degree=None,render_mode='RGB',backgrounds=None,absgrad=False)
    maps,alpha,_=original(**kw);a=alpha[0,...,0].cpu().numpy();maps=maps[0].cpu().numpy()
    avg=maps[...,0]/np.maximum(a,1e-6);shares=maps[...,1:]/np.maximum(a[...,None],1e-6)
    valid=np.ones((h,w),bool)
    if cam==3:valid[:int(250/1280*h),:int(550/1920*w)]=False;valid[:,:int(80/1920*w)]=False
    if cam==4:valid[:int(250/1280*h),int((1920-550)/1920*w):]=False;valid[:,int((1920-80)/1920*w):]=False
    if cam==10:valid[int(1500/2160*h):]=False
    mse=((pred-gt)**2).mean(-1);err=np.sqrt(mse)*255;high=valid&(mse>=np.percentile(mse[valid],90));support=valid&(a>.5)
    def stat(sel):
      return dict(pixels=int(sel.sum()),area_fraction=float(sel.sum()/valid.sum()),error_fraction=float(mse[sel].sum()/mse[valid].sum()),rmse=float(np.sqrt(mse[sel].mean())*255) if sel.any() else None,mean_radius=float(avg[sel].mean()) if sel.any() else None,mean_alpha=float(a[sel].mean()) if sel.any() else None)
    row=dict(image=name,psnr=float(-10*np.log10(mse[valid].mean())),all=stat(valid),high_error=stat(high),bins={},thresholds={},regions={})
    for lo,hi in [(0,4),(4,8),(8,16),(16,32),(32,80),(80,1e9)]:row['bins'][str(lo)+'-'+str(hi)]=stat(support&(avg>=lo)&(avg<hi))
    for j,t in enumerate([8,16,32,80]):
      dom=support&(shares[...,j]>.5);row['thresholds'][str(t)]=dict(**stat(dom),high_error_overlap=float((dom&high).sum()/high.sum()),high_error_mean_contribution=float(maps[...,j+1][high].mean()),all_mean_contribution=float(maps[...,j+1][valid].mean()))
    if cam==2:
      for label,box in {'left_trees':(0,0,.4,.55),'road':(0,.6,1,1),'distant_vehicles':(.43,.38,.65,.61)}.items():
        x0,y0,x1,y1=box;sel=np.zeros((h,w),bool);sel[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True
        row['regions'][label]=dict(**stat(sel&valid),contribution_gt16=float(maps[...,2][sel&valid].mean()),contribution_gt32=float(maps[...,3][sel&valid].mean()))
    saved=np.asarray(Image.open(base/'workers/gsplat/gsplat/images_25000'/(name+'.png')),dtype=np.float32)/255
    row['saved_render_max_difference']=float(np.abs(saved-pred).max())
    np.savez_compressed(out/(name+'.npz'),mean_radius=avg.astype('float16'),contribution_gt16=maps[...,2].astype('float16'),alpha=a.astype('float16'),rmse=err.astype('float16'),valid=valid)
    overlay=(gt*255*.55).astype('uint8');red=high&(shares[...,1]>.5)&(a>.5);cyan=high&~red
    overlay[red]=[255,50,40];overlay[cyan]=[0,230,255];overlay[~valid]=[90,90,90]
    panels=[('GT',gt*255),('gsplat 25k | PSNR %.2f'%row['psnr'],pred*255),('RMSE 0-60 /255',heat(err,60)),('Contribution-weighted radius 0-32 px',heat(avg,32)),('Contribution from radius >16 px: 0-1',heat(maps[...,2],1)),('Top10% error: red=large dominated; cyan=other',overlay)]
    sheet=Image.new('RGB',(1600,3*(round(h*.5)+42)),'white');d=ImageDraw.Draw(sheet)
    for k,(title,im) in enumerate(panels):
      x=(k%2)*800;y=(k//2)*(round(h*.5)+42);d.text((x+8,y+10),title,font=font,fill='black');sheet.paste(Image.fromarray(np.clip(im,0,255).astype('uint8')).resize((800,round(h*.5))),(x,y+42))
    sheet.save(out/(name+'.jpg'),quality=94)
    if cam==2:
      crop=Image.new('RGB',(1800,950),'white');dd=ImageDraw.Draw(crop)
      for k,(title,im) in enumerate(panels):
        x=k%3*600;y=k//3*475;dd.text((x+5,y+10),title[:48],font=font,fill='black');crop.paste(Image.fromarray(np.clip(im,0,255).astype('uint8')).crop((40,100,640,530)),(x,y+42))
      crop.save(out/'trees_detail.jpg',quality=96)
    rows.append(row);print(json.dumps(row),flush=True);captured.clear();del kw,feat,maps,r;torch.cuda.empty_cache()
(out/'stats.json').write_text(json.dumps(dict(rows=rows,definition='Radius=max(gsplat projected bounding radii x,y), pixels. Per-pixel size=sum(T*alpha_i*radius_i)/sum(T*alpha_i). Contribution maps=sum(T*alpha_i*indicator(radius>threshold)), preserve all occluders. Excludes sky cubemap; high error=top10% valid pixels. Descriptive association, not causal attribution.'),indent=2))
print('COMPLETE',flush=True)

