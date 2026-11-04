from bridge import *
from scene.gaussian_model import GaussianModel
from lib_bilagrid import BilateralGrid,slice
from utils.static_losses import masked_ssim
from PIL import Image,ImageDraw
import csv
@torch.no_grad()
def main():
 torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
 ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');m=ck['model'];bg.active_sh_degree=m[0];bg.max_sh_degree=m[1]
 for k,v in zip(PARAMS,m[2:8]):setattr(bg,k,v)
 scene=DynamicScene(ds,cams,bg,out,initialize=False);scene.sg.actor_pose.training_setup() if scene.sg.actor_pose is not None else None;scene.restore(ck['dynamic']);scene.sg.active_sh_degree=bg.active_sh_degree;sky=scene.sg.sky_cubemap
 grid=BilateralGrid(len(cams),16,16,8).cuda() if ck['grid'] is not None else None
 if grid is not None:grid.load_state_dict(ck['grid'])
 step=ck['iteration'];del ck,m
 pipe=SimpleNamespace(debug=False,antialiasing=False);background=torch.zeros(3,device='cuda');rows=[];xy={}
 def images(c):
  pkg=scene.render(c,bg,pipe,background,track_gradients=False);raw=pkg['render']+(1-pkg['acc'])*sky(c);h,w=c.image_height,c.image_width
  if (h,w) not in xy:
   y,x=torch.meshgrid((torch.arange(h,device='cuda')+.5)/h,(torch.arange(w,device='cuda')+.5)/w,indexing='ij');xy[h,w]=torch.stack([x,y],-1)
  pred=slice(grid,xy[h,w],raw.permute(1,2,0),torch.tensor([c.uid],device='cuda'))['rgb'].permute(2,0,1) if grid is not None else (scene.sg.color_correction(c,raw) if scene.sg.color_correction is not None else raw)
  return raw,pred
 limit=int(os.environ.get('EVAL_LIMIT',len(cams)))
 for i,c in enumerate(cams[:limit]):
  raw,pred=images(c);gt=c.original_image.cuda();mask=c.alpha_mask.cuda();mask=mask & ~c.guidance['obj_bound'].cuda().bool() if not cfg.model.nsg.include_obj else mask;dynamic=mask&c.guidance['obj_bound'].cuda().bool();static=mask&~dynamic;row=dict(image=c.image_name,camera=c.image_name.rsplit('_',1)[1],valid_pixels=int(mask.sum()),dynamic_pixels=int(dynamic.sum()))
  for name,img in [('raw',raw),('grid',pred)]:
   image=(img.clamp(0,1)*255+.5).byte().float()/255
   import cv2
   windows=cv2.erode(mask[0].cpu().numpy().astype('uint8'),np.ones((11,11),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0).any()
   row[name+'_ssim']=float(masked_ssim(image,gt,mask)) if windows else None
   for reg,mm in [('all',mask),('dynamic',dynamic),('static',static)]:row[name+'_'+reg+'_psnr']=float(-10*torch.log10(((image-gt)**2)[mm.expand_as(image)].mean().clamp_min(1e-15))) if mm.any() else None
  rows.append(row)
  if (i+1)%50==0:print('EVALUATED',i+1,flush=True)
 keys=[k for k in rows[0] if k.startswith(('raw_','grid_'))]
 def summary(rr):return {k:float(np.mean([z[k] for z in rr if z[k] is not None])) if any(z[k] is not None for z in rr) else None for k in keys}
 result=dict(metric_valid_images={k:sum(z[k] is not None for z in rows) for k in keys},images=len(rows),step=step,summary=summary(rows),cameras={c:summary([z for z in rows if z['camera']==c]) for c in sorted(set(z['camera'] for z in rows))},rows=rows,background=len(bg.get_xyz),actors=len(scene.sg.obj_list),actor_points=scene.actor_count(),sky_points=len(sky.get_xyz),protocol=('Dynamic' if cfg.model.nsg.include_obj else 'Static; object regions excluded')+'; training views, source poses (no BA claim). Static mode excludes object regions; ego mask retained. Round uint8, strict 11x11 SSIM. Dynamic region is source obj_bound guidance, not an independently annotated test mask.')
 (out/'results.json').write_text(json.dumps(result,indent=2));
 with (out/'per_image.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 parts=['<meta charset="utf-8"><h1>'+('Dynamic' if cfg.model.nsg.include_obj else 'Static')+' ImprovedGS + '+('Bilateral' if grid is not None else 'Affine')+' + '+cfg.model.sky.get('geometry','sphere')+'</h1><pre>'+json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2)+'</pre>']
 by={c.image_name:c for c in cams};selected=[]
 for camera in result['cameras']:
  rr=sorted([z for z in rows if z['camera']==camera],key=lambda z:z['image'])
  selected.extend([rr[len(rr)//5],rr[len(rr)//2],rr[len(rr)*4//5]])
 for row in selected:
  c=by[row['image']];raw,pred=images(c);ims=[c.original_image,raw.cpu(),pred.cpu()];h=round(480*c.image_height/c.image_width);panel=Image.new('RGB',(1440,h+32),'white');draw=ImageDraw.Draw(panel)
  for j,(v,label) in enumerate(zip(ims,['GT', 'Raw PSNR '+str(row['raw_all_psnr']), ('Bilateral' if grid is not None else 'Affine')+' PSNR '+str(row['grid_all_psnr'])])):
   im=Image.fromarray((v.clamp(0,1)*255+.5).byte().permute(1,2,0).numpy());panel.paste(im.resize((480,h)),(j*480,32));draw.text((j*480+8,8),label,fill='black')
  name=c.image_name+'.jpg';panel.save(out/name,quality=94);parts.append('<h2>'+c.image_name+'</h2><img style="max-width:100%" src="'+name+'">')
 (out/'report.html').write_text(''.join(parts))
 loader=getattr(cams[0],'_payload_loader',None)
 if loader is not None:loader.close()
 (out/'completion.json').write_text(json.dumps(dict(step=step,images=len(rows),loader_closed=True)))
 print('EVAL_COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
