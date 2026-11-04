from ground_extension import *
from scene.gaussian_model import GaussianModel
from PIL import Image,ImageDraw
@torch.no_grad()
def main():
 torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
 for k,v in zip(PARAMS,state[2:8]):setattr(bg,k,v)
 cfg.mode='evaluate';scene=DynamicScene(ds,cams,bg,out,initialize=False);scene.sg.actor_pose.training_setup() if scene.sg.actor_pose is not None else None;scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree;step=ck['iteration'];del ck,state;cfg.mode='evaluate'
 from export_ground import save_ground_ply
 save_ground_ply(scene.ground,out/'ground_gaussians.ply')
 pipe=SimpleNamespace(debug=False,antialiasing=False);background=torch.zeros(3,device='cuda');rows=[];test=sorted(ds.test_cameras[1],key=lambda c:c.image_name);limit=int(os.environ.get('EVAL_LIMIT',len(test)))
 for i,c in enumerate(test[:limit]):
  c.alpha_mask=masks_for_sky(c)['mask'];p=gsplat_render(c,scene.ground,pipe,background,track_gradients=False);pred=p['render'].clamp(0,1);gt=c.original_image.cuda();mask=c.alpha_mask.cuda();gm=scene.ground_mask(c).cuda()&mask;row=dict(image=c.image_name)
  for name,mm in [('all',mask),('static',mask&~c.guidance['obj_bound'].cuda().bool()),('ground',gm),('vehicle',mask&c.guidance['obj_bound'].cuda().bool())]:
   row[name+'_pixels']=int(mm.sum());row[name+'_psnr']=float(-10*torch.log10((pred-gt).square()[mm.expand_as(pred)].mean().clamp_min(1e-12))) if mm.any() else None;row[name+'_l1']=float((pred-gt).abs()[mm.expand_as(pred)].mean()) if mm.any() else None
  if scene.sg.color_correction is not None:
   raw=p['render'];final=scene.sg.color_correction(c,raw).clamp(0,1)
   for label,mm in [('all',mask),('static',mask&~c.guidance['obj_bound'].cuda().bool()),('ground',gm),('dynamic',mask&c.guidance['obj_bound'].cuda().bool())]:
    row['affine_'+label+'_psnr']=float(-10*torch.log10((final-gt).square()[mm.expand_as(final)].mean().clamp_min(1e-12))) if mm.any() else None
  else:final=pred
  from utils.static_losses import masked_ssim
  row['all_ssim']=float(masked_ssim(pred,gt,mask))
  eligible=torch.nn.functional.max_pool2d((~gm).float()[None],11,1)[0,0]==0
  row['ground_ssim']=float(masked_ssim(pred,gt,gm)) if eligible.any() else None
  row['affine_ground_ssim']=float(masked_ssim(final,gt,gm)) if eligible.any() else None
  if cfg.data.get('ground_split')=='train-all':row['affine_all_ssim']=float(masked_ssim(final,gt,mask))
  rows.append(row)
  sample=(c.meta['frame'] in [0,50,100]) if cfg.data.get('ground_split')=='train-all' else (i<7 or c.meta['frame']==49)
  if os.environ.get('SAVE_IMAGES','1')=='1' and sample:
   Image.fromarray((pred.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).save(out/(c.image_name+'_render.png'))
  if sample:
   canvas=Image.new('RGB',(1280,360))
   for j,v in enumerate([gt,final]):canvas.paste(Image.fromarray((v.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).resize((640,360)),(j*640,0))
   text=c.image_name+' Ground PSNR '+str(row['ground_psnr'])
   if 'affine_all_psnr' in row:text+=' Affine ground PSNR '+str(row.get('affine_ground_psnr'))
   banner=Image.new('RGB',(1280,388),'white');banner.paste(canvas,(0,28));ImageDraw.Draw(banner).text((8,7),text,fill='black');banner.save(out/(c.image_name+'.jpg'),quality=90)
  if i%20==0:print('EVAL',i+1,len(test),flush=True)
 keys=[k for k in rows[0] if k.endswith(('_psnr','_l1','_ssim'))];summary={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
 geo={}
 if scene.ground is not None:
  delta=(scene.ground.get_xyz-scene.ground.initial).norm(dim=1);geo=dict(split_events=len(scene.ground.events),mean_displacement=float(delta.mean()),max_displacement=float(delta.max()),count=len(delta));assert torch.isfinite(delta).all();assert float(delta.max())<=.10001
  assert cfg.model.nsg.include_obj or len(scene.sg.obj_list)==0
 (out/'results.json').write_text(json.dumps(dict(mode=MODE,step=step,images=len(rows),protocol=('Ground-only render on original training views; only ground metrics meaningful; no Affine fitting; not independent test.' if cfg.data.get('ground_split')=='train-all' else 'Held-out frame%5==4. Float RGB [0,1], raw rendering without affine correction or test fitting. Mean per-image PSNR. static excludes obj_bound and ego, includes sky; ground uses projected mask, excludes obj_bound and ego. all includes vehicles only in dynamic mode.'),summary=summary,ground_geometry=geo,rows=rows),indent=2));print('EVALUATION_COMPLETE',summary,flush=True)
if __name__=='__main__':
 main()
