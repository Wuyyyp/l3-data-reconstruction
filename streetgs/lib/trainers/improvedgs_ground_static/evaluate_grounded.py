from ground_extension import *
from scene.gaussian_model import GaussianModel
from PIL import Image,ImageDraw
@torch.no_grad()
def main():
 torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
 for k,v in zip(PARAMS,state[2:8]):setattr(bg,k,v)
 cfg.mode='evaluate';scene=DynamicScene(ds,cams,bg,out,initialize=False);scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree;step=ck['iteration'];del ck,state;cfg.mode='evaluate'
 pipe=SimpleNamespace(debug=False,antialiasing=False);background=torch.zeros(3,device='cuda');rows=[];test=sorted(ds.test_cameras[1],key=lambda c:c.image_name);limit=int(os.environ.get('EVAL_LIMIT',len(test)))
 for i,c in enumerate(test[:limit]):
  c.alpha_mask=masks_for_sky(c)['mask'];p=scene.render(c,bg,pipe,background,track_gradients=False);pred=(p['render']+(1-p['acc'])*scene.sg.sky_cubemap(c)).clamp(0,1);gt=c.original_image.cuda();mask=c.alpha_mask.cuda();gm=scene.ground_mask(c).cuda()&mask;row=dict(image=c.image_name)
  for name,mm in [('all',mask),('ground',gm),('vehicle',mask&c.guidance['obj_bound'].cuda().bool())]:
   row[name+'_pixels']=int(mm.sum());row[name+'_psnr']=float(-10*torch.log10((pred-gt).square()[mm.expand_as(pred)].mean().clamp_min(1e-12))) if mm.any() else None;row[name+'_l1']=float((pred-gt).abs()[mm.expand_as(pred)].mean()) if mm.any() else None
  from utils.static_losses import masked_ssim
  row['all_ssim']=float(masked_ssim(pred,gt,mask))
  eligible=torch.nn.functional.max_pool2d((~gm).float()[None],11,1)[0,0]==0
  row['ground_ssim']=float(masked_ssim(pred,gt,gm)) if eligible.any() else None
  rows.append(row)
  if i<7 or c.meta['frame']==49:
   canvas=Image.new('RGB',(1280,360))
   for j,v in enumerate([gt,pred]):canvas.paste(Image.fromarray((v.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).resize((640,360)),(j*640,0))
   canvas.save(out/(c.image_name+'.jpg'),quality=90)
  if i%20==0:print('EVAL',i+1,len(test),flush=True)
 keys=[k for k in rows[0] if k.endswith(('_psnr','_l1','_ssim'))];summary={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
 geo={}
 if scene.ground is not None:
  delta=(scene.ground.get_xyz-scene.ground.initial).norm(dim=1);geo=dict(split_events=len(scene.ground.events),mean_displacement=float(delta.mean()),max_displacement=float(delta.max()),count=len(delta));assert torch.isfinite(delta).all();assert float(delta.max())<=.10001
  assert len(scene.sg.obj_list)==0
 (out/'results.json').write_text(json.dumps(dict(mode=MODE,step=step,images=len(rows),protocol='Static-only held-out frames frame%5==4; raw rendering; union dynamic mask dilated 5 pixels excluded from metrics',summary=summary,ground_geometry=geo,rows=rows),indent=2));print('EVALUATION_COMPLETE',summary,flush=True)
if __name__=='__main__':
 main()
