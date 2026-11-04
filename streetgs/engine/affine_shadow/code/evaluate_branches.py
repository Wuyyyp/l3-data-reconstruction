from ground_extension import *
from scene.gaussian_model import GaussianModel
from PIL import Image,ImageDraw
@torch.no_grad()
def main():
 torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
 for k,v in zip(PARAMS,state[2:8]):setattr(bg,k,v)
 cfg.mode='evaluate';scene=DynamicScene(ds,cams,bg,out,initialize=False);scene.sg.actor_pose.training_setup() if scene.sg.actor_pose is not None else None;scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree;step=ck['iteration'];scene.iteration=step;del ck,state;cfg.mode='evaluate'
 pipe=SimpleNamespace(debug=False,antialiasing=False);black=torch.zeros(3,device='cuda');rows=[]
 g=scene.ground;bc=quat_scale_to_covar_preci(bg.get_rotation,bg.get_scaling,compute_covar=True,compute_preci=False,triu=False)[0]
 bm=SimpleNamespace(get_xyz=bg.get_xyz,covars=bc,get_opacity=bg.get_opacity,get_features_dc=bg.get_features_dc,get_features_rest=bg.get_features_rest,active_sh_degree=bg.active_sh_degree)
 for c in cams:
  if c.image_name not in ['000000_00','000000_03','000000_09']:continue
  c.alpha_mask=masks_for_sky(c)['mask'];gt=c.original_image.cuda();mask=c.alpha_mask.cuda();gm=scene.ground_mask(c).cuda()&mask
  p=scene.render(c,bg,pipe,black,track_gradients=False);full=scene.sg.color_correction(c,p['render']).clamp(0,1)
  gr=gsplat_render(c,g,pipe,black,track_gradients=False)['render'].clamp(0,1);br=gsplat_render(c,bm,pipe,black,track_gradients=False)['render'].clamp(0,1)
  rows.append(dict(image=c.image_name,ground_pixels=int(gm.sum()),note='standalone branches remove mutual occlusion'))
  for key,v in [('gt',gt*mask),('full_affine',full*mask),('ground_raw',gr*mask),('background_raw',br*mask)]:
   Image.fromarray((v.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).save(out/(c.image_name+'_'+key+'.png'))
  canvas=Image.new('RGB',(1280,760),'white')
  for j,(name,v) in enumerate([('GT valid',gt*mask),('Full Affine',full*mask),('Ground only RAW',gr*mask),('Background only RAW',br*mask)]):
   x=(j%2)*640;y=(j//2)*380;ImageDraw.Draw(canvas).text((x+5,y+3),name,fill='black');canvas.paste(Image.fromarray((v.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).resize((640,360)),(x,y+20))
  canvas.save(out/(c.image_name+'.jpg'),quality=93);print(rows[-1],flush=True)
 (out/'branch_metrics.json').write_text(json.dumps(rows,indent=2))
 loader=getattr(cams[0],'_payload_loader',None)
 if loader:loader.close()
 print('BRANCH_COMPLETE',flush=True)
if __name__=='__main__':main()
