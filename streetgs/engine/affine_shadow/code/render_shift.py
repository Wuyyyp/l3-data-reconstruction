from ground_extension import *
from scene.gaussian_model import GaussianModel
from PIL import Image,ImageDraw
@torch.no_grad()
def main():
 torch.set_num_threads(4);ck=torch.load(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
 for k,v in zip(PARAMS,state[2:8]):setattr(bg,k,v)
 cfg.mode='evaluate';scene=DynamicScene(ds,cams,bg,out,initialize=False);scene.sg.actor_pose.training_setup() if scene.sg.actor_pose is not None else None;scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree;step=ck['iteration'];del ck,state;cfg.mode='evaluate'

 import subprocess
 root=Path(cfg.source_path);lookup={c.image_name:c for c in ds.test_cameras[1]};assert len(lookup)==707
 layout={0:(0,0,'LEFT FRONT'),10:(1,0,'FRONT WIDE'),1:(2,0,'RIGHT FRONT'),3:(0,1,'LEFT REAR'),9:(1,1,'FRONT NARROW'),4:(2,1,'RIGHT REAR'),2:(1,2,'REAR')}
 pipe=SimpleNamespace(debug=False,antialiasing=False);background=torch.zeros(3,device='cuda')
 def writer(name,w,h):
  return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(name+'.partial.mp4'))],stdin=subprocess.PIPE)
 shifts=[('left_3m',-3.),('original',0.),('right_3m',3.)]
 writers={name:writer(name,1920,1152) for name,_ in shifts};writers['front_comparison']=writer('front_comparison',1920,384)
 audit=[];count=0
 try:
  for frame in range(101):
   ego=np.loadtxt(root/'ego_pose'/f'{frame:06d}.txt');right=ego[:3,0];right=right/np.linalg.norm(right)
   front=lookup[f'{frame:06d}_10'];assert np.dot(front.get_extrinsic()[:3,0],right)>.95
   comp=Image.new('RGB',(1920,384),(18,23,30));cd=ImageDraw.Draw(comp)
   for col,(name,offset) in enumerate(shifts):
    canvas=Image.new('RGB',(1920,1152),(18,23,30));draw=ImageDraw.Draw(canvas)
    draw.text((20,800),os.environ.get('VIDEO_CLIP','')+' | GROUND + 77 ACTORS 20K | SH '+os.environ['GROUND_SH_DEGREE'],fill='white');draw.text((20,825),name.upper()+' | AFFINE | Frame %03d/100'%frame,fill='white');draw.text((20,850),'Same rig orientation and actor time | 10 fps',fill='white')
    for cam,(x,y,label) in layout.items():
     c=lookup[f'{frame:06d}_{cam:02d}'];base=c.get_extrinsic().copy();moved=base.copy();moved[:3,3]+=offset*right;c.set_extrinsic(moved)
     assert np.allclose(moved[:3,:3],base[:3,:3]);assert abs(np.linalg.norm(moved[:3,3]-base[:3,3])-abs(offset))<1e-6
     c.alpha_mask=masks_for_sky(c)['mask'];p=scene.render(c,bg,pipe,background,track_gradients=False);raw=p['render'];pred=scene.sg.color_correction(c,raw).clamp(0,1)
     image=Image.fromarray((pred.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype('u1')).resize((640,360),Image.Resampling.LANCZOS)
     canvas.paste(image,(x*640,y*384+24));draw.text((x*640+8,y*384+6),f'CAM {cam:02d} - {label}',fill='white')
     if cam==10:comp.paste(image,(col*640,24));cd.text((col*640+8,6),name.upper()+f' | FRAME {frame:03d}',fill='white')
     audit.append(dict(frame=frame,cam=cam,view=name,delta_world=(offset*right).tolist()));count+=1;c.set_extrinsic(base);del p,raw,pred
    writers[name].stdin.write(canvas.tobytes())
    if frame in [0,50,100]:canvas.save(out/f'{name}_preview_{frame:03d}.jpg',quality=92)
   writers['front_comparison'].stdin.write(comp.tobytes())
   if frame in [0,50,100]:comp.save(out/f'comparison_preview_{frame:03d}.jpg',quality=92)
   if frame%10==0:print('FRAME',frame,'RENDERS',count,flush=True)
  assert count==2121
  probes={}
  for name,p in writers.items():
   p.stdin.close();assert p.wait()==0
   partial=out/(name+'.partial.mp4');probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(partial)],text=True));assert int(probe['streams'][0]['nb_read_frames'])==101;partial.rename(out/(name+'.mp4'));probes[name]=probe
  (out/'video_audit.json').write_text(json.dumps(dict(checkpoint_step=step,rendered_views=count,frames=101,fps=10,seconds=10.1,affine=True,shift_axis='paired optimized ego local X is right; left=-3m, right=+3m',orientation_unchanged=True,original_actor_time=True,layout=layout,probes=probes,poses=audit),indent=2));print('VIDEO_COMPLETE',flush=True)
 except Exception:
  for p in writers.values():
   if p.poll() is None:p.kill()
  raise
if __name__=='__main__':main()
