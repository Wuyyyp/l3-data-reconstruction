from ground_extension import *
from PIL import Image,ImageDraw
import subprocess

def writer(path,w,h):
 return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
@torch.no_grad()
def main():
 torch.set_num_threads(4);O=Path(__file__).resolve().parents[1];out=O/'shift_videos';ds,cams=dataset();lookup={c.image_name:c for c in ds.test_cameras[1]};assert len(lookup)==707
 expected={r['image']:r for r in json.loads((O/'full/eval_checkpoint_020000/results.json').read_text())['rows']}
 layout={0:(0,0,'LEFT FRONT'),10:(1,0,'FRONT WIDE'),1:(2,0,'RIGHT FRONT'),3:(0,1,'LEFT REAR'),9:(1,1,'FRONT NARROW'),4:(2,1,'RIGHT REAR'),2:(1,2,'REAR')}
 w=writer(out/'training_masked_GT_seven.mp4',1920,1152);front=writer(out/'training_masked_GT_front.mp4',640,384);audit=[]
 for frame in range(101):
  canvas=Image.new('RGB',(1920,1152),(18,23,30));d=ImageDraw.Draw(canvas);d.text((20,800),'TRAINING MASKED GT | ORIGINAL CAMERA',fill='white');d.text((20,825),'Ground OR dynamic region; ego mask retained',fill='white')
  for cam,(x,y,label) in layout.items():
   c=lookup[f'{frame:06d}_{cam:02d}'];fid,cc=c.image_name.split('_');a=np.load(Path(os.environ['GROUND_MASKS'])/('cam_'+cc)/(fid+'.npy'));a=np.asarray(Image.fromarray(a).resize((c.image_width,c.image_height),Image.Resampling.NEAREST)).copy();gm=torch.from_numpy(a)[None].bool()
   valid=c.guidance['mask'].cpu().bool();obj=c.guidance['obj_bound'].cpu().bool();gm=gm&valid&~obj;dm=valid&obj;mask=gm|dm
   assert int(gm.sum())==expected[c.image_name]['ground_pixels'];assert int(dm.sum())==expected[c.image_name]['vehicle_pixels'];assert int(mask.sum())==expected[c.image_name]['all_pixels']
   rgb=c.original_image.cpu();masked=rgb*mask;arr=(masked.permute(1,2,0).numpy()*255).clip(0,255).astype('u1');im=Image.fromarray(arr).resize((640,360),Image.Resampling.LANCZOS)
   canvas.paste(im,(x*640,y*384+24));d.text((x*640+8,y*384+6),f'MASKED GT CAM {cam:02d} | ORIGINAL',fill='white')
   if cam==10:
    ff=Image.new('RGB',(640,384));ff.paste(im,(0,24));ImageDraw.Draw(ff).text((8,6),'TRAINING GT | ORIGINAL VIEW REFERENCE',fill='white');front.stdin.write(ff.tobytes())
   if frame in [0,50,100]:
    Image.fromarray(arr).save(out/(c.image_name+'_training_GT.png'))
   audit.append(dict(image=c.image_name,ground_pixels=int(gm.sum()),dynamic_pixels=int(dm.sum()),union_pixels=int(mask.sum())))
  w.stdin.write(canvas.tobytes())
  if frame in [0,50,100]:canvas.save(out/f'masked_GT_preview_{frame:03d}.jpg',quality=92)
  if frame%20==0:print('GT_FRAME',frame,flush=True)
 for p in [w,front]:p.stdin.close();assert p.wait()==0
 for inputs,name in [(['training_masked_GT_front.mp4','front_comparison.mp4'],'GT_left_original_right.mp4'),(['training_masked_GT_seven.mp4','original.mp4'],'GT_original_seven_comparison.mp4')]:
  cmd=['ffmpeg','-y','-hide_banner','-loglevel','error']
  for i in inputs:cmd+=['-i',str(out/i)]
  cmd+=['-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/name)];subprocess.run(cmd,check=True)
 probes={}
 for p in out.glob('*.mp4'):
  j=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)],text=True));assert int(j['streams'][0]['nb_read_frames'])==101;probes[p.name]=101
 (out/'training_GT_audit.json').write_text(json.dumps(dict(images=707,mask_matches_evaluation=True,gt_is_original_view_only=True,probes=probes,rows=audit),indent=2));print('GT_COMPLETE',flush=True)
if __name__=='__main__':main()
