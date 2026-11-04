from ground_extension import *
from PIL import Image,ImageDraw
import subprocess,cv2

def writer(path,w,h):
 return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
@torch.no_grad()
def main():
 torch.set_num_threads(4);O=Path(__file__).resolve().parents[1];out=O/'opening_comparison';out.mkdir(exist_ok=True);ds,cams=dataset();lookup={c.image_name:c for c in ds.test_cameras[1]};assert len(lookup)==707
 expected={r['image']:r for r in json.loads((O/'full/eval_checkpoint_020000/results.json').read_text())['rows']}
 layout={0:(0,0,'LEFT FRONT'),10:(1,0,'FRONT WIDE'),1:(2,0,'RIGHT FRONT'),3:(0,1,'LEFT REAR'),9:(1,1,'FRONT NARROW'),4:(2,1,'RIGHT REAR'),2:(1,2,'REAR')}
 w=writer(out/'closing_vs_closing_opening_seven.mp4',3840,1152);front=writer(out/'closing_vs_closing_opening_front.mp4',1280,384);audit=[];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
 for frame in range(101):
  canvases=[Image.new('RGB',(1920,1152),(18,23,30)) for _ in range(2)]
  for j,title in enumerate(['CURRENT CLOSING GT','CLOSING + OPENING GT']):
   d=ImageDraw.Draw(canvases[j]);d.text((20,800),title,fill='white');d.text((20,825),'Ground mask only; dynamic and ego protection retained',fill='white');d.text((20,850),'5x5 ellipse, one opening | visualization only',fill='white')
  for cam,(x,y,label) in layout.items():
   c=lookup[f'{frame:06d}_{cam:02d}'];fid,cc=c.image_name.split('_');a=np.load(Path(os.environ['GROUND_MASKS'])/('cam_'+cc)/(fid+'.npy'));a=np.asarray(Image.fromarray(a).resize((c.image_width,c.image_height),Image.Resampling.NEAREST)).copy();gm=torch.from_numpy(a)[None].bool()
   valid=c.guidance['mask'].cpu().bool();obj=c.guidance['obj_bound'].cpu().bool();gm=gm&valid&~obj;dm=valid&obj;before=gm|dm
   assert int(gm.sum())==expected[c.image_name]['ground_pixels'];assert int(before.sum())==expected[c.image_name]['all_pixels']
   opened=cv2.morphologyEx(gm[0].numpy().astype('uint8'),cv2.MORPH_OPEN,kernel,iterations=1);go=torch.from_numpy(opened)[None].bool()&valid&~obj;after=go|dm
   assert not (go&~gm).any();assert torch.equal(after&obj,dm);assert not (after&~valid).any()
   rgb=c.original_image.cpu();pair=[];imgs=[]
   for j,m in enumerate([before,after]):
    arr=((rgb*m).permute(1,2,0).numpy()*255).clip(0,255).astype('u1');im=Image.fromarray(arr);imgs.append(im);im=im.resize((640,360),Image.Resampling.LANCZOS);pair.append(im);canvases[j].paste(im,(x*640,y*384+24));ImageDraw.Draw(canvases[j]).text((x*640+8,y*384+6),f'CAM {cam:02d} - '+['CLOSING','CLOSING + OPENING'][j],fill='white')
   if cam==10:
    ff=Image.new('RGB',(1280,384));df=ImageDraw.Draw(ff)
    for j,im in enumerate(pair):ff.paste(im,(j*640,24));df.text((j*640+8,6),['CURRENT CLOSING GT','AFTER OPENING 5x5'][j],fill='white')
    front.stdin.write(ff.tobytes())
    if frame in [0,50,100]:
     ff.save(out/f'front_comparison_{frame:03d}.jpg',quality=95)
     big=Image.new('RGB',(3200,930));bd=ImageDraw.Draw(big)
     for j,im in enumerate(imgs):big.paste(im,(j*1600,30));bd.text((j*1600+10,8),['CURRENT CLOSING GT','CLOSING + OPENING GT'][j],fill='white')
     big.save(out/f'front_native_comparison_{frame:03d}.png')
   audit.append(dict(image=c.image_name,ground_before=int(gm.sum()),ground_after=int(go.sum()),removed=int((gm&~go).sum()),added=0,dynamic_pixels=int(dm.sum()),union_before=int(before.sum()),union_after=int(after.sum())))
  combined=Image.new('RGB',(3840,1152));combined.paste(canvases[0],(0,0));combined.paste(canvases[1],(1920,0));w.stdin.write(combined.tobytes())
  if frame in [0,50,100]:combined.save(out/f'seven_comparison_{frame:03d}.jpg',quality=92)
  if frame%20==0:print('FRAME',frame,flush=True)
 for p in [w,front]:p.stdin.close();assert p.wait()==0
 probes={}
 for p in out.glob('*.mp4'):
  j=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)],text=True));assert int(j['streams'][0]['nb_read_frames'])==101;probes[p.name]=101
 (out/'audit.json').write_text(json.dumps(dict(images=707,input='current closed and protected ground masks',operation='5x5 ellipse opening once, then ego/dynamic protection, union unchanged dynamic mask',training_modified=False,total_ground_before=sum(r['ground_before'] for r in audit),total_ground_after=sum(r['ground_after'] for r in audit),removed=sum(r['removed'] for r in audit),added=0,probes=probes,rows=audit),indent=2));print('OPENING_COMPLETE',flush=True)
if __name__=='__main__':main()
