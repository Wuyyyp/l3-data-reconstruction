import hashlib,json,os,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
v=Path('/data/l3data-reconstruction-bingxing/new-pipeline-1031-samples-v3')
import argparse
p=argparse.ArgumentParser(description='Render a full-frame WYS L3 window as a seven-view video')
p.add_argument('window',type=Path);p.add_argument('output',type=Path)
p.add_argument('--branch',choices=('original','shifted'),default='original')
args=p.parse_args()
w=args.window.resolve();out=args.output.resolve();kind=args.branch
sid=w.name;direction=w.name.rsplit('_',1)[-1];case=w/kind;full=case/'full'
ckpt=full/'checkpoint_030000.pth'
from l3_wys_canvas import VERSION
audit=json.loads((w/'pose/input_audit.json').read_text())
if audit.get('geometry_version')!=VERSION:
 raise ValueError('This export expects full-frame WYS training inputs.')
out.mkdir(parents=True,exist_ok=True)
repo=(v/'code/street_gaussians').resolve()
sys.path[:0]=[str(repo/'output/engine_affine_shadow_1031/code'),str(v/'release'),str(v),str(repo),*[str(repo/'deps'/n) for n in ('improvedgs_ground','improvedgs_dynamic','gsplat')]]
config=json.loads((full/'launch_config.yaml').read_text())
config.update(source_path=str(case/'data/converted'),workspace=str(repo),model_path=str(out),mode='evaluate')
cfgfile=out/'render_config.yaml';cfgfile.write_text(json.dumps(config))
ground=json.loads((full/'pipeline_config.json').read_text())['model']['ground']
assert ground['masks'].endswith('/'+w.name+'/'+kind+'/full_road/road_masks') and ground['init'].endswith('/'+w.name+'/'+kind+'/surface/ground.npz')
ground['masks']=str(case/'full_road/road_masks');ground['init']=str(case/'surface/ground.npz')
os.environ.update(DYNAMIC_CONFIG=str(cfgfile),GROUND_INIT=ground['init'],GROUND_SHARED_SURFACE=ground['init'],GROUND_MASKS=ground['masks'],ACTOR_ANCHORS=str(case/'anchors'),SHADOW_TEMPLATES=str(case/'shadow_templates'),GROUND_SH_DEGREE=str(ground.get('sh_degree',0)),GROUND_MODE='anchored',GROUND_SPLIT='1',GROUND_WARMUP='3000',GROUND_POINT_CAP=str(ground.get('point_cap',1200000)),GROUND_SPLIT_THRESHOLD=str(ground.get('split_threshold',.0002)),FULL_BACKGROUND_SOURCE=str(w/'protected'),TORCH_EXTENSIONS_DIR=str(repo/'cache/improvedgs_static/torch_extensions'),CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.0')
os.chdir(repo)
import torch,cv2,numpy as np
from ground_extension import DynamicScene,dataset,masks_for_sky,cfg,PARAMS
from scene.gaussian_model import GaussianModel
CAMS={0:(0,0,'Left front'),9:(960,0,'Front FOV30'),10:(1920,0,'Front FOV120'),1:(2880,0,'Right front'),3:(480,560,'Left rear'),2:(1440,560,'Rear'),4:(2400,560,'Right rear')}

@torch.no_grad()
def run():
 torch.set_num_threads(2);cv2.setNumThreads(1)
 ck=torch.load(ckpt,map_location='cuda:0');assert ck['iteration']==30000
 ds,cams=dataset();assert ck['camera_names']==[c.image_name for c in cams]
 bg=GaussianModel(1,'ours_adam');state=ck['model'];bg.active_sh_degree=state[0];bg.max_sh_degree=state[1]
 for k,val in zip(PARAMS,state[2:8]):setattr(bg,k,val)
 scene=DynamicScene(ds,cams,bg,out,initialize=False)
 if scene.sg.actor_pose is not None:scene.sg.actor_pose.training_setup()
 scene.restore(ck['dynamic']);scene.audit_finite();scene.sg.active_sh_degree=bg.active_sh_degree
 assert scene.iteration==30000 and scene.joint_active()
 del ck,state
 if kind=='shifted':
  def poses(branch):
   return {row['name']: np.array(row['c2w']) for row in json.loads((w/branch/'data/input_manifest.json').read_text())['views']}
  orig,shift=poses('original'),poses('shifted')
  audit=json.loads((w/'shifted_render/render_audit.json').read_text())
  assert orig.keys()==shift.keys() and len(orig)==len(audit['poses'])==707
  for row in audit['poses']:
   key=row['name'];a,b=orig[key],shift[key]
   assert np.allclose(a[:3,:3],b[:3,:3]) and abs(np.linalg.norm(b[:3,3]-a[:3,3])-3)<0.001 and np.allclose(b[:3,3]-a[:3,3],row['delta_world'],atol=0.001)
 lookup={c.image_name:c for c in ds.test_cameras[1]};assert set(lookup)=={f'{i:06d}_{cid:02d}' for i in range(101) for cid in CAMS}
 black=torch.zeros(3,device='cuda');pipe=SimpleNamespace(debug=False,antialiasing=False)
 masks={}
 for cid in (3,4,10):
  mask=cv2.imread(str(w/'pose/source_fixed_masks'/f'000000_{cid:02d}.png'),cv2.IMREAD_GRAYSCALE)
  assert mask is not None and mask.shape==(900,1600)
  assert mask.any()
  masks[cid]=np.minimum(cv2.distanceTransform((mask>0).astype('uint8'),cv2.DIST_L2,3)/2,1)[:,:,None]
 name=sid+'_'+(kind if kind=='original' else kind+'_'+direction+'_3m')+'_30000_7views'
 temp=out/(name+'.partial.mp4');final=out/(name+'.mp4')
 cmd=['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','3840x1120','-r','10','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','19','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for f in range(101):
   grid=np.zeros((1120,3840,3),np.uint8)
   for cid,(x,y,label) in CAMS.items():
    key=f'{f:06d}_{cid:02d}';c=lookup[key];c.alpha_mask=masks_for_sky(c)['mask']
    pkg=scene.render(c,bg,pipe,black,track_gradients=False);pred=scene.sg.color_correction(c,pkg['render']).clamp(0,1)
    im=cv2.cvtColor(pred.mul(255).round().byte().permute(1,2,0).cpu().numpy(),cv2.COLOR_RGB2BGR)
    assert im.shape==(900,1600,3)
    if cid in masks:
     gt=cv2.imread(str(w/'original/data/converted/images'/(key+'.jpg')));assert gt.shape==im.shape
     a=masks[cid];im=np.rint(im*(1-a)+gt*a).astype(np.uint8)
    grid[y+20:y+560,x:x+960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
    cv2.putText(grid,label,(x+12,y+16),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    del pred,pkg
   p.stdin.write(grid.tobytes())
   if f%20==0:print('RENDERED',sid,f,flush=True)
  p.stdin.close();assert p.wait()==0
 finally:
  if p.poll() is None:p.kill();p.wait()
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,codec_name,avg_frame_rate,nb_read_frames,duration','-of','json',str(temp)]))['streams'][0]
 assert (probe['width'],probe['height'],probe['nb_read_frames'],probe['avg_frame_rate'],probe['codec_name'])==(3840,1120,'101','10/1','h264')
 temp.replace(final)
 m=dict(sample=w.name,sha256=hashlib.sha256(final.read_bytes()).hexdigest(),bytes=final.stat().st_size,filename=final.name,probe=probe,checkpoint=str(ckpt),geometry_version=VERSION,masks={k:int((a>0).sum()) for k,a in masks.items()})
 (out/'verified.json').write_text(json.dumps(m,indent=2));print('VERIFIED',sid,probe,flush=True)
run()

