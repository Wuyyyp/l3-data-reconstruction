from pathlib import Path
import json,time,collections,shutil,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from PIL import Image,ImageDraw
from plyfile import PlyData,PlyElement
from common import CFG,S,ROOT,O,G,selected_frames
if CFG.get('reuse_prepared_background'):
 import prepared_background
 raise SystemExit(0)
O.mkdir(exist_ok=False)
(O/'indices').mkdir();(O/'previews').mkdir();shutil.copy2(__file__,O/'export.py')
t=time.time();z=np.load(S/'pointcloud.npz',allow_pickle=True);pc=z['pointcloud'].item();proj=z['camera_projection'].item()
tracks=collections.defaultdict(list);moving_ids=set()
for line in (S/'track/track_info.txt').read_text().splitlines()[1:]:
 x=line.split();b=dict(id=int(x[1]),cls=x[2],h=float(x[4]),w=float(x[5]),l=float(x[6]),center=np.array(list(map(float,x[7:10]))),yaw=float(x[10]),speed=float(x[11]));tracks[int(x[0])].append(b)
 if abs(b['speed'])>.5:moving_ids.add(b['id'])
if CFG.get('mesh_source'):
 from mesh_inputs import readable
 moving_ids.update(a['id'] for a in json.loads(readable(Path(CFG['mesh_source'])/'actors.json').read_text()))
DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('frame_id','<u2'),('point_id','<u4'),('track_id','<i4'),('is_dynamic','u1')])
def write(name,a):PlyData([PlyElement.describe(a,'vertex')]).write(O/name)
def merge(a):
 xyz=np.column_stack([a[k] for k in 'xyz']);ix=np.unique(np.floor(xyz/.05).astype('int64'),axis=0,return_index=True)[1];return a[np.sort(ix)]
def colorize(fid,ids,tid,dynamic,E):
 p=pc[fid][ids];w=p.astype('f8')@E[:3,:3].T+E[:3,3];a=np.zeros(len(ids),DT)
 for j,k in enumerate('xyz'):a[k]=w[:,j]
 for k in ['red','green','blue']:a[k]=160
 a['frame_id']=fid;a['point_id']=ids;a['track_id']=tid[ids];a['is_dynamic']=dynamic[ids]
 pr=proj[fid][ids];colored=np.zeros(len(ids),bool)
 for path in sorted((S/'images').glob(f'{fid:06d}_*.jpg')):
  cam=int(path.stem.split('_')[1]);im=np.asarray(Image.open(path).convert('RGB'));h,w=im.shape[:2]
  for col in range(0,pr.shape[1],3):
   ii=np.flatnonzero((pr[:,col]==cam)&~colored);uv=pr[ii,col+1:col+3].astype(int);ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ii=ii[ok];uv=uv[ok]
   for j,k in enumerate(['red','green','blue']):a[k][ii]=im[uv[:,1],uv[:,0],j]
   colored[ii]=True
 return a
parts=[];rows=[];dyn50=None
for fid in selected_frames(pc):
 p=pc[fid];finite=np.isfinite(p).all(1);validids=np.flatnonzero(finite);tid=np.full(len(p),-1,np.int32)
 # Any track that moves during the clip is excluded at every time, including stops.
 for b in tracks[fid]:
  if b['id'] not in moving_ids:continue
  d=p-b['center'];c=np.cos(b['yaw']);s=np.sin(b['yaw']);xx=c*d[:,0]+s*d[:,1];yy=-s*d[:,0]+c*d[:,1]
  mask=(abs(xx)<b['l']/2+.15)&(abs(yy)<b['w']/2+.15)&(d[:,2]>=-b['h']/2)&(d[:,2]<=b['h']/2+.10);tid[mask]=b['id']
 g=np.load(G/'point_indices'/f'{fid:06d}.npz');gi=g['ground_indices'];E=g['ego_to_world'];ge=p[gi]
 dist,ix=cKDTree(ge[:,:2]).query(p[validids,:2],k=4,workers=CFG['workers']);dz=abs(p[validids,2]-np.median(ge[ix,2],axis=1));supported=dist[:,-1]<.5
 ground=np.zeros(len(p),bool);ground[validids]=supported&(dz<.10)&((tid[validids]<0)|(dz<.04));ground[gi]=True
 dynamic=finite&(tid>=0)&~ground;static=finite&~ground&~dynamic
 assert not np.any(static&dynamic) and int(static.sum()+dynamic.sum()+ground.sum())==int(finite.sum())
 np.savez_compressed(O/'indices'/f'{fid:06d}.npz',ground=np.flatnonzero(ground).astype('u4'),dynamic=np.flatnonzero(dynamic).astype('u4'),static=np.flatnonzero(static).astype('u4'))
 ids=np.flatnonzero(static);world=p[ids].astype('f8')@E[:3,:3].T+E[:3,3];ii=np.unique(np.floor(world/.05).astype('int64'),axis=0,return_index=True)[1];a=colorize(fid,ids[np.sort(ii)],tid,dynamic,E);parts.append(a)
 if len(parts)>=10:parts=[merge(np.concatenate(parts))]
 if fid==CFG['dynamic_frame']:
  dyn50=colorize(fid,np.flatnonzero(dynamic),tid,dynamic,E);write('dynamic_frame_world.ply',dyn50)
 row=dict(frame=fid,raw=len(p),finite=int(finite.sum()),ground=int(ground.sum()),dynamic=int(dynamic.sum()),static=int(static.sum()),static_voxel=len(a));rows.append(row)
 if fid%10==0:print(json.dumps(row),flush=True)
 (O/'status.json').write_text(json.dumps(dict(stage='processing',frames=len(rows),seconds=time.time()-t)))
print('FINAL_MERGE',flush=True);bg=merge(np.concatenate(parts));assert np.all(bg['is_dynamic']==0) and np.all(dyn50['frame_id']==CFG['dynamic_frame'])
write('static_no_ground_world_5cm.ply',bg);combined=np.concatenate([bg,dyn50]);write('scene_no_ground_dynamic_world.ply',combined)
rng=np.random.default_rng(42);check=combined[rng.choice(len(combined),min(10000,len(combined)),replace=False)];err=0
for fid in np.unique(check['frame_id']):
 a=check[check['frame_id']==fid];E=np.loadtxt(S/f'ego_pose/{fid:06d}.txt');expected=pc[int(fid)][a['point_id']].astype('f8')@E[:3,:3].T+E[:3,3];err=max(err,float(abs(expected-np.column_stack([a[k] for k in 'xyz'])).max()))
assert err<.0001
checks=[dict(file=p.name,bytes=p.stat().st_size,points=len(PlyData.read(p)['vertex']),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(O.glob('*.ply'))]
report=dict(source=str(S),pipeline_config=CFG,frames=len(rows),dynamic_frame=CFG['dynamic_frame'],ground_reference=str(G),static_points=len(bg),dynamic_points=len(dyn50),combined_points=len(combined),moving_track_ids=sorted(moving_ids),dynamic_rule='Track abs(speed)>0.5 in any frame (annotation speed units); include all classes, oriented box XY+15cm/top+10cm/bottom unchanged; exclude these tracks in every frame.',ground_rule='All original ground indices plus points supported by 4 ground neighbors within 0.5m XY and median-height residual <10cm; inside moving box <4cm.',sampling='Static 5cm world voxel representative; dynamic frame50 full resolution; raw IDs retained; original poses, no training.',limitations='Box-based dynamic extraction: unlabeled or out-of-box motion may remain; sparse unsupported ground may remain. Near-ego and other noise retained. Ego mask is applied during image preparation, not as a 3D point visibility classifier.',max_coordinate_error_m=err,outputs=checks,rows=rows,seconds=time.time()-t)
(O/'report.json').write_text(json.dumps(report,indent=2))
(O/'status.json').write_text(json.dumps(dict(stage='complete',outputs=checks,seconds=time.time()-t),indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
