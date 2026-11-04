"""Restore native actors/point clouds, replace only cars, and veto occluded mesh depth."""
from pathlib import Path
import os,sys,json,shutil,hashlib
ROOT=Path(sys.argv[1]);case=json.loads((ROOT/'case.json').read_text());BASE=Path(case['baseline']);OLD=Path(case['mesh_reference'])
# Config parsing validates the source directory before preparation creates its contents.
(ROOT/'converted').mkdir(exist_ok=True)
REPO=Path(os.environ['STREETGS_ROOT']);os.chdir(REPO);os.environ['PWD']=str(REPO);sys.path.insert(0,str(REPO));sys.argv=['prepare','--config',str(ROOT/'train.yaml')]
import numpy as np,cv2,torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
from lib.config import cfg
from lib.utils.m18_utils import get_obj_pose_tracking,_label2camera
from lib.utils.general_utils import quaternion_slerp
cv2.setNumThreads(1);torch.set_num_threads(2)
DATA=ROOT/'converted';OUT=ROOT/'model/input_ply';W,H=1600,900

def writejson(p,x):
 p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else a))
def loaddepth(p):
 a=np.load(p,allow_pickle=True).item();z=np.zeros((H,W),'f4');z[a['mask']]=a['value'];return z

def union_inside(points,boxes,tree=None):
 """Box membership in the common centered world frame; never a 2D crop."""
 tree=tree or cKDTree(points);mask=np.zeros(len(points),bool)
 for center,rot,half in boxes:
  idx=np.asarray(tree.query_ball_point(center,float(np.linalg.norm(half))),dtype=int)
  if len(idx):mask[idx[np.all(np.abs((points[idx]-center)@rot)<=half+1e-6,axis=1)]]=True
 return mask

corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
def foreground_boxes(mesh,E,K,T,boxes):
 """Ray/3D OBB intersection vetoes depth in front of, or ambiguous with, a mesh."""
 out=np.zeros((H,W),bool);ik=np.linalg.inv(K)
 for c,R,half in boxes:
  cp=(corners*half@R.T+c)@E[:3,:3].T+E[:3,3]
  if cp[:,2].max()<=.1:continue
  if cp[:,2].min()<=.1:lo=np.array([0,0]);hi=np.array([W,H])
  else:
   uv=cp@K.T;uv=uv[:,:2]/uv[:,2:];lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+2,[W,H])
  if np.any(hi<=lo):continue
  sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]];yy,xx=np.where((mesh[sl]>0)&~out[sl])
  if not len(xx):continue
  x=xx+lo[0];y=yy+lo[1];dirs=np.c_[x,y,np.ones(len(x))]@ik.T@T[:3,:3].T@R;org=(T[:3,3]-c)@R
  parallel=np.abs(dirs)<1e-10;safe=np.where(parallel,1.,dirs)
  a=(-half-org)/safe;b=(half-org)/safe
  near=np.where(parallel,-np.inf,np.minimum(a,b));far=np.where(parallel,np.inf,np.maximum(a,b))
  hit=(far.min(1)>=np.maximum(near.max(1),.1))&~np.any(parallel&(np.abs(org)>half),axis=1)
  hit&=near.max(1)<=mesh[y,x]+.3
  out[y[hit],x[hit]]=True
 return out

# Geometric regression checks: a nearer nonvehicle box must occlude; a box behind must not.
def selfcheck():
 p=np.array([[0,0,3],[0,0,6],[2,0,3]],float);b=[(np.array([0,0,3]),np.eye(3),np.ones(3)*.5)]
 assert union_inside(p,b).tolist()==[True,False,False]
 m=np.zeros((H,W),'f4');m[450,800]=5;k=np.array([[500,0,800],[0,500,450],[0,0,1.]])
 assert foreground_boxes(m,np.eye(4),k,np.eye(4),b)[450,800]
 assert not foreground_boxes(m,np.eye(4),k,np.eye(4),[(np.array([0,0,8]),np.eye(3),np.ones(3)*.5)])[450,800]
selfcheck();assert not (ROOT/'preparation.json').exists()
DATA.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
for d in ['images','sky_mask']:
 if not (DATA/d).exists():(DATA/d).symlink_to(BASE/'converted'/d,target_is_directory=True)
# Reuse the prepared fine ego masks below.

for name in ['valid_mask','fixed_mask']:
 if not (DATA/name).exists():(DATA/name).symlink_to(BASE/'converted'/name,target_is_directory=True)
for d in ['lidar_depth','mesh_depth','target_mask','foreground_protection']:(DATA/d).mkdir(exist_ok=True)
manifest=json.loads((OLD/'input_manifest.json').read_text());origin=np.array(manifest['world_origin']);views=manifest['views']
egos=np.stack([np.loadtxt(BASE/'converted/ego_pose'/f'{i:06d}.txt') for i in range(101)]);egos[:,:3,3]-=origin
world,vehicle,meta=get_obj_pose_tracking(str(BASE/'converted'),[0,100],egos,[0,1,2,3,4,9,10])
expected={int(k[4:]) for k in json.loads((BASE/'reader_validation.json').read_text())['point_counts'] if k.startswith('obj_')}
assert set(meta)==expected,(set(meta)^expected)
actors=json.loads((OLD/'actors.json').read_text());mesh_ids={a['id'] for a in actors};restored=sorted(expected-mesh_ids)
for a in actors:
 a['source']='mesh';shutil.copy2(OLD/'model/input_ply'/f"points3D_obj_{a['id']:03d}.ply",OUT/f"points3D_obj_{a['id']:03d}.ply")
ts=json.loads((BASE/'converted/timestamps.json').read_text());times=np.array([ts['FRAME'][f'{i:06d}'] for i in range(101)]);offset=min(min(v.values()) for v in ts.values());times-=offset
native_checks=[]
for tid in restored:
 idx=np.argwhere(vehicle[:,:,0]==tid);frames=idx[:,0];rows=vehicle[idx[:,0],idx[:,1]];info=meta[tid]
 st=max(float(times[frames.min()]-.1),0);et=min(float(times[frames.max()]+.1),max(max(v.values()) for v in ts.values())-offset)
 size=np.array([info[k] for k in ['length','width','height']])/cfg.data.box_scale
 def interpolate(t):
  i,j=np.argsort(np.abs(times[frames]-t))[:2];ratio=(t-times[frames[i]])/(times[frames[j]]-times[frames[i]])
  pos=rows[i,1:4]*(1-ratio)+rows[j,1:4]*ratio
  q=quaternion_slerp(torch.tensor(rows[i,4:8],dtype=torch.float32),torch.tensor(rows[j,4:8],dtype=torch.float32),ratio).numpy()
  return pos,Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
 assert len(frames)>1
 a=dict(id=tid,type=info['class'],source='native_lidar',native_meta=info,size=size.tolist(),frames=[],centers=[],rotations=[],transforms=[],view_transforms={})
 render_frames=sorted({v['frame'] for v in views if st<=ts[_label2camera[v['cam']]][f"{v['frame']:06d}"]-offset<=et})
 for fr in render_frames:
  pos,rot=interpolate(times[fr]);R=egos[fr,:3,:3]@rot;c=egos[fr,:3,:3]@pos+egos[fr,:3,3];M=np.eye(4);M[:3,:3]=R*size;M[:3,3]=c
  a['frames'].append(fr);a['centers'].append(c.tolist());a['rotations'].append(R.tolist());a['transforms'].append(M.tolist())
 for v in views:
  fr=v['frame'];t=ts[_label2camera[v['cam']]][f'{fr:06d}']-offset
  if not st<=t<=et:continue
  pos,rot=interpolate(t);M=np.eye(4);M[:3,:3]=(egos[fr,:3,:3]@rot)*size;M[:3,3]=egos[fr,:3,:3]@pos+egos[fr,:3,3];a['view_transforms'][v['name']]=M.tolist()
  # Same ego-space interpolation as native ActorPose, then ego-to-world.
  assert np.isfinite(M).all() and np.allclose((M[:3,:3]/size).T@(M[:3,:3]/size),np.eye(3),atol=1e-5)
 src=BASE/'model/input_ply'/f'points3D_obj_{tid:03d}.ply';dst=OUT/src.name;assert src.exists();shutil.copy2(src,dst)
 assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest()
 native_checks.append(dict(id=tid,kind=info['class'],original_frames=frames.tolist(),render_views=len(a['view_transforms']),initial_ply_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),points=len(PlyData.read(str(dst))['vertex'])))
 actors.append(a)
assert {a['id'] for a in actors}==expected|mesh_ids
writejson(ROOT/'actors.json',actors);print('ACTORS_RESTORED',len(restored),'TOTAL',len(actors),flush=True)
# All annotations protect non-mesh objects, including static targets the native reader keeps in background.
byframe={f:[] for f in range(101)};carboxes={f:[] for f in range(101)};nativeboxes={f:[] for f in range(101)}
for line in (BASE/'converted/track/track_info.txt').read_text().splitlines()[1:]:
 x=line.split();fr=int(x[0]);tid=int(x[1]);c=np.array(x[7:10],float);R=Rotation.from_euler('z',float(x[10])).as_matrix();dims=np.array([x[6],x[5],x[4]],float);box=(egos[fr,:3,:3]@c+egos[fr,:3,3],egos[fr,:3,:3]@R,dims/2+.10)
 if tid in mesh_ids:carboxes[fr].append(box)
 else:byframe[fr].append(box)
 if tid in restored:
  info=meta[tid];nativeboxes[fr].append((box[0],box[1],np.array([info[k] for k in ['length','width','height']])/2))
for a in actors:
 if a['source']=='mesh':
  pose_entries=list(zip(a['frames'],a['transforms']))+[(int(k.split('_')[0]),v) for k,v in a.get('view_transforms',{}).items()]
  for fr,T in pose_entries:
   T=np.array(T);sz=np.linalg.norm(T[:3,:3],axis=0);carboxes[fr].append((T[:3,3],T[:3,:3]/sz,sz/2+.15))
# Reuse native background (already excludes native moving actors) plus the exact same far DVP.
ply=PlyData.read(str(BASE/'model/input_ply/points3D_bkgd.ply'));v=ply['vertex'].data;xyz=np.c_[v['x'],v['y'],v['z']];tree=cKDTree(xyz);remove=np.zeros(len(xyz),bool);protect=np.zeros(len(xyz),bool)
for fr in range(101):
 remove|=union_inside(xyz,carboxes[fr],tree);protect|=union_inside(xyz,byframe[fr],tree)
keep=~remove|protect
PlyData([PlyElement.describe(v[keep].copy(),'vertex')],text=False).write(str(OUT/'points3D_bkgd.ply'))
# Baseline merge is LiDAR prefix followed by far DVP; preserve the exact split.
nlidar=json.loads((BASE/'initialization_merge.json').read_text())['lidar_background_points']
lv=PlyData.read(str(BASE/'model/input_ply/points3D_lidar.ply'))['vertex'].data
assert np.array_equal(v[:nlidar],lv)
PlyData([PlyElement.describe(v[:nlidar][keep[:nlidar]].copy(),'vertex')],text=False).write(str(OUT/'points3D_lidar.ply'))
writejson(ROOT/'point_split.json',dict(lidar=int(keep[:nlidar].sum()),far=int(keep[nlidar:].sum()),total=int(keep.sum())))
# Dynamic non-mesh geometry is exclusively retained in its original object-local PLYs.
manifest['initial_points']=int(keep.sum());lo=xyz[keep].min(0);hi=xyz[keep].max(0);manifest['baseline_sphere']=dict(center=((lo+hi)/2).tolist(),radius=float(np.linalg.norm(hi-lo)/2))
manifest['initialization']='Native non-mesh actors and LiDAR + mesh cars + original far DVP';writejson(ROOT/'input_manifest.json',manifest)
print('BACKGROUND',len(v),int(keep.sum()),'PROTECTED',int((remove&protect).sum()),flush=True)
rows=[]
for v in views:
 fr=v['frame'];name=v['name'];K=np.array(v['K']);T=np.array(v['c2w']);T[:3,3]-=origin;E=np.linalg.inv(T)
 valid=cv2.imread(str(DATA/'valid_mask'/(name+'.png')),0)>0;sky=cv2.imread(str(DATA/'sky_mask'/(name+'.png')),0)>0
 ld0=loaddepth(BASE/'converted/lidar_depth'/(name+'.npy'));ld=ld0.copy();oldmd=np.load(OLD/'converted/mesh_depth'/(name+'.npz'))['depth'];md=oldmd.copy()
 y,x=np.where(ld0>0);cam=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(K).T*ld0[y,x,None];points=cam@T[:3,:3].T+T[:3,3];tree=cKDTree(points)
 car=union_inside(points,carboxes[fr],tree);other=union_inside(points,byframe[fr],tree)
 # Target identity outranks a car OBB: overlapping boxes must not delete the person.
 erase=car&~other;ld[y[erase],x[erase]]=0
 protected=foreground_boxes(md,E,K,T,byframe[fr])
 protected[y[other],x[other]]=True
 measured=cv2.erode(np.where(ld>0,ld,1e6),np.ones((3,3),'u1'))
 near=(md>0)&(measured<md-.3);protected|=near
 md[protected|~valid|sky]=0;ld[(md>0)|~valid|sky]=0
 # Every non-mesh target's valid measured depth survives bit-for-bit.
 keep_other=other&valid[y,x]&~sky[y,x]
 assert np.array_equal(ld[y[keep_other],x[keep_other]],ld0[y[keep_other],x[keep_other]])
 assert not np.any((md>0)&(protected|(ld>0)|~valid|sky))
 np.save(DATA/'lidar_depth'/(name+'.npy'),dict(mask=ld>0,value=ld[ld>0]));np.savez_compressed(DATA/'mesh_depth'/(name+'.npz'),depth=md)
 cv2.imwrite(str(DATA/'target_mask'/(name+'.png')),(md>0).astype('u1')*255);cv2.imwrite(str(DATA/'foreground_protection'/(name+'.png')),protected.astype('u1')*255)
 rows.append(dict(image=name,original_lidar=int((ld0>0).sum()),retained_lidar=int((ld>0).sum()),nonmesh_depth_pixels=int(keep_other.sum()),nonmesh_depth_preserved=True,removed_car_depth=int(erase.sum()),old_mesh_pixels=int((oldmd>0).sum()),mesh_pixels=int((md>0).sum()),mesh_pixels_vetoed=int(((oldmd>0)&protected).sum()),foreground_mesh_conflicts=0))
 if len(rows)%7==0:print('DEPTH_FRAME',fr,'DONE',len(rows),flush=True)
report=dict(status='PASS',images=len(rows),actors=len(actors),mesh_actors=len(mesh_ids),restored_actors=len(restored),baseline_actor_ids=sorted(expected),mesh_actor_ids=sorted(mesh_ids),restored_actor_ids=restored,native_actors=native_checks,background_before=len(xyz),background_after=int(keep.sum()),background_car_points_removed=int((~keep).sum()),background_nonmesh_overlap_preserved=int((remove&protect).sum()),depth_audit=rows,nonmesh_original_ply_identical=True,all_gt_rgb_preserved=True,dynamic_mask=False,geometry_selfcheck='PASS')
report['preserve_nonmesh_tracks']=True
alltracks={}
for line in (BASE/'converted/track/track_info.txt').read_text().splitlines()[1:]:
 x=line.split();alltracks.setdefault(int(x[1]),dict(type=x[2],rows=[]))['rows'].append(line)
nonmesh={k:v for k,v in alltracks.items() if k not in mesh_ids}
report['nonmesh_track_ids']=sorted(nonmesh)
report['nonmesh_track_rows']=sum(len(v['rows']) for v in nonmesh.values())
report['native_actors_restored_ids']=restored
report['static_tracks_retained_in_background']=sorted(set(nonmesh)-set(restored))
(DATA/'track').symlink_to(BASE/'converted/track',target_is_directory=True)
assert (DATA/'track/track_info.txt').read_bytes()==(BASE/'converted/track/track_info.txt').read_bytes()
writejson(ROOT/'nonmesh_tracks.json',nonmesh)
writejson(ROOT/'preparation.json',report);print('PREPARATION_COMPLETE',len(rows),len(actors),flush=True)
