"""Read-only registered mesh bundle; explicit frame/pose checks and origin conversion."""
from pathlib import Path
import json,os,copy
import numpy as np

def readable(path):
 # Resolve each parent symlink too: many delivered directories link into /data.
 p=Path(path)
 for _ in range(64):
  old=os.environ.get('L3_OLD_PATH_PREFIX');new=os.environ.get('L3_NEW_PATH_PREFIX')
  if old and new and p.is_relative_to(Path(old)):p=Path(new)/p.relative_to(old)
  changed=False
  for ancestor in list(reversed(p.parents))+[p]:
   if ancestor.is_symlink():
    target=Path(os.readlink(ancestor));target=target if target.is_absolute() else ancestor.parent/target
    p=target/p.relative_to(ancestor);changed=True;break
  if changed:continue
  if p.is_file():return p
  raise FileNotFoundError(p)
 raise ValueError('Symlink cycle: '+str(path))

CAMERA_NAMES={0:'left_front_camera',1:'right_front_camera',2:'rear_camera',3:'left_rear_camera',4:'right_rear_camera',9:'front_camera_fov30',10:'front_camera_fov120'}
def verify_camera_authority(views,pose_csv,*,atol=1e-8):
 import csv
 rows={}
 with readable(pose_csv).open() as handle:csv_rows=list(csv.DictReader(handle))
 for r in csv_rows:
  key=(int(r['frame']),r['camera'])
  if key in rows:raise ValueError('Duplicate authoritative camera: '+str(key))
  rows[key]=np.array([float(r['T_cw_%d%d'%(i,j)]) for i in range(4) for j in range(4)]).reshape(4,4)
 errors=[];missing=[]
 for v in views:
  key=(v['frame'],CAMERA_NAMES[v['cam']])
  if key not in rows:missing.append(v['name']);continue
  pose=np.asarray(v['c2w']);authority=rows[key]
  if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.isfinite(authority).all():raise ValueError('Invalid camera pose')
  error=float(abs(pose-authority).max());errors.append(error)
  if error>atol:raise ValueError('Camera differs from explicit authority: '+v['name']+' max='+str(error))
 if missing:raise ValueError('Selected views absent from explicit camera authority: '+str(missing))
 return dict(path=str(readable(pose_csv)),views=len(errors),max_absolute_error=max(errors,default=0.),tolerance=atol)

def bundle(source,raw_source=None,frames=None,cameras=None,*,pose_csv=None):
 root=Path(source);m=json.loads(readable(root/'input_manifest.json').read_text());actors=json.loads(readable(root/'actors.json').read_text())
 views=[v for v in m['views'] if (frames is None or v['frame'] in frames) and (cameras is None or v['cam'] in cameras)]
 if not views or not actors:raise ValueError('Mesh bundle has no selected views/actors')
 origin=np.asarray(m['world_origin'],float);assert origin.shape==(3,) and np.isfinite(origin).all()
 if pose_csv is None:raise ValueError('Explicit paired optimized pose CSV is required; nearby raw poses are not sufficient')
 authority=verify_camera_authority(views,pose_csv)
 files=[readable(root/'input_manifest.json'),readable(root/'actors.json'),readable(pose_csv)]
 for v in views:
  pose=np.asarray(v['c2w']);K=np.asarray(v['K'])
  if pose.shape!=(4,4) or K.shape!=(3,3) or not np.isfinite(pose).all() or not np.isfinite(K).all():raise ValueError('Invalid mesh camera')
  for folder,ext in [('images','.jpg'),('valid_mask','.png'),('target_mask','.png'),('lidar_depth','.npy'),('mesh_depth','.npz')]:files.append(readable(root/'converted'/folder/(v.get('image_file',v['name']+ext) if folder=='images' else v['name']+ext)))
 selected={v['frame'] for v in views};kept=[]
 for actor in actors:
  ids=[i for i,f in enumerate(actor['frames']) if f in selected]
  if not ids:continue
  if len(set(actor['frames']))!=len(actor['frames']):raise ValueError('Duplicate actor frame')
  for key in ['centers','rotations','transforms']:
   if len(actor[key])!=len(actor['frames']):raise ValueError('Actor timeline shape mismatch')
  size=np.asarray(actor['size']);T=np.asarray(actor['transforms'])
  if size.shape!=(3,) or np.any(size<=0) or not np.isfinite(T).all() or T.shape[1:]!=(4,4):raise ValueError('Invalid actor transform')
  if not np.allclose(T[:,:3,3],actor['centers']):raise ValueError('Actor centers differ from transform translations')
  files.append(readable(root/'model/input_ply'/('points3D_obj_%03d.ply'%actor['id'])))
  kept.append(actor)
 if not kept:raise ValueError('No mesh actors overlap selected frames')
 return dict(root=root,manifest=m,views=views,actors=kept,origin=origin,files=files,audit=dict(actors=len(kept),views=len(views),camera_authority=authority,source=str(root)))

def shifted_actors(actors,old_origin,new_origin,frames):
 shift=np.asarray(old_origin)-np.asarray(new_origin);out=[]
 for original in actors:
  a=copy.deepcopy(original);ids=[i for i,f in enumerate(a['frames']) if f in frames]
  if not ids:continue
  for key in ['frames','centers','rotations','sizes','transforms','scale_ratios']:
   if key in a and isinstance(a[key],list) and len(a[key])==len(original['frames']):a[key]=[a[key][i] for i in ids]
  T=np.asarray(a['transforms'],float);T[:,:3,3]+=shift;a['transforms']=T.tolist();a['centers']=(np.asarray(a['centers'])+shift).tolist()
  # This conversion must preserve world-space locations and all affine linear parts.
  assert np.allclose(T[:,:3,3]+new_origin,np.asarray(original['transforms'])[ids,:3,3]+old_origin)
  assert np.array_equal(T[:,:3,:3],np.asarray(original['transforms'])[ids,:3,:3])
  if 'view_transforms' in a:
   shifted={}
   for name,matrix in a['view_transforms'].items():
    if int(name.split('_')[0]) not in frames:continue
    V=np.asarray(matrix,float).copy();V[:3,3]+=shift;shifted[name]=V.tolist()
    assert np.allclose(V[:3,3]+new_origin,np.asarray(matrix)[:3,3]+old_origin)
   a['view_transforms']=shifted
  out.append(a)
 return out
