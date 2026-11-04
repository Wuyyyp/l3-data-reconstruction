"""Fresh M18 preprocessing: empirical ego-return filtering and geometric visibility.
The ego template is vehicle/sample-specific, never inferred as a universal body box.
Source files remain read-only. A new training run must use a new output directory.
"""
from pathlib import Path
import numpy as np,json,time,hashlib,itertools
from scipy.spatial import cKDTree
from PIL import Image
import cv2

def ego_to_camera(path,convention):
 e=np.loadtxt(path)
 return e if convention=='ego_to_camera' else np.linalg.inv(e)

def prepare(source,output,template,extrinsics):
 S=Path(source).resolve();O=Path(output).resolve();template=Path(template).resolve() if template else None
 assert extrinsics in ('ego_to_camera','camera_to_ego')
 assert S.is_dir() and (template is None or template.is_file())
 assert not O.exists(), 'Preprocessing output exists: only same-run resume/evaluation may reuse it'
 O.mkdir(parents=True);D=O/'dataset';D.mkdir()
 for p in S.iterdir():
  if p.name in ('pointcloud.npz','lidar_depth','extrinsics','track'):continue
  (D/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
 (D/'extrinsics').mkdir();(D/'track').mkdir()
 for p in (S/'extrinsics').glob('*.txt'):
  np.savetxt(D/'extrinsics'/p.name,np.linalg.inv(ego_to_camera(p,extrinsics)))
 for p in (S/'track').iterdir():
  if p.name!='track_camera_vis.json':(D/'track'/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
 if template is not None:
  t=np.load(template);seed=t['seed_ego'];active=t['active_cells'];tree=cKDTree(seed)
 protect_boxes=bool(t['protect_track_boxes']) if template is not None and 'protect_track_boxes' in t else False
 cell_only=bool(t['cell_only']) if template is not None and 'cell_only' in t else False
 radius=float(t['roi_radius']) if template is not None and 'roi_radius' in t else 5.
 z_min=float(t['roi_z_min']) if template is not None and 'roi_z_min' in t else .8
 z_max=float(t['roi_z_max']) if template is not None and 'roi_z_max' in t else 2.3
 boxes={}
 if protect_boxes:
  for line in (S/'track/track_info.txt').read_text().splitlines()[1:]:
   row=line.split();boxes.setdefault(int(row[0]),[]).append(row)
 def protected(e,frame):
  keep=np.zeros(len(e),bool)
  for row in boxes.get(int(frame),[]):
   h,w,l=map(float,row[4:7]);a=float(row[10]);c,s=np.cos(a),np.sin(a)
   q=(e-np.array(row[7:10],float))@np.array([[c,-s,0],[s,c,0],[0,0,1]])
   keep|=(np.abs(q)<=np.array([l,w,h])/2+.15).all(1)
  return keep
 def keys(e):
  v=np.floor(e/.1).astype('i8');return (v[:,0]+100)*1000000+(v[:,1]+100)*1000+(v[:,2]+100)
 def classify(e,frame):
  if template is None:return np.zeros(len(e),bool)
  out=np.zeros(len(e),bool)
  ids=np.flatnonzero(np.isfinite(e).all(1)&(np.linalg.norm(e[:,:2],axis=1)<radius)&(e[:,2]>z_min)&(e[:,2]<z_max))
  if not cell_only:
   dist,_=tree.query(e[ids],distance_upper_bound=.06,workers=4);ids=ids[np.isfinite(dist)]
  ids=ids[np.isin(keys(e[ids]),active)]
  if protect_boxes:ids=ids[~protected(e[ids],frame)]
  out[ids]=True
  return out
 start=time.time();data=np.load(S/'pointcloud.npz',allow_pickle=True);pc=data['pointcloud'].item();proj=data['camera_projection'].item();rows=[];removed_xyz={}
 for f in sorted(pc):
  p=pc[f];assert len(proj[f])==len(p);bad=classify(p,f);removed_xyz[f]=p[bad].copy();pc[f]=p[~bad];proj[f]=proj[f][~bad];assert not classify(pc[f],f).any();rows.append(dict(frame=int(f),input=len(p),removed=int(bad.sum()),kept=len(pc[f])))
 np.savez(D/'pointcloud.npz',pointcloud=pc,camera_projection=proj);print('POINTS',sum(r['removed'] for r in rows),flush=True)
 (D/'lidar_depth').mkdir(exist_ok=True);depthrows=[]
 for im in sorted((S/'images').glob('*.jpg')):
  name=im.stem;f,cam=map(int,name.split('_'));src=S/'lidar_depth'/(name+'.npy');d=np.load(src,allow_pickle=True).item();mask=d['mask'];values=d['value'];v,u=np.where(mask);assert len(v)==len(values)
  E=ego_to_camera(S/'extrinsics'/(name+'.txt'),extrinsics);K=np.loadtxt(S/'intrinsics'/(name+'.txt'));rays=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(K).T;camera=rays*values[:,None];ego=(camera-E[:3,3])@E[:3,:3];bad=classify(ego,f)
  # Also delete depths matching projections of removed original points, including pixel quantization boundaries.
  p=removed_xyz[f];q=p@E[:3,:3].T+E[:3,3];uv=q@K.T
  with np.errstate(divide='ignore',invalid='ignore'):uv=uv[:,:2]/uv[:,2:]
  inside=(q[:,2]>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0]);uv=uv[inside].astype(int);z=q[inside,2];pix=uv[:,1]*mask.shape[1]+uv[:,0];origpix=v*mask.shape[1]+u;pos=np.searchsorted(origpix,pix);ok=pos<len(origpix);ok[ok]&=origpix[pos[ok]]==pix[ok];sel=np.flatnonzero(ok);match=np.abs(values[pos[sel]]-z[sel])<=.08;bad[pos[sel[match]]]=True
  if protect_boxes:bad &= ~protected(ego,f)
  newmask=mask.copy();newmask[v[bad],u[bad]]=False;out=dict(d);out['mask']=newmask;out['value']=values[~bad];assert np.array_equal(out['value'],values[~bad]);assert not classify(ego[~bad],f).any();np.save(D/'lidar_depth'/(name+'.npy'),out);depthrows.append(dict(image=name,input=len(values),removed=int(bad.sum()),kept=int((~bad).sum())))
  if len(depthrows)%100==0:print('DEPTH',len(depthrows),sum(r['removed'] for r in depthrows),flush=True)
 assert len(depthrows)==len(list((S/'images').glob('*.jpg'))) and depthrows
 info=dict(source=str(S),dataset=str(D),pointcloud=rows,depth=depthrows,removed_points=sum(r['removed'] for r in rows),removed_depth_samples=sum(r['removed'] for r in depthrows),template=str(template),ego_filter_enabled=template is not None,protect_track_boxes=protect_boxes,cell_only=cell_only,method=('No ego filter: no validated template for this sample' if template is None else 'Empirical recurrent 10cm cells; 6cm seed distance unless cell_only; optional all-track-box protection with 15cm margin; matching depth within 8cm'),fresh_training=True,seconds=time.time()-start)
 (O/'cleaning_audit.json').write_text(json.dumps(info,indent=2));print('CLEAN_DONE',info['removed_points'],info['removed_depth_samples'],flush=True)

 sg=np.array(list(itertools.product([-1,1],repeat=3)))
 edges=[(i,j) for i in range(8) for j in range(i) if (sg[i]!=sg[j]).sum()==1]
 def area(p,E,K,w,h):
  q=p@E[:3,:3].T+E[:3,3];pts=list(q[q[:,2]>=.1])
  for i,j in edges:
   a,b=q[i],q[j]
   if (a[2]>=.1)!=(b[2]>=.1):pts.append(a+(.1-a[2])/(b[2]-a[2])*(b-a))
  if len(pts)<3:return 0.
  z=np.array(pts)@K.T;uv=(z[:,:2]/z[:,2:]).astype('f4');hull=cv2.convexHull(uv)
  return float(cv2.intersectConvexConvex(hull,np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype='f4'))[0])
 cal={}
 for im in sorted((S/'images').glob('*.jpg')):
  f,c=map(int,im.stem.split('_'))
  with Image.open(im) as image:w,h=image.size
  cal[f,c]=(ego_to_camera(S/'extrinsics'/(im.stem+'.txt'),extrinsics),np.loadtxt(S/'intrinsics'/(im.stem+'.txt')),w,h)
 old=json.loads((S/'track/track_camera_vis.json').read_text());new={};changes=[]
 rows=[r.split() for r in (S/'track/track_info.txt').read_text().splitlines()[1:]]
 for r in rows:
  f=int(r[0]);t=r[1];h,w,l=map(float,r[4:7]);yaw=float(r[10]);c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);p=(sg*np.array([l,w,h])/2)@rot.T+np.array(r[7:10],float)
  vis=[cam for ff,cam in sorted(cal) if ff==f and area(p,*cal[ff,cam])>1.]
  new.setdefault(t,{})[str(f)]=vis
  if old.get(t,{}).get(str(f),[])!=vis:changes.append(dict(track=t,frame=f,old=old.get(t,{}).get(str(f),[]),new=vis))
 (D/'track/track_camera_vis.json').write_text(json.dumps(new,indent=2))
 audit=dict(source=str(S),tracks=len(new),rows=len(rows),changed_rows=len(changes),changes=changes,source_extrinsics=extrinsics,output_extrinsics='camera_to_ego',method='Near-plane clipped 3D-box silhouette intersects image >1 pixel squared; geometric FOV only, no occlusion claim',template_sha256=hashlib.sha256(template.read_bytes()).hexdigest() if template is not None else None)
 (O/'visibility_audit.json').write_text(json.dumps(audit,indent=2));(O/'complete.json').write_text(json.dumps(dict(dataset=str(D),images=len(cal),rows=len(rows))))
 return D
