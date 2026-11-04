from pathlib import Path
import sys,json,time,collections,shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from PIL import Image,ImageDraw
from plyfile import PlyData,PlyElement
import open3d as o3d
from common import CFG,S,ROOT,O,REPO,selected_frames
sys.path.insert(0,__import__('os').environ.get('GROUND_PREPROCESS_DEPS',str(REPO/'deps/ground_preprocess')))
import pypatchworkpp as pw
trial=CFG['frames'] is not None
O.mkdir(exist_ok=False)
for d in ['per_frame_world','point_indices','previews']:(O/d).mkdir()
shutil.copy2(__file__,O/'extract.py');start=time.time()
(O/'status.json').write_text(json.dumps(dict(stage='loading',source=str(S))))
z=np.load(S/'pointcloud.npz',allow_pickle=True);pc=z['pointcloud'].item();projection=z['camera_projection'].item()
frames=selected_frames(pc);sample_frames=[frames[0],frames[len(frames)//2],frames[-1]]
DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('frame_id','<u2'),('point_id','<u4')])
def write(path,a):PlyData([PlyElement.describe(a,'vertex')]).write(path)
def plane_fit(p):
 r=np.linalg.norm(p[:,:2],axis=1);near=p[(r>2.7)&(r<12)&(abs(p[:,0])<5)&(abs(p[:,2])<.6)]
 cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near));cloud=cloud.voxel_down_sample(.05)
 assert len(cloud.points)>100,'Insufficient near-ground support'
 o3d.utility.random.seed(42);plane,idx=cloud.segment_plane(.05,3,1000);plane=np.asarray(plane)
 if plane[2]<0:plane=-plane
 ratio=len(idx)/len(cloud.points)
 assert plane[2]>CFG['normal_z_min'] and ratio>.4,(plane,ratio)
 return plane,ratio
def connected(q,plane):
 keys,iv=np.unique(np.floor(q[:,:2]/2).astype(int),axis=0,return_inverse=True);order=np.argsort(iv);groups=np.split(order,np.flatnonzero(np.diff(iv[order]))+1)
 med=np.array([np.median(q[g,2]) for g in groups]);counts=np.array([len(g) for g in groups]);lookup={tuple(k):i for i,k in enumerate(keys)}
 center=(keys+.5)*2;zg=-(plane[0]*center[:,0]+plane[1]*center[:,1]+plane[3])/plane[2]
 seeds=np.flatnonzero((np.linalg.norm(center,axis=1)<10)&(abs(med-zg)<.18)&(counts>=6));visited=set(seeds.tolist());queue=collections.deque(seeds.tolist())
 while queue:
  i=queue.popleft()
  for dx,dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
   j=lookup.get((keys[i,0]+dx,keys[i,1]+dy))
   if j is not None and j not in visited and counts[j]>=6 and abs(med[i]-med[j])<.12+.3*np.hypot(dx,dy):visited.add(j);queue.append(j)
 keep=np.zeros(len(q),bool)
 for i in visited:
  g=groups[i];pts=q[g];seed=pts[abs(pts[:,2]-med[i])<.15]
  if len(seed)<6:continue
  mid=np.median(seed,axis=0);_,_,vt=np.linalg.svd(seed-mid,full_matrices=False);normal=vt[-1]
  if abs(normal[2])<CFG['normal_z_min']:continue
  keep[g]=abs((pts-mid)@normal)<.08
 return keep,len(visited)
parts=[];rows=[];raw_preview=[]
for fid in frames:
 p=pc[fid];r=np.linalg.norm(p[:,:2],axis=1);ids=np.flatnonzero(np.isfinite(p).all(1)&(r>=2.7))
 _,ix=np.unique(np.floor(p[ids]/.05).astype('int64'),axis=0,return_index=True);ids=ids[ix];q=p[ids]
 d,_=cKDTree(q).query(q,k=4,workers=CFG['workers']);ids=ids[d[:,-1]<(.12+.004*np.linalg.norm(q[:,:2],axis=1))];q=p[ids]
 plane,ratio=plane_fit(q)
 params=pw.Parameters();params.sensor_height=float(2+plane[3]/plane[2]);params.enable_RNR=False;params.enable_RVPF=True;params.enable_TGR=True;params.uprightness_thr=CFG['normal_z_min'];params.th_dist=.08;params.th_seeds=.15;params.min_range=2.7;params.max_range=float(np.ceil(r[ids].max())+1);params.verbose=False
 # Fused input is ego-frame, not a single physical lidar. Classifier-only virtual height.
 engine=pw.patchworkpp(params);shift=q.copy();shift[:,2]-=2;engine.estimateGround(np.c_[shift,np.ones(len(q))].astype('f4'))
 gi=np.asarray(engine.getGroundIndices()).reshape(-1).astype(int);assert len(gi)>0 and gi.min()>=0 and gi.max()<len(q)
 candidates=ids[gi];good,ncells=connected(p[candidates],plane);selected=candidates[good];assert len(selected)>1000,(fid,len(selected))
 E=np.loadtxt(S/f'ego_pose/{fid:06d}.txt');assert E.shape==(4,4) and np.allclose(E[3],[0,0,0,1])
 world=p[selected].astype('f8')@E[:3,:3].T+E[:3,3];a=np.empty(len(selected),DT)
 for i,k in enumerate('xyz'):a[k]=world[:,i]
 for k in ['red','green','blue']:a[k]=160
 a['frame_id']=fid;a['point_id']=selected
 pr=projection[fid][selected];colored=np.zeros(len(selected),bool)
 for path in sorted((S/'images').glob(f'{fid:06d}_*.jpg')):
  cam=int(path.stem.split('_')[1]);im=Image.open(path).convert('RGB');img=np.asarray(im);h,w=img.shape[:2];dots=np.zeros((h,w),bool) if fid in sample_frames and (CFG['cameras'] is None or cam in CFG['cameras']) else None
  for col in range(0,pr.shape[1],3):
   ii=np.flatnonzero(pr[:,col]==cam);uv=pr[ii,col+1:col+3].astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ii=ii[valid];uv=uv[valid]
   fresh=~colored[ii]
   for j,k in enumerate(['red','green','blue']):a[k][ii[fresh]]=img[uv[fresh,1],uv[fresh,0],j]
   colored[ii]=True
   if dots is not None:dots[uv[:,1],uv[:,0]]=True
  if dots is not None:
   sh=round(h*960/w);small=np.array(im.resize((960,sh)));yy,xx=np.where(dots);dmask=np.zeros((sh,960),bool);dmask[np.minimum((yy*sh/h).astype(int),sh-1),np.minimum((xx*960/w).astype(int),959)]=True;dmask=maximum_filter(dmask,size=2)
   small[dmask]=(small[dmask]*.3+np.array([0,255,60])*.7).astype('uint8');preview=Image.fromarray(small);ImageDraw.Draw(preview).text((8,8),path.stem+' | green: selected LiDAR ground',fill='white');preview.save(O/'previews'/(path.stem+'.jpg'),quality=92)
 write(O/'per_frame_world'/f'{fid:06d}.ply',a);parts.append(a)
 np.savez_compressed(O/'point_indices'/f'{fid:06d}.npz',ground_indices=selected,patchwork_candidate_indices=candidates,ego_to_world=E)
 row=dict(frame=fid,raw=len(p),voxel=len(ix),after_isolation=len(q),candidates=len(candidates),ground=len(a),near_plane=plane.tolist(),near_inlier_ratio=ratio,connected_cells=ncells,max_range=float(r[selected].max()),uncolored=int((~colored).sum()));rows.append(row);print(json.dumps(row),flush=True)
 (O/'status.json').write_text(json.dumps(dict(stage='extracting',frames_done=len(rows),frames_total=len(frames),last_frame=fid,seconds=time.time()-start)))
 if fid in sample_frames:raw_preview.append(p[::50]@E[:3,:3].T+E[:3,3])
a=np.concatenate(parts);xyz=np.column_stack([a[k] for k in 'xyz']);before=len(a)
index=np.unique(np.floor(xyz/.05).astype('int64'),axis=0,return_index=True)[1];a=a[np.sort(index)];write(O/'ground_world_5cm.ply',a)
xyz=np.column_stack([a[k] for k in 'xyz']).astype('f8');E0=np.loadtxt(S/f'ego_pose/{min(frames):06d}.txt');local=(xyz-E0[:3,3])@np.linalg.inv(E0[:3,:3]).T
b=a.copy()
for j,k in enumerate('xyz'):b[k]=local[:,j]
write(O/'ground_first_ego_5cm.ply',b)
# Check random output representatives against original raw point IDs and source poses.
rng=np.random.default_rng(42);error=0
for fid in frames:
 ids=np.flatnonzero(a['frame_id']==fid)
 if not len(ids):continue
 ids=rng.choice(ids,min(100,len(ids)),replace=False);E=np.loadtxt(S/f'ego_pose/{fid:06d}.txt');expected=pc[fid][a['point_id'][ids]].astype('f8')@E[:3,:3].T+E[:3,3];error=max(error,float(abs(expected-xyz[ids]).max()))
assert error<.0001 and np.isfinite(xyz).all()
report=dict(source=str(S),pipeline_config=CFG,method='Patchwork++ + adaptive isolation + connected local ground planes',frames=len(rows),trial=trial,points_before_merge=before,points=len(a),merge_voxel_m=.05,world_bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],first_ego_to_world=E0.tolist(),max_raw_xyz_verification_error_m=error,semantics_used=False,training=False,virtual_origin_shift_only_for_classifier_m=2,near_plane_estimated_per_frame=True,upper_distance_cutoff=None,limitations='Conservative connected ground: may omit sparse/disconnected ground, high curbs and slopes; no labeled accuracy. RGB from stored camera projections, not visibility-revalidated.',rows=rows,seconds=time.time()-start)
(O/'report.json').write_text(json.dumps(report,indent=2))
(O/'status.json').write_text(json.dumps(dict(stage='complete',frames=len(rows),points=len(a),seconds=time.time()-start),indent=2));print('COMPLETE',len(a),flush=True)
