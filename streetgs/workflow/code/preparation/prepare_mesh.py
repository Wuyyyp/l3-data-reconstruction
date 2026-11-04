"""Use supplied glTF placements, remove vehicle LiDAR per frame, and preserve GT supervision."""
from pathlib import Path
import sys,os,json,pickle,hashlib,copy
# Runtime dependencies come from the validated StreetGS environment.
import numpy as np,cv2,trimesh,open3d as o3d,yaml
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
cv2.setNumThreads(1)
ROOT=Path(sys.argv[1]);BASE=Path(json.loads((ROOT/'case.json').read_text())['baseline'])
PLUGIN=ROOT/'mesh_registration'
OLD=BASE;DATA=ROOT/'converted';W,H=1600,900
NAMES={0:'left_front_camera',1:'right_front_camera',2:'rear_camera',3:'left_rear_camera',4:'right_rear_camera',9:'front_camera_fov30',10:'front_camera_fov120'}
class U(pickle.Unpickler):
 def find_class(self,m,n):return super().find_class(m.replace('numpy._core','numpy.core'),n)
def saveply(path,x,c):
 a=np.zeros(len(x),dtype=[(k,'f4') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue']])
 for j,k in enumerate('xyz'):a[k]=x[:,j]
 for j,k in enumerate(['red','green','blue']):a[k]=np.clip(np.rint(c[:,j]*255),0,255)
 PlyData([PlyElement.describe(a,'vertex')],text=False).write(str(path))
def loadply(p):
 a=PlyData.read(str(p))['vertex'];return np.column_stack([a[k] for k in 'xyz']),np.column_stack([a[k] for k in ['red','green','blue']])/255.
def link(a,b):
 if not b.exists():b.symlink_to(a,target_is_directory=a.is_dir())
def samples(v,f,dims,n,seed):
 rng=np.random.default_rng(seed);tri=v[f];t=tri*dims;area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1);idx=rng.choice(len(f),n,p=area/area.sum());uv=rng.random((n,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];return (tri[idx]*np.c_[1-uv.sum(1),uv][:,:,None]).sum(1).astype('f4')
def observed(cam,fr):
 masks={};p=PLUGIN/'registration/mask-videos/masks'/NAMES[cam]/f'{fr:06d}.npz'
 if p.exists():
  with np.load(p) as a:
   for k in a.files:
    if k.endswith('_bbox'):
     tid=int(k.split('_')[1]);x,y,w,h=map(int,a[k]);v=np.zeros((H,W),bool);v[y:y+h,x:x+w]=np.unpackbits(a[k[:-5]+'_bits'],count=w*h).reshape(h,w)>0;masks[tid]=v
 return masks

def main():
 assert not (ROOT/'preparation.json').exists()
 for d in ['lidar_depth','mesh_depth','target_mask','vehicle_lidar_exclusion']: (DATA/d).mkdir(parents=True,exist_ok=True)
 for d in ['images','sky_mask']:link(OLD/'converted'/d,DATA/d)
 sys.path.insert(0,os.environ['STREETGS_ROOT'])
 # Reuse the prepared fine ego masks below.
 link(OLD/'converted/valid_mask',DATA/'valid_mask');link(OLD/'converted/fixed_mask',DATA/'fixed_mask')
 (ROOT/'model/input_ply').mkdir(parents=True,exist_ok=True);(ROOT/'meshes').mkdir(exist_ok=True);(ROOT/'alignment_preview').mkdir(exist_ok=True)
 infos=U((PLUGIN/'result_interpolation_optimized_first101_with_meshes.pkl').open('rb')).load()['infos'];assert len(infos)==101
 placements=json.loads((PLUGIN/'placements.json').read_text())['frames'];assert len(placements)==101;per_view=json.loads((PLUGIN/'view_transforms.json').read_text())
 manifest=json.loads((OLD/'input_manifest.json').read_text());views=manifest['views'];assert len(views)==707
 egos=np.stack([np.loadtxt(OLD/'converted/ego_pose'/f'{i:06d}.txt') for i in range(101)]);origin=egos[:,:3,3].mean(0)
 by={};frame_objects={}
 for fr,rec in enumerate(placements):
  assert rec['frame']==fr;frame_objects[fr]={}
  for o in rec['objects']:
   tid=o['track_id'];p=infos[fr]['mesh_poses_9dof'].get(str(tid),infos[fr]['mesh_poses_9dof'].get(tid));T=np.array(p['world_from_unit_mesh']);G=np.array(o['world_from_gltf_scene']);normal=np.linalg.inv(T)@G
   np.testing.assert_allclose(G,egos[fr]@np.array(o['source_ego_from_gltf_scene']),atol=1e-6)
   dims=np.linalg.norm(T[:3,:3],axis=0);R=T[:3,:3]/dims
   np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-5);assert np.linalg.det(R)>.999
   state=by.setdefault(tid,dict(id=tid,asset=o['asset_path'],normal=normal,frames=[],centers=[],rotations=[],sizes=[],transforms=[]))
   np.testing.assert_allclose(normal,state['normal'],atol=1e-5)
   state['frames'].append(fr);state['centers'].append((T[:3,3]-origin).tolist());state['rotations'].append(R.tolist());state['sizes'].append(dims.tolist());Tc=T.copy();Tc[:3,3]-=origin;state['transforms'].append(Tc.tolist());frame_objects[fr][tid]=Tc
 assert len(by)==len(json.loads((PLUGIN/'asset_matches.json').read_text())) and all(t>=0 for t in by)
 assets={};states={};actors=[]
 for tid,a in sorted(by.items()):
  key=a['asset']
  if key not in assets:
   scene=trimesh.load(key,force='scene',process=False);meshes=[]
   for node in scene.graph.nodes_geometry:
    transform,name=scene.graph[node];mesh=scene.geometry[name];v=np.asarray(mesh.vertices);v=v@transform[:3,:3].T+transform[:3,3];v=v@a['normal'][:3,:3].T+a['normal'][:3,3];meshes.append(trimesh.Trimesh(v,mesh.faces,process=False))
   mesh=trimesh.util.concatenate(meshes);v=np.asarray(mesh.vertices,dtype='f4');f=np.asarray(mesh.faces,dtype='u4');assert np.isfinite(v).all()
   np.testing.assert_allclose(v.min(0),[-.5]*3,atol=.01);np.testing.assert_allclose(v.max(0),[.5]*3,atol=.01)
   ray=o3d.t.geometry.RaycastingScene(nthreads=4);ray.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f));assets[key]=dict(v=v,f=f,ray=ray)
  asset=assets[key];size=np.median(a['sizes'],axis=0);pts=samples(asset['v'],asset['f'],size,40000,tid)
  states[tid]=dict(points=pts,rgb=np.full(pts.shape,.5,'f4'),quality=np.zeros(len(pts)),asset=asset)
  a.update(size=size.tolist(),type='Car',static=False,scale_ratios=(np.array(a['sizes'])/size).tolist());a.pop('normal');actors.append(a)
  a['view_transforms']={}
  for name,world_T in per_view[str(tid)].items():
   V=np.array(world_T);V[:3,3]-=origin;a['view_transforms'][name]=V.tolist()
  np.savez_compressed(ROOT/'meshes'/f'obj_{tid:03d}.npz',unit_points=pts,canonical_points=pts*size)
 print('ASSETS_READY',len(actors),len(assets),flush=True)
 clouds=np.load(OLD/'converted/pointcloud.npz',allow_pickle=True)['pointcloud'].item();bgx=[];bgc=[];audits=[];removed_total=0
 corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
 for fr in range(101):
  raw=np.asarray(clouds[fr],dtype='f4');good=np.isfinite(raw).all(1)&(np.linalg.norm(raw,axis=1)>1)
  removed=np.zeros(len(raw),bool)
  for o in infos[fr]['objects']:
   if int(o['id']) not in by:continue
   c=np.array(o['location']);sz=np.array(o['size']);rot=Rotation.from_euler('z',o['rotation'][2]).as_matrix();inside=np.all(np.abs((raw-c)@rot)<sz/2+.15,axis=1);removed|=inside
  world=raw@egos[fr,:3,:3].T+egos[fr,:3,3]-origin
  for tid,T in frame_objects[fr].items():
   q=(world-T[:3,3])@np.linalg.inv(T[:3,:3]).T;removed|=np.all(np.abs(q)<.5+.15/np.linalg.norm(T[:3,:3],axis=0),axis=1)
  removed_total+=int((good&removed).sum());xyz=world[good&~removed];rgb=np.full(xyz.shape,.5,'f4');quality=np.zeros(len(xyz),'f4')
  for view in [v for v in views if v['frame']==fr]:
   cam=view['cam'];name=view['name'];K=np.array(view['K']);T=np.array(view['c2w']);T[:3,3]-=origin;E=np.linalg.inv(T);valid=cv2.imread(str(DATA/'valid_mask'/(name+'.png')),0)>0;sky=cv2.imread(str(DATA/'sky_mask'/(name+'.png')),0)>0;gt=cv2.imread(str(DATA/'images'/(name+'.jpg')))
   current_objects={tid:np.array(per_view[str(tid)][name]) for tid in frame_objects[fr]}
   for V in current_objects.values():V[:3,3]-=origin
   nearest=np.full((H,W),np.inf,'f4');owner=np.full((H,W),-1,'i4')
   for tid,M in current_objects.items():
    box=(corners@M[:3,:3].T+M[:3,3])@E[:3,:3].T+E[:3,3]
    if box[:,2].max()<.1:continue
    if box[:,2].min()<.1:lo=np.array([0,0]);hi=np.array([W,H])
    else:
     uv=box@K.T;uv=uv[:,:2]/uv[:,2:];lo=np.maximum(np.floor(uv.min(0)).astype(int)-2,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+3,[W,H])
    if np.any(hi<=lo):continue
    x,y=np.meshgrid(np.arange(lo[0],hi[0]),np.arange(lo[1],hi[1]));inverse=np.linalg.inv(M[:3,:3]);dirs=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T@T[:3,:3].T@inverse.T;org=(T[:3,3]-M[:3,3])@inverse.T
    rays=np.concatenate([np.broadcast_to(org,dirs.shape),dirs],-1).astype('f4');z=states[tid]['asset']['ray'].cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy();z[z<.1]=np.inf
    sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]];take=z<nearest[sl];nearest[sl][take]=z[take];owner[sl][take]=tid
   cp=xyz@E[:3,:3].T+E[:3,3];z=cp[:,2];uv=cp@K.T;ij=np.rint(uv[:,:2]/np.maximum(z[:,None],.001)).astype(int);x,y=ij.T;ids=np.flatnonzero((z>.1)&(x>=0)&(x<W)&(y>=0)&(y<H));lidar=np.full(H*W,np.inf,'f4');np.minimum.at(lidar,y[ids]*W+x[ids],z[ids]);lidar=lidar.reshape(H,W);lidar[~np.isfinite(lidar)]=0
   masks=observed(cam,fr);sam=np.zeros((H,W),bool)
   for m in masks.values():sam|=m
   foreground=(lidar>0)&np.isfinite(nearest)&(lidar<nearest-.3)&~sam
   vehicle=(owner>=0)|sam;exclude=cv2.dilate(vehicle.astype('u1'),np.ones((3,3),'u1'))>0;exclude&=~foreground
   ld=np.where(valid&~sky&~exclude&(lidar>=.1)&(lidar<=200),lidar,0).astype('f4')
   measured_near=cv2.erode(np.where(lidar>0,lidar,1e6),np.ones((3,3),'u1'));occluded=(measured_near<nearest-.3)&~sam
   target=(owner>=0)&valid&~sky&~occluded
   for tid,m in masks.items():target[owner==tid]&=m[owner==tid]
   md=np.where(target,nearest,0).astype('f4');md[~np.isfinite(md)]=0
   # Depth sources never compete on the same pixel, and no fixed/sky pixel gets depth.
   ld[md>0]=0;assert not np.any((md>0)&((ld>0)|~valid|sky))
   np.save(DATA/'lidar_depth'/(name+'.npy'),dict(mask=ld>0,value=ld[ld>0]));np.savez_compressed(DATA/'mesh_depth'/(name+'.npz'),depth=md);cv2.imwrite(str(DATA/'target_mask'/(name+'.png')),target.astype('u1')*255);cv2.imwrite(str(DATA/'vehicle_lidar_exclusion'/(name+'.png')),exclude.astype('u1')*255)
   ids=ids[valid[y[ids],x[ids]]&~sky[y[ids],x[ids]]&~exclude[y[ids],x[ids]]];ids=ids[np.abs(z[ids]-lidar[y[ids],x[ids]])<.2+.005*z[ids]];score=K[0,0]*K[1,1]/z[ids]**2;ids=ids[score>quality[ids]];quality[ids]=K[0,0]*K[1,1]/z[ids]**2;rgb[ids]=gt[y[ids],x[ids]][:,::-1]/255.
   for tid,M in current_objects.items():
    st=states[tid];pts=st['points'];cp=(pts@M[:3,:3].T+M[:3,3])@E[:3,:3].T+E[:3,3];uv=cp@K.T;ij=np.rint(uv[:,:2]/np.maximum(cp[:,2:],.001)).astype(int);x,y=ij.T;ids=np.flatnonzero((cp[:,2]>.1)&(x>=0)&(x<W)&(y>=0)&(y<H));ids=ids[target[y[ids],x[ids]]&(owner[y[ids],x[ids]]==tid)];ids=ids[np.abs(cp[ids,2]-nearest[y[ids],x[ids]])<.08+.002*cp[ids,2]];q=K[0,0]*K[1,1]/cp[ids,2]**2;ids=ids[q>st['quality'][ids]];st['quality'][ids]=K[0,0]*K[1,1]/cp[ids,2]**2;st['rgb'][ids]=gt[y[ids],x[ids]][:,::-1]/255.
   audits.append(dict(image=name,lidar_pixels=int((ld>0).sum()),mesh_pixels=int((md>0).sum()),vehicle_excluded_pixels=int(exclude.sum())))
   if fr in [0,50,100]:
    preview=gt.copy();contours,_=cv2.findContours(target.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,(0,255,0),2);cv2.imwrite(str(ROOT/'alignment_preview'/(name+'.jpg')),cv2.resize(preview,(800,450)))
  seen=quality>0;bgx.append(xyz[seen]);bgc.append(rgb[seen]);print('FRAME',fr,'removed_vehicle_points',int(removed.sum()),'background_points',int(seen.sum()),flush=True)
 del clouds
 x=np.concatenate(bgx);c=np.concatenate(bgc);_,idx=np.unique(np.floor(x/.10).astype('i4'),axis=0,return_index=True);x=x[idx];c=c[idx]
 saveply(ROOT/'model/input_ply/points3D_lidar.ply',x,c);lidar_count=len(x)
 far=json.loads((BASE/'initialization_merge.json').read_text())['far_ply']
 if far:
  fx,fc=loadply(Path(far));fx=fx-origin
 else:
  fx,fc=np.empty((0,3)),np.empty((0,3))
 # Existing far PLY must not seed vehicle geometry either.
 keep=np.ones(len(fx),bool)
 for objects in frame_objects.values():
  for M in objects.values():
   # Query an inexpensive enclosing sphere first.
   center=M[:3,3];radius=np.linalg.norm(M[:3,:3],axis=0).sum()/2+.3;ids=np.flatnonzero(keep&(np.linalg.norm(fx-center,axis=1)<radius))
   if len(ids):q=(fx[ids]-center)@np.linalg.inv(M[:3,:3]).T;keep[ids[np.all(np.abs(q)<.55,axis=1)]]=False
 x=np.concatenate([x,fx[keep]]);c=np.concatenate([c,fc[keep]]);saveply(ROOT/'model/input_ply/points3D_bkgd.ply',x,c)
 actor_audit=[]
 for a in actors:
  st=states[a['id']];colored=st['quality']>0
  if colored.any():st['rgb'][~colored]=np.median(st['rgb'][colored],axis=0)
  saveply(ROOT/'model/input_ply'/f"points3D_obj_{a['id']:03d}.ply",st['points']*a['size'],st['rgb'])
  actor_audit.append(dict(id=a['id'],points=len(st['points']),gt_colored=int(colored.sum()),vehicle_lidar_points=0))
 (ROOT/'actors.json').write_text(json.dumps(actors,indent=2));lo=x.min(0);hi=x.max(0)
 manifest.update(world_origin=origin.tolist(),initial_points=len(x),baseline_sphere=dict(center=((lo+hi)/2).tolist(),radius=float(np.linalg.norm(hi-lo)/2)),initialization='Nonvehicle per-frame LiDAR + existing far DVP; all selected motor vehicles mesh-only',lambda_depth_mesh=.05,lambda_depth_ply=0.,dynamic_mask=False)
 (ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2));report=dict(status='PASS',images=707,actors=len(actors),vehicle_lidar_points_removed=removed_total,background_lidar_points=lidar_count,far_points_kept=int(keep.sum()),far_points_removed=int((~keep).sum()),background_points=len(x),actors_audit=actor_audit,depth_audit=audits,mesh_texture_used=False,all_gt_rgb_preserved=True)
 (ROOT/'preparation.json').write_text(json.dumps(report,indent=2));print('PREPARATION_COMPLETE',len(x),flush=True)
if __name__=='__main__':main()
