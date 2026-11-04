from pathlib import Path
import numpy as np,json,shutil,time,concurrent.futures,py_compile
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter,binary_closing,binary_erosion,maximum_filter
from PIL import Image,ImageDraw
from plyfile import PlyData,PlyElement
from common import CFG,S,ROOT,O,G,V,REPO,mask_path,ego_path,binary_mask
from mesh_inputs import bundle,readable,shifted_actors
M=bundle(CFG['mesh_source'],frames=CFG['frames'],cameras=CFG['cameras'],pose_csv=CFG['pose_csv']) if CFG.get('mesh_source') else None
DYNAMIC=CFG.get('scene_mode','static')=='dynamic'
USE_SKY=CFG.get('sky_mode','cubemap')=='cubemap'
F=Path(CFG['ego_mask'])
O.mkdir(exist_ok=False);(O/'actors.json').write_text('[]');P=O/'prepared';P.mkdir();D=O/'converted';D.mkdir()
for name in ['images','valid_mask','sky_mask','target_mask','lidar_depth','mesh_depth']:(D/name).mkdir()
for name in ['static_excluded','ground_masks','previews']:(O/name).mkdir()
for name in ['ego_pose','extrinsics','intrinsics','track','timestamps.json','timestamps_specific.json']:
 if (S/name).exists():(D/name).symlink_to(S/name)
shutil.copy2(__file__,O/'prepare.py');start=time.time()
def xyz(a):return np.column_stack([a[k] for k in 'xyz']).astype('f8')
def voxel(a,size):return a[np.sort(np.unique(np.floor(xyz(a)/size).astype('int64'),axis=0,return_index=True)[1])]
g=PlyData.read(G/'ground_world_5cm.ply')['vertex'].data;r=PlyData.read(V/'additional_removed_ground_world.ply')['vertex'].data;a=np.empty(len(g)+len(r),g.dtype)
for k in g.dtype.names:a[k]=np.concatenate([g[k],r[k]])
a=voxel(a,CFG['ground_voxel']);gp=xyz(a);tree=cKDTree(gp);normals=[];smooth=[];valids=[]
for st in range(0,len(gp),100000):
 q=gp[st:st+100000];dist,ix=tree.query(q,k=24,workers=CFG['workers']);nn=gp[ix];mean=nn.mean(1);v=nn-mean[:,None];val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',v,v)/24);n=vec[:,:,0];n*=np.where(n[:,2]<0,-1,1)[:,None];delta=np.einsum('ni,ni->n',q-mean,n);good=(dist[:,-1]<CFG['support_radius'])&(n[:,2]>CFG['normal_z_min'])&(np.sqrt(np.maximum(val[:,0],0))<CFG['plane_rms_max'])&(abs(delta)<CFG['ground_tolerance'])&(val[:,1]>.0025);valids.append(good);normals.append(n);smooth.append(q-n*delta[:,None])
good=np.concatenate(valids);n=np.concatenate(normals)[good];gp=np.concatenate(smooth)[good];a=a[good];origin=np.loadtxt(S/'ego_pose'/('%06d.txt'%CFG['origin_frame']))[:3,3]
x=np.cross(np.broadcast_to([0.,1.,0.],n.shape),n);x/=np.linalg.norm(x,axis=1)[:,None];y=np.cross(n,x);quat=Rotation.from_matrix(np.stack([x,y,n],axis=2)).as_quat()[:,[3,0,1,2]]
np.savez_compressed(P/'ground.npz',xyz=(gp-origin).astype('f4'),normals=n.astype('f4'),rotation=quat.astype('f4'),rgb=np.column_stack([a[k] for k in ['red','green','blue']]).astype('f4')/255)
bg_source=PlyData.read(V/'static_no_ground_world_5cm.ply')['vertex'].data
bg=(bg_source if CFG.get('reuse_prepared_background') else voxel(bg_source,CFG['background_voxel'])).copy();bp=xyz(bg)
for j,k in enumerate('xyz'):bg[k]-=origin[j]
PlyData([PlyElement.describe(bg,'vertex')]).write(P/'nonground.ply');center=(bp.min(0)+bp.max(0))/2-origin;radius=float(np.linalg.norm(bp-origin-center,axis=1).max()*1.1)
manifest=json.loads((S/'input_manifest.json').read_text());views=[];camcodes=[]
for rec in manifest['records']:
 name=Path(rec['label']).stem;fid,cs=name.split('_');cam=int(cs)
 if CFG['frames'] is not None and int(fid) not in CFG['frames']:continue
 if CFG['cameras'] is not None and cam not in CFG['cameras']:continue
 cal=manifest['calibrations'][rec['camera']];w=CFG['width'];h=int(cal['height']*w/cal['width']);K=np.array(cal['K']);K[0]*=w/cal['width'];K[1]*=h/cal['height'];views.append(dict(name=name,frame=int(fid),cam=cam,K=K.tolist(),c2w=rec['camera_to_world'],width=w,height=h));camcodes.append(cam)
if M is not None:
 views=[]
 for v in M['views']:
  v=dict(v);ow,oh=M['manifest']['resolution'];w=CFG['width'];h=int(oh*w/ow);K=np.array(v['K']);K[0]*=w/ow;K[1]*=h/oh;v.update(width=w,height=h,K=K.tolist());views.append(v)
 camcodes=[v['cam'] for v in views]
 if DYNAMIC:
  actors=shifted_actors(M['actors'],M['origin'],origin,{v['frame'] for v in views});(O/'actors.json').write_text(json.dumps(actors,indent=2));(P/'actors').mkdir()
  for actor in actors:
   name='points3D_obj_%03d.ply'%actor['id'];(P/'actors'/name).symlink_to(readable(M['root']/'model/input_ply'/name))
 (O/'mesh_coordinate_audit.json').write_text(json.dumps(dict(authority=M['audit'],old_origin=M['origin'].tolist(),new_origin=origin.tolist(),affine_world_positions_preserved=True),indent=2))
m=dict(views=views,cameras=sorted(set(camcodes)),world_origin=origin.tolist(),baseline_sphere=dict(center=center.tolist(),radius=radius),static_only=not DYNAMIC,source=str(S),fixed_mask_source=str(F),fixed_mask_transfer='Nearest resize of explicit ego masks; per-frame mask preferred, then frame000000 camera template. Source path recorded; cross-clip transfer requires visual review.')
pose_audit=F.parent/'input_audit.json'
if pose_audit.is_file():
 version=json.loads(pose_audit.read_text())['geometry_version']
 if version=='l3-wys-full-frame-v1':m['geometry_version']=version
(O/'input_manifest.json').write_text(json.dumps(m,indent=2))
view_frames=sorted({r['frame'] for r in views});preview_frames={view_frames[0],view_frames[len(view_frames)//2],view_frames[-1]}
def zbuf(p,c2w,K,w,h):
 W=np.linalg.inv(c2w);q=p@W[:3,:3].T+W[:3,3];q=q[q[:,2]>.1];uv=q@K.T;uv=uv[:,:2]/uv[:,2:];ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[ok].astype(int);z=np.full(w*h,np.inf,np.float32);np.minimum.at(z,uv[:,1]*w+uv[:,0],q[ok,2]);return z.reshape(h,w)
def make_view(r):
 name=r['name'];w,h=r['width'],r['height'];cam=r['cam'];sz=(w,h);im=Image.open(readable(M['root']/'converted/images'/r.get('image_file',name+'.jpg')) if M is not None else S/'images'/(name+'.jpg')).convert('RGB').resize(sz,Image.Resampling.LANCZOS);im.save(D/'images'/r.get('image_file',name+'.jpg'),**({} if r.get('image_file','').endswith('.png') else {'quality':95}))
 fixed=binary_mask(ego_path(name),sz);valid=~fixed
 # The selected precise ego template replaces the legacy coarse ego exclusion.
 if M is not None and F.resolve()!=(REPO/'configs/preprocessing/ego_mask_l3_recon_20261030').resolve():valid &= binary_mask(readable(M['root']/'converted/valid_mask'/(name+'.png')),sz)
 dyn=binary_mask(readable(M['root']/'converted/target_mask'/(name+'.png')) if M is not None else mask_path(S/'dynamic_mask',name),sz);dyn=maximum_filter(dyn,size=CFG['dynamic_dilation'])
 if M is not None:
  protection=M['root']/'converted/foreground_protection'/(name+'.png')
  if protection.exists() or protection.is_symlink():dyn |= binary_mask(readable(protection),sz)
 sky=binary_mask(mask_path(S/'sky_mask',name),sz,127) if USE_SKY else np.zeros((h,w),bool)
 for path,mask in [(D/'valid_mask',valid),(D/'target_mask',dyn),(O/'static_excluded',dyn),(D/'sky_mask',sky)]:Image.fromarray(mask.astype('u1')*255).save(path/(name+'.png'))
 src=np.load(readable(M['root']/'converted/lidar_depth'/(name+'.npy')) if M is not None else S/'lidar_depth'/(name+'.npy'),allow_pickle=True).item();depth=np.zeros(src['mask'].shape,np.float32);depth[src['mask']]=src['value'];depth=np.asarray(Image.fromarray(depth).resize(sz,Image.Resampling.NEAREST)).copy();ok=np.isfinite(depth)&(depth>0)&valid&(~dyn if not DYNAMIC else True)&~sky;np.save(D/'lidar_depth'/(name+'.npy'),dict(mask=ok,value=depth[ok]))
 if DYNAMIC:
  with np.load(readable(M['root']/'converted/mesh_depth'/(name+'.npz'))) as mdz:md=np.asarray(mdz['depth'],np.float32)
  md=np.asarray(Image.fromarray(md).resize(sz,Image.Resampling.NEAREST)).copy();md[~valid|sky|~np.isfinite(md)|(md<0)]=0;np.savez_compressed(D/'mesh_depth'/(name+'.npz'),depth=md)
 z=zbuf(gp,np.array(r['c2w']),np.array(r['K']),w,h);z=minimum_filter(z,size=5);observed=minimum_filter(np.where(depth>0,depth,np.inf),size=5)
 # Static environment depth prevents a projected road from labeling a visible guardrail/bridge.
 scene=minimum_filter(zbuf(bp,np.array(r['c2w']),np.array(r['K']),w,h),size=3);front=np.minimum(scene,observed)
 visible=np.isfinite(z)&((~np.isfinite(front))|(z<=front+CFG['occlusion_tolerance']));gm=binary_erosion(binary_closing(visible,iterations=1),iterations=1)&valid&~dyn&~sky
 folder=O/'ground_masks'/f'cam_{cam:02d}';folder.mkdir(exist_ok=True);np.save(folder/f'{r["frame"]:06d}.npy',gm)
 if r['frame'] in preview_frames:
  pic=np.asarray(im).copy();pic[gm]=(pic[gm]*.4+np.array([0,255,50])*.6).astype('u1');pic[~valid]=(pic[~valid]*.4+np.array([255,0,255])*.6).astype('u1');pic[dyn]=(pic[dyn]*.4+np.array([255,20,0])*.6).astype('u1');pic=Image.fromarray(pic);pic.thumbnail((960,640));pic.save(O/'previews'/(name+'.jpg'))
 return dict(name=name,ground=int(gm.sum()),valid_static=int((valid&~dyn).sum()),dynamic=int(dyn.sum()),ego=int(fixed.sum()),depth=int(ok.sum()))
rows=[]
with concurrent.futures.ThreadPoolExecutor(CFG['workers']) as pool:
 for row in pool.map(make_view,views):
  rows.append(row)
  if len(rows)%50==0:print('views',len(rows),flush=True)
  (O/'prepare_status.json').write_text(json.dumps(dict(stage='views',done=len(rows),total=len(views))))
report=dict(source=str(S),pipeline_config=CFG,ground_points=len(gp),background_points=len(bg),views=len(rows),train=sum(r['frame']%5!=4 for r in views),test=sum(r['frame']%5==4 for r in views),rows=rows,seconds=time.time()-start,ground_preparation='24-neighbor PCA and projection onto supported local planes; thresholds and voxel size in pipeline_config; +/-10cm learnable normal offset',background='Refined non-ground static voxel cloud; voxel size in pipeline_config; no attraction applied to unclassified background',ground_mask='5px ground splat + background and LiDAR occlusion at configured tolerance; sky, dynamic and ego excluded',ego_mask=m['fixed_mask_transfer']);(O/'preparation_report.json').write_text(json.dumps(report,indent=2));(O/'prepare_status.json').write_text(json.dumps(dict(stage='complete',ground_points=len(gp),background_points=len(bg))));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
