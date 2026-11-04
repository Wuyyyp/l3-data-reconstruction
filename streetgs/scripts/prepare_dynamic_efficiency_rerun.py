from pathlib import Path
import os,sys,json,pickle,math,time
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,cv2,open3d as o3d
cv2.setNumThreads(1)
for n in ['multiarray','numeric']:sys.modules['numpy._core.'+n]=__import__('numpy.core.'+n,fromlist=[n])
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');RUN=ROOT/'runs/dynamic_efficiency_20261013';OUT=RUN/'converted';OUT.mkdir(parents=True,exist_ok=True)
SRC=Path('/mnt/nuplan/l3data-reconstruction-bingxing/samples-restored/l3-data/clip_M18-2_07_20251230153515_DF');BA=SRC/'colmap-ba'
OLD=Path('/mnt/nuplan/l3data-reconstruction-bingxing/samples/sample_00380_clip_M18-2_07_20251230153515_DF_f184_284_right/converted')
CAMS={'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov30':9,'front_camera_fov120':10}
NAMES={k:k.replace('front_camera_fov','center_camera_fov') for k in CAMS}
for n in ['images','intrinsics','extrinsics','ego_pose','track','sky_mask','lidar_depth','frames_cache']:(OUT/n).mkdir(exist_ok=True)
def status(**kw):
 kw.update(updated=time.time());p=RUN/'prepare_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(kw,indent=2));t.replace(p)
def link(src,dst):
 assert src.is_file(),str(src)
 if not dst.exists():dst.symlink_to(src)
infos=pickle.load(open(SRC/'result_interpolation_optimized.pkl','rb'))['infos'];oldtimes=json.loads((OLD/'timestamps_specific.json').read_text())
manifest=json.loads((BA/'input/03_converted/input_manifest.json').read_text());lookup={(x['original_frame'],int(x['camera_code'])):x for x in manifest['records']}
timestamps={'FRAME':{}};timestamps.update({NAMES[n]:{} for n in CAMS});tracks=['frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed'];vis={};mapping=[]
start=time.time()
for fi,orig in enumerate(range(184,285)):
 f=infos[orig];assert f['timestamp']==oldtimes[fi]['timestamp'];frame=f'{fi:06d}'
 ego=np.asarray(f['ego2global_transformation_matrix_camera_reference_offset_optimized']);np.savetxt(OUT/'ego_pose'/(frame+'.txt'),ego)
 timestamps['FRAME'][frame]=f['dynamic_compensation_reference_time_s']
 projections=[]
 for name,cam in CAMS.items():
  rec=lookup[orig,cam];label=f'{frame}_{cam:02d}';oldlabel=Path(rec['label']).stem
  source_image=Path(rec['raw_image'].replace('/data/mnt/','/mnt/'));assert int(source_image.stem)==f['sensors']['cams'][name]['timestamp']
  link(source_image,OUT/'images'/(label+'.jpg'))
  sky=OLD/'sky_mask'/(label+'.jpg');link(sky,OUT/'sky_mask'/(label+'.jpg'))
  k=np.loadtxt(BA/'input/03_converted/intrinsics'/(oldlabel+'.txt'));c2w=np.loadtxt(BA/'converted_after/ego_pose'/(oldlabel+'.txt'))
  np.savetxt(OUT/'intrinsics'/(label+'.txt'),k);np.savetxt(OUT/'ego_pose'/(label+'.txt'),c2w);np.savetxt(OUT/'extrinsics'/(label+'.txt'),np.linalg.inv(ego)@c2w)
  cal=manifest['calibrations'][name];projections.append((cam,label,k,np.linalg.inv(c2w)@ego,cal['width'],cal['height']))
  timestamps[NAMES[name]][frame]=f['sensors']['cams'][name]['timestamp']/1e6
 for obj in f['objects']:
  tid=int(obj['id']);l,w,h=obj['size'];x,y,z=obj['location'];yaw=obj['rotation'][2];speed=np.linalg.norm(obj['velocity']);cls=obj['type']
  cameras=[]
  for bb in obj.get('info2d',[]):
   n=bb['camera'].replace('center_camera_fov','front_camera_fov')
   if n in CAMS:cameras.append(CAMS[n])
  if not cameras:continue
  tracks.append(f'{fi} {tid} {cls} -10 {h} {w} {l} {x} {y} {z} {yaw} {speed}')
  vis.setdefault(str(tid),{})[str(fi)]=sorted(set(cameras))
 cache=OUT/'frames_cache'/(frame+'.npz')
 if not cache.exists():
  pcd=SRC/f['sensors']['lidar']['perception']['data_path'];xyz=np.asarray(o3d.io.read_point_cloud(str(pcd)).points,dtype=np.float32);assert len(xyz)>0,str(pcd)
  xyz=xyz[np.isfinite(xyz).all(1)];proj=np.full((len(xyz),3),-1,np.int32)
  for cam,label,k,transform,w,h in projections:
   cp=xyz@transform[:3,:3].T+transform[:3,3];uv=cp@k.T;z=cp[:,2];uv=uv[:,:2]/np.maximum(z[:,None],1e-6)
   valid=(z>.1)&(z<200)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
   idx=np.flatnonzero(valid);xy=uv[idx].astype(np.int32);fresh=proj[idx,0]<0;proj[idx[fresh]]=np.column_stack([np.full(fresh.sum(),cam),xy[fresh]])
   dep=np.full(h*w,np.inf,np.float32);np.minimum.at(dep,xy[:,1]*w+xy[:,0],z[idx].astype(np.float32));mask=np.isfinite(dep).reshape(h,w)
   np.save(OUT/'lidar_depth'/(label+'.npy'),dict(mask=mask,value=dep[np.isfinite(dep)]))
  np.savez(cache,xyz=xyz,projection=proj)
 mapping.append(dict(frame=fi,original_frame=orig,timestamp=f['timestamp']))
 status(state='converting',frame=fi,frames=101,elapsed_s=time.time()-start);print('FRAME',fi,flush=True)
(OUT/'timestamps.json').write_text(json.dumps(timestamps));(OUT/'track/track_info.txt').write_text('\n'.join(tracks)+'\n');(OUT/'track/track_camera_vis.json').write_text(json.dumps(vis))
status(state='packing_pointcloud',elapsed_s=time.time()-start)
points={};project={}
for fi in range(101):
 a=np.load(OUT/'frames_cache'/f'{fi:06d}.npz');points[fi]=a['xyz'];project[fi]=a['projection']
np.savez(OUT/'pointcloud.npz',pointcloud=points,camera_projection=project)
(OUT/'input_manifest.json').write_text(json.dumps(dict(source=str(BA),frame_mapping=mapping,image_count=707,sky_masks=str(OLD/'sky_mask'),lidar='compensated_perception with optimized ego; depth reprojected using BA cameras',tracks='optimized pickle objects; original info2d camera visibility',note='Fresh data and initialization; not numerically identical to deleted 20261006 experiment'),indent=2))
status(state='complete',frames=101,images=707,tracks=len(vis),elapsed_s=time.time()-start)
print('PREPARATION_COMPLETE',flush=True)
