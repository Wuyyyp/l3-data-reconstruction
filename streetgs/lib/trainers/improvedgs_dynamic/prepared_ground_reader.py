"""Prepared V2/V3 scene reader; preserves calibrated cameras and actor meshes."""
from pathlib import Path
import json,os,numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation

def read_prepared(path,**kwargs):
 from lib.config import cfg
 from lib.datasets.base_readers import CameraInfo,SceneInfo
 from lib.utils.graphics_utils import BasicPointCloud,focal2fov
 root=Path(path).parent;m=json.loads((root/'input_manifest.json').read_text());origin=np.array(m['world_origin']);cams=[]
 timestamps={c:{'train_timestamps':[],'test_timestamps':[]} for c in m['cameras']}
 for uid,v in enumerate(m['views']):
  if os.environ.get('SMOKE')=='1' and v['frame']>4:continue
  fid=v['frame'];name=v['name'];cam=v['cam'];K=np.array(v['K']);C=np.array(v['c2w']);C[:3,3]-=origin;E=np.linalg.inv(C)
  image=Image.open(Path(path)/'images'/f'{name}.jpg').convert('RGB');w,h=image.size
  valid=Image.open(Path(path)/'valid_mask'/f'{name}.png').convert('L');sky=Image.open(Path(path)/'sky_mask'/f'{name}.png').convert('L');obj=Image.open(Path(path)/'target_mask'/f'{name}.png').convert('L')
  a=np.load(Path(path)/'lidar_depth'/f'{name}.npy',allow_pickle=True).item();d=np.zeros((h,w),'f4');d[a['mask']]=a['value']
  t=fid/10.;isval=fid%5==4;timestamps[cam]['test_timestamps' if isval else 'train_timestamps'].append(t)
  cams.append(CameraInfo(uid=uid,R=E[:3,:3].T,T=E[:3,3],K=K,FovX=focal2fov(K[0,0],w),FovY=focal2fov(K[1,1],h),image=image,image_path=str(Path(path)/'images'/f'{name}.jpg'),image_name=name,width=w,height=h,metadata=dict(frame=fid,frame_idx=fid,cam=cam,is_val=isval,timestamp=t,ego_pose=np.eye(4),extrinsic=np.eye(4)),guidance=dict(mask=valid,sky_mask=sky,obj_bound=obj,lidar_depth=d)))
 actors=json.loads((root/'actors.json').read_text());tracks=np.full((101,len(actors),8),-1.,'f4');info={}
 for j,a in enumerate(actors):
  f=np.array(a['frames']);q=Rotation.from_matrix(a['rotations']).as_quat()[:,[3,0,1,2]];tracks[f,j]=np.c_[np.full(len(f),a['id']),a['centers'],q];info[a['id']]=dict(track_id=a['id'],**{'class':'car'},class_label=0,deformable=False,start_frame=int(f.min()),end_frame=max(int(f.max()),int(f.min())+1),start_timestamp=float(f.min()/10),end_timestamp=float(f.max()/10),length=a['size'][0]*1.35,width=a['size'][1]*1.35,height=a['size'][2]*1.35,valid_frames=f.tolist(),column=j)
 md=dict(obj_tracklets=tracks,obj_meta=info,tracklet_timestamps=np.arange(101)/10.,camera_timestamps=timestamps,num_images=len(cams),num_cams=len(m['cameras']),num_frames=101,scene_center=np.zeros(3),scene_radius=20.,sphere_center=np.array(m['baseline_sphere']['center']),sphere_radius=m['baseline_sphere']['radius'])
 p=Path(cfg.model_path)/'input_ply/points3D_bkgd.ply';a=PlyData.read(p)['vertex'];x=np.column_stack([a[k] for k in 'xyz']);c=np.column_stack([a[k] for k in ['red','green','blue']])/255.;pcd=BasicPointCloud(x,c,np.zeros_like(x));norm=dict(center=np.zeros(3),translate=np.zeros(3),radius=20.)
 return SceneInfo(pcd,[c for c in cams if not c.metadata['is_val']],[c for c in cams if c.metadata['is_val']],norm,str(p),md,[])

