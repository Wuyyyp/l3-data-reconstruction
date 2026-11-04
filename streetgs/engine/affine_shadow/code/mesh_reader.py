"""Read the T3 BA cameras, explicit masks and centered initialization PLY."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_t3(path, **kwargs):
    from lib.config import cfg
    from lib.datasets.base_readers import CameraInfo, SceneInfo
    from lib.utils.graphics_utils import BasicPointCloud, focal2fov
    from plyfile import PlyData
    root=Path(path).parent
    m=json.loads((root/'input_manifest.json').read_text())
    nframes=max(r['frame'] for r in m['views'])+1
    cameras=[]
    timestamps={c:{'train_timestamps':[],'test_timestamps':[]} for c in m['cameras']}
    for uid,r in enumerate(m['views']):
        if r['cam'] not in cfg.data.cameras:continue
        if cfg.data.get('frame_ids') is not None and r['frame'] not in cfg.data.frame_ids:continue
        name=r['name'];w=r.get('width',m.get('resolution',[None,None])[0]);h=r.get('height',m.get('resolution',[None,None])[1]);K=np.array(r['K']);c2w=np.array(r['c2w'])
        c2w[:3,3]-=np.array(m['world_origin']);w2c=np.linalg.inv(c2w)
        image=Image.new('RGB',(1,1))
        with Image.open(Path(path)/'valid_mask'/(name+'.png')) as f:
            if f.size!=(w,h):raise ValueError(f'Unexpected valid mask size for {name}: {f.size}, expected {(w,h)}')
            valid=Image.fromarray(np.asarray(f.convert('L'))>0)
        if cfg.data.get('use_sky_mask',True):
            with Image.open(Path(path)/'sky_mask'/(name+'.png')) as f:sky=Image.fromarray(np.asarray(f.convert('L'))>0)
        else:sky=Image.fromarray(np.zeros((h,w),bool))
        guidance=dict(mask=valid,sky_mask=sky,obj_bound=Image.open(Path(path)/'target_mask'/(name+'.png')).convert('L').copy())
        excluded_path=root/'static_excluded'/(name+'.png')
        excluded=np.asarray(Image.open(excluded_path))>0 if excluded_path.exists() else np.asarray(guidance['obj_bound'])>0
        guidance['mask']=Image.fromarray((np.asarray(valid)>0) & (~excluded if not cfg.model.nsg.include_obj else True))
        guidance['obj_bound']=Image.fromarray(excluded)
        guidance['lidar_depth']=None
        t=r['frame']/10.
        timestamps[r['cam']]['test_timestamps' if r['frame']%5==4 else 'train_timestamps'].append(t)
        cameras.append(CameraInfo(uid=uid,R=w2c[:3,:3].T,T=w2c[:3,3],K=K,
            FovX=focal2fov(K[0,0],w),FovY=focal2fov(K[1,1],h),image=image,
            image_path=str(Path(path)/'images'/r.get('image_file',name+'.jpg')),image_name=name,width=w,height=h,
            metadata=dict(frame=r['frame'],frame_idx=r['frame'],cam=r['cam'],is_val=False,timestamp=t,
                          ego_pose=np.eye(4),extrinsic=np.eye(4)),guidance=guidance))
    norm=dict(center=np.zeros(3),translate=np.zeros(3),radius=float(cfg.data.extent))
    metadata=dict(obj_tracklets=np.empty((nframes,0,8)),obj_meta={},tracklet_timestamps=np.arange(nframes)/10.,
        camera_timestamps=timestamps,num_images=len(m['views']),num_cams=len(m['cameras']),num_frames=nframes,scene_center=np.zeros(3),
        scene_radius=float(cfg.data.extent),sphere_center=np.array(m['baseline_sphere']['center']),
        sphere_radius=m['baseline_sphere']['radius'])
    from scipy.spatial.transform import Rotation
    actors=json.loads((root/'actors.json').read_text());tracklets=np.full((nframes,len(actors),8),-1.,dtype=np.float32);info={}
    for column,a in enumerate(actors):
        tid=a['id'];frames=np.array(a['frames']);q=Rotation.from_matrix(a['rotations']).as_quat()[:,[3,0,1,2]]
        tracklets[frames,column]=np.c_[np.full(len(frames),tid),a['centers'],q]
        info[tid]=dict(track_id=tid,**{'class':'car'},class_label=0,deformable=a['type'] in ['Non_motor_rider','Pedestrian'],start_frame=int(frames.min()),end_frame=max(int(frames.max()),int(frames.min())+1),start_timestamp=float(frames.min()/10),end_timestamp=float(frames.max()/10),length=a['size'][0]*1.35,width=a['size'][1]*1.35,height=a['size'][2]*1.35,valid_frames=frames.tolist(),column=column)
        if a.get('native_meta'):
            for key in ['class','class_label','deformable','length','width','height']:info[tid][key]=a['native_meta'][key]
    metadata.update(obj_tracklets=tracklets,obj_meta=info)
    pcd=None;ply=Path(cfg.model_path)/'input_ply/points3D_bkgd.ply'
    if cfg.mode=='train':
        v=PlyData.read(str(ply))['vertex']
        xyz=np.stack([v[k] for k in ['x','y','z']],1)
        rgb=np.stack([v[k] for k in ['red','green','blue']],1).astype(np.float32)/255.
        pcd=BasicPointCloud(xyz,rgb,np.zeros_like(xyz))
        assert np.isfinite(xyz).all()
    all_views=cfg.data.get('ground_split','heldout')=='train-all'
    train=[c for c in cameras if all_views or c.metadata['frame']%5!=4];test=[c._replace(metadata=dict(c.metadata,is_val=True)) for c in cameras if all_views or c.metadata['frame']%5==4]
    return SceneInfo(pcd,train,test,norm,str(ply),metadata,[])
