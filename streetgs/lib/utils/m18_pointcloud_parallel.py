"""Deterministic frame-parallel M18 point-cloud initialization (CPU only)."""
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points
from lib.utils.general_utils import quaternion_to_matrix_numpy


def build_lidar_frames(pts3d_dict, pts2d_dict, start_frame, end_frame, num_cameras,
                       cameras, cams, image_filenames, ego_frame_poses,
                       object_tracklets_vehicle, object_info, workers=1, progress=True):
    keys=['bkgd']+[f'obj_{track_id:03d}' for track_id in object_info]

    def one_frame(i):
        frame=start_frame+i
        xyz={k:[] for k in keys}
        rgb={k:[] for k in keys}
        raw_2d=pts2d_dict[frame]
        visible=np.isin(raw_2d[...,0],cameras)
        points=pts3d_dict[frame][visible]
        points=np.concatenate([points,np.ones_like(points[:,:1])],axis=-1)
        world=points @ ego_frame_poses[frame].T
        colors=np.ones_like(points[:,:3])
        projection=raw_2d[visible]
        for idx in range(i*num_cameras,(i+1)*num_cameras):
            mask=projection[:,0]==cams[idx]
            # Match the old float64 division -> point dtype assignment exactly.
            image=cv2.imread(image_filenames[idx])
            if image is None:
                raise ValueError('Unreadable image: '+image_filenames[idx])
            pixels=image[projection[mask,2],projection[mask,1]][:,[2,1,0]]/255.
            colors[mask]=pixels
        object_mask=np.zeros(len(points),dtype=np.bool_)
        for tracklet in object_tracklets_vehicle[i]:
            track_id=int(tracklet[0])
            if track_id < 0:
                continue
            pose=np.eye(4)
            pose[:3,:3]=quaternion_to_matrix_numpy(tracklet[4:8])
            pose[:3,3]=tracklet[1:4]
            local=(points @ np.linalg.inv(pose).T)[...,:3]
            info=object_info[track_id]
            length,width,height=info['length'],info['width'],info['height']
            corners=bbox_to_corner3d([[-length/2,-width/2,-height/2],[length/2,width/2,height/2]])
            inside=inbbox_points(local,corners)
            object_mask=np.logical_or(object_mask,inside)
            key=f'obj_{track_id:03d}'
            xyz[key].append(local[inside])
            rgb[key].append(colors[inside])
        xyz['bkgd'].append(world[~object_mask][...,:3])
        rgb['bkgd'].append(colors[~object_mask])
        return xyz,rgb

    count=end_frame-start_frame+1
    workers=min(max(1,workers),count)
    xyz={k:[] for k in keys}
    rgb={k:[] for k in keys}
    old_cv_threads=cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        # BLAS threads inside frame threads would oversubscribe the node.
        with threadpool_limits(limits=1,user_api='blas'), ThreadPoolExecutor(max_workers=workers) as pool:
            results=pool.map(one_frame,range(count))
            for frame_xyz,frame_rgb in tqdm(results,total=count,desc=f'LiDAR init ({workers} CPU workers)',disable=not progress):
                # Consume in frame order, preserving voxel and random-sampling input order.
                for key in keys:
                    xyz[key].extend(frame_xyz[key])
                    rgb[key].extend(frame_rgb[key])
    finally:
        cv2.setNumThreads(old_cv_threads)
    return xyz,rgb
