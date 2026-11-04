"""World-space motion evidence and lossless masks, independent of mesh assets."""
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist

PROMPTS = ('vehicle', 'person', 'bicycle', 'motorcycle', 'electric scooter', 'electric bicycle', 'tricycle')
CAMERA_IDS = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov30':9,'front_camera_fov120':10}


def category_group(name):
    text=str(name).lower().replace('-', '_').replace(' ', '_')
    if text.startswith('vehicle_') and text!='vehicle_else':text=text[len('vehicle_'):]
    text=text.removeprefix('bike_')
    if any(x in text for x in ('pedestrian','person','human')):return 'person'
    # The shared cycle/rider matching group also includes three-wheel cycles.
    if any(x in text for x in ('bicycle','tricycle','cyclist','rider','motorcycle','scooter','bike')):return 'two_wheeler'
    if text in {'car','suv','vehicle','van','bus','truck','ambulance','trailer','other_vehicle','pickup','mpv','huge_vehicle','vehicle_else','firetruck'}:return 'vehicle'
    return 'other'


def compatible(prompt, source_class):
    group=category_group(source_class)
    if prompt=='person':return group=='person' or 'rider' in source_class.lower()
    if prompt=='vehicle':return group in {'vehicle','two_wheeler'}
    return group=='two_wheeler'


def motion_decision(positions, threshold=1., *, source='annotation', radii=None):
    """Dynamic means max pairwise 3D world distance > threshold, not path length."""
    p=np.asarray(positions,dtype=float).reshape(-1,3)
    if not np.isfinite(p).all():raise ValueError('non-finite world trajectory')
    result=dict(state='uncertain',source=source,observations=len(p),max_displacement_m=None,threshold_m=float(threshold),rule='max pairwise world XYZ distance strictly greater than threshold')
    if len(p)<2:result['reason']='fewer_than_two_world_positions';return result
    displacement=float(pdist(p).max());result['max_displacement_m']=displacement
    if source=='lidar_surface_centroid':
        # A visible surface is not an object centre. Keep ambiguous near-threshold
        # changes out of the moving mask instead of pretending they are certain.
        if len(p)<3:result['reason']='insufficient_lidar_observations';return result
        if radii is None or len(radii)!=len(p) or not np.isfinite(radii).all() or np.any(np.asarray(radii)<0):
            result['reason']='invalid_surface_extent_evidence';return result
        # Tens of metres of depth spread inside a road-user mask indicates mixed
        # foreground/background returns, not a dependable single-object centre.
        if float(np.median(radii))>10.:
            result['reason']='incoherent_lidar_surface_extent';return result
        margin=2.*float(np.median(radii))
        result['surface_uncertainty_margin_m']=margin
        if displacement>threshold and displacement-margin<=threshold:
            result['reason']='surface_centroid_displacement_uncertain';return result
    result['state']='moving' if displacement>threshold else 'static'
    result['reason']='world_displacement_above_threshold' if displacement>threshold else 'world_displacement_not_above_threshold'
    return result


def consistent_annotation(observations):
    votes=Counter(o['annotation_id'] for o in observations if o.get('annotation_id') is not None)
    if not votes:return None
    identity,count=votes.most_common(1)[0]
    # Reject a SAM identity that switches between different annotated objects.
    return identity if count>=2 and count/sum(votes.values())>=.8 and count/len(observations)>=.5 else None


def camera_matrices(info, name):
    camera=info['sensors']['cams'][name]
    world_from_ego=np.asarray(info['ego2global_transformation_matrix_camera_reference_offset_optimized'],float)
    if camera.get('camera_to_world_optimized') is not None:
        world_from_camera=np.asarray(camera['camera_to_world_optimized'],float)
    else:world_from_camera=world_from_ego@np.linalg.inv(np.asarray(camera['extrinsic'],float))
    return np.asarray(camera['cam_intrinsic'],float),world_from_camera,world_from_ego


def annotation_objects(info):
    if 'objects' in info:
        return [(int(o['id']),str(o['type']),np.asarray(o['location']+o['size']+[o['rotation'][2]],float)) for o in info['objects']]
    return [(int(i),str(n),np.asarray(b,float)) for i,n,b in zip(info['track_id'],info['gt_names'],info['gt_boxes'])]


def annotation_world_tracks(infos):
    tracks={}
    for fi,info in enumerate(infos):
        T=np.asarray(info['ego2global_transformation_matrix_camera_reference_offset_optimized'],float)
        for identity,category,box in annotation_objects(info):
            if category_group(category)=='other':continue
            row=tracks.setdefault(identity,dict(category=category,frames=[],positions=[]))
            row['frames'].append(fi);row['positions'].append((T@np.r_[box[:3],1.])[:3].tolist())
    return tracks


def pack_mask(mask):
    mask=np.asarray(mask,bool)
    yy,xx=np.nonzero(mask)
    if not len(xx):raise ValueError('empty instance mask')
    x,y=int(xx.min()),int(yy.min());w,h=int(xx.max()+1-x),int(yy.max()+1-y)
    return np.asarray([x,y,w,h],np.int32),np.packbits(mask[y:y+h,x:x+w],axis=None)


def unpack_mask(bbox,bits,shape):
    x,y,w,h=map(int,bbox);H,W=shape
    if not (0<=x<x+w<=W and 0<=y<y+h<=H):raise ValueError('mask bbox outside image')
    if len(bits)!=(w*h+7)//8:raise ValueError('invalid packed mask length')
    mask=np.zeros((H,W),bool);mask[y:y+h,x:x+w]=np.unpackbits(bits,count=w*h).reshape(h,w)
    return mask


def depth_world_centroid(mask,depth,K,world_from_camera):
    import cv2
    interior=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    ys,xs=np.nonzero(interior & np.isfinite(depth) & (depth>.1) & (depth<200.))
    if len(xs)<20:return None
    z=depth[ys,xs];median=float(np.median(z));mad=float(np.median(np.abs(z-median)))
    keep=np.abs(z-median)<=max(.3,3.*1.4826*mad);xs,ys,z=xs[keep],ys[keep],z[keep]
    if len(z)<20:return None
    rays=np.linalg.solve(K,np.stack([xs,ys,np.ones_like(xs)]));camera=(rays*z).T
    world=camera@world_from_camera[:3,:3].T+world_from_camera[:3,3]
    center=np.median(world,axis=0)
    radius=float(np.percentile(np.linalg.norm(world-center,axis=1),75))
    return dict(world_position=center.tolist(),depth_points=len(z),surface_radius_m=radius)
