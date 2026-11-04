"""Compute unique track motion and camera visibility using world coordinates."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.stats import theilslopes
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(__file__).resolve().parents[1] / 'input'
try:
    from .box_projection import project_track_box
except ImportError:  # Direct utility invocation from this folder.
    from box_projection import project_track_box
CANONICAL_FROM_SOURCE=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
SOURCE_FROM_CANONICAL=np.linalg.inv(CANONICAL_FROM_SOURCE)


def statistics(input_root,speed_threshold,min_observations,min_displacement):
    input_root=Path(input_root).resolve()
    payload=pickle.load((input_root/'result_interpolation_optimized_first101.pkl').open('rb'))
    infos=payload['infos'];manifest=json.loads((input_root/'manifest.json').read_text())
    records=defaultdict(list)
    for fi,info in enumerate(infos):
        reference=np.array(info['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        cameras=[]
        for item in manifest['videos']:
            cam=info['sensors']['cams'][item['camera']]
            extrinsic=np.linalg.inv(np.array(cam['camera_to_world_optimized']))@reference if 'camera_to_world_optimized' in cam else np.array(cam['extrinsic'])
            cameras.append((item,np.array(cam['cam_intrinsic']),extrinsic@SOURCE_FROM_CANONICAL))
        for identity,name,box,velocity in zip(info['track_id'],info['gt_names'],info['gt_boxes'],info['vels']):
            if not (name.startswith('VEHICLE_') or name.startswith('BIKE_')):continue
            box=np.array(box,dtype=float);velocity=np.array(velocity,dtype=float)
            if box.shape!=(7,) or velocity.shape!=(3,) or not np.isfinite(box).all() or not np.isfinite(velocity).all():raise ValueError((fi,identity))
            center=(CANONICAL_FROM_SOURCE@np.r_[box[:3],1.])[:3]
            visible=[]
            for item,intrinsic,extrinsic in cameras:
                projected=project_track_box(length=box[3],width=box[4],height=box[5],center_xyz=tuple(center),
                    heading=box[6]+np.pi/2,intrinsic=intrinsic,extrinsic=extrinsic,
                    image_width=item['width'],image_height=item['height'])
                bbox=projected.visible_bbox_xywh
                if bbox is not None and bbox[2]*bbox[3]>=25:visible.append(item['camera'])
            world_center=(reference@np.r_[box[:3],1.])[:3]
            records[int(identity)].append(dict(frame=fi,object_class=name,speed=float(np.linalg.norm(velocity[:2])),
                world=world_center,range_m=float(np.linalg.norm(box[:3])),views=visible))
    result=[]
    for identity,observations in sorted(records.items()):
        speeds=np.array([r['speed'] for r in observations])
        visible=[r for r in observations if r['views']]
        positions=np.array([r['world'] for r in observations]);times=np.array([r['frame']/manifest['fps'] for r in observations])
        # Split at gaps longer than 0.3 s: unrelated brief appearances must
        # not add up to a convincing trajectory. Theil-Sen limits outlier impact.
        frame_ids=np.array([r['frame'] for r in observations]);cuts=np.where(np.diff(frame_ids)>3)[0]+1
        segments=np.split(np.arange(len(observations)),cuts)
        segment_evidence=[]
        for segment in segments:
            if len(segment)<min_observations:continue
            segment_positions=positions[segment,:2];segment_times=times[segment]
            displacement=float(np.linalg.norm(segment_positions[-1]-segment_positions[0]))
            vx=float(theilslopes(segment_positions[:,0],segment_times).slope)
            vy=float(theilslopes(segment_positions[:,1],segment_times).slope)
            robust_speed=float(np.hypot(vx,vy))
            interval_speeds=np.linalg.norm(np.diff(segment_positions,axis=0),axis=1)/np.diff(segment_times)
            sustained_fraction=float(np.mean(interval_speeds>=speed_threshold))
            segment_evidence.append(dict(start_frame=int(frame_ids[segment[0]]),end_frame=int(frame_ids[segment[-1]]),
                observations=len(segment),displacement_xy_m=displacement,robust_speed_xy_mps=robust_speed,
                intervals_over_speed_fraction=sustained_fraction,
                passed=displacement>=min_displacement and robust_speed>=speed_threshold and sustained_fraction>=0.6))
        fitted_speed=max((s['robust_speed_xy_mps'] for s in segment_evidence),default=0.)
        dynamic=any(s['passed'] for s in segment_evidence) and bool(visible)
        observations_count=sum(len(r['views']) for r in visible)
        median_range=float(np.median([r['range_m'] for r in visible])) if visible else None
        result.append(dict(track_id=identity,object_class=observations[0]['object_class'],
            group='motor_vehicle' if observations[0]['object_class'].startswith('VEHICLE_') else 'bicycle',
            dynamic=dynamic,total_observed_frames=len(observations),
            annotated_velocity_observations_over_threshold=int((speeds>=speed_threshold).sum()),
            visible_frames=len(visible),
            cameras=sorted({camera for r in visible for camera in r['views']}),camera_frame_observations=observations_count,
            median_annotated_speed_mps=float(np.median(speeds)),max_annotated_speed_mps=float(speeds.max()),
            world_displacement_xy_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),
            world_fitted_speed_xy_mps=fitted_speed,median_visible_range_m=median_range,
            motion_segments=segment_evidence,
            observed_frame_ids=[r['frame'] for r in observations],
            visible_frame_ids=[r['frame'] for r in visible],
            visible_frame_ids_by_camera={item['camera']:[r['frame'] for r in observations if item['camera'] in r['views']] for item in manifest['videos']},
            short_track=not any(s['observations']>=min_observations for s in segment_evidence),
            visibility_distance_score=observations_count/max(median_range,1.) if median_range is not None else 0.))
    return result,manifest


