"""Repair floating/twisted fits using ground contact and sequence constraints.

Arguments: --gpus lists physical GPU IDs; --evaluations is the moving-frame
search budget (default 140); --static-evaluations is the joint stationary-sequence
budget (default 240); --overwrite recomputes completed object jobs.
Example: python utils/optimize_grounded_temporal.py --gpus 4 5 6 7
"""
from __future__ import annotations
import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import optimize_9dof as opt
import register_baseline as base

DATA=base.INPUT
VERSION='grounded_world_sequence_v4_dynamic_movers_scale_limited'
WORLD=[]
PREVIOUS=None
LOWER=np.array([-.7,-.7,-.35,np.log(.85),np.log(.85),np.log(.85),-.4,-.4,-6.])
UPPER=np.array([.7,.7,.35,np.log(1.2),np.log(1.2),np.log(1.2),.4,.4,6.])
STEP=np.array([.10,.10,.04,.025,.025,.025,.10,.10,1.])


def initialize(queue):
    global WORLD,PREVIOUS
    opt.initialize(queue)
    WORLD=[np.array(info['ego2global_transformation_matrix_camera_reference_offset_optimized'])@base.SOURCE_FROM_CANONICAL for info in opt.INFOS]
    PREVIOUS=pickle.load((DATA/'pkl/unconstrained-backup/result_interpolation_optimized_first101_mesh_9dof.pkl').open('rb'))
    if os.environ.get('MULTIVIEW_PERF_ROOT'):
        import benchmark_runtime; benchmark_runtime.install('optimize')


def smooth(values,window=11):
    values=np.asarray(values,dtype=float)
    size=min(window,len(values) if len(values)%2 else len(values)-1)
    return savgol_filter(values,size,2,axis=0,mode='interp') if size>=3 else values.copy()


def context_for(identity):
    frames=[index for index,tracks in enumerate(opt.TRACKS) if identity in tracks]
    poses=np.array([WORLD[fi]@opt.TRACKS[fi][identity] for fi in frames])
    lengths=np.array([np.linalg.norm(p[:3,:3],axis=0) for p in poses])
    yaw=np.unwrap(np.arctan2(poses[:,1,0],poses[:,0,0]))
    grounds=[];speeds=[];previous=[]
    for fi in frames:
        info=opt.INFOS[fi];j=info['track_id'].index(identity);box=np.array(info['gt_boxes'][j])
        ground=np.r_[box[:3],1.];ground[2]-=box[5]/2
        T=np.array(info['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        grounds.append((T@ground)[2]);speeds.append(float(np.linalg.norm(info['vels'][j])))
        previous.append(WORLD[fi]@np.array(PREVIOUS['infos'][fi]['mesh_poses_9dof'][str(identity)]['canonical_ego_from_unit_mesh']))
    return dict(identity=identity,frames=frames,baseline_world=poses,lengths=np.median(lengths,axis=0),
        yaw=yaw,grounds=np.array(grounds),speeds=np.array(speeds),previous=np.array(previous),
        stationary=bool(np.median(speeds)<.2 and identity not in {int(value) for value in (DATA/'dynamic_vehicle_ids.txt').read_text().split()}))


def contact_geometry(geometry):
    primitive=geometry['full'].primitives[0]
    vertices=np.asarray(primitive.positions,dtype=float)
    # Linear support in any direction is attained on the convex hull.
    return vertices[ConvexHull(vertices).vertices]


def feasible_state(state,lengths):
    state=np.clip(np.asarray(state,dtype=float),LOWER,UPPER).copy()
    # Limit the wheel-to-wheel height spread to 2.5 cm, rather than allowing
    # roll and pitch to trade visible silhouette for a lifted axle.
    anchors=np.array([[x,y,0.] for x in [-.33,.33] for y in [-.38,.38]])
    for _ in range(2):
        R=Rotation.from_euler('xyz',state[6:],degrees=True).as_matrix()
        heights=(anchors*(lengths*np.exp(state[3:6])))@R[2]
        spread=np.ptp(heights)
        if spread<=.025:break
        state[6:8]*=.025/spread
    return state


def world_pose(center,yaw,lengths,ground,state,support):
    state=feasible_state(state,lengths)
    R=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('xyz',state[6:],degrees=True).as_matrix()
    A=R@np.diag(lengths*np.exp(state[3:6]))
    pose=np.eye(4);pose[:3,:3]=A
    pose[:2,3]=center[:2]+state[:2]
    pose[2,3]=ground+state[2]-float((support@A[2]).min())
    return pose,state


def wheel_contact_loss(pose_world,frame_id,views,support):
    anchors=[]
    for sx in [-1,1]:
        for sy in [-1,1]:
            patch=support[(support[:,0]*sx>.12)&(support[:,1]*sy>.22)&(support[:,2]<-.3)]
            if not len(patch):continue
            low=patch[:,2].min();patch=patch[patch[:,2]<low+.01]
            anchor=np.mean(patch,axis=0);anchor[2]=low;anchors.append(anchor)
    if not anchors:return 0.
    homogeneous=np.c_[anchors,np.ones(len(anchors))]
    canonical=np.linalg.inv(WORLD[frame_id])@pose_world
    errors=[]
    for view in views:
        x,y,w,h=view['full_bbox']
        if h<24 or w<16:continue
        if 'lower_contour' not in view:
            pixels=view['full_mask']
            view['lower_contour']=np.where(pixels,np.arange(h)[:,None]+y,-np.inf).max(axis=0)
        points=(view['extrinsic']@canonical@homogeneous.T).T[:,:3]
        points=points[points[:,2]>.1]
        uv=(view['intrinsic']@points.T).T
        uv=uv[:,:2]/uv[:,2:]
        chosen=[]
        for px,py in sorted(uv,key=lambda point:-point[1]):
            if not x<=px<x+w or any(abs(px-other)<w*.2 for other in chosen):continue
            radius=max(2,round(w*.03));left=max(0,round(px-x)-radius);right=min(w,round(px-x)+radius+1)
            bottom=view['lower_contour'][left:right];bottom=bottom[np.isfinite(bottom)]
            if not len(bottom):continue
            target=float(np.percentile(bottom,85))
            error=(target-py)/h
            # A visible lower contour below the predicted tire is evidence of
            # floating; an upper contour may instead be foreground occlusion.
            errors.append(min(.5,max(error,0.)+.15*max(-error,0.)))
            chosen.append(px)
            if len(chosen)==2:break
    return float(np.mean(errors)) if errors else 0.


def search(objective,start,budget):
    cache={};best=start.copy();best_value=-np.inf
    def evaluate(state):
        nonlocal best,best_value
        state=np.clip(np.asarray(state),LOWER,UPPER)
        key=tuple(np.round(state,8))
        if key not in cache:
            value=objective(state);cache[key]=value
            if value>best_value:best_value=value;best=state.copy()
        return -cache[key]
    evaluate(start)
    for axis in range(9):
        for sign in [-1,1]:
            candidate=start.copy();candidate[axis]+=sign*STEP[axis];evaluate(candidate)
    simplex=[best.copy()]
    for axis in range(9):
        candidate=best.copy();candidate[axis]+=STEP[axis];simplex.append(np.clip(candidate,LOWER,UPPER))
    minimize(evaluate,best,method='Nelder-Mead',bounds=list(zip(LOWER,UPPER)),options=dict(
        maxfev=max(10,budget-len(cache)-18),xatol=.001,fatol=.0001,adaptive=True,initial_simplex=np.array(simplex)))
    for axis in range(9):
        for sign in [-1,1]:
            candidate=best.copy();candidate[axis]+=sign*STEP[axis]*.2;evaluate(candidate)
    return best,len(cache)


def record_pose(identity,fi,pose_world,ground,support,before,before_views,after,after_views,evaluations,mode):
    canonical=np.linalg.inv(WORLD[fi])@pose_world
    original=opt.TRACKS[fi][identity]
    dimensions=np.linalg.norm(canonical[:3,:3],axis=0);original_dimensions=np.linalg.norm(original[:3,:3],axis=0)
    rotation=canonical[:3,:3]/dimensions
    original_rotation=original[:3,:3]/original_dimensions
    euler=Rotation.from_matrix(original_rotation.T@rotation).as_euler('xyz',degrees=True)
    state=np.r_[canonical[:3,3]-original[:3,3],np.log(dimensions/original_dimensions),euler]
    rotation_world=pose_world[:3,:3]/dimensions
    world_euler=Rotation.from_matrix(rotation_world).as_euler('xyz',degrees=True)
    bottom=float((support@pose_world[2,:3]+pose_world[2,3]).min())
    record=dict(object_id=identity,frame=fi,status='grounded_temporal_optimized',constraint_mode=mode,
        state=state.tolist(),translation_ego_xyz_m=state[:3].tolist(),scale_xyz=np.exp(state[3:6]).tolist(),
        euler_local_xyz_degrees=euler.tolist(),world_euler_xyz_degrees=world_euler.tolist(),
        canonical_ego_from_unit_mesh=canonical.tolist(),source_ego_from_unit_mesh=(base.SOURCE_FROM_CANONICAL@canonical).tolist(),
        world_from_unit_mesh=pose_world.tolist(),ground_world_z_m=float(ground),mesh_bottom_world_z_m=bottom,
        ground_contact_error_m=bottom-ground,views=list(after_views),mean_iou_before=before,mean_iou_after=after,
        per_view_iou_before=before_views,per_view_iou_after=after_views,evaluations=evaluations)
    if not after_views:
        record['status']='no_qualified_mask_temporal_propagation'
        for key in ['mean_iou_before','mean_iou_after','per_view_iou_before','per_view_iou_after']:record.pop(key,None)
    return record


def optimize_stationary(context,geometry,support,budget):
    identity=context['identity'];frames=context['frames'];lengths=context['lengths']
    center=np.median(context['baseline_world'][:,:3,3],axis=0);yaw=float(np.median(context['yaw']));ground=float(np.median(context['grounds']))
    views=[opt.load_views(identity,fi) for fi in frames]
    scene,node=opt.make_scene(geometry['proxy'])
    previous_lengths=np.linalg.norm(context['previous'][:,:3,:3],axis=1)
    start=np.zeros(9);start[3:6]=np.clip(np.median(np.log(previous_lengths/lengths),axis=0),LOWER[3:6],UPPER[3:6])
    def objective(state):
        pose,_=world_pose(center,yaw,lengths,ground,state,support)
        values=[]
        for fi,frame_views in zip(frames,views):
            if frame_views:
                value,_=opt.score(scene,node,np.linalg.inv(WORLD[fi])@pose,frame_views)
                values.append(value-.55*wheel_contact_loss(pose,fi,frame_views,support))
        # Every observed frame participates. One physical parked-car pose and
        # one set of dimensions are shared over the entire sequence.
        return (float(np.mean(values)) if values else 0.)-.002*float(np.mean((state[6:8]/.4)**2))-.003*(state[2]/.25)**2
    best,evaluations=search(objective,start,budget)
    pose,best=world_pose(center,yaw,lengths,ground,best,support)
    full_scene,full_node=opt.make_scene(geometry['full']);results=[]
    for fi,frame_views in zip(frames,views):
        before,before_views=opt.score(full_scene,full_node,opt.TRACKS[fi][identity],frame_views,full=True)
        canonical=np.linalg.inv(WORLD[fi])@pose
        after,after_views=opt.score(full_scene,full_node,canonical,frame_views,full=True)
        record=record_pose(identity,fi,pose,ground+best[2],support,before,before_views,after,after_views,evaluations,'stationary_shared_world_pose')
        record.update(ground_annotation_world_z_m=ground,fitted_ground_correction_m=float(best[2]))
        results.append(record)
    print('STATIONARY DONE',identity,'frames',len(frames),'IoU',np.mean([r['mean_iou_after'] for r in results if 'mean_iou_after' in r]) if any('mean_iou_after' in r for r in results) else None,flush=True)
    return results


def optimize_moving(context,geometry,support,budget):
    identity=context['identity'];frames=context['frames'];lengths=context['lengths']
    reference_centers=smooth(context['previous'][:,:3,3],11)
    previous_yaw=np.unwrap(np.arctan2(context['previous'][:,1,0],context['previous'][:,0,0]))
    reference_yaw=smooth(previous_yaw,15)
    ground=smooth(context['grounds'],15)
    shared_scale=np.clip(np.median(np.log(np.linalg.norm(context['previous'][:,:3,:3],axis=1)/lengths),axis=0),LOWER[3:6],UPPER[3:6])
    scene,node=opt.make_scene(geometry['proxy']);states=[];warm=None;all_views=[];counts=[]
    for index,fi in enumerate(frames):
        views=opt.load_views(identity,fi);all_views.append(views)
        start=np.zeros(9);start[3:6]=shared_scale
        if warm is not None:start=warm.copy()
        if not views:
            states.append(start);counts.append(0);warm=start
            continue
        def objective(state):
            pose,_=world_pose(reference_centers[index],reference_yaw[index],lengths,ground[index],state,support)
            value,_=opt.score(scene,node,np.linalg.inv(WORLD[fi])@pose,views)
            # Spatial changes are corrections to a smooth moving trajectory,
            # not penalties on the legitimate motion of the vehicle itself.
            penalty=.018*np.mean((state[:2]/.3)**2)+.025*np.mean(((state[3:6]-shared_scale)/.04)**2)
            penalty+=.004*np.mean((state[6:8]/.4)**2)+.006*(state[8]/3.)**2
            if warm is not None:
                penalty+=.018*np.mean(((state[:2]-warm[:2])/.15)**2)
                penalty+=.015*np.mean(((state[3:6]-warm[3:6])/.025)**2)
                penalty+=.01*((state[8]-warm[8])/1.5)**2
            penalty+=.01*(state[2]/.15)**2
            if warm is not None:penalty+=.02*((state[2]-warm[2])/.04)**2
            return value-float(penalty)-.55*wheel_contact_loss(pose,fi,views,support)
        best,evaluations=search(objective,start,budget)
        _,best=world_pose(reference_centers[index],reference_yaw[index],lengths,ground[index],best,support)
        states.append(best);counts.append(evaluations);warm=best
        if fi%10==0:print('MOVING FIT',identity,fi,flush=True)
    # Smooth the complete sequence, including worker/chunk boundaries. All
    # candidates remain coupled to ground after filtering, not merely shifted.
    states=smooth(np.array(states),11)
    # Sparse/short tracks can retain a sharp log-scale kink after quadratic
    # smoothing. Bound adjacent observed poses to a 1.9% scale step.
    max_log_scale_step=float(np.log(1.019))
    for index in range(1,len(states)):
        states[index,3:6]=np.clip(states[index,3:6],
            states[index-1,3:6]-max_log_scale_step,states[index-1,3:6]+max_log_scale_step)
    full_scene,full_node=opt.make_scene(geometry['full']);results=[]
    for index,(fi,views) in enumerate(zip(frames,all_views)):
        pose,state=world_pose(reference_centers[index],reference_yaw[index],lengths,ground[index],states[index],support)
        before,before_views=opt.score(full_scene,full_node,opt.TRACKS[fi][identity],views,full=True)
        after,after_views=opt.score(full_scene,full_node,np.linalg.inv(WORLD[fi])@pose,views,full=True)
        record=record_pose(identity,fi,pose,ground[index]+state[2],support,before,before_views,after,after_views,counts[index],'moving_grounded_temporal')
        record.update(ground_annotation_world_z_m=float(ground[index]),fitted_ground_correction_m=float(state[2]))
        results.append(record)
    return results


def run_job(job):
    identity,budget,static_budget,overwrite=job
    path=DATA/'pkl'/'grounded_jobs'/f'obj-{identity:03d}.json'
    if path.exists() and not overwrite:
        saved=json.loads(path.read_text())
        if saved['version']==VERSION:return saved['results']
    start=time.monotonic();context=context_for(identity);geometry=opt.model_geometry(identity);support=contact_geometry(geometry)
    print('GROUND/TEMPORAL START',identity,'stationary',context['stationary'],'GPU',opt.GPU,flush=True)
    results=optimize_stationary(context,geometry,support,static_budget) if context['stationary'] else optimize_moving(context,geometry,support,budget)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(version=VERSION,results=results),indent=2)+'\n')
    print('GROUND/TEMPORAL DONE',identity,round(time.monotonic()-start,1),'seconds',flush=True)
    return results


def finalize_metadata(args):
    path=DATA/'pkl'/'result_interpolation_optimized_first101_mesh_9dof.pkl';payload=pickle.load(path.open('rb'))
    payload['metadata']['multiview_mesh_9dof'].update(schema_version=2,constraint_version=VERSION,
        constraints='Stationary vehicles share a constant world pose/scale across all frames; moving vehicle has sequence-smoothed world trajectory, scales and angles; adjacent observed scale changes are limited to 1.9%. All full-mesh supports remain within 5 mm of ground; near-upright roll/pitch and <=2.5 cm nominal axle height spread.',
        ground_reference='Annotated world track-box bottom initializes ground; shared ground-height correction is fitted against multi-view visible lower mask contours at predicted tire contact points, within +/-0.35 m. Stationary ground is constant, moving corrections are sequence regularized and filtered.',
        objective='Joint seven-view mask IoU subject to physical ground/upright/sequence constraints. Stationary objects optimize all observed frames jointly. Physical validity takes precedence over per-frame IoU gain.',
        world_search_lower=LOWER.tolist(),world_search_upper=UPPER.tolist())
    for key in ['lower_bounds','upper_bounds']:
        payload['metadata']['multiview_mesh_9dof'].pop(key,None)
    payload['metadata']['multiview_mesh_9dof']['static_max_evaluations']=args.static_evaluations
    with path.open('wb') as handle:pickle.dump(payload,handle,protocol=pickle.HIGHEST_PROTOCOL)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6,7]);p.add_argument('--evaluations',type=int,default=140);p.add_argument('--static-evaluations',type=int,default=240);p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    backup=DATA/'pkl'/'unconstrained-backup';backup.mkdir(parents=True,exist_ok=True)
    for name in ['result_interpolation_optimized_first101_mesh_9dof.pkl','optimization_summary.json','validation_report.json']:
        destination=backup/name
        if not destination.exists() and (DATA/'pkl'/name).exists():shutil.copy2(DATA/'pkl'/name,destination)
    context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in args.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
            chunks=list(pool.map(run_job,[(identity,args.evaluations,args.static_evaluations,args.overwrite) for identity in base.asset_rows()]))
    results=sorted([r for chunk in chunks for r in chunk],key=lambda r:(r['frame'],r['object_id']))
    opt.export_results(results,args)
    finalize_metadata(args)
    report={}
    for identity in base.asset_rows():
        rows=[r for r in results if r['object_id']==identity]
        poses=np.array([r['world_from_unit_mesh'] for r in rows]);angles=np.array([r['world_euler_xyz_degrees'] for r in rows]);lengths=np.linalg.norm(poses[:,:3,:3],axis=1)
        report[str(identity)]=dict(mode=rows[0]['constraint_mode'],frames=len(rows),
            max_ground_contact_error_m=float(max(abs(r['ground_contact_error_m']) for r in rows)),
            max_abs_world_roll_pitch_deg=float(np.abs(angles[:,:2]).max()),
            max_frame_center_step_m=float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).max(initial=0)),
            max_frame_scale_relative_step=float(np.abs(np.diff(np.log(lengths),axis=0)).max(initial=0)))
    (DATA/'pkl'/'grounded_temporal_report.json').write_text(json.dumps(dict(version=VERSION,objects=report),indent=2)+'\n')
    print('GROUNDED TEMPORAL COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
