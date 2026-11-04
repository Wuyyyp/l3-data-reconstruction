"""Jointly fit one per-object/per-frame 9DOF pose to all seven SAM3 views.

Arguments: --gpus lists EGL physical GPU IDs; --evaluations limits coarse search
per frame (default 220); --chunk-size controls CPU/GPU work units (default 8);
--overwrite recomputes existing pose chunks. This is an isolated trimesh test.
Example: python utils/optimize_9dof.py --gpus 4 5 6 7
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
import sys
import time
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh
import pyglet
pyglet.options['shadow_window']=False
import pyrender
import register_baseline as base

DATA=base.INPUT
OUT=DATA/'pkl'
LOWER=np.array([-1.,-1.,-.45, np.log(.75),np.log(.75),np.log(.75), -12.,-12.,-20.])
UPPER=np.array([1.,1.,.45, np.log(1.3),np.log(1.3),np.log(1.3), 12.,12.,20.])
INITIAL_STEP=np.array([.16,.16,.07,.04,.04,.04,2.,2.,3.])
CACHE={}
RENDERER=None
INFOS=None
TRACKS=None
ITEMS=None
GPU=None


def initialize(gpu_queue):
    global RENDERER,INFOS,TRACKS,ITEMS,GPU
    GPU=int(gpu_queue.get())
    os.environ['EGL_DEVICE_ID']=str(GPU)
    cv2.setNumThreads(1)
    INFOS,TRACKS,manifest=base.inputs()
    ITEMS=manifest['videos']
    RENDERER=pyrender.OffscreenRenderer(256,256)


def model_geometry(identity):
    row=base.asset_rows()[identity]
    cache_key=row['model_id']
    if cache_key in CACHE:return CACHE[cache_key]
    scene=trimesh.load(str(Path(row['asset_path']).expanduser()),force='scene',process=False)
    meshes=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        mesh=scene.geometry[name].copy()
        mesh.apply_transform(base.BLENDER_FROM_GLTF@transform)
        meshes.append(mesh)
    bounds=np.array([m.bounds for m in meshes]);lo=bounds[:,0].min(axis=0);hi=bounds[:,1].max(axis=0)
    normal=np.eye(4);normal[:3,:3]=np.diag(1/(hi-lo));normal[:3,3]=-(lo+hi)/2/(hi-lo)
    for mesh in meshes:mesh.apply_transform(normal)
    full=trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices,faces=m.faces,process=False) for m in meshes])
    # Vertex clustering is a temporary silhouette proxy, never a modified asset.
    quantized=np.round(full.vertices*128).astype(np.int32)
    _,inverse=np.unique(quantized,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    vertices=np.column_stack([np.bincount(inverse,weights=full.vertices[:,axis])/count for axis in range(3)])
    faces=inverse[full.faces]
    valid=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])
    faces=faces[valid]
    _,unique=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
    proxy=trimesh.Trimesh(vertices=vertices,faces=faces[unique],process=False)
    result=dict(full=pyrender.Mesh.from_trimesh(full,smooth=False),proxy=pyrender.Mesh.from_trimesh(proxy,smooth=False),
        normalization=normal,full_triangles=len(full.faces),proxy_triangles=len(proxy.faces),asset=row)
    CACHE[cache_key]=result
    print('MODEL GPU',GPU,identity,'faces',len(full.faces),'proxy',len(proxy.faces),flush=True)
    return result


def decode_pose(baseline,state):
    lengths=np.linalg.norm(baseline[:3,:3],axis=0)
    original_rotation=baseline[:3,:3]/lengths
    pose=np.eye(4)
    pose[:3,:3]=original_rotation@Rotation.from_euler('xyz',state[6:],degrees=True).as_matrix()@np.diag(lengths*np.exp(state[3:6]))
    # Translation is explicit canonical ego XYZ (right, forward, up), in meters.
    pose[:3,3]=baseline[:3,3]+state[:3]
    return pose


def load_views(identity,frame_id):
    views=[]
    for item in ITEMS:
        path=DATA/'mask-videos'/'masks'/item['camera']/f'{frame_id:06d}.npz'
        key=f'obj_{identity:03d}'
        with np.load(path) as data:
            if key+'_bbox' not in data:continue
            x,y,w,h=map(int,data[key+'_bbox'])
            pixels=np.unpackbits(data[key+'_bits'],count=w*h).reshape(h,w).astype(bool)
        # Tiny detections carry too little shape information for nine parameters.
        if int(pixels.sum())<64:continue
        K,E=base.camera_for(INFOS[frame_id],item['camera'])
        observed=base.scene_depth(item['camera'],frame_id)
        if observed is not None:
            corners=np.array([[a,b,c,1.] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5]])
            near=float((E@TRACKS[frame_id][identity]@corners.T)[2].min())
            # Ignore a recovered mask when most pixels belong to a nearer occluder.
            support=pixels & (observed[y:y+h,x:x+w]+2.>=near)
            if np.count_nonzero(support)<.25*np.count_nonzero(pixels):continue
        padding=max(48,round(max(w,h)*.75))
        x0=max(0,x-padding);y0=max(0,y-padding)
        x1=min(item['width'],x+w+padding);y1=min(item['height'],y+h+padding)
        crop=np.zeros((y1-y0,x1-x0),np.uint8)
        crop[y-y0:y-y0+h,x-x0:x-x0+w]=pixels
        scale=min(1.,320/max(crop.shape))
        rw=max(8,round(crop.shape[1]*scale));rh=max(8,round(crop.shape[0]*scale))
        target=cv2.resize(crop,(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
        A=np.array([[rw/crop.shape[1],0.,-x0*rw/crop.shape[1]],
                    [0.,rh/crop.shape[0],-y0*rh/crop.shape[0]],[0.,0.,1.]])
        views.append(dict(name=item['camera'],item=item,intrinsic=K,extrinsic=E,proxy_intrinsic=A@K,
            target=target,full_bbox=(x,y,w,h),full_mask=pixels))
        if observed is not None:
            views[-1].update(scene_depth=observed,proxy_scene_depth=cv2.resize(observed[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_NEAREST))
    return views


def make_scene(mesh):
    scene=pyrender.Scene()
    node=scene.add(mesh)
    return scene,node


def render_mask(scene,node,pose,view,full=False):
    item=view['item']
    if full:
        width,height=item['width'],item['height'];K=view['intrinsic']
    else:
        height,width=view['target'].shape;K=view['proxy_intrinsic']
    RENDERER.viewport_width=width;RENDERER.viewport_height=height
    camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.1,zfar=300.)
    camera_node=scene.add(camera,pose=np.linalg.inv(view['extrinsic'])@base.CV_FROM_GL)
    scene.set_pose(node,pose=pose)
    depth=RENDERER.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY|pyrender.RenderFlags.SKIP_CULL_FACES)
    scene.remove_node(camera_node)
    observed=view.get('scene_depth' if full else 'proxy_scene_depth')
    return (depth>0) if observed is None else (depth>0)&(depth<=observed+2.)


def view_iou(mask,view,full=False):
    if full:
        x,y,w,h=view['full_bbox'];target=view['full_mask']
        intersection=int((mask[y:y+h,x:x+w]&target).sum())
        union=int(mask.sum())+int(target.sum())-intersection
    else:
        target=view['target'];intersection=int((mask&target).sum());union=int((mask|target).sum())
    return intersection/union if union else 0.


def score(scene,node,pose,views,full=False):
    if not views:return 0.,{}
    values={view['name']:view_iou(render_mask(scene,node,pose,view,full),view,full) for view in views}
    return float(np.mean(list(values.values()))),values


def optimize_frame(identity,frame_id,geometry,budget,warm):
    baseline=TRACKS[frame_id][identity]
    views=load_views(identity,frame_id)
    if not views:
        return dict(object_id=identity,frame=frame_id,status='no_qualified_mask',state=np.zeros(9).tolist(),
            canonical_ego_from_unit_mesh=baseline.tolist(),source_ego_from_unit_mesh=(base.SOURCE_FROM_CANONICAL@baseline).tolist(),
            translation_ego_xyz_m=[0.,0.,0.],scale_xyz=[1.,1.,1.],euler_local_xyz_degrees=[0.,0.,0.],views=[],evaluations=0),np.zeros(9)
    scene,node=make_scene(geometry['proxy'])
    cache={}
    best_state=np.zeros(9);best_score=-1.
    def objective(state):
        nonlocal best_state,best_score
        state=np.clip(state,LOWER,UPPER)
        key=tuple(np.round(state,8))
        if key not in cache:
            value,_=score(scene,node,decode_pose(baseline,state),views)
            cache[key]=value
            if value>best_score+1e-8 or (abs(value-best_score)<=1e-8 and np.linalg.norm(state)<np.linalg.norm(best_state)):
                best_score=value;best_state=state.copy()
        return -cache[key]
    objective(np.zeros(9))
    if warm is not None:objective(warm)
    start=best_state.copy()
    # Both signs on each axis avoid a simplex initially facing the wrong side.
    for axis in range(9):
        for sign in [-1.,1.]:
            candidate=start.copy();candidate[axis]+=sign*INITIAL_STEP[axis];objective(candidate)
    start=best_state.copy()
    simplex=[start]
    for axis in range(9):
        candidate=start.copy();candidate[axis]+=INITIAL_STEP[axis]
        simplex.append(np.clip(candidate,LOWER,UPPER))
    remaining=max(10,budget-len(cache)-36)
    minimize(objective,start,method='Nelder-Mead',bounds=list(zip(LOWER,UPPER)),
             options=dict(maxfev=remaining,xatol=.002,fatol=.00015,initial_simplex=np.array(simplex),adaptive=True))
    # Finish with a shrinking bidirectional pattern search in all nine axes.
    for fraction in [.5,.2]:
        start=best_state.copy()
        for axis in range(9):
            for sign in [-1.,1.]:
                if len(cache)>=budget:break
                candidate=best_state.copy();candidate[axis]+=sign*INITIAL_STEP[axis]*fraction;objective(candidate)
    state=best_state.copy()
    full_scene,full_node=make_scene(geometry['full'])
    before,before_views=score(full_scene,full_node,baseline,views,full=True)
    candidates=[np.zeros(9),state]
    if warm is not None:candidates.append(warm)
    accepted_state=np.zeros(9);after=before;after_views=before_views
    for candidate in candidates[1:]:
        value,values=score(full_scene,full_node,decode_pose(baseline,candidate),views,full=True)
        if value>after:
            after=value;after_views=values;accepted_state=candidate.copy()
    pose=decode_pose(baseline,accepted_state)
    result=dict(object_id=identity,frame=frame_id,status='optimized' if after>before+1e-6 else 'baseline_retained',
        state=accepted_state.tolist(),translation_ego_xyz_m=accepted_state[:3].tolist(),
        scale_xyz=np.exp(accepted_state[3:6]).tolist(),euler_local_xyz_degrees=accepted_state[6:].tolist(),
        canonical_ego_from_unit_mesh=pose.tolist(),source_ego_from_unit_mesh=(base.SOURCE_FROM_CANONICAL@pose).tolist(),
        views=[v['name'] for v in views],mean_iou_before=before,mean_iou_after=after,
        per_view_iou_before=before_views,per_view_iou_after=after_views,evaluations=len(cache))
    return result,accepted_state


def run_chunk(payload):
    identity,frames,budget,overwrite=payload
    path=OUT/'chunks'/f'obj-{identity:03d}_{frames[0]:06d}_{frames[-1]:06d}.json'
    if path.exists() and not overwrite:return json.loads(path.read_text())
    geometry=model_geometry(identity)
    results=[];warm=None;start=time.monotonic()
    for frame_id in frames:
        result,warm=optimize_frame(identity,frame_id,geometry,budget,warm)
        results.append(result)
        print('FIT GPU',GPU,'obj',identity,'frame',frame_id,result['status'],
            round(result.get('mean_iou_before',0),3),'->',round(result.get('mean_iou_after',0),3),flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n')
    print('CHUNK DONE',identity,frames[0],round(time.monotonic()-start,1),'seconds',flush=True)
    return results


def export_results(results,args):
    source=DATA/'result_interpolation_optimized_first101.pkl'
    payload=pickle.load(source.open('rb'))
    from concurrent.futures import ThreadPoolExecutor
    def normalization_for(item):
        identity,asset=item
        model=trimesh.load(str(Path(asset['asset_path']).expanduser()),force='scene',process=False)
        model.apply_transform(base.BLENDER_FROM_GLTF)
        lo,hi=model.bounds
        transform=np.eye(4);transform[:3,:3]=np.diag(1/(hi-lo));transform[:3,3]=-(lo+hi)/2/(hi-lo)
        return identity,transform@base.BLENDER_FROM_GLTF
    with ThreadPoolExecutor(max_workers=4) as pool:
        normalizations=dict(pool.map(normalization_for,base.asset_rows().items()))
    by_frame={index:{} for index in range(101)}
    for result in results:
        identity=result['object_id']
        pose=np.array(result['canonical_ego_from_unit_mesh'])
        result['canonical_ego_from_gltf_scene']=(pose@normalizations[identity]).tolist()
        result['source_ego_from_gltf_scene']=(base.SOURCE_FROM_CANONICAL@pose@normalizations[identity]).tolist()
        reference=np.array(payload['infos'][result['frame']]['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        result['world_from_gltf_scene']=(reference@base.SOURCE_FROM_CANONICAL@pose@normalizations[identity]).tolist()
        by_frame[result['frame']][str(identity)]=result
    for index,info in enumerate(payload['infos']):
        info['mesh_poses_9dof']=by_frame[index]
    payload['metadata']['multiview_mesh_9dof']=dict(
        schema_version=1,source_pkl=source.name,objects=base.asset_rows(),
        parameter_order=['translation_x_m','translation_y_m','translation_z_m','log_scale_x','log_scale_y','log_scale_z',
                         'local_euler_x_deg','local_euler_y_deg','local_euler_z_deg'],
        translation_space='canonical ego: X right, Y forward, Z up',
        rotation='scipy Rotation.from_euler xyz, extrinsic XYZ; local rotation composed after original box yaw, before scale',
        pose='4x4 left-multiplying column vectors of unit-AABB-centered mesh; a single shared pose per object/frame in all cameras',
        objective='Mean instance silhouette IoU over every qualified matched camera; exact native-resolution full-mesh verification; no lower mean accepted',
        lower_bounds=LOWER.tolist(),upper_bounds=UPPER.tolist(),max_evaluations=args.evaluations,
        mask_min_area_px=64,proxy_vertex_grid=128,proxy_roi_longest_edge=320,
        renderer='trimesh + pyrender/EGL, isolated experiment; full assets unmodified',
        original_annotations='All existing source fields including gt_boxes preserved; optimized matrices are in infos[i].mesh_poses_9dof')
    destination=OUT/'result_interpolation_optimized_first101_mesh_9dof.pkl'
    with destination.open('wb') as handle:pickle.dump(payload,handle,protocol=pickle.HIGHEST_PROTOCOL)
    qualified=[r for r in results if 'mean_iou_before' in r]
    summary=dict(pkl=str(destination.relative_to(DATA)),object_frames=len(results),qualified_object_frames=len(qualified),
        improved=sum(r['mean_iou_after']>r['mean_iou_before']+1e-6 for r in qualified),
        mean_iou_before=float(np.mean([r['mean_iou_before'] for r in qualified])),
        mean_iou_after=float(np.mean([r['mean_iou_after'] for r in qualified])),objects={})
    for identity in base.asset_rows():
        rows=[r for r in qualified if r['object_id']==identity]
        summary['objects'][str(identity)]=dict(qualified_frames=len(rows),
            mean_iou_before=float(np.mean([r['mean_iou_before'] for r in rows])) if rows else None,
            mean_iou_after=float(np.mean([r['mean_iou_after'] for r in rows])) if rows else None)
    (OUT/'optimization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('OPTIMIZATION COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6,7])
    p.add_argument('--evaluations',type=int,default=220)
    p.add_argument('--chunk-size',type=int,default=8)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    infos,tracks,_=base.inputs()
    jobs=[]
    for identity in base.asset_rows():
        frames=[fi for fi,t in enumerate(tracks) if identity in t]
        for start in range(0,len(frames),args.chunk_size):
            jobs.append((identity,frames[start:start+args.chunk_size],args.evaluations,args.overwrite))
    context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in args.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=context,
                initializer=initialize,initargs=(queue,)) as pool:
            chunks=list(pool.map(run_chunk,jobs))
    results=sorted([r for chunk in chunks for r in chunk],key=lambda x:(x['frame'],x['object_id']))
    export_results(results,args)

if __name__=='__main__':main()
