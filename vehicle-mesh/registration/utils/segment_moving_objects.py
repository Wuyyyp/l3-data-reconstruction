"""Segment movable classes independently of selected mesh tracks; retain >1 m movers.

Needs prepared manifest.json, videos, and result_interpolation_optimized_first101.pkl.
No asset manifest or model mapping is required. Instance IDs are camera/prompt-local;
annotation_id, when available, supplies the source cross-camera identity.
"""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,pickle,subprocess,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from motion_mask_utils import (PROMPTS,CAMERA_IDS,annotation_objects,annotation_world_tracks,
    camera_matrices,compatible,consistent_annotation,depth_world_centroid,motion_decision,pack_mask,unpack_mask)
from sam3_video import Sam3VideoTracker,checkpoint,match_mask_indices
from box_projection import project_track_box

TRACKER=None
GPU=None


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def initialize(queue):
    global TRACKER,GPU
    GPU=int(queue.get());os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1);TRACKER=Sam3VideoTracker()


def projections(info,item,prompt):
    K,C,T=camera_matrices(info,item['camera']);E=np.linalg.inv(C)@T;boxes=[]
    for identity,category,b in annotation_objects(info):
        if not compatible(prompt,category):continue
        p=project_track_box(length=b[3],width=b[4],height=b[5],center_xyz=b[:3],heading=b[6],intrinsic=K,extrinsic=E,image_width=item['width'],image_height=item['height'])
        if p.visible_bbox_xywh:boxes.append((identity,p.visible_bbox_xywh))
    return boxes


def load_depth(root,name,fi,shape):
    if not root:return None
    if name not in CAMERA_IDS:return None
    p=Path(root)/'lidar_depth'/f'{fi:06d}_{CAMERA_IDS[name]:02d}.npy'
    if not p.exists():return None
    arrays=np.load(p,allow_pickle=True).item()
    if arrays['mask'].shape!=shape:raise ValueError(f'depth/image shape mismatch: {p}')
    depth=np.full(shape,np.nan,np.float32);depth[arrays['mask']]=arrays['value'];return depth


def encoder(path,width,height,fps,pixel_format,log):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt',pixel_format,'-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-profile:v','high','-pix_fmt','yuv420p','-crf','18','-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=log)


def export_camera(root,source,item,manifest,tracks,prompts):
    name=item['camera'];height,width=item['height'],item['width'];n=manifest['frames']
    folder=root/'masks'/name;unions=root/'union-masks'/name;folder.mkdir(parents=True,exist_ok=True);unions.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source/item['file']));counts=defaultdict(int);total_masks=0
    with (root/'logs'/f'{name}-encode.log').open('w') as log:
        writers=[encoder(root/'videos'/f'{name}.mp4',width,height,manifest['fps'],'gray',log),encoder(root/'overlays'/f'{name}.mp4',width,height,manifest['fps'],'bgr24',log)]
        try:
            for fi in range(n):
                ok,frame=cap.read()
                if not ok:raise ValueError(f'source video ends at {fi}: {name}')
                union=np.zeros((height,width),bool);arrays={};overlay=frame.copy()
                for pi,_ in enumerate(prompts):
                    with np.load(root/'candidates'/name/f'p{pi}'/f'{fi:06d}.npz') as data:
                        for key in data.files:
                            if not key.endswith('_bbox'):continue
                            identity=key[:-5];track=tracks[identity]
                            if track['motion']['state']!='moving':continue
                            mask=unpack_mask(data[key],data[identity+'_bits'],(height,width));union|=mask
                            arrays[key]=data[key];arrays[identity+'_bits']=data[identity+'_bits'];counts[track['prompt']]+=1;total_masks+=1
                            color=np.asarray([(41,190,255),(75,255,85),(255,140,40),(220,80,210),(210,230,55),(230,120,70)][pi%6])
                            overlay[mask]=(overlay[mask]*.5+color*.5).astype(np.uint8)
                np.savez_compressed(folder/f'{fi:06d}.npz',**arrays)
                if not cv2.imwrite(str(unions/f'{fi:06d}.png'),union.astype(np.uint8)*255):raise IOError('cannot write union mask')
                cv2.putText(overlay,f'MOVING > {manifest["displacement_m"]:g} m | {name} | {fi}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                writers[0].stdin.write((union.astype(np.uint8)*255).tobytes());writers[1].stdin.write(overlay.tobytes())
                if fi in {0,n//2,n-1}:cv2.imwrite(str(root/'review'/f'{name}_{fi:06d}.jpg'),overlay)
            if cap.read()[0]:raise ValueError(f'video has more than manifest frames: {name}')
            for writer in writers:writer.stdin.close()
            for writer in writers:
                if writer.wait()!=0:raise RuntimeError(f'video encoding failed: {name}')
        finally:
            cap.release()
            for writer in writers:
                if writer.poll() is None:writer.terminate();writer.wait()
    return dict(instance_frames=total_masks,per_prompt_instance_frames=dict(counts))


def run_camera(payload):
    item,cfg=payload;source=Path(cfg['input']);root=Path(cfg['output']);name=item['camera'];shape=(item['height'],item['width'])
    final_marker=root/'masks'/name/'assignments.json'
    if final_marker.exists():return json.load(final_marker.open())['summary']
    infos=pickle.load((source/'result_interpolation_optimized_first101.pkl').open('rb'))['infos'];world_tracks=annotation_world_tracks(infos)
    all_tracks={};started=time.time()
    for pi,prompt in enumerate(cfg['prompts']):
        directory=root/'candidates'/name/f'p{pi}';directory.mkdir(parents=True,exist_ok=True);marker=directory/'tracks.json'
        if marker.exists():all_tracks.update(json.load(marker.open()));continue
        observations=defaultdict(list);print('PROMPT START',name,prompt,'GPU',GPU,flush=True)
        def consume(fi,outputs):
            arrays={};candidates=[]
            for sid,score,pixels in zip(outputs['out_obj_ids'],outputs['out_probs'],outputs['out_binary_masks']):
                mask=np.asarray(pixels,bool)
                if mask.shape!=shape:raise ValueError('SAM3 output resolution mismatch')
                if float(score)<cfg['confidence'] or int(mask.sum())<cfg['min_area']:continue
                candidates.append((int(sid),float(score),mask))
            boxes=projections(infos[fi],item,prompt)
            matches=match_mask_indices(np.stack([x[2] for x in candidates]),boxes,.05) if candidates else {}
            by_candidate={v:k for k,v in matches.items()};depth=None;K,C,_=camera_matrices(infos[fi],name)
            for ci,(sid,score,mask) in enumerate(candidates):
                identity=f'p{pi}_obj_{sid}';bbox,bits=pack_mask(mask);arrays[identity+'_bbox']=bbox;arrays[identity+'_bits']=bits
                row=dict(frame=fi,confidence=score,pixels=int(mask.sum()),annotation_id=by_candidate.get(ci))
                if row['annotation_id'] is None and cfg['scene_depth_root']:
                    if depth is None:depth=load_depth(cfg['scene_depth_root'],name,fi,shape)
                    if depth is not None:
                        x,y,w,h=map(int,bbox)
                        # Truncation changes the sampled surface as a target enters the image.
                        if x>1 and y>1 and x+w<shape[1]-1 and y+h<shape[0]-1:
                            position=depth_world_centroid(mask,depth,K,C)
                            if position:row.update(position)
                observations[identity].append(row)
            np.savez_compressed(directory/f'{fi:06d}.npz',**arrays)
        TRACKER.infer(source/item['file'],checkpoint(),prompt,len(infos),shape,frame_consumer=consume)
        result={}
        for identity,rows in observations.items():
            ann_id=consistent_annotation(rows)
            if ann_id is not None:
                trajectory=world_tracks[ann_id];motion=motion_decision(trajectory['positions'],cfg['displacement_m']);motion['annotation_id']=ann_id;motion['trajectory_frames']=trajectory['frames']
            else:
                points=[r for r in rows if 'world_position' in r]
                motion=motion_decision([r['world_position'] for r in points],cfg['displacement_m'],source='lidar_surface_centroid',radii=[r['surface_radius_m'] for r in points])
            result[identity]=dict(camera=name,prompt=prompt,sam3_id=int(identity.rsplit('_',1)[1]),motion=motion,observations=rows)
        atomic_json(marker,result);all_tracks.update(result)
        print('PROMPT DONE',name,prompt,'tracks',len(result),'moving',sum(x['motion']['state']=='moving' for x in result.values()),flush=True)
    exported=export_camera(root,source,item,cfg,all_tracks,cfg['prompts'])
    states={s:sum(t['motion']['state']==s for t in all_tracks.values()) for s in ['moving','static','uncertain']}
    summary=dict(camera=name,frames=cfg['frames'],gpu=GPU,seconds=time.time()-started,track_states=states,**exported)
    atomic_json(final_marker,dict(summary=summary,tracks=all_tracks));print('CAMERA DONE',json.dumps(summary),flush=True);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--scene-depth-root',type=Path);parser.add_argument('--displacement-m',type=float,default=1.)
    parser.add_argument('--prompts',nargs='+',default=list(PROMPTS));parser.add_argument('--confidence',type=float,default=.35);parser.add_argument('--min-area',type=int,default=16)
    a=parser.parse_args();source=a.input.resolve();root=a.output.resolve()
    if source==root or source in root.parents:parser.error('output must be outside the input directory')
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or any(g<0 for g in a.gpus):parser.error('unique nonnegative GPU IDs required')
    if not np.isfinite(a.displacement_m) or a.displacement_m<=0:parser.error('positive finite displacement required')
    if len(set(a.prompts))!=len(a.prompts) or any(p not in PROMPTS for p in a.prompts):parser.error(f'choose unique prompts from {PROMPTS}')
    if not 0<a.confidence<=1 or a.min_area<1:parser.error('invalid mask thresholds')
    manifest=json.load((source/'manifest.json').open());infos=pickle.load((source/'result_interpolation_optimized_first101.pkl').open('rb'))['infos']
    if len(infos)!=manifest['frames'] or not infos:raise ValueError('annotation/manifest frame mismatch')
    names=[v['camera'] for v in manifest['videos']]
    if len(names)!=len(set(names)):raise ValueError('duplicate camera names')
    for item in manifest['videos']:
        if Path(item['camera']).name!=item['camera']:raise ValueError('invalid camera name')
        if not (source/item['file']).is_file():raise FileNotFoundError(item['file'])
        for info in infos:
            K,C,T=camera_matrices(info,item['camera'])
            if K.shape!=(3,3) or C.shape!=(4,4) or T.shape!=(4,4) or not all(np.isfinite(x).all() for x in [K,C,T]):raise ValueError('invalid camera calibration')
    cfg=dict(input=str(source),output=str(root),frames=manifest['frames'],fps=manifest['fps'],prompts=a.prompts,confidence=a.confidence,min_area=a.min_area,displacement_m=a.displacement_m,scene_depth_root=str(a.scene_depth_root.resolve()) if a.scene_depth_root else None,
             input_sha256=hashlib.sha256((source/'result_interpolation_optimized_first101.pkl').read_bytes()).hexdigest(),manifest=manifest,
             video_stats={v['camera']:[(source/v['file']).stat().st_size,(source/v['file']).stat().st_mtime_ns] for v in manifest['videos']},checkpoint=str(checkpoint()))
    if (root/'config.json').exists():
        if json.load((root/'config.json').open())!=cfg:raise ValueError('output belongs to different inputs/settings; use a fresh directory')
    elif root.exists() and any(root.iterdir()):raise ValueError('nonempty output without config; use a fresh directory')
    for name in ['logs','review','masks','union-masks','videos','overlays','candidates']:(root/name).mkdir(parents=True,exist_ok=True)
    atomic_json(root/'config.json',cfg);started=time.time();context=mp.get_context('spawn')
    with context.Manager() as manager:
        q=manager.Queue()
        for g in a.gpus:q.put(g)
        with ProcessPoolExecutor(max_workers=len(a.gpus),mp_context=context,initializer=initialize,initargs=(q,)) as pool:
            results=list(pool.map(run_camera,[(item,cfg) for item in manifest['videos']]))
    atomic_json(root/'manifest.json',dict(complete=True,mode='actual_motion',motion_rule='clip maximum pairwise world XYZ displacement > threshold',threshold_m=a.displacement_m,frames=cfg['frames'],fps=cfg['fps'],prompts=a.prompts,cameras=results,seconds=time.time()-started,
        mask_storage='Exact cropped binary NPZ instances; full-resolution lossless PNG unions. MP4 is preview only.',
        identities='Camera/prompt-local SAM3 IDs; annotation_id provides source identity where matched. Prompt instances may overlap; union masks count each pixel once.',
        uncertainty='Unmatched tracks use LiDAR visible-surface centroids, not exact object centres. Missing or ambiguous metric evidence is uncertain and omitted from moving masks.',
        selected_vehicle_masks_unchanged=True))
    print('MOVING MASKS COMPLETE',flush=True)

if __name__=='__main__':main()
