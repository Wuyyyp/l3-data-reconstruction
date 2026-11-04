#!/usr/bin/env python3
"""Select world-motion dynamic motor vehicles and create seven-view box videos.

Arguments: --input is a source directory containing manifest.json and the source
PKL; --output is a new deliverable directory; --jobs controls CPU camera-video
workers; --speed-threshold, --min-observations, and --min-displacement define
the world-trajectory filter. This CPU step has no --gpus option.
Example: python select_dynamic_vehicles.py --input /data/sequence --output /data/selected --jobs 4
"""
from __future__ import annotations
import argparse, json, os, pickle, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, imageio_ffmpeg, numpy as np, yaml
from utils.filter_dynamic_vehicles import statistics
from utils.box_projection import project_track_box

PKL='result_interpolation_optimized_first101.pkl'
CANONICAL_FROM_SOURCE=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])

def draw_camera(input_root:Path, output_root:Path, manifest:dict, infos:list, item:dict, selected:set[int]):
    width,height=int(item['width']),int(item['height']); cap=cv2.VideoCapture(str(input_root/item['file']))
    if width%2 or height%2:raise ValueError(f"H.264 yuv420p requires even dimensions, got {width}x{height}")
    if not cap.isOpened():raise RuntimeError(f"cannot open {item['file']}")
    out=output_root/'box-videos'/f"{item['camera']}.mp4";out.parent.mkdir(parents=True,exist_ok=True)
    encoder=imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    cmd=[encoder,'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(manifest['fps']),'-i','pipe:0','-an','-c:v','libx264','-profile:v','high','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for fi,info in enumerate(infos):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f"short video {item['file']} at frame {fi}")
            if frame.shape[:2]!=(height,width):raise RuntimeError(f"frame size differs from manifest in {item['file']} at frame {fi}")
            cam=info['sensors']['cams'][item['camera']];ref=np.asarray(info['ego2global_transformation_matrix_camera_reference_offset_optimized'])
            ext=np.linalg.inv(np.asarray(cam['camera_to_world_optimized']))@ref if 'camera_to_world_optimized' in cam else np.asarray(cam['extrinsic'])
            ext=ext@np.linalg.inv(CANONICAL_FROM_SOURCE);K=np.asarray(cam['cam_intrinsic'])
            for ident,name,box in zip(info['track_id'],info['gt_names'],info['gt_boxes']):
                ident=int(ident)
                if ident not in selected or not name.startswith('VEHICLE_'):continue
                b=np.asarray(box,float);center=(CANONICAL_FROM_SOURCE@np.r_[b[:3],1.])[:3]
                result=project_track_box(length=b[3],width=b[4],height=b[5],center_xyz=center,heading=b[6]+np.pi/2,intrinsic=K,extrinsic=ext,image_width=width,image_height=height)
                if result.visible_bbox_xywh is None:continue
                x,y,w,h=result.visible_bbox_xywh;cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2);cv2.putText(frame,str(ident),(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2,cv2.LINE_AA)
            proc.stdin.write(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).tobytes())
        if cap.read()[0]:raise RuntimeError(f"video has extra frames: {item['file']}")
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f"video encoder failed: {out}")
    finally:
        cap.release()
        if proc.poll() is None:proc.kill();proc.wait()
    return str(out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--jobs',type=int,default=4);p.add_argument('--speed-threshold',type=float,default=.5);p.add_argument('--min-observations',type=int,default=8);p.add_argument('--min-displacement',type=float,default=1.5);a=p.parse_args()
    src=a.input.resolve();out=a.output.resolve()
    if a.jobs<1:p.error('--jobs must be positive')
    if out==src or src in out.parents:p.error('--output must be outside --input and must not equal it')
    if out.exists() and any(out.iterdir()):p.error(f'--output must be empty or absent: {out}')
    if not (src/'manifest.json').is_file() or not (src/PKL).is_file():p.error(f'input must contain manifest.json and {PKL}')
    out.mkdir(parents=True,exist_ok=True); infos=pickle.loads((src/PKL).read_bytes())['infos'];manifest=json.loads((src/'manifest.json').read_text())
    records,_=statistics(src,a.speed_threshold,a.min_observations,a.min_displacement)
    selected=sorted(r['track_id'] for r in records if r['dynamic'] and r['group']=='motor_vehicle')
    if not selected:raise SystemExit('No world-coordinate dynamic motor vehicles passed the filter.')
    (out/'obj_selected.txt').write_text(''.join(f'{x}\n' for x in selected));(out/'dynamic_vehicle_ids.txt').write_text(''.join(f'{x}\n' for x in selected))
    shutil.copy2(Path(__file__).resolve().parent/'class_whitelist.txt',out/'class_whitelist.txt')
    (out/'dynamic_vehicle_statistics.json').write_text(json.dumps({'coordinate_space':'world','policy':'world XY trajectory; dynamic VEHICLE_* only','distinct_dynamic_motor_vehicle_count':len(selected),'selected_ids':selected,'tracks':records},indent=2)+'\n')
    config={'version':1,'input':str(src),'selected_ids':selected,'objects':{str(i):{'category':'','model_path':''} for i in selected}}
    (out/'registration.yaml').write_text('# Fill in an existing model file for every selected ID before registration.\n'+yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    with ThreadPoolExecutor(max_workers=min(a.jobs,len(manifest['videos']))) as pool:
        videos=list(pool.map(lambda item:draw_camera(src,out,manifest,infos,item,set(selected)),manifest['videos']))
    print(json.dumps({'coordinate_space':'world','dynamic_motor_vehicle_count':len(selected),'selected_ids':selected,'box_videos':videos,'config':str(out/'registration.yaml')},indent=2))

if __name__=='__main__':main()
