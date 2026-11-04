"""Resolve per-window input from samples before using narrowly scoped raw supplements."""
from pathlib import Path
import csv,json,os,shutil
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from l3_wys_canvas import geometry

NAMES={'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'center_camera_fov30':9,'center_camera_fov120':10}

def fill_metadata_gaps(c,out,infos):
    # Keep complete shared inputs in place. Synthesize only absent files from the aligned window.
    dest=out/'supplements';tid=c/'track'
    if all((c/f).is_file() for f in ('track/track_info.txt','track/track_camera_vis.json','timestamps.json','timestamps_specific.json')):return
    dest.mkdir(exist_ok=True);tracks=dest/'track';tracks.mkdir(exist_ok=True)
    lines=['frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed'];visible={};tt={'FRAME':{}};details=[]
    for fi,info in enumerate(infos):
        key=f'{fi:06d}';ts=int(info['timestamp']);tt['FRAME'][key]=ts/1e9;ct={}
        for name,cam in info['sensors']['cams'].items():
            nm=name.replace('front_camera_fov','center_camera_fov');ct[nm]=dict(cam_idx=NAMES[nm],timestamp=cam['timestamp']);tt.setdefault(nm,{})[key]=cam['timestamp']/1e6
        details.append(dict(FRAME_IDX=fi,FRAME_NAME=key,timestamp=ts,cameras=ct))
        for o in info['objects']:
            i=int(o['id']);l,w,h=o['size'];v=o.get('velocity')
            speed=float(np.linalg.norm(np.asarray(v)[:2])) if isinstance(v,(tuple,list,np.ndarray)) and len(v)>=2 else -1
            lines.append(' '.join(map(str,(fi,i,o['type'],-10,h,w,l,*o['location'],o['rotation'][2],speed))))
            found=sorted({NAMES[x['camera']] for x in o.get('info2d',[]) if x.get('camera') in NAMES})
            visible.setdefault(str(i),{})[str(fi)]=found
    for name,contents in [('track/track_info.txt','\n'.join(lines)+'\n'),('track/track_camera_vis.json',json.dumps(visible)),('timestamps.json',json.dumps(tt)),('timestamps_specific.json',json.dumps(details))]:
        p=dest/name;source=c/name
        if source.is_file():p.symlink_to(source)
        else:p.write_text(contents)
