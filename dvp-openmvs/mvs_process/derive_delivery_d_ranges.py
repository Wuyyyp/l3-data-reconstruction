#!/usr/bin/env python3
"""Sparse-Z bounds; use full delivered BA points in-frustum when a window has no tracks."""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

from derive_sparse_depth_ranges import read_points, read_images, quaternion_to_rotation, frame_position


def full_points(path):
    xyz=[]
    with path.open('rb') as source:
        for _ in range(struct.unpack('<Q',source.read(8))[0]):
            record=struct.unpack('<QdddBBBd',source.read(43))
            xyz.append(record[1:4])
            source.seek(8*struct.unpack('<Q',source.read(8))[0],1)
        if source.read(1): raise ValueError(f'Unexpected BA points payload: {path}')
    return np.asarray(xyz,dtype=np.float64)


def cameras(path):
    result={}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'): continue
        fields=line.split()
        if fields[1]!='PINHOLE':raise ValueError(f'Expected PINHOLE camera: {line}')
        result[int(fields[0])]=tuple(map(float,fields[2:]))
    return result


def image_camera_ids(path):
    ids={}
    lines=[line for line in path.read_text().splitlines() if not line.startswith('#')]
    if len(lines)%2:raise ValueError(f'Malformed images.txt: {path}')
    for line in lines[0::2]:
        fields=line.split()
        if len(fields)<10:raise ValueError(f'Malformed image header: {line}')
        ids[Path(fields[9]).stem]=int(fields[8])
    return ids


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--full-ba-points',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    points=read_points(args.model/'points3D.txt')
    images=read_images(args.model/'images.txt')
    cam=cameras(args.model/'cameras.txt');cam_ids=image_camera_ids(args.model/'images.txt')
    ranges={}; missing=[]; positions={label:frame_position(label,i) for i,(label,*_) in enumerate(images)}
    for order,(label,q,t,point_ids) in enumerate(images):
        R=quaternion_to_rotation(q)
        depths=np.asarray([(R@points[pid]+t)[2] for pid in set(point_ids) if pid in points],dtype=np.float64)
        depths=depths[np.isfinite(depths)&(depths>0)]
        if not len(depths):missing.append((label,R,t));continue
        ranges[label]=dict(depth_min_m=float(.9*depths.min()),depth_max_m=float(1.1*depths.max()),sparse_points=len(depths),source='window_observed_BA_sparse_minmax_0.9_1.1')
    full=full_points(args.full_ba_points) if missing else None
    frustum_count=0;nearest_count=0
    for label,R,t in missing:
        width,height,fx,fy,cx,cy=cam[cam_ids[label]]
        xyz=full@R.T+t;xyz=xyz[np.isfinite(xyz).all(1)&(xyz[:,2]>0)]
        uv=xyz[:,:2]/xyz[:,2,None]
        inside=(fx*uv[:,0]+cx>=0)&(fx*uv[:,0]+cx<width)&(fy*uv[:,1]+cy>=0)&(fy*uv[:,1]+cy<height)
        depths=xyz[inside,2]
        if len(depths):
            ranges[label]=dict(depth_min_m=float(.9*depths.min()),depth_max_m=float(1.1*depths.max()),sparse_points=len(depths),source='full_delivered_BA_points_in_window_image_frustum_minmax_0.9_1.1')
            frustum_count+=1
        else:
            camera,position=positions[label]
            candidates=[(abs(positions[name][1]-position),name,values) for name,values in ranges.items() if positions[name][0]==camera]
            if not candidates:raise RuntimeError(f'No BA sparse points or same-camera range: {label}')
            _,neighbor,values=min(candidates)
            ranges[label]=dict(values,sparse_points=0,source=f'nearest_same_camera_BA_range:{neighbor}')
            nearest_count+=1
    if len(ranges)!=len(images):raise ValueError('BA depth-range/image count mismatch')
    minimums=[row['depth_min_m'] for row in ranges.values()];maximums=[row['depth_max_m'] for row in ranges.values()]
    report=dict(method='Delivered BA sparse camera-Z min*0.9/max*1.1',near_scale=.9,far_scale=1.1,fixed_0_100_fallback=False,
        image_count=len(images),direct_count=len(images)-len(missing),full_BA_frustum_fallback_count=frustum_count,
        nearest_same_camera_fallback_count=nearest_count,
        depth_min_m_summary=dict(min=min(minimums),median=float(np.median(minimums)),max=max(minimums)),
        depth_max_m_summary=dict(min=min(maximums),median=float(np.median(maximums)),max=max(maximums)),ranges=dict(sorted(ranges.items())))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');os.replace(temporary,args.output)
    print(json.dumps({key:value for key,value in report.items() if key!='ranges'}))


if __name__=='__main__':main()
