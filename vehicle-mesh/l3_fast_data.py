"""L3 data, deterministic asset search, and in-memory mask contexts."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import hashlib,json,shutil
import cv2
import numpy as np
import trimesh
import torch


CAM_IDS={'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov30':9,'front_camera_fov120':10}
VIEWS={'front_camera_fov30':'front','front_camera_fov120':'front_wide','left_front_camera':'left_front','right_front_camera':'right_front','rear_camera':'rear','left_rear_camera':'left_rear','right_rear_camera':'right_rear'}
VEHICLES={'Car','Suv','Truck','Huge_vehicle','Vehicle_else','Bus','Firetruck','Ambulance'}

def copy_local(source,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
    assert Path(source).stat().st_size==dest.stat().st_size
    return dest

def load_asset(path,asset_root,cache):
    path=Path(path);relative=path.relative_to(asset_root)
    local=copy_local(path,cache/'assets'/relative)
    scene=trimesh.load(local,force='scene',process=False)
    align=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
    parts=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node];m=scene.geometry[name].copy();m.apply_transform(align@transform);parts.append(m)
    bounds=np.array([m.bounds for m in parts]);lo=bounds[:,0].min(0);hi=bounds[:,1].max(0);size=hi-lo
    assert np.isfinite(size).all() and min(size)>0
    norm=np.eye(4);norm[:3,:3]=np.diag(1/size);norm[:3,3]=-(lo+hi)/2/size
    for m in parts:m.apply_transform(norm)
    vertices=[];faces=[];offset=0
    for m in parts:
        vertices.append(np.asarray(m.vertices,dtype=np.float32));faces.append(np.asarray(m.faces,dtype=np.int32)+offset);offset+=len(m.vertices)
    v=np.concatenate(vertices);f=np.concatenate(faces)
    rng=np.random.default_rng(20261030);tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    chosen=rng.choice(len(f),4096,p=area/area.sum());uv=rng.random((4096,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    samples=(tri[chosen]*np.c_[1-uv.sum(1),uv][:,:,None]).sum(1).astype(np.float32)
    return dict(path=str(path),local=str(local),category=relative.parts[0],size=size,vertices=v,faces=f,parts=parts,samples=samples,sha256=hashlib.sha256(local.read_bytes()).hexdigest())

def inventory(infos):
    tracks=defaultdict(list)
    for fi,info in enumerate(infos):
        for obj in info['objects']:
            if obj['type'] in VEHICLES:tracks[int(obj['id'])].append((fi,obj))
    return dict(tracks)

def collect_lidar(infos,tracks,cloud_path):
    clouds=np.load(cloud_path,allow_pickle=True)['pointcloud'].item();points=defaultdict(list)
    for fi,info in enumerate(infos):
        if fi%5:continue
        raw=np.asarray(clouds[fi])[:,:3];raw=raw[np.isfinite(raw).all(1)]
        for obj in info['objects']:
            tid=int(obj['id'])
            if tid not in tracks:continue
            angle=obj['rotation'][2];c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            size=np.array(obj['size']);local=(raw-np.array(obj['location']))@rot
            local=local[np.all(np.abs(local)<size/2+.08,axis=1)&(local[:,2]>-size[2]/2+.18)]
            if len(local):points[tid].append(local[np.linspace(0,len(local)-1,min(256,len(local)),dtype=int)])
    result={}
    for tid in tracks:
        p=np.concatenate(points[tid]) if points[tid] else np.empty((0,3))
        result[tid]=p[np.linspace(0,len(p)-1,min(1024,len(p)),dtype=int)].astype(np.float32) if len(p) else p.astype(np.float32)
    return result

@torch.inference_mode()
def match_assets(tracks,points,catalog):
    matches={};audit=[]
    for tid,observations in sorted(tracks.items()):
        size=np.median([o['size'] for _,o in observations],axis=0);category=observations[0][1]['type']
        family={'Car':{'car','suv','van'},'Suv':{'car','suv','van'},'Truck':{'truck'},'Huge_vehicle':{'truck','bus'},'Bus':{'bus'},'Ambulance':{'ambulance','van'}}.get(category,{'car','suv','van','truck','bus','ambulance'})
        def rank(a):return float(np.square(np.log(size/a['size'])).sum())+(0 if a['category'] in family else 2.)
        candidates=sorted(catalog,key=lambda a:(rank(a),a['path']))[:5];rows=[];p=points[tid]
        if len(p)>=50:
            queries=torch.as_tensor(p,device='cuda')[None].expand(len(candidates),-1,-1)
            surfaces=torch.as_tensor(np.stack([a['samples']*size for a in candidates]),dtype=torch.float32,device='cuda')
            distances=torch.cdist(queries,surfaces).amin(-1).clamp(max=.35).mean(-1).cpu().numpy()
        else:distances=np.zeros(len(candidates))
        for a,d in zip(candidates,distances):
            rows.append(dict(asset=a['path'],category=a['category'],dimension_class_score=rank(a),surface_distance_m=float(d) if len(p)>=50 else None,score=float(d)+.03*rank(a)))
        winner=min(range(len(rows)),key=lambda i:rows[i]['score']);matches[tid]=candidates[winner]
        audit.append(dict(track_id=tid,source_category=category,lidar_points=len(p),median_lwh=size.tolist(),asset_path=matches[tid]['path'],confidence='medium' if len(p)>=200 else 'low',candidates=rows))
    return matches,audit

