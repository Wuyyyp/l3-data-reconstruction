#!/usr/bin/env python3
"""Prepare one delivered BA window for the DVP D pipeline without rerunning BA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CAMERAS = {
    'left_front_camera': '00', 'right_front_camera': '01',
    'rear_camera': '02', 'left_rear_camera': '03', 'right_rear_camera': '04',
    'front_camera_fov30': '09', 'center_camera_fov30': '09',
    'front_camera_fov120': '10', 'center_camera_fov120': '10',
}
OLD_PREFIX = os.environ.get('DVP_OLD_IMAGE_PREFIX')
NEW_PREFIX = os.environ.get('DVP_NEW_IMAGE_PREFIX')
ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COLMAP = Path(os.environ.get('DVP_DELIVERY_COLMAP', ROOT.parent/'tools/colmap/bin/colmap'))
MVS = Path(os.environ.get('OPENMVS_BIN_DIR', ROOT/'vendor/openMVS/install/bin/OpenMVS'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as out:
        subprocess.run([str(x) for x in command], stdout=out, stderr=subprocess.STDOUT, check=True)


def rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def model_images(path, rename=None):
    """Read/optionally rename COLMAP binary images without changing any poses or tracks."""
    records = {}
    temp = path.with_suffix('.renamed') if rename else None
    with path.open('rb') as source:
        out = temp.open('wb') if temp else None
        try:
            count_raw = source.read(8)
            count = struct.unpack('<Q', count_raw)[0]
            if out: out.write(count_raw)
            for _ in range(count):
                raw = source.read(64)
                image_id, *rest = struct.unpack('<i7di', raw)
                name = bytearray()
                while (char := source.read(1)) != b'\0':
                    if not char: raise EOFError(path)
                    name.extend(char)
                name = name.decode()
                nraw = source.read(8)
                observations = source.read(24 * struct.unpack('<Q', nraw)[0])
                if len(observations) != 24 * struct.unpack('<Q', nraw)[0]: raise EOFError(path)
                T = np.eye(4); T[:3, :3] = rot(rest[:4]); T[:3, 3] = rest[4:7]
                if name in records: raise ValueError('Duplicate BA image ' + name)
                records[name] = dict(id=image_id, camera_id=rest[7], w2c=T)
                if out: out.write(raw + rename[name].encode() + b'\0' + nraw + observations)
            if source.read(1): raise ValueError('Unexpected images.bin data')
        finally:
            if out: out.close()
    if temp: temp.replace(path)
    return records


def model_cameras(path, scales=None):
    records = {}; payload = bytearray()
    with path.open('rb') as source:
        count_raw = source.read(8); payload.extend(count_raw)
        for _ in range(struct.unpack('<Q', count_raw)[0]):
            raw = source.read(24)
            camera_id, model_id, width, height = struct.unpack('<iiQQ', raw)
            if model_id != 1: raise ValueError(f'Only PINHOLE is supported: {camera_id} model={model_id}')
            params = np.array(struct.unpack('<4d', source.read(32)))
            scale = (0.5 if width==3840 else 1) if scales is None else scales.get(camera_id, 1)
            out_width, out_height = round(width * scale), round(height * scale)
            if (width, height) not in ((1920, 1280), (3840, 2160)) or (out_width, out_height) not in ((1920, 1280), (1920, 1080)):
                raise ValueError(f'Unsupported BA image dimensions {camera_id}: {width}x{height}')
            records[camera_id] = dict(size=(width, height), output_size=(out_width, out_height), K=np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]]))
            payload.extend(struct.pack('<iiQQ4d', camera_id, model_id, out_width, out_height, *(params*scale)))
        if source.read(1): raise ValueError('Unexpected cameras.bin data')
    if scales: path.write_bytes(payload)
    return records


def objects(info):
    boxes, ids, names = info.get('gt_boxes', []), info.get('track_id', []), info.get('gt_names', [])
    return [(str(ids[i]), str(names[i]), np.asarray(box, float)) for i, box in enumerate(boxes) if i < len(ids) and str(ids[i]) not in ('', '-1', 'None', 'nan')]


def tracks_for_window(infos, start, end):
    tracks = {}
    for frame in range(start, end + 1):
        ego = np.asarray(infos[frame]['ego2global_transformation_matrix_camera_reference_offset_optimized'], float)
        seen = set()
        for track_id, name, box in objects(infos[frame]):
            if track_id in seen: raise ValueError(f'Duplicate track {track_id} in frame {frame}')
            seen.add(track_id)
            xyz = ego[:3, :3] @ box[:3] + ego[:3, 3]
            tracks.setdefault(track_id, dict(type=name, frames=[], world_xy=[]))['frames'].append(frame)
            tracks[track_id]['world_xy'].append(xyz[:2].tolist())
    for track_id, row in tracks.items():
        points = np.asarray(row['world_xy']); distance = float(np.linalg.norm(points[:, None]-points[None, :], axis=-1).max())
        row['max_displacement_m'] = distance
        row['selected'] = len(points) > 1 and distance > 1.0
    return tracks


def box_polygon(box, ego_to_camera, K, width, height):
    x, y, z, length, width_box, height_box, yaw = box
    local = np.array([[sx*length/2, sy*width_box/2, sz*height_box/2] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    c, s = np.cos(yaw), np.sin(yaw)
    local[:, :2] = local[:, :2] @ np.array([[c, s], [-s, c]])
    xyz = local + [x, y, z]
    xyz = xyz @ ego_to_camera[:3,:3].T + ego_to_camera[:3,3]
    near=.1;visible=[point for point in xyz if point[2]>=near]
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if i<j and (xyz[i,2]<near)!=(xyz[j,2]<near):
                visible.append(xyz[i]+(xyz[j]-xyz[i])*(near-xyz[i,2])/(xyz[j,2]-xyz[i,2]))
    if len(visible)<3:return None
    uvz = np.asarray(visible) @ K.T; uv = (uvz[:,:2] / uvz[:,2:]).astype(np.float32)
    hull = cv2.convexHull(uv)
    screen = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    area, clipped = cv2.intersectConvexConvex(hull, screen)
    return None if area <= 0 else np.rint(clipped).astype(np.int32)


def image_path(row):
    raw = row['image_path']
    candidate = Path(NEW_PREFIX + raw[len(OLD_PREFIX):]) if OLD_PREFIX and NEW_PREFIX and raw.startswith(OLD_PREFIX) else Path(raw)
    if not candidate.is_file(): raise FileNotFoundError(candidate)
    return candidate


def same_semantic_source(meta,image):
    source=meta['source'];stat=image.stat()
    return Path(source['path']).resolve()==image.resolve() and source['size']==stat.st_size and source['mtime_ns']==stat.st_mtime_ns


def inventory(clip, start, end, output):
    csv_path = clip/'colmap/poses/image_camera_to_world_after_ba.csv'
    model = clip/'colmap/sparse_ba'
    for path in [csv_path, model/'images.bin', model/'points3D.bin', model/'cameras.bin', clip/'result_interpolation_optimized.pkl']:
        if not path.is_file(): raise FileNotFoundError(path)
    with (clip/'result_interpolation_optimized.pkl').open('rb') as source: infos = pickle.load(source)['infos']
    if not 0 <= start <= end < len(infos): raise ValueError(f'Invalid frame window {start}-{end}')
    original = model_images(model/'images.bin')
    original_cameras = model_cameras(model/'cameras.bin')
    rows = []; excluded = []; labels = set(); selected = {}
    for row in csv.DictReader(csv_path.open()):
        frame = int(row['frame'])
        if not start <= frame <= end: continue
        camera = row['camera']; key = f"{camera}/{row['image_filename']}"
        if camera not in CAMERAS or key not in original: raise ValueError(f'BA CSV/model mismatch: {key}')
        label = f"{frame-start:06d}_{CAMERAS[camera]}.jpg"
        if label in labels: raise ValueError(f'Duplicate window label {label}')
        labels.add(label)
        T = np.array([[float(row[f'T_cw_{i}{j}']) for j in range(4)] for i in range(4)])
        if np.max(np.abs(np.linalg.inv(T) - original[key]['w2c'])) > 1e-6: raise ValueError(f'BA CSV/model pose mismatch: {key}')
        source_image = image_path(row)
        camera_id = original[key]['camera_id']
        spec = original_cameras[camera_id]
        sensor = infos[frame]['sensors']['cams'][camera]
        if not sensor.get('images_undistorted', False): raise ValueError(f'Input image is not undistorted: {key}')
        if Path(sensor['data_path']).name != row['image_filename']: raise ValueError(f'PKL/image mismatch: {key}')
        record = dict(frame=frame,camera=camera,code=CAMERAS[camera],timestamp_ns=int(row['timestamp_ns']),ba_name=key,ba_id=original[key]['id'],label=label,source_image=str(source_image),camera_id=camera_id,c2w=T.tolist(),K=spec['K'].tolist(),source_size=list(spec['size']),output_size=list(spec['output_size']))
        rows.append(record); selected[key] = label
    if not rows: raise ValueError('No delivered BA images in window')
    for frame in range(start, end+1):
        for camera, sensor in infos[frame]['sensors']['cams'].items():
            if camera in CAMERAS and f"{camera}/{Path(sensor['data_path']).name}" not in selected:
                excluded.append(dict(frame=frame,camera=camera,image=Path(sensor['data_path']).name,reason='not_in_BA_csv'))
    rows.sort(key=lambda row:(row['frame'], row['code']))
    inv=dict(status='PASS',clip=str(clip),frames=[start,end],images=len(rows),records=rows,excluded=excluded,selected=selected,tracks=tracks_for_window(infos,start,end),source_sha256=dict(csv=digest(csv_path),images=digest(model/'images.bin'),cameras=digest(model/'cameras.bin'),points3D=digest(model/'points3D.bin'),pkl=digest(clip/'result_interpolation_optimized.pkl')))
    audit_path=output/'audit/delivery_inventory.json'
    if audit_path.exists():
        previous=json.loads(audit_path.read_text())
        if previous['source_sha256']!=inv['source_sha256'] or previous['frames']!=inv['frames']:
            raise ValueError('Delivery inputs changed; use a fresh result root')
    save(output/'audit/delivery_inventory.json', inv)
    return inv, infos


def prepare_model(clip, prepared, inv, log):
    sparse = prepared/'07_sparse/triangulated_binary'; sparse.mkdir(parents=True,exist_ok=True)
    source = clip/'colmap/sparse_ba'
    for name in ('images.bin','points3D.bin','cameras.bin','rigs.bin','frames.bin'):
        if (source/name).exists(): shutil.copy2(source/name,sparse/name)
    all_images = model_images(sparse/'images.bin')
    selected = inv['selected']
    delete_names = [name for name in all_images if name not in selected]
    if delete_names:
        names = prepared/'audit/delete_images.txt'; names.write_text('\n'.join(delete_names)+'\n')
        output = prepared/'audit/ba_window_subset'; output.mkdir(exist_ok=True)
        run([COLMAP,'image_deleter','--input_path',sparse,'--output_path',output,'--image_names_path',names],log)
        shutil.rmtree(sparse); output.replace(sparse)
    subset = model_images(sparse/'images.bin',selected)
    if set(subset) != set(selected): raise ValueError('COLMAP window subset image mismatch')
    for name,record in subset.items():
        if np.max(np.abs(record['w2c']-all_images[name]['w2c']))>1e-8:
            raise ValueError(f'COLMAP deletion changed BA pose: {name}')
    original_cameras = model_cameras(sparse/'cameras.bin')
    scales = {row['camera_id']: row['output_size'][0]/row['source_size'][0] for row in inv['records']}
    camera_specs = model_cameras(sparse/'cameras.bin',scales)
    text = prepared/'07_sparse/triangulated_text'; text.mkdir(exist_ok=True)
    run([COLMAP,'model_converter','--input_path',sparse,'--output_path',text,'--output_type','TXT'],log)
    if set(model_images(sparse/'images.bin')) != set(selected.values()): raise ValueError('Renamed model image mismatch')
    return camera_specs


def prepare_images(prepared, inv, infos):
    converted = prepared/'03_converted'
    for folder in ('images','intrinsics','ego_pose','extrinsics','lidar_depth'):(converted/folder).mkdir(parents=True,exist_ok=True)
    for row in inv['records']:
        label = row['label']; stem = Path(label).stem
        image = cv2.imread(row['source_image'])
        if image is None or list(image.shape[1::-1]) != row['source_size']: raise ValueError(f'RGB size mismatch: {label}')
        target = converted/'images'/label
        if row['source_size'] == row['output_size']:
            if not target.exists(): target.symlink_to(row['source_image'])
        elif not target.exists():
            image = cv2.resize(image,tuple(row['output_size']),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(target),image,[cv2.IMWRITE_JPEG_QUALITY,95]): raise OSError(target)
        T=np.asarray(row['c2w']); original_K=np.asarray(row['K'])
        scale=row['output_size'][0]/row['source_size'][0]; K=original_K.copy();K[:2]*=scale
        np.savetxt(converted/'intrinsics'/(stem+'.txt'),K,fmt='%.18e')
        np.savetxt(converted/'ego_pose'/(stem+'.txt'),T,fmt='%.18e')
        ego=np.asarray(infos[row['frame']]['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        np.savetxt(converted/'extrinsics'/(stem+'.txt'),np.linalg.inv(T)@ego,fmt='%.18e')


def generate_semantics(prepared, inv, gpu):
    """Use the existing Dongfeng Mask2Former cache; sky and ego are separate labels."""
    import torch
    sys.path.insert(0,str(Path(os.environ.get('POSE_ROOT',ROOT.parent/'pose-repair'))/'src'))
    from dongfeng_registration.semantic_masks import DynamicMaskConfig,MaskCache,create_predictor,generate_cached_masks_batch
    torch.set_num_threads(4)
    cache_root=prepared/'06_masks/semantic_cache'
    model=Path(os.environ.get('DVP_MASK2FORMER_WEIGHTS',ROOT.parent/'weights/mask2former'))
    config=DynamicMaskConfig(cache_root=cache_root,device='cuda:0',model_name=str(model),batch_size=4,write_workers=4,preview_ratio=0)
    cache=MaskCache(config); predictor=None
    for offset in range(0,len(inv['records']),4):
        batch=inv['records'][offset:offset+4];pending=[]
        for row in batch:
            image=prepared/'03_converted/images'/row['label']
            matches=list((cache_root/'backends/mask2former').glob(f'*/{row["camera"]}/metadata/{row["timestamp_ns"]}.json'))
            if len(matches)==1 and same_semantic_source(json.loads(matches[0].read_text()),image): continue
            pending.append((image,row['camera'],row['timestamp_ns']))
        if pending:
            if predictor is None: predictor=create_predictor(config)
            generate_cached_masks_batch(pending,predictor,cache,write_preview_flags=[False]*len(pending),write_workers=4)
        if offset%40==0: print(f'semantics {min(offset+4,len(inv["records"]))}/{len(inv["records"])}',flush=True)


def prepare_masks(prepared, inv, infos):
    root=prepared/'06_masks'; masks=root/'openmvs_masks'; masks.mkdir(parents=True,exist_ok=True)
    profile=dict(description='Fixed fov120 hood/wipers and connected black borders',camera10_topline=[[0,.666],[.15,.67],[.35,.676],[.55,.675],[.8,.67],[1,.664]],edge_margin_at_960=2)
    save(root/'camera10_rig_profile.json',profile)
    rows=[]; selected={key for key,row in inv['tracks'].items() if row['selected']}
    fixed_counts={};camera_counts={}
    for row in inv['records']:
        matches=list((root/'semantic_cache/backends/mask2former').glob(f'*/{row["camera"]}/metadata/{row["timestamp_ns"]}.json'))
        if len(matches)!=1: raise ValueError(f'Expected one semantic mask for {row["label"]}, found {len(matches)}')
        meta=json.loads(matches[0].read_text());classes={name:int(index) for index,name in meta['class_id_to_name'].items()}
        labels=cv2.imread(meta['class_mask_path'],cv2.IMREAD_UNCHANGED)
        if labels is None:raise ValueError(f'Missing semantic class mask: {row["label"]}')
        camera=row['camera']
        if camera not in fixed_counts:fixed_counts[camera]=np.zeros(labels.shape,np.uint16)
        fixed_counts[camera]+=(labels==classes.get('ego_vehicle',-1))
        camera_counts[camera]=camera_counts.get(camera,0)+1
    for row in inv['records']:
        image_path=prepared/'03_converted/images'/row['label'];image=cv2.imread(str(image_path));h,w=image.shape[:2]
        matches=list((root/'semantic_cache/backends/mask2former').glob(f'*/{row["camera"]}/metadata/{row["timestamp_ns"]}.json'))
        if len(matches)!=1: raise ValueError(f'Expected one semantic mask for {row["label"]}, found {len(matches)}')
        metadata=matches[0]
        meta=json.loads(metadata.read_text())
        if not same_semantic_source(meta,image_path): raise ValueError(f'Semantic source mismatch: {row["label"]}')
        labels=cv2.imread(meta['class_mask_path'],cv2.IMREAD_UNCHANGED)
        if labels is None or labels.shape != (h,w): raise ValueError(f'Semantic size mismatch: {row["label"]}')
        classes={name:int(index) for index,name in meta['class_id_to_name'].items()}
        sky=labels==classes.get('sky',-1)
        ego_mask=(labels==classes.get('ego_vehicle',-1))|(fixed_counts[row['camera']]>=max(2,(camera_counts[row['camera']]+1)//2))
        dynamic=np.zeros((h,w),np.uint8)
        T=np.linalg.inv(np.asarray(row['c2w'])) @ np.asarray(infos[row['frame']]['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        K=np.loadtxt(prepared/'03_converted/intrinsics'/(Path(row['label']).stem+'.txt'))
        for track_id,name,box in objects(infos[row['frame']]):
            if track_id not in selected: continue
            polygon=box_polygon(box,T,K,w,h)
            if polygon is None: continue
            inside=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(inside,polygon,1)
            if name.startswith('VEHICLE'): semantic=['car','truck','bus','trailer','train']
            elif 'PEDESTRIAN' in name: semantic=['person','rider']
            elif 'BIKE' in name: semantic=['bicycle','motorcycle','rider']
            else: semantic=[]
            ids=[classes[c] for c in semantic if c in classes]
            dynamic|=(inside.astype(bool) & np.isin(labels,ids)).astype(np.uint8) if ids else inside
        dynamic=cv2.dilate(dynamic,np.ones((11,11),np.uint8))>0
        black=(image.max(2)<=2).astype(np.uint8)
        _,components=cv2.connectedComponents(black,8)
        border=np.unique(np.concatenate([components[0],components[-1],components[:,0],components[:,-1]]));border=border[border>0]
        boundary=np.isin(components,border)
        ignore=dynamic|sky|ego_mask|boundary
        if row['code']=='10':
            top=np.asarray(profile['camera10_topline']);poly=np.rint(top*np.array([w-1,h-1])).astype(np.int32)
            poly=np.vstack([poly,[[w-1,h-1],[0,h-1]]])
            rig=np.zeros((h,w),np.uint8);cv2.fillPoly(rig,[poly],1);ignore|=rig>0
        ignore=cv2.dilate(ignore.astype(np.uint8),np.ones((5,5),np.uint8))*255
        target=masks/(Path(row['label']).stem+'.mask.png')
        if not cv2.imwrite(str(target),ignore): raise OSError(target)
        rows.append(dict(image=row['label'],ignored_pixels=int((ignore>0).sum()),dynamic_pixels=int(dynamic.sum()),sky_pixels=int(sky.sum()),fixed_ego_pixels=int(ego_mask.sum()),sha256=digest(target)))
    policy='map-xy-window-displacement-gt1m-v1'
    save(root/'dynamic_mask_audit.json',dict(status='PASS',image_count=len(rows),dynamic_mask_count=len(rows),dynamic_policy=policy,dynamic_threshold_m=1.0,tracks=inv['tracks'],rows=rows))
    save(root/'mask_audit.json',dict(status='PASS',images=len(rows),rows=rows))


def read_pcd(path):
    with path.open('rb') as source:
        header={}
        while True:
            line=source.readline().decode('ascii').strip()
            if not line: raise ValueError(f'Invalid PCD header: {path}')
            if line.startswith('#'): continue
            key,*values=line.split();header[key]=values
            if key=='DATA': break
        if header['DATA']!=['binary']: raise ValueError(f'Only binary PCD is supported: {path}')
        names=header['FIELDS'];sizes=list(map(int,header['SIZE']));types=header['TYPE'];counts=list(map(int,header.get('COUNT',['1']*len(names))))
        formats=[]
        for size,kind,count in zip(sizes,types,counts):
            if kind=='F' and size in (4,8): code=f'<f{size}'
            elif kind in ('U','I') and size in (1,2,4,8): code=f'<{kind.lower()}{size}'
            else: raise ValueError(f'Unsupported PCD field: {path} {kind}{size}')
            formats.append((code,(count,)) if count>1 else code)
        dtype=np.dtype({'names':names,'formats':formats})
        points=int(header['POINTS'][0]);raw=source.read()
        if len(raw)!=points*dtype.itemsize: raise ValueError(f'PCD length mismatch: {path}')
        cloud=np.frombuffer(raw,dtype=dtype,count=points)
        return np.column_stack([cloud[name] for name in ('x','y','z')]).astype(np.float64)


def prepare_lidar(clip,prepared,inv,infos):
    selected={key for key,row in inv['tracks'].items() if row['selected']}
    by_frame={frame:[] for frame in range(inv['frames'][0],inv['frames'][1]+1)}
    for row in inv['records']: by_frame[row['frame']].append(row)
    converted=prepared/'03_converted/lidar_depth'
    for frame, rows in by_frame.items():
        pcd=clip/infos[frame]['sensors']['lidar']['perception']['data_path']
        if not pcd.is_file(): raise FileNotFoundError(pcd)
        xyz=read_pcd(pcd)
        if not len(xyz): raise ValueError(f'Empty PCD: {pcd}')
        keep=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz,axis=1)>1)
        for track_id,_,box in objects(infos[frame]):
            if track_id not in selected: continue
            x,y,z,length,width,height,yaw=box
            dx=xyz[:,0]-x;dy=xyz[:,1]-y;c=np.cos(yaw);s=np.sin(yaw)
            inside=(np.abs(xyz[:,2]-z)<=height/2+.4)&(np.abs(dx*c+dy*s)<=length/2+.4)&(np.abs(-dx*s+dy*c)<=width/2+.4)
            keep&=~inside
        xyz=xyz[keep]
        ego=np.asarray(infos[frame]['ego2global_transformation_matrix_camera_reference_offset_optimized'])
        for row in rows:
            K=np.loadtxt(prepared/'03_converted/intrinsics'/(Path(row['label']).stem+'.txt'))
            T=np.linalg.inv(np.asarray(row['c2w']))@ego
            pts=xyz@T[:3,:3].T+T[:3,3];pts=pts[np.isfinite(pts).all(1)&(pts[:,2]>.1)]
            uvz=pts@K.T;uv=np.rint(uvz[:,:2]/uvz[:,2:]).astype(np.int64)
            w,h=row['output_size'];valid=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            depth=np.full(w*h,np.inf,np.float32);np.minimum.at(depth,uv[valid,1]*w+uv[valid,0],pts[valid,2]);depth=depth.reshape(h,w)
            mask=cv2.imread(str(prepared/'06_masks/openmvs_masks'/(Path(row['label']).stem+'.mask.png')),0)>0
            good=np.isfinite(depth)&~mask
            np.save(converted/(Path(row['label']).stem+'.npy'),dict(mask=good,value=depth[good]))
        if (frame-inv['frames'][0]+1)%20==0: print(f'LiDAR {frame}/{inv["frames"][1]}',flush=True)


def prepare_scene(prepared,result_root,inv,log):
    work=result_root/'10_openmvs_fusion/dmaps';work.mkdir(parents=True,exist_ok=True)
    colmap_input=work/'colmap_input';colmap_input.mkdir(exist_ok=True)
    sparse=prepared/'07_sparse/triangulated_binary'
    link=colmap_input/'sparse'
    if not link.exists(): link.symlink_to(sparse)
    scene=work/'scene.mvs'
    run([MVS/'InterfaceCOLMAP','-w',work,'-i',colmap_input,'-o',scene,'--image-folder',prepared/'03_converted/images','--binary','1','--archive-type','2','-v','2'],log)
    neighbors=work/'neighbors_diverse_pm20_top20.txt'
    audit=work/'neighbors_diverse_pm20_top20.json'
    run([PYTHON,ROOT/'data_preprocess/steps/08_build_diverse_neighbors.py','--scene',scene,'--images-txt',prepared/'07_sparse/triangulated_text/images.txt','--output',neighbors,'--audit',audit,'--radius','20','--sources','20','--cross-camera-quota','6'],log)
    import re
    names=re.findall(rb'[0-9]{6}_(?:00|01|02|03|04|09|10)\.jpg',scene.read_bytes())
    expected={row['label'].encode() for row in inv['records']}
    if len(names)!=len(expected) or set(names)!=expected: raise ValueError('OpenMVS scene/image mismatch')
    global_dir=prepared/'08_openmvs_input/global';global_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(scene,global_dir/'scene.mvs');shutil.copy2(neighbors,global_dir/neighbors.name)
    shutil.copy2(neighbors.with_suffix('.tsv'),prepared/'audit'/neighbors.with_suffix('.tsv').name)
    if len(list((prepared/'03_converted/images').glob('*.jpg'))) != inv['images']: raise ValueError('Prepared image count mismatch')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--clip-root',type=Path,required=True)
    parser.add_argument('--frame-start',type=int,required=True)
    parser.add_argument('--frame-end',type=int,required=True)
    parser.add_argument('--result-root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--semantics-only',action='store_true')
    args=parser.parse_args();clip=args.clip_root.resolve();result=args.result_root.resolve()
    if args.semantics_only:
        inv=json.loads((result/'audit/delivery_inventory.json').read_text())
        generate_semantics(result/'prepared',inv,args.gpu)
        return
    inv,infos=inventory(clip,args.frame_start,args.frame_end,result)
    print(f'Audited {inv["images"]} BA images, {len(inv["excluded"])} excluded, {len(inv["tracks"])} tracks',flush=True)
    if args.check:return
    prepared=result/'prepared';prepared.mkdir(parents=True,exist_ok=True)
    (prepared/'audit').mkdir(exist_ok=True)
    log=result/'logs/delivery-preparation.log'
    prepare_model(clip,prepared,inv,log)
    prepare_images(prepared,inv,infos)
    vision=Path(os.environ.get('DVP_DELIVERY_VISION_PYTHON',sys.executable))
    semantic_args=[vision,__file__,'--clip-root',clip,'--frame-start',args.frame_start,'--frame-end',args.frame_end,'--result-root',result,'--gpu',args.gpu,'--semantics-only']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    with log.open('a') as out: subprocess.run([str(x) for x in semantic_args],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    prepare_masks(prepared,inv,infos)
    prepare_lidar(clip,prepared,inv,infos)
    prepare_scene(prepared,result,inv,log)
    save(prepared/'audit/preparation.json',dict(status='PASS',delivery=str(clip),frames=inv['frames'],images=inv['images'],ba_model_sha256=inv['source_sha256'],dynamic_threshold_m=1.0,internal_ba=False))
    print('Prepared',prepared,flush=True)


if __name__=='__main__': main()
