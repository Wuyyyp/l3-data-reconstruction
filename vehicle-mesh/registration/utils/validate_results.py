"""Validate experiment PKL consistency, exact masks, video streams and decoding.

Arguments: --workers sets parallel CPU video decoders (default 4).
Example: python utils/validate_results.py --workers 4
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import re
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
DATA=Path(os.environ.get('MULTIVIEW_RUN_ROOT',str(Path(__file__).resolve().parents[1] / 'input'))).resolve()


def equal_source(source,output):
    if isinstance(source,dict):
        return all(key in output and equal_source(value,output[key]) for key,value in source.items())
    if isinstance(source,np.ndarray):return np.array_equal(source,output,equal_nan=True) if np.issubdtype(source.dtype,np.number) else np.array_equal(source,output)
    if isinstance(source,(list,tuple)):return len(source)==len(output) and all(equal_source(a,b) for a,b in zip(source,output))
    return bool(source==output)


def atoms(path):
    result=[]
    with path.open('rb') as handle:
        while True:
            header=handle.read(8)
            if len(header)<8:break
            size=int.from_bytes(header[:4],'big');result.append(header[4:]);header_size=8
            if size==1:size=int.from_bytes(handle.read(8),'big');header_size=16
            if size==0:break
            handle.seek(size-header_size,1)
    return result


def validate_video(job):
    item,directory=job
    path=DATA/directory/(item['camera']+'.mp4')
    cap=cv2.VideoCapture(str(path))
    actual=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),cap.get(cv2.CAP_PROP_FPS))
    cap.release();assert actual==(item['width'],item['height'],101,10.),(path,actual)
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0,(path,result.stderr)
    assert 'h264 (High)' in result.stderr and 'yuv420p' in result.stderr,path
    assert re.findall(r'frame=\s*(\d+)',result.stderr)[-1]=='101',path
    order=atoms(path);assert order.index(b'moov')<order.index(b'mdat'),path
    return dict(file=str(path.relative_to(DATA)),size_wh=actual[:2],frames=101,fps=10,codec='h264',profile='High',pixel_format='yuv420p',faststart=True,decodable=True)


def mask_overlay(item,frame_id):
    name=item['camera'];cap=cv2.VideoCapture(str(DATA/'videos'/f'{name}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,image=cap.read();cap.release();assert ok
    with np.load(DATA/'mask-videos'/'masks'/name/f'{frame_id:06d}.npz') as data:
        selected=[int(value) for value in (DATA/'obj_selected.txt').read_text().split()]
        for index,identity in enumerate(selected):
            key=f'obj_{identity:03d}'
            if key+'_bbox' not in data:continue
            x,y,w,h=map(int,data[key+'_bbox']);mask=np.unpackbits(data[key+'_bits'],count=w*h).reshape(h,w).astype(bool)
            color=np.array([(0,255,255),(255,100,0),(0,255,0),(255,0,255),(0,0,255)][index%5])
            patch=image[y:y+h,x:x+w];patch[mask]=(patch[mask]*.5+color*.5).astype(np.uint8)
            cv2.putText(image,f'ID {identity}',(x,max(20,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.75,color.tolist(),2)
    return image


def contact_sheet(items):
    tile_width,tile_height=640,390
    canvas=np.zeros((len(items)*tile_height,3*tile_width,3),np.uint8)
    for row,item in enumerate(items):
        name=item['camera'];assignments=json.loads((DATA/'mask-videos'/'masks'/name/'assignments.json').read_text())
        frame=max(assignments['frames'],key=lambda f:sum(x['pixels'] for x in f['objects'].values()))['frame']
        images=[]
        baseline='register-videos' if (DATA/'register-videos'/f'{name}.mp4').is_file() else 'videos'
        for directory in [baseline,'register-videos-optimized']:
            cap=cv2.VideoCapture(str(DATA/directory/f'{name}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();cap.release();assert ok;images.append(image)
        images.insert(1,mask_overlay(item,frame))
        for column,(image,label) in enumerate(zip(images,['source' if baseline=='videos' else 'baseline','SAM3 selected masks','shared 9DOF optimized'])):
            tile=np.zeros((tile_height,tile_width,3),np.uint8)
            scale=min(tile_width/image.shape[1],360/image.shape[0]);w,h=round(image.shape[1]*scale),round(image.shape[0]*scale);x=(tile_width-w)//2
            tile[30:30+h,x:x+w]=cv2.resize(image,(w,h))
            cv2.putText(tile,f'{name} #{frame} {label}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            canvas[row*tile_height:(row+1)*tile_height,column*tile_width:(column+1)*tile_width]=tile
    path=DATA/'register-videos-optimized'/'comparison_overview.jpg';assert cv2.imwrite(str(path),canvas)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    source=pickle.load((DATA/'result_interpolation_optimized_first101.pkl').open('rb'))
    optimized=pickle.load((DATA/'pkl'/'result_interpolation_optimized_first101_mesh_9dof.pkl').open('rb'))
    assert equal_source(source,optimized),'An existing source field was changed'
    canonical_from_source=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    selected={int(identity) for identity in optimized['metadata']['multiview_mesh_9dof']['objects']}
    results=[];exact_masks=0
    items=json.loads((DATA/'manifest.json').read_text())['videos']
    for index,info in enumerate(optimized['infos']):
        expected=set(source['infos'][index]['track_id']) & selected
        assert {int(identity) for identity in info['mesh_poses_9dof']}==expected
        for identity,record in info['mesh_poses_9dof'].items():
            pose=np.array(record['canonical_ego_from_unit_mesh']);other=np.array(record['source_ego_from_unit_mesh'])
            assert pose.shape==(4,4) and np.isfinite(pose).all() and np.linalg.det(pose[:3,:3])>0
            assert np.allclose(pose,canonical_from_source@other)
            assert np.allclose(pose[3],[0,0,0,1])
            state=np.array(record['state']);assert state.shape==(9,) and np.isfinite(state).all()
            if 'mean_iou_before' in record:
                if optimized['metadata']['multiview_mesh_9dof'].get('schema_version',1)<2:
                    assert record['mean_iou_after']>=record['mean_iou_before']-1e-10
                assert set(record['per_view_iou_after'])==set(record['views'])
                assert np.isclose(record['mean_iou_after'],np.mean(list(record['per_view_iou_after'].values())))
            if optimized['metadata']['multiview_mesh_9dof'].get('schema_version',1)>=2:
                assert abs(record['ground_contact_error_m'])<=.005001
                assert np.abs(record['world_euler_xyz_degrees'][:2]).max()<=.40001
            results.append(record)
        for item in items:
            with np.load(DATA/'mask-videos'/'masks'/item['camera']/f'{index:06d}.npz') as arrays:
                for key in arrays.files:
                    if not key.endswith('_bbox'):continue
                    x,y,w,h=map(int,arrays[key]);bits=arrays[key[:-5]+'_bits']
                    assert 0<=x<x+w<=item['width'] and 0<=y<y+h<=item['height']
                    assert bits.dtype==np.uint8 and len(bits)==(w*h+7)//8
                    pixels=np.unpackbits(bits,count=w*h);assert pixels.any() and np.isin(pixels,[0,1]).all();exact_masks+=1
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        videos=list(pool.map(validate_video,[(item,directory) for item in items for directory in ['mask-videos','register-videos-optimized']]))
    render_manifest=json.loads((DATA/'register-videos-optimized'/'registration_manifest.json').read_text())
    for camera in render_manifest['cameras']:
        assert camera['frame_ids']==list(range(101))
        for fi,processed in enumerate(camera['processed_track_ids_per_frame']):
            expected=set(source['infos'][fi]['track_id'])&selected
            assert set(processed)==expected,(camera['camera'],fi,expected,processed)
    contact_sheet(items)
    physical_report={}
    for identity in sorted(selected):
        rows=[r for r in results if r['object_id']==identity]
        if 'world_from_unit_mesh' not in rows[0]:continue
        world=np.array([r['world_from_unit_mesh'] for r in rows])
        dimensions=np.linalg.norm(world[:,:3,:3],axis=1)
        if rows[0]['constraint_mode']=='stationary_shared_world_pose':
            assert np.allclose(world,world[0],atol=1e-8),identity
        scale_step=float(np.abs(np.diff(np.log(dimensions),axis=0)).max(initial=0))
        assert scale_step<.02,(identity,scale_step)
        physical_report[str(identity)]=dict(ground_contact_valid=True,upright=True,
            stationary_world_pose_constant=rows[0]['constraint_mode']=='stationary_shared_world_pose',
            maximum_relative_scale_step=scale_step)
    backup=DATA/'pkl/unconstrained-backup/result_interpolation_optimized_first101_mesh_9dof.pkl'
    if physical_report and backup.exists() and not os.environ.get('MULTIVIEW_PERF_ROOT'):
        previous=pickle.load(backup.open('rb'))
        if previous.get('metadata',{}).get('multiview_mesh_9dof'):
            source_from_canonical=np.linalg.inv(canonical_from_source)
            for identity in sorted(selected):
                rows=[r for r in results if r['object_id']==identity]
                if len(rows)<3:continue
                new=np.array([r['world_from_unit_mesh'] for r in rows])
                old=[]
                for record in rows:
                    fi=record['frame']
                    reference=np.array(source['infos'][fi]['ego2global_transformation_matrix_camera_reference_offset_optimized'])
                    prior=np.array(previous['infos'][fi]['mesh_poses_9dof'][str(identity)]['canonical_ego_from_unit_mesh'])
                    old.append(reference@source_from_canonical@prior)
                old=np.array(old)
                before=float(np.linalg.norm(np.diff(old[:,:3,3],n=2,axis=0),axis=1).mean())
                after=float(np.linalg.norm(np.diff(new[:,:3,3],n=2,axis=0),axis=1).mean())
                physical_report[str(identity)]['mean_position_second_difference_before_m']=before
                physical_report[str(identity)]['mean_position_second_difference_after_m']=after
                # Ground-contact corrections can perturb short-track differences below 0.1 mm.
                tolerance_m=1e-4
                physical_report[str(identity)]['smoothness_tolerance_m']=tolerance_m
                physical_report[str(identity)]['position_second_difference_change_m']=after-before
                assert after<=before+tolerance_m,(identity,before,after)
    report=dict(all_selected_track_frames_processed_in_all_cameras=True,source_fields_preserved=True,shared_pose_matrices_valid=True,physical_constraints=physical_report,
        frames=101,object_frames=len(results),exact_instance_masks=exact_masks,videos=videos)
    (DATA/'pkl'/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VALIDATED',len(videos),'videos;',len(results),'shared object/frame poses;',exact_masks,'exact instance masks')

if __name__=='__main__':main()
