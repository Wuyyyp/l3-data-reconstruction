#!/usr/bin/env python3
"""Run SAM3 masks and temporally grounded seven-view mesh registration.

Arguments: --input is the source sequence root; --config is registration.yaml
created by select_dynamic_vehicles.py and edited with a whitelisted category and
one model_path per ID;
--output is a fresh run directory; --jobs limits concurrent workers; --gpus
lists physical GPU IDs available for independent rendering/inference workers;
--resume continues an existing run without replacing successful stage outputs.
SAM3_CHECKPOINT must point to the local model checkpoint.
Example: SAM3_CHECKPOINT=/models/sam3.pt python run_registration.py --input /data/sequence --config /data/selected/registration.yaml --output /data/registration --jobs 2 --gpus 0 1
"""
from __future__ import annotations
import argparse, json, os, pickle, shutil, subprocess, sys
from pathlib import Path
import numpy as np, yaml

TASK=Path(__file__).resolve().parent; UTILS=TASK/'utils'; PKL='result_interpolation_optimized_first101.pkl'
CANONICAL_FROM_SOURCE=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])

def link_or_copy(source:Path,destination:Path):
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():shutil.copytree(source,destination,dirs_exist_ok=True)
        else:shutil.copy2(source,destination)

def prepare(source:Path,output:Path,config:dict,manifest:dict,initialize_poses=True):
    ids=[int(x) for x in config['selected_ids']]
    whitelist_path=Path(config['_config_path']).parent/'class_whitelist.txt'
    if not whitelist_path.is_file():raise FileNotFoundError(f'missing category whitelist: {whitelist_path}')
    allowed_categories={line.strip() for line in whitelist_path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')}
    if not allowed_categories:raise ValueError(f'category whitelist is empty: {whitelist_path}')
    assets={}
    for identity in ids:
        record=config['objects'][str(identity)];category=str(record.get('category','')).strip()
        if category not in allowed_categories:raise ValueError(f'category for object {identity} must be in {sorted(allowed_categories)}')
        model=Path(str(record.get('model_path',''))).expanduser()
        if not model.is_absolute():model=(Path(config['_config_path']).parent/model).resolve()
        if not model.is_file():raise FileNotFoundError(f'model_path for object {identity} must name an existing model file: {model}')
        assets[str(identity)]={'object_id':identity,'category':category,'model_id':model.stem,'asset_path':str(model.resolve()),'description':'Manually specified model'}
    output.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json',PKL]:link_or_copy(source/name,output/name)
    for item in manifest['videos']:link_or_copy(source/item['file'],output/item['file'])
    (output/'obj_selected.txt').write_text(''.join(f'{x}\n' for x in ids));(output/'dynamic_vehicle_ids.txt').write_text(''.join(f'{x}\n' for x in config.get('dynamic_ids',ids)))
    (output/'asset_manifest.json').write_text(json.dumps({'objects':assets},indent=2)+'\n')
    if not initialize_poses:return
    # The source annotations and calibrated camera records are read-only inputs.
    initial=pickle.loads((source/PKL).read_bytes());infos=initial['infos']
    for info in infos:
        poses={}
        for identity,box in zip(info['track_id'],info['gt_boxes']):
            identity=int(identity)
            if identity not in ids:continue
            b=np.asarray(box,float);pose=np.eye(4);yaw=b[6]+np.pi/2
            pose[:3,:3]=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])@np.diag(b[3:6])
            pose[:3,3]=(CANONICAL_FROM_SOURCE@np.r_[b[:3],1])[:3]
            poses[str(identity)]={'canonical_ego_from_unit_mesh':pose.tolist()}
        info['mesh_poses_9dof']=poses
    (output/'pkl').mkdir(exist_ok=True)
    with (output/'pkl'/'result_interpolation_optimized_first101_mesh_9dof.pkl').open('wb') as f:pickle.dump(initial,f,pickle.HIGHEST_PROTOCOL)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--jobs',type=int,default=4);p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--evaluations',type=int,default=140);p.add_argument('--static-evaluations',type=int,default=240);p.add_argument('--include-static',action='store_true');p.add_argument('--selection-evidence',type=Path);p.add_argument('--scene-depth-root',type=Path);p.add_argument('--masks-only',action='store_true',help='Generate selected vehicle and all-moving-object masks, then stop before mesh fitting');p.add_argument('--motion-displacement-m',type=float,default=1.);a=p.parse_args()
    if a.jobs<1 or not a.gpus:p.error('--jobs must be positive and --gpus must list at least one device')
    if not np.isfinite(a.motion_displacement_m) or a.motion_displacement_m<=0:p.error('--motion-displacement-m must be positive and finite')
    if not os.environ.get('SAM3_CHECKPOINT'):p.error('set SAM3_CHECKPOINT to a local SAM3 checkpoint')
    source=a.input.resolve();output=a.output.resolve();conf_path=a.config.resolve()
    if source==output or source in output.parents:p.error('--output must be outside --input')
    if output.exists() and any(output.iterdir()) and not a.resume:p.error('non-empty --output requires --resume')
    config=yaml.safe_load(conf_path.read_text());config['_config_path']=str(conf_path);ids=[int(i) for i in config['selected_ids']]
    if len(ids)!=len(set(ids)) or set(ids)!={int(x) for x in config['objects']}:p.error('selected_ids and objects must contain the same unique IDs')
    evidence_path=a.selection_evidence or conf_path.parent/'dynamic_vehicle_statistics.json'
    if a.include_static and a.selection_evidence is None:p.error('--include-static requires explicit --selection-evidence')
    if not evidence_path.is_file():p.error(f'missing dynamic-selection evidence: {evidence_path}')
    evidence=json.loads(evidence_path.read_text())
    if not a.include_static and evidence.get('policy','').startswith('all_'):p.error('all-vehicle evidence requires --include-static')
    eligible={int(i) for i in evidence['selected_ids']}
    if a.include_static:
        config['dynamic_ids']=[int(i) for i in evidence['dynamic_ids'] if int(i) in ids]
    if not set(ids).issubset(eligible):p.error(f'only motor-vehicle IDs covered by selection evidence may be registered; eligible IDs are {sorted(eligible)}')
    manifest=json.loads((source/'manifest.json').read_text())
    if a.scene_depth_root and not (a.scene_depth_root/'lidar_depth').is_dir():p.error('--scene-depth-root must contain lidar_depth/')
    if not a.resume:prepare(source,output,config,manifest,initialize_poses=not a.masks_only)
    elif not (output/'asset_manifest.json').is_file() or not (output/'obj_selected.txt').is_file():p.error('--resume requires an initialized output directory from an earlier run')
    active_gpus=a.gpus[:min(a.jobs,len(a.gpus))]
    env=dict(os.environ,MULTIVIEW_RUN_ROOT=str(output),MULTIVIEW_ASSET_MANIFEST=str(output/'asset_manifest.json'),MULTIVIEW_LOG_ROOT=str(output/'logs'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYOPENGL_PLATFORM='egl')
    if a.scene_depth_root:env['MULTIVIEW_SCENE_DEPTH_ROOT']=str(a.scene_depth_root.resolve())
    moving_args=['--input',str(source),'--output',str(output/'moving-object-candidates'),'--gpus',*map(str,active_gpus),'--displacement-m',str(a.motion_displacement_m)]
    if a.scene_depth_root:moving_args+=['--scene-depth-root',str(a.scene_depth_root.resolve())]
    whole_args=['--candidates',str(output/'moving-object-candidates'),'--output',str(output/'moving-object-masks'),'--gpus',*map(str,active_gpus)]
    stages=[('segment_masks.py',['--gpus',*map(str,active_gpus)],output/'mask-videos'/'mask_manifest.json'),('refine_missing_masks.py',['--gpus',*map(str,active_gpus)],None),('segment_moving_objects.py',moving_args,None),('assemble_moving_objects.py',whole_args,None)]
    fitting_stages=[('optimize_9dof.py',['--gpus',*map(str,active_gpus),'--evaluations',str(a.evaluations)],output/'pkl'/'optimization_summary.json'),('optimize_grounded_temporal.py',['--gpus',*map(str,active_gpus),'--evaluations',str(a.evaluations),'--static-evaluations',str(a.static_evaluations)],output/'pkl'/'grounded_temporal_report.json'),('render_optimized.py',['--gpus',*map(str,active_gpus)],output/'register-videos-optimized'/'registration_manifest.json'),('validate_results.py',['--workers',str(a.jobs)],output/'pkl'/'validation_report.json')]
    if not a.masks_only:stages+=fitting_stages
    (output/'logs').mkdir(parents=True,exist_ok=True)
    for script,args,marker in stages:
        if marker is not None and marker.exists() and a.resume:continue
        cmd=[sys.executable,str(UTILS/script),*args];log=output/'logs'/f'{script}.log'
        with log.open('w') as stream:subprocess.run(cmd,cwd=TASK,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'output':str(output),'selected_vehicle_ids':ids,'include_static':a.include_static,'stages':'masks_complete' if a.masks_only else 'complete','mask_outputs':{'selected_vehicles':str(output/'mask-videos'),'moving_objects':str(output/'moving-object-masks')}},indent=2))

if __name__=='__main__':main()
