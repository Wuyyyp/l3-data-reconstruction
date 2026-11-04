"""One entry: converted LiDAR -> road/background -> masks -> anchored static ImprovedGS."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
MODULES=REPO/'lib/preprocessing/ground_pipeline'
TEMPLATE=REPO/'configs/pipelines/ground_static_train_template.json'
DEFAULTS=dict(width=1600,workers=4,frames=None,cameras=None,ground_voxel=.10,
    background_voxel=.20,normal_z_min=.94,plane_rms_max=.06,
    support_radius=1.,ground_tolerance=.15,occlusion_tolerance=.15,
    dynamic_dilation=11,ground_point_cap=1200000,background_budget=3403964,
    warmup=3000,iterations=25000,scene_mode='static',sky_mode='cubemap',sky_max_points=0,mesh_source=None,pose_csv=None,reuse_prepared_background=False,split='heldout')
STAGES=[('extract','preprocess/extraction'),('separate','preprocess/separation'),
        ('refine','preprocess/refined'),('prepare','data')]

def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def integer_list(value):
    result=[]
    for part in value.split(','):
        if ':' in part:
            lo,hi=map(int,part.split(':'));result.extend(range(lo,hi+1))
        else:result.append(int(part))
    return sorted(set(result))

def find_mask(folder,name,template=False):
    stems=[name,'000000_'+name.split('_')[1]] if template else [name]
    for stem in stems:
        for ext in ['.png','.jpg','.jpeg']:
            p=Path(folder)/(stem+ext)
            if p.is_file():return p
    raise FileNotFoundError(str(Path(folder)/name))

def preflight(cfg,require_split):
    import numpy as np
    s=Path(cfg['source']);ego=Path(cfg['ego_mask'])
    required=[s/'pointcloud.npz',s/'input_manifest.json',s/'track/track_info.txt']
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
    with np.load(s/'pointcloud.npz',allow_pickle=True) as z:
        if not {'pointcloud','camera_projection'}.issubset(z.files):
            raise ValueError('pointcloud.npz requires pointcloud and camera_projection dictionaries')
    manifest=json.loads((s/'input_manifest.json').read_text())
    if not {'records','calibrations'}.issubset(manifest):
        raise ValueError('Source must be original converted records/calibrations format')
    records=[]
    available_frames=set();available_cams=set()
    for rec in manifest['records']:
        name=Path(rec['label']).stem;fid,cam=map(int,name.split('_'))
        available_frames.add(fid);available_cams.add(cam)
        if cfg['frames'] is not None and fid not in cfg['frames']:continue
        if cfg['cameras'] is not None and cam not in cfg['cameras']:continue
        cal=manifest['calibrations'][rec['camera']]
        K=np.asarray(cal['K']);pose=np.asarray(rec['camera_to_world'])
        if K.shape!=(3,3) or pose.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(pose).all():
            raise ValueError('Invalid calibration: '+name)
        if cal['width']<=0 or cal['height']<=0:raise ValueError('Invalid image dimensions')
        for p in [s/'images'/(name+'.jpg'),s/'lidar_depth'/(name+'.npy'),s/'ego_pose'/('%06d.txt'%fid)]:
            if not p.is_file():raise FileNotFoundError(p)
            required.append(p)
        required.extend([find_mask(s/'sky_mask',name),find_mask(s/'dynamic_mask',name),
                         find_mask(ego,name,template=True)])
        records.append((fid,cam,name))
    if cfg['frames'] is not None and set(cfg['frames'])-available_frames:raise ValueError('Requested frames absent')
    if cfg['cameras'] is not None and set(cfg['cameras'])-available_cams:raise ValueError('Requested cameras absent')
    if not records:raise ValueError('No selected views')
    frames=sorted({r[0] for r in records})
    if require_split and cfg.get('split','heldout')=='heldout' and not (any(f%5!=4 for f in frames) and any(f%5==4 for f in frames)):
        raise ValueError('Need training and held-out frames (frame%5==4)')
    # Colorization may use every source camera in each selected LiDAR frame.
    for fid in frames:required.extend((s/'images').glob('%06d_*.jpg'%fid))
    for fid in frames:
        p=s/'ego_pose'/('%06d.txt'%fid);required.append(p)
        E=np.loadtxt(p)
        if E.shape!=(4,4) or not np.isfinite(E).all() or not np.allclose(E[3],[0,0,0,1]):
            raise ValueError('Invalid ego pose: '+str(p))
    cfg['origin_frame']=frames[0]
    cfg['dynamic_frame']=frames[len(frames)//2]
    if cfg.get('mesh_source'):
        sys.path.insert(0,str(MODULES))
        from mesh_inputs import bundle
        paired=bundle(cfg['mesh_source'],frames=frames,cameras=cfg['cameras'],pose_csv=cfg.get('pose_csv'))
        selected_source={Path(v['label']).stem:np.asarray(v['camera_to_world']) for v in manifest['records']}
        for v in paired['views']:
            if v['name'] not in selected_source:raise ValueError('Mesh camera missing from LiDAR source manifest')
            np.testing.assert_allclose(selected_source[v['name']],np.asarray(v['c2w']),atol=1e-8,rtol=0,err_msg='Source/mesh camera authority differs')
        required.extend(paired['files'])
        if cfg.get('reuse_prepared_background'):
            from mesh_inputs import readable
            required.append(readable(Path(cfg['mesh_source'])/'model/input_ply/points3D_bkgd.ply'))
    fingerprint=[(str(p.absolute()),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(set(required))]
    source_hash=hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    files=sorted(MODULES.glob('*.py'))+[Path(__file__).resolve(),TEMPLATE]
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    return dict(views=len(records),frames=frames,cameras=sorted({r[1] for r in records}),
                source_fingerprint=source_hash,implementation_fingerprint=code_hash)

def validate_prepared(data):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData
    manifest=json.loads((data/'input_manifest.json').read_text())
    count=0
    for view in manifest['views']:
        name=view['name'];fid,cam=name.split('_');size=(view['height'],view['width'])
        gm=np.load(data/'ground_masks'/('cam_'+cam)/(fid+'.npy'))
        def read(folder):return np.asarray(Image.open(data/'converted'/folder/(name+'.png')))>0
        valid,dyn,sky=read('valid_mask'),read('target_mask'),read('sky_mask')
        if gm.shape!=size or any(x.shape!=size for x in [valid,dyn,sky]):raise ValueError('Mask dimension mismatch '+name)
        if np.any(gm&(~valid|dyn|sky)):raise ValueError('Ground mask intersects exclusions '+name)
        with Image.open(data/'converted/images'/(name+'.jpg')) as im:
            if im.size!=size[::-1]:raise ValueError('Image dimension mismatch '+name)
        depth=np.load(data/'converted/lidar_depth'/(name+'.npy'),allow_pickle=True).item()
        if depth['mask'].shape!=size or len(depth['value'])!=int(depth['mask'].sum()):raise ValueError('Depth shape mismatch')
        if not np.isfinite(depth['value']).all() or np.any(depth['value']<=0):raise ValueError('Invalid depth')
        if np.any(depth['mask']&(~valid|(dyn if manifest.get('static_only',True) else False)|sky)):raise ValueError('Depth intersects exclusions')
        count+=int(gm.sum())
    if count==0:raise ValueError('No ground pixels; inspect projected masks')
    with np.load(data/'prepared/ground.npz') as ground:
        n=len(ground['xyz'])
        if not n or not all(np.isfinite(ground[k]).all() for k in ground.files):raise ValueError('Invalid ground initialization')
        if not np.allclose(np.linalg.norm(ground['normals'],axis=1),1,atol=.001):raise ValueError('Invalid ground normals')
    background=PlyData.read(str(data/'prepared/nonground.ply'))['vertex'].data
    if not len(background):raise ValueError('No static background points')
    result=dict(passed=True,views=len(manifest['views']),ground_points=n,background_points=len(background),
                ground_pixels=count,mask_exclusions_checked=True,depth_checked=True)
    write_json(data/'input_validation.json',result)
    return result

def artifacts(folder):
    return {str(p.relative_to(folder)):p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}

def recover_curve(curve,step):
    """Archive uncheckpointed rows, including an interrupted final JSON write."""
    lines=curve.read_text().splitlines()
    nonempty=[(i,line) for i,line in enumerate(lines) if line.strip()]
    rows=[];truncated=False
    for i,line in nonempty:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=nonempty[-1][0]:raise ValueError('Corrupted middle of training curve')
            truncated=True
    kept=[r for r in rows if r['iteration']<=step]
    if [r['iteration'] for r in kept]!=list(range(1,step+1)):
        raise ValueError('Invalid checkpoint/curve sequence')
    if truncated or len(kept)!=len(rows):
        shutil.copy2(curve,curve.with_name('curve.before_resume_'+str(time.time_ns())+'.jsonl'))
        curve.write_text(''.join(json.dumps(r)+'\n' for r in kept))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',help='Pipeline recipe JSON (not a trainer configuration)')
    for name in ['source','output','ego-mask']:p.add_argument('--'+name)
    p.add_argument('--split',choices=['heldout','train-all'])
    p.add_argument('--scene-mode',choices=['static','dynamic'])
    p.add_argument('--sky-mode',choices=['cubemap','hemisphere'])
    p.add_argument('--sky-max-points',type=int)
    p.add_argument('--reuse-prepared-background',action='store_true',default=None)
    p.add_argument('--mesh-source')
    p.add_argument('--pose-csv')
    p.add_argument('--smoke',action='store_true',help='Force bounded topology checks in short training')
    p.add_argument('--frames',type=integer_list,help='e.g. 0,1,4 or 0:100 (inclusive)')
    p.add_argument('--cameras',type=integer_list)
    for name in ['width','workers','dynamic-dilation','ground-point-cap','background-budget','warmup','iterations']:
        p.add_argument('--'+name,type=int)
    for name in ['ground-voxel','background-voxel','normal-z-min','plane-rms-max','support-radius','ground-tolerance','occlusion-tolerance']:
        p.add_argument('--'+name,type=float)
    p.add_argument('--stage',choices=['validate','extract','separate','refine','prepare','train','evaluate','all'],default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--allow-share',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--eval-limit',type=int,default=0)
    args=p.parse_args()
    recipe=json.loads(Path(args.config).read_text()) if args.config else {}
    if args.output:recipe['output']=args.output
    if 'output' not in recipe:p.error('--output is required')
    root=Path(recipe['output']).absolute()
    if not root.resolve().is_relative_to(REPO/'output'):p.error('--output must be a NEW directory under repository/output')
    prior=root/'pipeline_config.json'
    if args.resume and prior.exists():
        saved=json.loads(prior.read_text());recipe={**saved,**recipe}
    cfg={**DEFAULTS,**recipe}
    for key in ['source','ego_mask']+list(DEFAULTS):
        val=getattr(args,key,None)
        if val is not None:cfg[key]=val
    if 'source' not in cfg:p.error('--source is required for a new pipeline')
    cfg['source']=str(Path(cfg['source']).absolute());cfg['output']=str(root)
    cfg['ego_mask']=str(Path(cfg.get('ego_mask',str(REPO/'configs/preprocessing/ego_mask_l3_recon_20261030'))).absolute())
    if not 32<=cfg['width']<=1600:p.error('Validated image width range is 32..1600')
    if cfg['workers']<1 or cfg['dynamic_dilation']<1 or cfg['dynamic_dilation']%2!=1:p.error('Invalid workers/dilation')
    if cfg['warmup']<0 or cfg['warmup']>=cfg['iterations']:p.error('Require 0 <= warmup < iterations')
    for key in ['ground_voxel','background_voxel','plane_rms_max','support_radius','ground_tolerance']:
        if cfg[key]<=0:p.error(key+' must be positive')
    if not 0<cfg['normal_z_min']<1:p.error('normal-z-min must be between 0 and 1')
    if cfg['scene_mode']=='dynamic' and not cfg['mesh_source']:p.error('Dynamic ground requires --mesh-source and --pose-csv')
    if cfg['mesh_source'] and not cfg['pose_csv']:p.error('Mesh source requires explicit paired --pose-csv')
    info=preflight(cfg,args.stage not in ['validate','extract','separate','refine'])
    if args.stage=='validate':print(json.dumps(info,indent=2));return
    if root.exists() and not args.resume:raise FileExistsError('Existing output: use --resume or a new --output')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if prior.exists():
        if json.loads(prior.read_text())!=cfg:raise ValueError('Recipe changed; use a new output directory')
        old=json.loads((root/'input_fingerprint.json').read_text())
        if old!=info:raise ValueError('Inputs or preprocessing implementation changed; use a new output directory')
    else:
        write_json(prior,cfg);write_json(root/'input_fingerprint.json',info)
    for folder in ['logs','preprocess','stages']:(root/folder).mkdir(exist_ok=True)
    def status(stage,**extra):write_json(root/'pipeline_status.json',dict(stage=stage,time=time.time(),**extra))
    deps=[REPO/'deps/ground_preprocess',REPO/'deps/improvedgs_ground',REPO/'deps/improvedgs_dynamic',REPO.parent/'gsplat_shape_recovery_20261013/deps']
    env=dict(os.environ,PYTHONPATH=':'.join(map(str,deps)),ROAD_PIPELINE_CONFIG=str(prior),
             OMP_NUM_THREADS=str(cfg['workers']),OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for key in ['BENCHMARK_START','SMOKE','RESUME','EVAL_LIMIT']:env.pop(key,None)
    try:
        if args.stage not in ['train','evaluate']:
            for stage,relative in STAGES:
                target=root/relative;marker=root/'stages'/(stage+'.json')
                if marker.exists():
                    expected=json.loads(marker.read_text())['artifacts']
                    if artifacts(target)!=expected:raise ValueError('Stage artifacts changed: '+stage)
                    print('REUSE',stage,flush=True)
                else:
                    temporary=target.with_name(target.name+'.incomplete')
                    for partial in [temporary,target]:
                        if partial.exists():partial.rename(partial.with_name(partial.name+'.unfinished_'+str(time.time_ns())))
                    log=root/'logs'/(stage+'_'+str(time.time_ns())+'.log')
                    with log.open('x') as stream:
                        child=subprocess.Popen([sys.executable,str(MODULES/(stage+'.py'))],cwd=REPO,
                            env=dict(env,ROAD_STAGE_OUTPUT=str(temporary)),stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                        status(stage,pid=child.pid,log=str(log));print('RUN',stage,str(log),flush=True)
                        rc=child.wait()
                    if rc:raise RuntimeError(stage+' failed; see '+str(log))
                    if stage=='prepare':validate_prepared(temporary)
                    temporary.rename(target)
                    write_json(marker,dict(stage=stage,artifacts=artifacts(target),completed=time.time()))
                if args.stage==stage and stage!='prepare':status(stage+'_complete');return
        data=root/'data'
        if not (root/'stages/prepare.json').exists():raise RuntimeError('Run --stage prepare first')
        if artifacts(data)!=json.loads((root/'stages/prepare.json').read_text())['artifacts']:
            raise ValueError('Prepared data changed')
        prepared=json.loads((data/'input_validation.json').read_text())
        c=json.loads(TEMPLATE.read_text())
        c['source_path']=str(data/'converted');c['model_path']=str(root/'train')
        for key,folder in [('record_dir','record'),('point_cloud_dir','point_cloud'),('trained_model_dir','trained_model')]:
            c[key]=str(root/'train'/folder)
        c['workspace']=str(REPO);c['data']['cameras']=info['cameras'];c['data']['selected_frames']=[min(info['frames']),max(info['frames'])]
        c['train']['iterations']=cfg['iterations'];c['model']['nsg'].update(include_obj=cfg['scene_mode']=='dynamic',opt_track=False)
        if cfg['sky_mode']=='hemisphere':
            c['model']['sky'].update(type='gaussian_sphere',geometry='hemisphere',max_points=cfg['sky_max_points'],coverage_weight=0.,multi_layer_init=False)
            c['data']['use_sky_mask']=False;c['optim']['lambda_sky']=0.0
        c['data']['frame_ids']=info['frames']
        c['data']['ground_split']=cfg['split']
        c['model']['ground'].update(enabled=True,mode='anchored',init=str(data/'prepared/ground.npz'),
            background_ply=str(data/'prepared/nonground.ply'),masks=str(data/'ground_masks'),split=True,
            warmup=cfg['warmup'],point_cap=cfg['ground_point_cap'],split_threshold=.0002,actor_ply_dir=str(data/'prepared/actors'))
        c['improvedgs']['background_budget']=max(cfg['background_budget'],prepared['background_points'])
        if cfg['ground_point_cap']<=prepared['ground_points']:raise ValueError('Increase --ground-point-cap in a new recipe')
        train_config=root/'train_config.json'
        if train_config.exists():
            previous = json.loads(train_config.read_text())
            # On resume use the original ground degree, including legacy configs without this field.
            ground = previous['model']['ground']
            if 'sh_degree' in ground:
                c['model']['ground']['sh_degree'] = ground['sh_degree']
            else:
                c['model']['ground'].pop('sh_degree', None)
            if previous != c:raise ValueError('Generated training config changed')
        write_json(train_config,c)
        launch=[sys.executable,str(REPO/'scripts/train_improvedgs_ground.py'),'--config',str(train_config),'--gpu',str(args.gpu)]
        if args.allow_share:launch+=['--allow-share']
        subprocess.run(launch+['--stage','validate'],env=env,cwd=REPO,check=True,pass_fds=(lock.fileno(),))
        if args.stage=='prepare':status('prepared',validation=prepared);return
        stop=args.stop_after or cfg['iterations']
        train=root/'train';checkpoint=train/('checkpoint_%06d.pth'%stop)
        if args.stage!='evaluate' and not checkpoint.exists():
            extra=[]
            checkpoints=sorted(train.glob('checkpoint_*.pth')) if train.exists() else []
            if checkpoints:
                latest=checkpoints[-1];step=int(latest.stem.split('_')[-1])
                if step>stop:raise ValueError('Requested stop is before latest checkpoint')
                curve=train/'curve.jsonl'
                recover_curve(curve,step)
                extra=['--resume',str(latest)]
            elif train.exists() and (train/'curve.jsonl').exists():
                train.rename(root/('train.uncheckpointed_'+str(time.time_ns())))
            status('training',gpu=args.gpu,target=stop)
            subprocess.run(launch+(['--stage','smoke'] if args.smoke else [])+['--stop-after',str(stop)]+extra,env=env,cwd=REPO,check=True,pass_fds=(lock.fileno(),))
        if not checkpoint.exists():raise FileNotFoundError(checkpoint)
        if args.stage in ['all','evaluate']:
            evaluation=train/('eval_'+checkpoint.stem)
            if not (evaluation/'results.json').exists():
                if evaluation.exists():evaluation.rename(evaluation.with_name(evaluation.name+'.unfinished_'+str(time.time_ns())))
                extra=['--eval-limit',str(args.eval_limit)] if args.eval_limit else []
                status('evaluating',checkpoint=str(checkpoint))
                subprocess.run(launch+['--stage','evaluate','--checkpoint',str(checkpoint)]+extra,env=env,cwd=REPO,check=True,pass_fds=(lock.fileno(),))
        status('complete',checkpoint=str(checkpoint),requested_stage=args.stage)
        print('COMPLETE',str(root),flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc));raise

if __name__=='__main__':main()
