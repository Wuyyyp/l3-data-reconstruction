"""Config-driven ImprovedGS ground/anchor training, checkpoint resume and evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / 'lib/trainers/improvedgs_ground'

def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    ground = config['model']['ground']
    if 'sh_degree' not in ground:
        saved = Path(config['model_path']) / 'pipeline_config.json'
        if saved.is_file():
            # Older runs omit this field and used SH1. Preserve their config on resume.
            previous = json.loads(saved.read_text())['model']['ground']
            if 'sh_degree' in previous:
                ground['sh_degree'] = previous['sh_degree']
        else:
            ground['sh_degree'] = 0
    return config

def validate(config):
    ground = config.get('model', {}).get('ground', {})
    if not ground.get('enabled'):
        raise ValueError('model.ground.enabled must be true')
    if ground.get('sh_degree', 1) not in (0, 1):
        raise ValueError('Ground SH degree must be 0 or 1')
    if ground.get('mode', 'anchored') != 'anchored':
        raise ValueError('This entry supports anchored ground')
    if ground.get('split', True) is not True:
        raise ValueError('Ground splitting is required')
    if config['data']['type'] != 'T3Prepared':
        raise ValueError('Ground pipeline requires T3Prepared camera/mask/point-cloud format')
    if config['model']['gaussian']['sh_degree'] != 1:
        raise ValueError('Background and actors currently require SH degree 1')
    sky=config['model']['sky']
    if sky.get('type','cubemap') not in ['cubemap','gaussian_sphere']:raise ValueError('Unsupported ground sky type')
    if sky.get('type')=='gaussian_sphere' and (sky.get('geometry')!='hemisphere' or config['data'].get('use_sky_mask',True)):
        raise ValueError('Ground Gaussian sky requires explicit hemisphere and use_sky_mask=false')
    if config['model']['nsg'].get('opt_track', False):
        raise ValueError('Prepared mesh affine poses must remain fixed')
    source = Path(config['source_path']).absolute()
    for name in ['input_manifest.json', 'actors.json']:
        if not (source.parent / name).is_file():
            raise FileNotFoundError(source.parent / name)
    for name in ['images','valid_mask','sky_mask','target_mask','lidar_depth']:
        if not (source / name).is_dir():
            raise FileNotFoundError(source / name)
    import numpy as np
    path = Path(ground['init'])
    with np.load(path) as a:
        for name, width in [('xyz',3),('normals',3),('rotation',4),('rgb',3)]:
            if name not in a or a[name].ndim != 2 or a[name].shape[1] != width:
                raise ValueError('ground.npz missing/invalid '+name)
            if len(a[name]) != len(a['xyz']) or not np.isfinite(a[name]).all():
                raise ValueError('ground.npz inconsistent/nonfinite '+name)
        n = len(a['xyz'])
        if not n or not np.allclose(np.linalg.norm(a['normals'],axis=1),1,atol=1e-3):
            raise ValueError('Ground requires nonempty unit normals')
        if not np.allclose(np.linalg.norm(a['rotation'],axis=1),1,atol=1e-3):
            raise ValueError('Ground rotations must be unit quaternions')
        w,x,y,z=a['rotation'].T
        normal_axis=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=1)
        if not np.allclose(np.abs((normal_axis*a['normals']).sum(1)),1,atol=1e-3):
            raise ValueError('Ground rotation thin axis must align with supplied normal')
        if a['rgb'].min()<0 or a['rgb'].max()>1:
            raise ValueError('Ground RGB must be in [0,1]')
    if int(ground.get('point_cap',1200000)) <= n:
        raise ValueError('Ground point cap must exceed initial count')
    if not Path(ground['background_ply']).is_file():
        raise FileNotFoundError(ground['background_ply'])
    manifest=json.loads((source.parent/'input_manifest.json').read_text())
    views=[r for r in manifest['views'] if r['cam'] in config['data']['cameras'] and
           (config['data'].get('frame_ids') is None or r['frame'] in config['data']['frame_ids'])]
    if config['data'].get('ground_split','heldout')=='heldout' and (not any(r['frame']%5!=4 for r in views) or not any(r['frame']%5==4 for r in views)):
        raise ValueError('Need both train and validation views')
    for r in views:
        fid, cam=r['name'].split('_')
        mask=Path(ground['masks'])/('cam_'+cam)/(fid+'.npy')
        if not mask.is_file():raise FileNotFoundError(mask)
        m=np.load(mask,mmap_mode='r')
        if m.ndim!=2:raise ValueError('Ground masks must be 2D: '+str(mask))
    if config['model']['nsg']['include_obj']:
        actors=json.loads((source.parent/'actors.json').read_text())
        if not actors:raise ValueError('Dynamic mode needs actors.json with registered mesh poses')
        actor_dir=Path(ground['actor_ply_dir'])
        for actor in actors:
            name='points3D_obj_%03d.ply'%actor['id']
            if not (actor_dir/name).is_file():raise FileNotFoundError(actor_dir/name)
    return dict(ground_points=n,views=len(views),train=len(views) if config['data'].get('ground_split')=='train-all' else sum(r['frame']%5!=4 for r in views),
                test=sum(r['frame']%5==4 for r in views),actors_enabled=config['model']['nsg']['include_obj'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--stage',choices=['validate','train','smoke','evaluate'],default='train')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--resume')
    p.add_argument('--checkpoint')
    p.add_argument('--eval-limit',type=int,default=0)
    p.add_argument('--allow-share',action='store_true')
    p.add_argument('--no-images',action='store_true')
    args=p.parse_args()
    c=load_config(args.config)
    report=validate(c)
    if args.stage=='validate':
        print(json.dumps(report,indent=2));return
    out=Path(c['model_path']).absolute()
    if not out.resolve().is_relative_to(REPO/'output'):
        raise ValueError('Output must be within repository/output')
    if args.gpu<0:raise ValueError('GPU must be nonnegative')
    stop=args.stop_after or int(c['train']['iterations'])
    if stop<1 or stop>c['train']['iterations']:raise ValueError('Invalid stop-after')
    is_eval=args.stage=='evaluate'
    if not is_eval and (out/'curve.jsonl').exists() and not args.resume:
        raise FileExistsError('Existing run requires --resume; use a new model_path')
    if args.resume and Path(args.resume).resolve().parent!=out.resolve():
        raise ValueError('Resume checkpoint must belong to model_path')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    util,free=map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if not args.allow_share and util:
        raise RuntimeError('GPU is busy; select another GPU or explicitly use --allow-share')
    if free<4000:raise RuntimeError('Less than 4 GiB free GPU memory')
    ground=c['model']['ground']
    inp=out/'input_ply';inp.mkdir(exist_ok=True)
    files={'points3D_bkgd.ply':Path(ground['background_ply'])}
    if c['model']['nsg']['include_obj']:
        files.update({f.name:f for f in Path(ground['actor_ply_dir']).glob('points3D_obj_*.ply')})
    for name,source in files.items():
        dest=inp/name
        if dest.exists() or dest.is_symlink():
            if dest.resolve()!=source.resolve():raise FileExistsError('Different input already exists: '+str(dest))
        else:dest.symlink_to(source.resolve())
    c['data'].update(regenerate_pcd=False,load_pcd_from=False,streaming_load=True,shuffle=False)
    c['mode']='train'
    c['model_path']=str(out)
    for name in ['point_cloud','record','trained_model']:
        c[name+'_dir']=str(out/name)
    # Custom configuration is owned by this launcher, not the legacy yacs schema.
    runtime=json.loads(json.dumps(c))
    runtime['model'].pop('ground')
    runtime.pop('improvedgs',None)
    config_file=out/('evaluation_config.yaml' if is_eval else 'launch_config.yaml')
    if (is_eval or args.resume) and (out/'pipeline_config.json').exists():
        old=json.loads((out/'pipeline_config.json').read_text())
        if old!=c:raise ValueError('Configuration differs from saved pipeline_config.json')
    config_file.write_text(json.dumps(runtime,indent=2))
    if not is_eval:(out/'pipeline_config.json').write_text(json.dumps(c,indent=2))
    (out/'input_validation.json').write_text(json.dumps(report,indent=2))
    cache=REPO/'cache/improvedgs_static';cache.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['RESUME','SMOKE','EVAL_LIMIT']:env.pop(key,None)
    deps=[REPO/'deps/improvedgs_ground',REPO/'deps/improvedgs_dynamic',REPO.parent/'gsplat_shape_recovery_20261013/deps']
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),DYNAMIC_CONFIG=str(config_file),PWD=str(REPO),
        PYTHONPATH=':'.join(map(str,deps)),CUDA_HOME='/usr/local/cuda',
        TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='4',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        GROUND_INIT=str(ground['init']),GROUND_MASKS=str(ground['masks']),GROUND_MODE='anchored',
        GROUND_SH_DEGREE=str(ground.get('sh_degree', 1)),
        GROUND_SPLIT='1',GROUND_POINT_CAP=str(ground.get('point_cap',1200000)),
        GROUND_SPLIT_THRESHOLD=str(ground.get('split_threshold',.0002)),
        GROUND_WARMUP=str(ground.get('warmup',3000)),STOP_AFTER=str(stop),
        SAVE_IMAGES='0' if args.no_images else '1')
    if args.stage=='smoke':env['SMOKE']='1'
    if args.resume:env['RESUME']=str(Path(args.resume).resolve())
    if args.eval_limit:env['EVAL_LIMIT']=str(args.eval_limit)
    if is_eval:
        ck=Path(args.checkpoint) if args.checkpoint else out/('checkpoint_%06d.pth'%stop)
        if not ck.is_file():raise FileNotFoundError(ck)
        evaluation=out/('eval_'+ck.stem)
        if evaluation.exists():raise FileExistsError(evaluation)
        cmd=[sys.executable,str(ENGINE/'evaluate_grounded.py'),str(ck),str(evaluation)]
    else:
        base=json.loads((REPO/'lib/trainers/improvedgs_static/base_args.json').read_text())
        values={'-s':c['source_path'],'-m':str(out),'--iterations':c['train']['iterations'],
                '--position_lr_max_steps':c['train']['iterations'],'--sh_degree':1,
                '--use_bilateral_grid':'false',
                '--budget':c.get('improvedgs',{}).get('background_budget',4000000)}
        for flag,value in values.items():base[base.index(flag)+1]=str(value)
        cmd=[sys.executable,str(ENGINE/'train.py')]+base
    status=out/'pipeline_status.json'
    with (out/('%s_%06d.log'%('evaluate' if is_eval else 'resume' if args.resume else args.stage,stop))).open('a') as log:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.write_text(json.dumps(dict(stage=args.stage,state='running',pid=proc.pid,gpu=args.gpu)))
        code=proc.wait()
    status.write_text(json.dumps(dict(stage=args.stage,state='complete' if code==0 else 'failed',returncode=code)))
    if code:raise RuntimeError('Pipeline failed; inspect logs in '+str(out))
    print(str(out),flush=True)
if __name__=='__main__':main()

