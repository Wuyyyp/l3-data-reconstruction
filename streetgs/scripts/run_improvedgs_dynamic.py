"""Unified static/dynamic ImprovedGS with fresh preprocessing, asynchronous loading and configurable Gaussian sky."""
import argparse, json, os, subprocess, sys, time, fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 from improvedgs_pipeline_router import dispatch_ground
 if dispatch_ground(sys.argv[1:]):return
 p=argparse.ArgumentParser(description=__doc__,epilog='Ground pipeline: --pipeline ground --help (separate recipe, anchored ground, Cubemap sky).')
 p.add_argument('--pipeline',choices=['scene','ground'],default='scene',help='scene: static/dynamic Gaussian sky; ground: extraction, ground separation and anchored static training')
 p.add_argument('--variant',choices=['improvedgs','improvedgs_bilateral'],default='improvedgs')
 p.add_argument('--scene-mode',choices=['static','dynamic'],default=None)
 p.add_argument('--sky-mode',choices=['sphere','hemisphere','multisphere'],default=None)
 p.add_argument('--sky-mask',choices=['on','off'],default=None)
 p.add_argument('--sky-max-points',type=int,default=None,help='0 means unlimited for hemisphere/multisphere')
 p.add_argument('--depth-mode',choices=['inverse','metric_trim'],default=None)
 p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--output',required=True)
 p.add_argument('--source',help='Converted source dataset; defaults to the validated M18 data')
 p.add_argument('--preprocess',action='store_true',help='Fresh ego-return filtering and 3D-box camera association inside this run')
 p.add_argument('--skip-ego-filter',action='store_true',help='Explicitly preserve ego points when no validated template is available; still rebuild camera associations')
 p.add_argument('--build-ego-template',action='store_true',help='Build candidate template from this source inside each fresh output; protect labeled boxes')
 p.add_argument('--ego-min-fraction',type=float,default=.69)
 p.add_argument('--ego-min-motion',type=float,default=10.)
 p.add_argument('--ego-radius',type=float,default=5.)
 p.add_argument('--ego-z-min',type=float,default=.8)
 p.add_argument('--ego-z-max',type=float,default=2.3)
 p.add_argument('--ego-template',help='Explicit vehicle-specific template NPZ (alternative to --build-ego-template)')
 p.add_argument('--source-extrinsics',choices=['ego_to_camera','camera_to_ego'],default='camera_to_ego')
 p.add_argument('--smoke',action='store_true',help='3 steps, restore to 6, then evaluate 2 images')
 p.add_argument('--no-merge-actor-adam',action='store_true')
 p.add_argument('--resume',help='Complete checkpoint inside OUTPUT/full; only same-run resume is allowed')
 a=p.parse_args();out=Path(a.output);out=(R/out).resolve() if not out.is_absolute() else out.resolve()
 if R/'output' not in out.parents:p.error('Output must be under repository/output')
 if a.preprocess and (not a.source or sum(map(bool,[a.ego_template,a.skip_ego_filter,a.build_ego_template]))!=1):p.error('--preprocess requires --source and exactly one of --build-ego-template, --ego-template, --skip-ego-filter')
 if a.build_ego_template and (not a.preprocess or a.resume):p.error('--build-ego-template requires fresh --preprocess; same-run resume uses saved template')
 if not (0<a.ego_min_fraction<=1 and a.ego_min_motion>0 and 0<a.ego_radius<9 and -9<a.ego_z_min<a.ego_z_max<9):p.error('Invalid ego candidate thresholds/ROI')
 if a.gpu<0:p.error('GPU must be nonnegative')
 if a.resume and any(v is not None for v in [a.scene_mode,a.sky_mode,a.sky_mask,a.sky_max_points,a.depth_mode]):p.error('Resume uses saved model options; omit scene/sky/depth overrides')
 if a.sky_max_points is not None and a.sky_max_points<0:p.error('sky-max-points must be >=0')
 if a.resume and a.smoke:p.error('Smoke already includes a restore test; omit --resume')
 if out.exists() and not a.resume:p.error('Use a new output directory for fresh training')
 dest=out/'full';resume=Path(a.resume).resolve() if a.resume else None
 if resume and (resume.parent!=dest or not resume.is_file()):p.error('Resume checkpoint must belong to this run/full')
 cache=R/'cache/improvedgs_static';cache.mkdir(parents=True,exist_ok=True)
 lock=(cache/f'dynamic_gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for i in range(2):
  u,m=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
  avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
  if u!=0 or m>=1000 or avail<250*1024**2:raise RuntimeError('GPU busy or host available memory <250GiB')
  if i==0:time.sleep(5)
 if resume:
  cfg=json.loads((out/'config.json').read_text())
  saved=json.loads((out/'run_options.json').read_text())
  if saved['variant']!=a.variant:p.error('Resume variant must match the saved run')
  if a.source and saved.get('source')!=a.source:p.error('Resume source must match the saved run')
 else:
  dest.mkdir(parents=True)
  cfg=json.loads((R/'configs/experiments_m18'/('dynamic_async_'+a.variant+'.json')).read_text())
  if a.source:cfg['source_path']=str(Path(a.source).resolve())
  if a.preprocess:
   sys.path.insert(0,str(R))
   from lib.utils.m18_preprocess import prepare
   template=a.ego_template
   if a.build_ego_template:
    from lib.utils.m18_ego_template import build
    template=build(a.source,out/'ego_template',a.source_extrinsics,a.ego_min_fraction,a.ego_min_motion,a.ego_radius,a.ego_z_min,a.ego_z_max)
   cfg['source_path']=str(prepare(a.source,out/'preprocess',template,a.source_extrinsics))
  cfg.update(model_path=str(dest),point_cloud_dir=str(dest/'point_cloud'),record_dir=str(dest/'record'),trained_model_dir=str(dest/'trained_model'),workspace=str(R))
  mode=a.scene_mode or 'dynamic';geometry=a.sky_mode or ('hemisphere' if mode=='static' else 'sphere')
  use_mask=(a.sky_mask=='on') if a.sky_mask is not None else geometry=='sphere'
  if geometry!='sphere' and use_mask:p.error('Hemisphere modes currently require --sky-mask off')
  if geometry=='sphere' and not use_mask:p.error('Use hemisphere for mask-free sky')
  if geometry=='multisphere' and mode!='static':p.error('Multisphere coverage is validated for static mode only')
  limit=a.sky_max_points if a.sky_max_points is not None else (100000 if geometry=='sphere' else 0)
  if geometry=='sphere' and limit==0:p.error('Unlimited sky requires hemisphere or multisphere')
  cfg['model']['nsg']['include_obj']=mode=='dynamic'
  cfg['model']['nsg']['opt_track']=mode=='dynamic'
  cfg['model']['use_color_correction']=not a.variant.endswith('bilateral')
  cfg['model']['sky'].update(geometry=geometry,max_points=limit,multi_layer_init=geometry=='multisphere',candidates_per_layer=100000,initial_retained_points=50000,layer_radius_multipliers=[.5,1.,2.],coverage_window_fraction=.005)
  cfg['data'].update(unified_masks=True,use_sky_mask=use_mask,use_source_dynamic_mask=mode=='dynamic')
  cfg['optim']['depth_mode']=a.depth_mode or 'inverse'
  if not use_mask:cfg['optim']['lambda_sky']=0.;cfg['model']['sky']['coverage_weight']=0.
  cfg['data']['load_pcd_from']=False;cfg['data']['regenerate_pcd']=True;cfg['optim']['merge_actor_adam']=not a.no_merge_actor_adam
  if a.smoke:cfg['data']['selected_frames']=[0,2]
  (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'run_options.json').write_text(json.dumps(vars(a),indent=2))
 engine=R/'lib/trainers/improvedgs_dynamic';env=os.environ.copy()
 for key in ['RESUME','SMOKE','EVAL_LIMIT','STOP_AFTER','MERGE_ACTOR_ADAM']:env.pop(key,None)
 env.update(PWD=str(R),CUDA_VISIBLE_DEVICES=str(a.gpu),STREETGS_INIT_WORKERS='4',PYTHONPATH=':'.join(map(str,[engine,R/'deps/improvedgs_dynamic',R.parent/'gsplat_shape_recovery_20261013/deps'])),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_HOME='/usr/local/cuda',TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),TORCH_CUDA_ARCH_LIST='8.0')
 py=R.parent/'Improved-GS/.venv/bin/python';args=json.loads((R/'lib/trainers/improvedgs_static/base_args.json').read_text())
 for flag,value in [('-s',cfg['source_path']),('-m',str(dest)),('--use_bilateral_grid','true' if a.variant.endswith('bilateral') else 'false')]:args[args.index(flag)+1]=value
 def run(stage,cmd,fresh=False):
  marker=out/(stage+'_exit.json')
  if marker.exists():raise FileExistsError('Existing stage: '+str(marker))
  cfg['data']['regenerate_pcd']=fresh;runtime=out/(stage+'_config.yaml');runtime.write_text(json.dumps(cfg,indent=2));env['DYNAMIC_CONFIG']=str(runtime)
  log=out/(stage+'.log')
  with log.open('x') as f:
   start=time.time();child=subprocess.Popen(list(map(str,cmd)),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
   (out/'status.json').write_text(json.dumps(dict(stage=stage,pid=child.pid,gpu=a.gpu,start=start)))
   rc=child.wait()
  result=dict(stage=stage,returncode=rc,seconds=time.time()-start);marker.write_text(json.dumps(result));(out/'status.json').write_text(json.dumps(result))
  if rc:raise RuntimeError('Failed; inspect '+str(log))
 if a.smoke:
  env['SMOKE']='1';env['STOP_AFTER']='3';run('train3',[py,engine/'train.py']+args,True)
  env['RESUME']=str(dest/'checkpoint_000003.pth');env['STOP_AFTER']='6';run('resume6',[py,engine/'train.py']+args)
 else:
  env['STOP_AFTER']='25000'
  if resume:env['RESUME']=str(resume)
  run('resume_'+str(int(time.time())) if resume else 'train',[py,engine/'train.py']+args,not resume)
 env.pop('RESUME',None);env.pop('SMOKE',None)
 if a.smoke:env['EVAL_LIMIT']='2'
 step=6 if a.smoke else 25000
 run('eval_'+str(int(time.time())) if resume else 'eval',[py,engine/'evaluate.py',dest/f'checkpoint_{step:06d}.pth',dest/'eval'])
 result=json.loads((dest/'eval/results.json').read_text());assert result['step']==step and result['images']==(2 if a.smoke else len(json.loads((dest/'initialization.json').read_text())['camera_names'])) and (result['actors']>0 if cfg['model']['nsg']['include_obj'] else result['actors']==0)
 (out/'complete.json').write_text(json.dumps(dict(step=step,images=result['images'],variant=a.variant)))
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
