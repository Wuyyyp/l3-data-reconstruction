from pathlib import Path
import os,sys,json,subprocess,time,fcntl,traceback
R=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');O=Path(os.environ.get('GROUND_RUN_ROOT',str(R/'output/sample00024_static_anchored_20261026')));C=R/'lib/trainers/improvedgs_ground_static';P=O/'prepared';B=R/'output/ground_static_split_v4_20261026';py=str(R.parent/'Improved-GS/.venv/bin/python');gpu=1
lock=(O/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 p=O/'status.tmp';p.write_text(json.dumps(dict(stage=stage,time=time.time(),gpu=gpu,**kw),indent=2));p.replace(O/'status.json');print(stage,kw,flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=':'.join(map(str,[O/'deps',C,R/'deps/improvedgs_dynamic',R.parent/'gsplat_shape_recovery_20261013/deps'])),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='8.0',TORCH_EXTENSIONS_DIR=str(R/'output/sample01020_mesh_cube_fast_20261025/torch_extensions'),MAX_JOBS='4',GROUND_INIT=str(P/'ground.npz'),GROUND_MASKS=str(O/'ground_masks'),GROUND_MODE='anchored',GROUND_SPLIT='1',GROUND_POINT_CAP='1200000')
def run(phase,steps,resume=None,evaluate=False):
 out=O/'anchored'/phase;out.mkdir(parents=True,exist_ok=True);inp=out/'input_ply';inp.mkdir(exist_ok=True);q=inp/'points3D_bkgd.ply'
 if not q.exists():q.symlink_to(P/'nonground.ply')
 cfg=json.loads((B/'control/full/config.yaml').read_text());cfg.update(source_path=str(O/'converted'),model_path=str(out),point_cloud_dir=str(out/'point_cloud'),record_dir=str(out/'record'),trained_model_dir=str(out/'trained_model'));cfg['model']['nsg'].update(include_obj=False,opt_track=False);(out/'config.yaml').write_text(json.dumps(cfg,indent=2))
 e=env.copy();e.update(DYNAMIC_CONFIG=str(out/'config.yaml'),GROUND_WARMUP='5' if phase=='smoke' else '3000',STOP_AFTER=str(steps))
 if phase=='smoke':e.update(SMOKE='1',EVAL_LIMIT='2')
 if resume:e['RESUME']=str(resume)
 args=json.loads((R/'lib/trainers/improvedgs_static/base_args.json').read_text());prep=json.loads((O/'preparation_report.json').read_text());budget=max(4000000-prep['ground_points'],prep['background_points'])
 for flag,value in [('-s',str(O/'converted')),('-m',str(out)),('--use_bilateral_grid','false'),('--budget',str(budget))]:args[args.index(flag)+1]=value
 tag=('eval' if evaluate else 'resume' if resume else 'train')+f'_{steps:06d}';cmd=[py,str(C/'evaluate_grounded.py'),str(out/f'checkpoint_{steps:06d}.pth'),str(out/'eval')] if evaluate else [py,str(C/'train.py')]+args
 with (out/(tag+'.log')).open('x') as f:
  proc=subprocess.Popen(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT);status(tag,phase=phase,pid=proc.pid);rc=proc.wait()
 (out/(tag+'_exit.json')).write_text(json.dumps(dict(returncode=rc)));assert rc==0,str(out/(tag+'.log'));return out/f'checkpoint_{steps:06d}.pth'
try:
 assert json.loads((O/'prepare_status.json').read_text())['stage']=='complete'
 assert json.loads((O/'input_validation.json').read_text())['passed']
 status('waiting_for_gpu',policy='GPU1 idle memory<1GiB, utilization<5%; sharing only if explicitly authorized')
 while True:
  mem,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  if (mem<1000 and util<5) or (O/'allow_share_gpu1').exists():break
  time.sleep(30)
 if '--direct' in sys.argv:
  run('full',25000);run('full',25000,evaluate=True);status('complete')
 else:
  ck=run('smoke',10);run('smoke',20,ck);run('smoke',20,evaluate=True)
  assert json.loads((O/'anchored/smoke/static_validation.json').read_text())['actors']==0
  (O/'anchored/smoke/passed.json').write_text(json.dumps(dict(passed=True,static_only=True)))
  ck=run('full',3200);run('full',25000,ck);run('full',25000,evaluate=True);status('complete')
except BaseException as exc:status('failed',error=repr(exc));traceback.print_exc();sys.exit(1)
