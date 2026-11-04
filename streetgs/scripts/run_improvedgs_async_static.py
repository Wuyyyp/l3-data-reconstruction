"""Fresh static ImprovedGS runs with corrected ego->camera object masks."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--variant',choices=['improvedgs','improvedgs_bilateral'],required=True)
 p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--output',required=True,help='New output directory inside this repository')
 p.add_argument('--smoke',action='store_true',help='6 steps + 2 image evaluation, instead of 25k + 707')
 a=p.parse_args();out=Path(a.output);out=(R/out).resolve() if not out.is_absolute() else out.resolve()
 if R not in out.parents:p.error('Output must be inside the repository')
 if out.exists():p.error('Output already exists: use a NEW directory; this launcher never reuses another run cache')
 lockdir=R/'cache/improvedgs_static';lockdir.mkdir(parents=True,exist_ok=True)
 lock=(lockdir/f'async_static_gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for i in range(2):
  u,m=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
  avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
  if u!=0 or m>=1000 or avail<250*1024**2:raise RuntimeError('GPU is busy or host available memory <250GiB; no job launched')
  if i==0:time.sleep(5)
 out.mkdir();dest=out/'full';dest.mkdir();cfg=json.loads((R/'configs/experiments_m18'/('static_async_'+a.variant+'.json')).read_text())
 cfg.update(model_path=str(dest),point_cloud_dir=str(dest/'point_cloud'),record_dir=str(dest/'record'),trained_model_dir=str(dest/'trained_model'),workspace=str(R))
 if a.smoke:cfg['data']['selected_frames']=[0,2]
 config=out/'config.yaml';config.write_text(json.dumps(cfg,indent=2))
 engine=R/'lib/trainers/improvedgs_async';env=os.environ.copy()
 for key in ['RESUME','SMOKE','EVAL_LIMIT','STOP_AFTER']:env.pop(key,None)
 env.update(PWD=str(R),CUDA_VISIBLE_DEVICES=str(a.gpu),DYNAMIC_CONFIG=str(config),STREETGS_INIT_WORKERS='4',PYTHONPATH=':'.join(map(str,[R/'output/static_four_stages_20261017/deps',engine,R/'deps/improvedgs_dynamic',R.parent/'gsplat_shape_recovery_20261013/deps'])),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_HOME='/usr/local/cuda',TORCH_EXTENSIONS_DIR=str(lockdir/'torch_extensions'),TMPDIR=str(lockdir),XDG_CACHE_HOME=str(lockdir),TORCH_CUDA_ARCH_LIST='8.0',STOP_AFTER='6' if a.smoke else '25000')
 if a.smoke:env['SMOKE']='1'
 py=R.parent/'Improved-GS/.venv/bin/python';args=json.loads((R/'lib/trainers/improvedgs_static/base_args.json').read_text())
 for flag,value in [('-s',cfg['source_path']),('-m',str(dest)),('--use_bilateral_grid','true' if a.variant.endswith('bilateral') else 'false')]:args[args.index(flag)+1]=value
 def run(stage,cmd):
  start=time.time();(out/'status.json').write_text(json.dumps(dict(stage=stage,state='running',gpu=a.gpu,start=start)))
  print(stage, 'log:',out/(stage+'.log'),flush=True)
  with (out/(stage+'.log')).open('w') as log:rc=subprocess.call(list(map(str,cmd)),cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  result=dict(stage=stage,returncode=rc,seconds=time.time()-start);(out/(stage+'_exit.json')).write_text(json.dumps(result));(out/'status.json').write_text(json.dumps(result))
  if rc:raise RuntimeError(f'{stage} failed; retained log: {out/(stage+".log")}')
 run('train',[py,engine/'train.py']+args)
 # Only this run's evaluation may reuse its freshly constructed input_ply.
 cfg['data']['regenerate_pcd']=False;ec=out/'eval_config.yaml';ec.write_text(json.dumps(cfg,indent=2));env['DYNAMIC_CONFIG']=str(ec)
 if a.smoke:env['EVAL_LIMIT']='2'
 step=6 if a.smoke else 25000;run('eval',[py,engine/'evaluate.py',dest/f'checkpoint_{step:06d}.pth',dest/'eval'])
 result=json.loads((dest/'eval/results.json').read_text());assert result['images']==(2 if a.smoke else 707) and result['step']==step and result['actors']==0
 (out/'complete.json').write_text(json.dumps(dict(step=step,images=result['images'],variant=a.variant)));print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
