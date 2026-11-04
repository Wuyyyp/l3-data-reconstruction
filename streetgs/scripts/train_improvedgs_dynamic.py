import argparse,os,sys,json,subprocess,fcntl,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--stop-after',type=int,default=25000);p.add_argument('--resume');a=p.parse_args()
 if a.gpu<0:raise ValueError('GPU must be nonnegative')
 import yaml
 c=yaml.safe_load(Path(a.config).read_text());out=Path(c['model_path']).resolve()
 if not out.is_relative_to(R/'output'):raise ValueError('Output must remain inside repository/output')
 if (out/'curve.jsonl').exists() and not a.resume:raise FileExistsError('Existing run requires --resume; use a fresh output for new training')
 if a.resume and Path(a.resume).resolve().parent!=out:raise ValueError('Resume checkpoint must belong to output directory')
 out.mkdir(parents=True,exist_ok=True);lock=open(out/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lock.fileno(),True)
 for i in range(2):
  v,f=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
  if v or f<24000:raise RuntimeError('GPU busy or insufficient free memory')
  if i==0:time.sleep(10)
 args=json.loads((R/'lib/trainers/improvedgs_static/base_args.json').read_text());args[args.index('-m')+1]=str(out);args[args.index('-s')+1]=c['source_path'];args[args.index('--use_bilateral_grid')+1]='false' if c['model'].get('use_color_correction',False) else 'true'
 c['data']['regenerate_pcd']=not bool(a.resume);c['data']['load_pcd_from']=False
 runtime=out/'launch_config.yaml';runtime.write_text(json.dumps(c,indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),DYNAMIC_CONFIG=str(runtime),STREETGS_INIT_WORKERS='4',STOP_AFTER=str(a.stop_after),PWD=str(R),PYTHONPATH=str(R/'deps/improvedgs_dynamic')+':'+str(R.parent/'gsplat_shape_recovery_20261013/deps'),CUDA_HOME='/usr/local/cuda',TORCH_EXTENSIONS_DIR=str(R/'cache/improvedgs_static/torch_extensions'),TMPDIR=str(R/'cache/improvedgs_static'),XDG_CACHE_HOME=str(R/'cache/improvedgs_static'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='4')
 if a.resume:env['RESUME']=a.resume
 os.execve(sys.executable,[sys.executable,str(R/'lib/trainers/improvedgs_dynamic/train.py')]+args,env)
if __name__=='__main__':main()
