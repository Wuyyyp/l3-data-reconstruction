"""Single GPU-1 supervisor. Never kills external processes or launches duplicates."""
from pathlib import Path
import os,sys,time,json,subprocess,fcntl,yaml,traceback
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dynamic_efficiency_20261013';RUN.mkdir(parents=True,exist_ok=True)
lock=open(RUN/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'state':'waiting_data','pid':os.getpid(),'gpu':1,'started':time.time()}
def status(**kw):
 state.update(kw);state['updated']=time.time();p=RUN/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p);print(json.dumps(state),flush=True)
def execute(name,cmd,extra=None):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(RUN/'deps')+':'+str(ROOT),PWD=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',TORCH_EXTENSIONS_DIR=str(RUN/'torch_extensions_cpp17'),TORCH_CUDA_ARCH_LIST='8.0')
 env['PATH']=str(RUN/'deps/ninja/data/bin')+':'+env.get('PATH','')
 if extra:env.update(extra)
 log=open(RUN/(name+'.log'),'a');start=time.time();p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT);status(state=name,child_pid=p.pid,stage_started=start)
 result=p.wait();status(last_stage=name,last_returncode=result,last_elapsed_s=time.time()-start)
 if result:raise RuntimeError(name+' failed; see '+str(RUN/(name+'.log')))
 return time.time()-start
try:
 status()
 while True:
  p=RUN/'prepare_status.json'
  if p.exists() and json.loads(p.read_text()).get('state')=='complete':break
  pid=RUN/'prepare.pid'
  if pid.exists() and not Path('/proc/'+pid.read_text().strip()).exists():raise RuntimeError('Data preparation exited before complete; see prepare.log')
  time.sleep(10)
 status(state='waiting_gpu1',reason='Require 6 consecutive idle readings and at least 50 GiB free; residual memory allowed')
 stable=0
 while stable<6:
  row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');util,free=[int(x.strip()) for x in row];stable=stable+1 if util==0 and free>=51200 else 0
  state.update(gpu_util=util,gpu_free_mib=free,idle_readings=stable)
  status();time.sleep(10)
 cfg=yaml.safe_load((RUN/'train.yaml').read_text());smoke=RUN/'smoke';scfg=yaml.safe_load((RUN/'train.yaml').read_text());scfg.update(model_path=str(smoke),record_dir=str(smoke/'record'),trained_model_dir=str(smoke/'trained_model'),point_cloud_dir=str(smoke/'point_cloud'));scfg['train'].update(iterations=2,save_iterations=[2],checkpoint_iterations=[2],test_iterations=[]);(RUN/'smoke.yaml').write_text(yaml.safe_dump(scfg))
 execute('smoke_train',['/usr/bin/python','-u',str(ROOT/'scripts/run_dynamic_efficiency_train.py'),'--config',str(RUN/'smoke.yaml')])
 execute('smoke_compare',['/usr/bin/python','-u',str(ROOT/'scripts/benchmark_dynamic_efficiency.py'),'--config',str(RUN/'smoke.yaml')],dict(EFF_OUT=str(RUN/'smoke_compare'),EFF_CHECKPOINT=str(smoke/'trained_model/iteration_2.pth'),EFF_ROUNDS='1'))
 # Reuse only immutable LiDAR initialization, not the two-step learned state.
 (RUN/'model').mkdir(exist_ok=True);dst=RUN/'model/input_ply'
 if not dst.exists():dst.symlink_to(smoke/'input_ply',target_is_directory=True)
 elapsed=execute('train_25000',['/usr/bin/python','-u',str(ROOT/'scripts/run_dynamic_efficiency_train.py'),'--config',str(RUN/'train.yaml')]);state['training_wall_s']=elapsed
 execute('benchmark',['/usr/bin/python','-u',str(ROOT/'scripts/benchmark_dynamic_efficiency.py'),'--config',str(RUN/'train.yaml')],dict(EFF_OUT=str(RUN/'comparison'),EFF_CHECKPOINT=str(RUN/'model/trained_model/iteration_25000.pth'),EFF_ROUNDS='5'))
 status(state='complete',finished=time.time())
except Exception:
 status(state='failed',error=traceback.format_exc());raise
