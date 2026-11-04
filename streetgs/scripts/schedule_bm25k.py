import os,sys,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');OUT=ROOT/'output/bm25k_20261007';OUT.mkdir(exist_ok=True)
lock=(OUT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def status(**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
def event(**kw):
 with (OUT/'scheduler_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**kw))+'\n')
 print(kw,flush=True)
def alive(pid):
 try:os.kill(pid,0);return True
 except OSError:return False
completed=[]
for name in ['baseline','bilateral','mcmc']:
 w=OUT/'workers'/name;w.mkdir(parents=True,exist_ok=True)
 if read(w/'status.json').get('state')=='complete':completed.append(name);continue
 old=read(w/'launch.json')
 if old and alive(old['pid']):
  status(state='watching_existing',variant=name,pid=old['pid'],completed=completed)
  while alive(old['pid']) and read(w/'status.json').get('state') not in ['complete','failed']:time.sleep(30)
  if read(w/'status.json').get('state')=='complete':completed.append(name);continue
  status(state='failed',variant=name,completed=completed);sys.exit(1)
 # Do not automatically overwrite or resume partial experiment data.
 if list((w/name).glob('curve.jsonl')):status(state='needs_resume',variant=name,completed=completed);sys.exit(1)
 idle=0
 while idle<3:
  raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True);util,free=map(int,raw.strip().split(','))
  idle=idle+1 if util==0 and free>25000 else 0
  status(state='waiting_for_gpu1',variant=name,completed=completed,util=util,free_mb=free);time.sleep(5)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',STAGE67_OUT=str(w),STAGE67_VARIANT=name,STAGE67_FINAL='25000',BM_SMOKE='0')
 cmd=['bash','scripts/run_gsplat_eval.sh','scripts/run_bm25k.py','--config','output/sample_00380_pose_ba_25000_20261001_172042/evaluate.yaml']
 log=(w/'train.log').open('a');proc=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();(w/'launch.json').write_text(json.dumps(dict(pid=proc.pid,time=time.time(),cmd=cmd,gpu=1)))
 event(event='started',variant=name,pid=proc.pid)
 while proc.poll() is None:
  status(state='running',variant=name,pid=proc.pid,completed=completed,worker=read(w/'status.json'));time.sleep(30)
 if proc.returncode or read(w/'status.json').get('state')!='complete':
  event(event='failed',variant=name,code=proc.returncode);status(state='failed',variant=name,completed=completed);sys.exit(1)
 completed.append(name);event(event='completed',variant=name)
status(state='building_report',completed=completed)
result=subprocess.run(['/usr/bin/python','scripts/build_bm25k_report.py'],cwd=str(ROOT),stdout=(OUT/'report.log').open('w'),stderr=subprocess.STDOUT)
status(state='complete' if result.returncode==0 else 'report_failed',completed=completed,report_exit=result.returncode)
