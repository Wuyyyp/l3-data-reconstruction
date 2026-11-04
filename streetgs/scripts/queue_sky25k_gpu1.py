import os,json,time,subprocess,fcntl
from pathlib import Path
R=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');O=R/'output/sky_sphere_vs_cubemap_25000_20261007'
lock=(O/'gpu1_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def alive(pid):
 try:
  s=Path('/proc/%d/stat'%pid).read_text().split();return s[2]!='Z'
 except FileNotFoundError:return False
def status(**kw):
 p=O/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated=time.time(),**kw),indent=2));tmp.replace(p)
idle=0
while True:
 # Wait for the existing GPU1 scheduler's entire queue, not a gap between its jobs.
 if alive(421537) or alive(438416):
  idle=0;status(state='waiting_existing_gpu1_queue',predecessor_scheduler=421537);time.sleep(30);continue
 raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
 uid,util,free=[x.strip() for x in raw.strip().split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 busy=any(line.split(',')[0].strip()==uid for line in apps.splitlines())
 idle=idle+1 if not busy and int(util)==0 and int(free)>30000 else 0
 status(state='waiting_gpu1_idle',idle_samples=idle)
 if idle>=3:break
 time.sleep(30)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',STAGE67_OUT=str(O),STAGE67_VARIANT='gsplat',SKY_SMOKE_STEPS='0',SKY_RESUME_STEP='1000')
cmd=['bash','scripts/run_gsplat_eval.sh','scripts/run_sky25k.py','--config','output/sample_00380_pose_ba_25000_20261001_172042/evaluate.yaml']
with (O/'train_gpu1.log').open('a') as log:
 proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(O/'gpu1_launch.json').write_text(json.dumps(dict(pid=proc.pid,gpu=1,resume_step=1000,command=cmd),indent=2));status(state='running_gpu1',pid=proc.pid,resume_step=1000)
rc=proc.wait();status(state='finished' if rc==0 else 'failed',exit_code=rc,pid=proc.pid)
