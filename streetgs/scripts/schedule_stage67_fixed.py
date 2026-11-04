"""One training process per idle GPU. GPU 1 may have residual memory by user authorization."""
import os,sys,json,time,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/gsplat_stage67_fixed15k_20261006'
OUT.mkdir(exist_ok=True)
lock=(OUT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(ROOT/'scripts'))
from build_stage67_fixed_report import build
NAMES=['legacy','gsplat','packed','radius1'];running={};completed=[];failed=[];pending=NAMES.copy();idle={};started=time.time();bench_started=False;bench_finished=False
assert not (OUT/'scheduler_started.json').exists(),'Existing run: do not duplicate jobs'
(OUT/'scheduler_started.json').write_text(json.dumps(dict(time=started,pid=os.getpid(),names=NAMES)))
def event(kind,**kw):
    value=dict(time=time.time(),event=kind,**kw)
    with (OUT/'scheduler_events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    rawp=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    procs={}
    for line in rawp.splitlines():
        cols=[x.strip() for x in line.split(',')]
        if len(cols)==2:procs.setdefault(cols[0],[]).append(cols[1])
    values=[]
    for line in raw.splitlines():
        i,u,util,used,free=[x.strip() for x in line.split(',')]
        values.append(dict(index=int(i),uuid=u,util=int(util),used=int(used),free=int(free),pids=procs.get(u,[])))
    return values
def start(name,gpu,bench=False):
    worker='gsplat' if bench else name;w=OUT/'workers'/worker;w.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),STAGE67_OUT=str(w),STAGE67_VARIANT=worker,STAGE67_BENCH='1' if bench else '0')
    cmd=['bash','scripts/run_gsplat_eval.sh','scripts/run_stage67_fixed.py','--config','output/sample_00380_pose_ba_25000_20261001_172042/evaluate.yaml']
    log=(w/('benchmark.log' if bench else 'train.log')).open('w')
    p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close()
    running[name]=dict(process=p,gpu=gpu,worker=w,bench=bench,start=time.time())
    (w/('benchmark_launch.json' if bench else 'launch.json')).write_text(json.dumps(dict(pid=p.pid,gpu=gpu,time=time.time(),command=cmd),indent=2))
    event('started',name=name,gpu=gpu,pid=p.pid)
try:
    while True:
        for name,v in list(running.items()):
            code=v['process'].poll()
            if code is None:continue
            s=read(v['worker']/'status.json',{})
            expected='benchmark_complete' if v['bench'] else 'complete'
            if code==0 and s.get('state')==expected:
                if v['bench']:bench_finished=True
                else:completed.append(name)
                event('completed',name=name,wall_seconds=time.time()-v['start'])
            else:
                failed.append(name);event('failed',name=name,exit_code=code,worker_state=s)
            del running[name]
        gpus=snapshot();busy={v['gpu'] for v in running.values()}
        for g in gpus:
            i=g['index'];ok=i==1 and i not in busy and not running and g['util']==0 and not g['pids'] and g['free']>30000
            idle[i]=idle.get(i,0)+1 if ok else 0
        # Require three consecutive 30-second samples. Recheck immediately before dispatch.
        available=[g['index'] for g in gpus if idle[g['index']]>=3]
        for gpu in available:
            fresh=next(g for g in snapshot() if g['index']==gpu)
            if fresh['util'] or fresh['pids']:idle[gpu]=0;continue
            if pending:
                name=pending.pop(0);start(name,gpu);idle[gpu]=0
            elif not running and not bench_started and 'gsplat' in completed:
                start('benchmark',gpu,bench=True);bench_started=True;idle[gpu]=0
        state='training' if running else 'waiting_for_idle_gpu'
        if not pending and not running and (bench_finished or 'benchmark' in failed or 'gsplat' in failed):state='complete' if not failed else 'failed'
        worker_states={n:read(OUT/'workers'/n/'status.json',{}) for n in NAMES}
        payload=dict(state=state,updated=time.time(),started=started,pending=pending,running={n:dict(pid=v['process'].pid,gpu=v['gpu']) for n,v in running.items()},completed=completed,failed=failed,gpus=gpus,workers=worker_states,benchmark_complete=bench_finished)
        tmp=OUT/'status.json.tmp';tmp.write_text(json.dumps(payload,indent=2));tmp.replace(OUT/'status.json')
        build()
        if state in ['complete','failed']:break
        time.sleep(30)
except Exception:
    event('scheduler_failed',error=traceback.format_exc())
    # Do not kill successfully running workers on a reporting/scheduler error.
    (OUT/'scheduler_error.txt').write_text(traceback.format_exc());raise
