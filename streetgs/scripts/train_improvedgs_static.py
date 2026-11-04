"""Repository entry point for static ImprovedGS + gsplat + bilateral grid + sky."""
import argparse, fcntl, json, math, os, subprocess, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / 'lib/trainers/improvedgs_static'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO/'configs/experiments_m18/improvedgs_static_sky.json'))
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--stage', choices=['smoke','train','evaluate'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--eval-limit', type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config.get('actors') or config.get('pose_optimization'):
        raise ValueError('This trainer supports static fixed-BA scenes only')
    if args.gpu <= 0:
        raise ValueError('GPU 0 is excluded')
    out = Path(config['output']).resolve()
    if not out.is_relative_to(REPO/'output'):
        raise ValueError('Outputs must stay under this repository/output')
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out/'integration.lock','a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Only launch on an idle nonzero GPU; never stop an existing process.
    for check in range(2):
        result = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'], text=True)
        utilization, free = map(int,result.strip().split(','))
        if utilization != 0 or free < 20000:
            raise RuntimeError('GPU is busy or has less than 20 GB free')
        if check == 0: time.sleep(10)
    cache = REPO/'cache/improvedgs_static'
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),STREET_IMPROVEDGS_CONFIG=str(config_path),
        PYTHONPATH=config['dependencies'],PYTHONDONTWRITEBYTECODE='1',PWD=str(REPO),
        CUDA_HOME='/usr/local/cuda',TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),
        TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.0',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    base = json.loads((ENGINE/'base_args.json').read_text())
    base[base.index('-s')+1] = config['data_cache']
    base[base.index('-m')+1] = str(out/'full')
    def run(label, script, argv, extra=None):
        start=time.time()
        with (out/(label+'.log')).open('a') as log:
            child=subprocess.Popen([config['python'],str(ENGINE/script)]+argv,cwd=REPO,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT)
            (out/'integration_status.json').write_text(json.dumps(dict(state=label,pid=child.pid,started=start,gpu=args.gpu)))
            rc=child.wait()
        if rc:
            (out/'integration_status.json').write_text(json.dumps(dict(state='failed',stage=label,exit_code=rc)))
            raise RuntimeError(label+' failed; inspect '+str(out/(label+'.log')))
        return time.time()-start
    if args.stage == 'smoke':
        smoke=base.copy();smoke[smoke.index('-m')+1]=str(out/'smoke')
        smoke += ['--densify_from_iter','1','--densification_interval','2','--opacity_reset_interval','2','--rap_prune_offset','2','--mu_start_iter','2','--mu_interval','2','--mu_second_start_iter','5','--mu_second_interval','3','--budget','2909257','--budget_warmup_until_offset','25000']
        for stop in [3,6]:
            if (out/('smoke/checkpoint_%06d.pth'%stop)).exists(): continue
            extra=dict(STOP_AFTER=str(stop),CAMERA_LIMIT='21',SMOKE='1')
            if stop==6: extra['RESUME']=str(out/'smoke/checkpoint_000003.pth')
            run('smoke' if stop==3 else 'smoke_resume','train_aligned.py',smoke,extra)
        rows=[json.loads(x) for x in (out/'smoke/curve.jsonl').read_text().splitlines()]
        assert [x['iteration'] for x in rows]==list(range(1,7))
        assert all(math.isfinite(x['loss']) and x['actors']==0 for x in rows)
        assert len(set(x['points'] for x in rows))>1
        assert rows[-1]['rap_reset_count']>0
        if config.get('sky'): assert len(set(x['sky_points'] for x in rows))>1
        assert json.loads((out/'smoke/resume_validation.json').read_text())['pending_gradients']
        run('smoke_eval','evaluate_aligned.py',[str(out/'smoke/checkpoint_000006.pth'),str(out/'smoke_evaluation')],dict(CAMERA_LIMIT='21',EVAL_LIMIT='2'))
        (out/'smoke_validation.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
    elif args.stage == 'train':
        if not (out/'smoke_validation.json').exists(): raise RuntimeError('Run --stage smoke first')
        if (out/'full/checkpoint_025000.pth').exists(): raise RuntimeError('25k already completed; use --stage evaluate')
        extra={}
        if args.resume: extra['RESUME']=str(Path(args.resume).resolve())
        wall=run('train','train_aligned.py',base,extra)
        (out/'wall_seconds.json').write_text(json.dumps(dict(seconds=wall)))
        rows=[json.loads(x) for x in (out/'full/curve.jsonl').read_text().splitlines()]
        assert [x['iteration'] for x in rows]==list(range(1,25001))
        assert all(math.isfinite(x['loss']) and x['actors']==0 and x['points']<=4000000 and x['sky_points']<=100000 for x in rows)
        assert len(set((x['points'],x['sky_points']) for x in rows if x['iteration']>=15000))==1
        (out/'training_validation.json').write_text(json.dumps(dict(passed=True,steps=25000,points=rows[-1]['points'],sky_points=rows[-1]['sky_points'])))
        run('evaluate','evaluate_aligned.py',[str(out/'full/checkpoint_025000.pth'),str(out/'evaluation')])
    else:
        ck=Path(args.checkpoint).resolve() if args.checkpoint else out/'full/checkpoint_025000.pth'
        extra={'EVAL_LIMIT':str(args.eval_limit)} if args.eval_limit else {}
        run('evaluate','evaluate_aligned.py',[str(ck),str(out/'evaluation')],extra)
    (out/'integration_status.json').write_text(json.dumps(dict(state=args.stage+'_complete',gpu=args.gpu,completed=time.time())))
    print(args.stage+' complete: '+str(out),flush=True)

if __name__=='__main__': main()
