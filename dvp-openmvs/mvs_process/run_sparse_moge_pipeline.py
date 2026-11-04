from pathlib import Path
import subprocess,os,time,json,shutil,hashlib,traceback
repo=Path('/mnt/zhoukaixuan_workspace/code/DVP-MVS');root=Path('/mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap-ba-dvp-final/2026_04_25_14_32_57_dlp_pilotGtParser/after/dvp_sparse_moge_20260924')
assert (root/'calibration_report.json').exists()
logs=root/'logs';logs.mkdir(exist_ok=True);binary=root/'APD-run-binary';shutil.copy2(repo/'build/APD',binary)
status={'profile':'openmvs','gpu_ids':list(range(1,8)),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'stage':'depth'};processes=[];handles=[];start=time.monotonic()
def save():(root/'experiment_status.json').write_text(json.dumps(status,indent=2))
save()
try:
 for worker,gpu in enumerate(range(1,8)):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),DVP_SKIP_FUSION='1',DVP_ITERATION_PROFILE='openmvs',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=(logs/f'dvp-worker-{worker}.log').open('w');handles.append(log);processes.append(subprocess.Popen([str(binary),str(root/'09_dvp/scene'),'0',str(worker),'7'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(p.poll() is None for p in processes):
  if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('DVP worker failed; inspect worker logs')
  if time.monotonic()-start>4*3600:raise TimeoutError('DVP exceeded four hours')
  time.sleep(5)
 status['depth_seconds']=time.monotonic()-start;assert len(list((root/'09_dvp/scene/APD').glob('*/depths.dmb')))==707
 status['stage']='prior_filter';save();print('DEPTH COMPLETE',status['depth_seconds'],flush=True)
 t=time.monotonic()
 with (logs/'prior-filter.log').open('w') as log:subprocess.run(['python3','-u',str(repo/'mvs_process/filter_sparse_moge_result.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 status['prior_filter_seconds']=time.monotonic()-t;status['stage']='fusion';save();print('PRIOR FILTER COMPLETE',flush=True)
 with (logs/'fusion-runner.log').open('w') as log:subprocess.run(['python3','-u',str(repo/'mvs_process/run_sparse_moge_fusion.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 assert json.loads((root/'masked_fusion_status.json').read_text())['returncode']==0
 status['stage']='complete';status['returncode']=0
except Exception:
 status.update(stage='failed',returncode=1,error=traceback.format_exc());print(status['error'],flush=True)
 for p in processes:
  if p.poll() is None:p.terminate()
finally:
 for f in handles:f.close()
 status['total_seconds']=time.monotonic()-start;save();print(json.dumps(status),flush=True)
