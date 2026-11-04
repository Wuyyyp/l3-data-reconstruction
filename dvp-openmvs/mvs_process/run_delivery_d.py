#!/usr/bin/env python3
"""One-command delivered BA clip -> method D -> one PLY per 101-frame window."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
BASE=REPO.parent
RUNTIME=Path(os.environ.get('DVP_RUNTIME',REPO/'runtime'))
PREP=REPO/'mvs_process/delivery_d_prepare.py'
RANGES=REPO/'mvs_process/derive_delivery_d_ranges.py'
CALIBRATE=REPO/'mvs_process/delivery_d_calibrate.py'
FILTER=REPO/'mvs_process/delivery_d_filter.py'
PREVIEW=REPO/'mvs_process/delivery_d_depth_preview.py'
GENERIC=REPO/'mvs_process/run_batch_mvsnet_to_openmvs_ply.sh'
CONVERT=REPO/'mvs_process/convert_dvp_to_openmvs_dmap.py'
MVS=Path(os.environ.get('OPENMVS_BIN_DIR',REPO/'vendor/openMVS/install/bin/OpenMVS'))
PYTHON=Path(os.environ.get('PYTHON_BIN',str(RUNTIME/'venv/bin/python')))
FAR_BIN=Path(os.environ.get('DVP_FAR_BIN',REPO/'build-far-only/APD'))


def call(command, environment, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN', ' '.join(str(x) for x in command), flush=True)
    with log.open('a') as out:
        subprocess.run([str(x) for x in command],cwd=REPO,env=environment,stdout=out,stderr=subprocess.STDOUT,check=True)


def environment(gpu):
    env=dict(os.environ)
    # Only the explicit CLI switch enables far computation in this entry point.
    env.pop('DVP_FAR_MASK_DIR',None)
    native=RUNTIME/'native/usr/lib/x86_64-linux-gnu'
    openmvs=Path(os.environ.get('OPENMVS_RUNTIME',MVS.parent/'lib'))
    colmap=os.environ.get('COLMAP_RUNTIME','/usr')
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),DVP_GPU_IDS=str(gpu),
        DVP_ITERATION_PROFILE='legacy19',DVP_CPU_PIPELINE='1',DVP_DYNAMIC_THRESHOLD_M='1',
        DVP_SKIP_FUSION='1',DVP_FAST_MODE='0',DVP_EXEC_OPT='0',DVP_STATIC_CACHE='0',DVP_INPUT_OVERLAP='0',
        DVP_COLMAP_BA='0',DVP_USE_PRECOMPUTED_RANGES='1',PYTHON_BIN=str(PYTHON),DVP_MOGE_ROOT=os.environ.get('DVP_MOGE_ROOT',str(REPO/'vendor/MoGe')),
        DVP_MOGE_PYTHON=str(PYTHON),DVP_MOGE_WEIGHTS=str(Path(os.environ.get('DVP_MOGE_WEIGHTS',RUNTIME/'weights/moge-3-vitl/model.pt'))),
        DVP_BIN=str(Path(os.environ.get('DVP_FULL_BIN',REPO/'build-cpu-pipeline/APD'))),OPENMVS_BIN_DIR=str(MVS),
        OPENMVS_RUNTIME=str(openmvs),COLMAP_RUNTIME=colmap,OPENMVS_THREADS='8',
        QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        TRITON_CACHE_DIR=str(RUNTIME/'cache/triton'),TORCH_HOME=str(RUNTIME/'cache/torch'),
        XDG_CACHE_HOME=str(RUNTIME/'cache'),
    )
    env['LD_LIBRARY_PATH']=':'.join([str(native),str(openmvs),colmap+'/usr/lib',colmap+'/usr/lib/x86_64-linux-gnu',colmap+'/usr/lib/x86_64-linux-gnu/atlas',colmap+'/lib/x86_64-linux-gnu',env.get('LD_LIBRARY_PATH','')])
    return env


def pair10(scene):
    full=scene/'pair.txt';rows=full.read_text().splitlines();count=int(rows[0])
    if len(rows)!=1+2*count:raise ValueError('Invalid DVP pair.txt')
    output=[rows[0]]
    for i in range(count):
        reference=rows[1+2*i];parts=rows[2+2*i].split();n=int(parts[0])
        if len(parts)!=1+2*n or n<10:raise ValueError(f'Insufficient DVP neighbors for {reference}')
        output.extend([reference,'10 '+' '.join(parts[1:21])])
    shutil.copy2(full,scene/'pair20.txt')
    (scene/'pair10.txt').write_text('\n'.join(output)+'\n')
    shutil.copy2(scene/'pair10.txt',full)


def ply_count(path):
    with path.open('rb') as source:
        for _ in range(100):
            line=source.readline()
            if line.startswith(b'element vertex '):return int(line.split()[-1])
    raise ValueError('No PLY vertex header: '+str(path))


def verify_inputs(saved,clip):
    from hashlib import sha256
    sha=lambda path:sha256(path.read_bytes()).hexdigest()
    model=clip/'colmap/sparse_ba'
    current=dict(csv=sha(clip/'colmap/poses/image_camera_to_world_after_ba.csv'),images=sha(model/'images.bin'),cameras=sha(model/'cameras.bin'),points3D=sha(model/'points3D.bin'),pkl=sha(clip/'result_interpolation_optimized.pkl'))
    if saved['ba_model_sha256']!=current:raise ValueError('Delivery BA inputs changed; use a fresh result root')


def run_window(clip,output,start,end,gpu,check=False,prepare_only=False,far_only=False):
    root=output/clip.name/(f'f{start}_{end}'+('_far50' if far_only else ''));root.mkdir(parents=True,exist_ok=True)
    policy=None
    if far_only:
        from delivery_d_far import POLICY,prepare_masks,trajectory,crop_ply
        policy=POLICY
        if not FAR_BIN.is_file():raise FileNotFoundError('Build the far-only binary first: '+str(FAR_BIN))
    config=root/'audit/run_config.json'
    if config.exists():
        saved_mode=json.loads(config.read_text())
        if bool(saved_mode.get('far_only',False))!=far_only or (far_only and saved_mode.get('far_policy')!=policy):
            raise ValueError('Result mode/policy differs; use a fresh result directory')
    elif far_only and any((root/p).exists() for p in ['audit/dvp_depths.complete','prior_filter_report.json','09_dvp/scene/APD','10_openmvs_fusion']):
        raise ValueError('Unidentified existing reconstruction in far-only directory; use a fresh result directory')
    env=environment(gpu);log=root/'logs/delivery-d.log'
    args=[PYTHON,PREP,'--clip-root',clip,'--frame-start',start,'--frame-end',end,'--result-root',root,'--gpu',gpu]
    if check:
        call(args+['--check'],env,log)
        return str(root)
    final=root/'10_openmvs_fusion/openmvs_filtered_fused.ply'
    if final.is_file() and (ply_count(final)>0 or far_only) and (not far_only or (root/'audit/D.complete').exists()):
        complete=json.loads((root/'audit/D.complete').read_text())
        if complete.get('status')!='PASS' or complete.get('method')!='D':raise ValueError('Existing PLY is not a completed D result')
        if bool(complete.get('far_only',False))!=far_only or (far_only and complete.get('far_policy')!=policy):
            raise ValueError('Completed PLY has a different far-region policy')
        saved=json.loads((root/'prepared/audit/preparation.json').read_text())
        verify_inputs(saved,clip)
        return str(final)
    prepared=root/'prepared';done=prepared/'audit/preparation.json'
    if not done.is_file():call(args,env,log)
    audit=json.loads(done.read_text())
    if audit.get('status')!='PASS' or audit.get('frames')!=[start,end] or audit.get('delivery')!=str(clip):raise ValueError('Prepared input audit mismatch')
    if far_only:verify_inputs(audit,clip)
    (root/'audit').mkdir(exist_ok=True)
    (root/'audit/run_config.json').write_text(json.dumps(dict(method='D',delivery=str(clip),frames=[start,end],gpu=gpu,
        far_only=far_only,far_policy=policy,
        ba_csv='colmap/poses/image_camera_to_world_after_ba.csv',ba_model='colmap/sparse_ba',internal_ba=False,
        dynamic_mask_displacement_m='>1',dvp_profile='legacy19',dvp_sources=10,postfilter_sources=20,
        cpu_pipeline=True,fusion_sources=20,fusion_depth_difference=.015,fusion_min_views=2),indent=2)+'\n')
    if prepare_only:return str(prepared)
    run_root=root/'09_dvp';stage1=root/'09_dvp_stage1'
    calibration=root/'calibration_report.json'
    if not calibration.exists():
        if not stage1.exists():
            bounds=root/'sparse_depth_ranges_openmvs.json'
            if not bounds.exists():
                call([PYTHON,RANGES,'--model',prepared/'07_sparse/triangulated_text',
                      '--full-ba-points',clip/'colmap/sparse_ba/points3D.bin','--output',bounds],env,log)
            call(['bash',GENERIC,'--sample-root',prepared,'--result-root',root,'--gpu-ids',gpu,'--dvp-profile','legacy19','--no-colmap-ba','--to-stage','1','--resume'],env,log)
            run_root.rename(stage1)
        base_ranges=root/'audit/base_sparse_depth_ranges.json'
        if not base_ranges.exists():shutil.copy2(root/'sparse_depth_ranges_openmvs.json',base_ranges)
        call([PYTHON,CALIBRATE,'--source',stage1,'--prepared',prepared,'--result-root',root],env,log)
        pair10(run_root/'scene')
    scene=run_root/'scene';depth_marker=root/'audit/dvp_depths.complete'
    if calibration.exists() and not (scene/'pair10.txt').exists():pair10(scene)
    if far_only and not depth_marker.exists():
        env['DVP_FAR_MASK_DIR']=str(prepare_masks(root,clip,start,end))
    if not depth_marker.exists():
        if (scene/'pair.txt').read_bytes()!=(scene/'pair10.txt').read_bytes():raise ValueError('DVP must use the top 10 neighbors')
        shutil.rmtree(scene/'APD',ignore_errors=True)
        (scene/'APD').mkdir(exist_ok=True)
        call([FAR_BIN if far_only else Path(os.environ.get('DVP_FULL_BIN',REPO/'build-cpu-pipeline/APD')),scene,0,0,1],env,log)
        env.pop('DVP_FAR_MASK_DIR',None)
        expected=int((scene/'pair10.txt').read_text().splitlines()[0]);actual=len(list((scene/'APD').glob('*/depths.dmb')))
        if actual!=expected:raise ValueError(f'Expected {expected} DVP depths, got {actual}')
        depth_marker.parent.mkdir(exist_ok=True);depth_marker.write_text('PASS\n')
    if not (root/'09_dvp/APD_before_prior_filter/previews/README.txt').exists():
        call([PYTHON,PREVIEW,'--result-root',root],env,log)
    if not (root/'prior_filter_report.json').exists():
        shutil.copy2(scene/'pair20.txt',scene/'pair.txt')
        call([PYTHON,FILTER,'--result-root',root],env,log)
    if (scene/'pair.txt').read_bytes()!=(scene/'pair20.txt').read_bytes():raise ValueError('Filtering must use the original 20 neighbors')
    work=root/'10_openmvs_fusion/dmaps';work.mkdir(parents=True,exist_ok=True)
    conversion=work.parent/'conversion_stats.json'
    if not conversion.exists():
        call([PYTHON,CONVERT,'--run',run_root,'--source',prepared/'03_converted',
              '--scene',prepared/'08_openmvs_input/global/scene.mvs',
              '--images-txt',prepared/'07_sparse/triangulated_text/images.txt',
              '--neighbors',prepared/'08_openmvs_input/global/neighbors_diverse_pm20_top20.txt',
              '--output',work,'--depth-ranges-json',root/'sparse_depth_ranges_openmvs.json'],env,log)
    output_mvs=root/'10_openmvs_fusion/openmvs_filtered_fused.mvs'
    call([MVS/'DensifyPointCloud','-w',work,'-i',prepared/'08_openmvs_input/global/scene.mvs','-o',output_mvs,
          '--dense-config-file',REPO/'mvs_process/openmvs_dvp_depth_filter.cfg',
          '--mask-path',root/'06_masks/effective_masks','--ignore-mask-label',255,
          '--view-neighbors-file',prepared/'08_openmvs_input/global/neighbors_diverse_pm20_top20.txt',
          '--cuda-device',-2,'--max-threads',8,'--resolution-level',1,'--max-resolution',960,
          '--min-resolution',640,'--sub-resolution-levels',2,'--number-views',20,
          '--number-views-fuse',2,'--iters',4,'--geometric-iters',0,'--fusion-mode',0,
          '--fusion-filter',2,'--fusion-depth-diff-threshold',0.015,
          '--fusion-reprojection-threshold',2.5,'--postprocess-dmaps',1,
          '--estimate-colors',2,'--estimate-normals',2,'--tower-mode',0,'--estimate-roi',0,
          '--crop-to-roi',0,'--filter-point-cloud',0,'--remove-dmaps',0,'--archive-type',2,'-v',3],env,log)
    points=ply_count(final)
    if points<1 and not far_only:raise ValueError(f'Empty D point cloud: {final}')
    if far_only:
        (root/'far_only').mkdir(exist_ok=True)
        points=crop_ply(final,trajectory(clip,start,end),root/'far_only/crop_report.json')
    (root/'audit/D.complete').write_text(json.dumps(dict(status='PASS',method='D',frames=[start,end],gpu=gpu,points=points,ply=str(final),far_only=far_only,far_policy=policy),indent=2)+'\n')
    return str(final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-root',type=Path,required=True,help='Delivered clip with colmap/sparse_ba and BA CSV')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--gpu-ids',default=os.environ.get('DVP_GPU_IDS','0'))
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--far-only',action='store_true',help='Keep only points >50m in XY from every ego position in each window; default off')
    args=parser.parse_args();clip=args.clip_root.resolve();output=args.output_root.resolve()
    if not clip.is_dir():parser.error(f'Clip missing: {clip}')
    if output==clip or clip in output.parents:parser.error('--output-root must be outside the delivered clip')
    gpus=[int(x) for x in args.gpu_ids.split(',')]
    if not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0:parser.error('Use unique non-negative GPU IDs')
    required=[PYTHON,PREP,RANGES,CALIBRATE,FILTER,PREVIEW,GENERIC,CONVERT,MVS/'InterfaceCOLMAP',MVS/'DensifyPointCloud',Path(os.environ.get('DVP_FULL_BIN',REPO/'build-cpu-pipeline/APD')),Path(os.environ.get('DVP_MOGE_WEIGHTS',RUNTIME/'weights/moge-3-vitl/model.pt')),Path(os.environ['DVP_DELIVERY_COLMAP']),Path(os.environ.get('DVP_DELIVERY_VISION_PYTHON',PYTHON))]
    if args.far_only:required.append(FAR_BIN)
    for path in required:
        if not path.is_file():parser.error(f'Missing D dependency: {path}')
    if args.check:
        call(['bash',GENERIC,'--sample-root',clip,'--to-stage','1','--check'],environment(gpus[0]),output/clip.name/'preflight.log')
    with (clip/'result_interpolation_optimized.pkl').open('rb') as source:
        import pickle
        frames=len(pickle.load(source)['infos'])
    windows=[(start,min(start+100,frames-1)) for start in range(0,frames,101)]
    if args.check or len(gpus)==1:
        results=[run_window(clip,output,start,end,gpus[i%len(gpus)],args.check,args.prepare_only,args.far_only) for i,(start,end) in enumerate(windows)]
    else:
        def worker(gpu,assigned):
            return [run_window(clip,output,start,end,gpu,False,args.prepare_only,args.far_only) for start,end in assigned]
        assignments={gpu:[window for i,window in enumerate(windows) if gpus[i%len(gpus)]==gpu] for gpu in gpus}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu,assignments[gpu]) for gpu in gpus if assignments[gpu]]
            results=[result for future in futures for result in future.result()]
    print(json.dumps(dict(status='PASS',method='D',far_only=args.far_only,clip=str(clip),windows=len(windows),results=results),indent=2))


if __name__=='__main__':main()
