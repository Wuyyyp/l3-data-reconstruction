"""Render the shared optimized PKL poses into seven full-resolution videos.

Arguments: --gpus lists physical EGL GPU IDs (default 4 5 6 7). Reads pkl/
result_interpolation_optimized_first101_mesh_9dof.pkl; outputs register-videos-optimized/.
Example: python utils/render_optimized.py --gpus 4 5 6 7
"""
from __future__ import annotations
import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import register_baseline as base
import numpy as np
DATA=base.INPUT
GPU=None

def initialize(queue):
    global GPU
    GPU=int(queue.get());os.environ['EGL_DEVICE_ID']=str(GPU)
    if os.environ.get('MULTIVIEW_PERF_ROOT'):
        import benchmark_runtime; benchmark_runtime.install('render')


def render(item):
    original_inputs=base.inputs
    payload=pickle.load((DATA/'pkl'/'result_interpolation_optimized_first101_mesh_9dof.pkl').open('rb'))
    def inputs_with_optimized_pose():
        infos,tracks,manifest=original_inputs()
        for fi,info in enumerate(payload['infos']):
            for identity,record in info['mesh_poses_9dof'].items():
                tracks[fi][int(identity)]=np.array(record['canonical_ego_from_unit_mesh'])
        return infos,tracks,manifest
    base.inputs=inputs_with_optimized_pose
    base.RESTORE_VERTICAL_FOREGROUND=True
    base.OUTPUT=DATA/'register-videos-optimized';base.OUTPUT.mkdir(parents=True,exist_ok=True)
    try:return base.render_camera(item)
    finally:base.inputs=original_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpus',type=int,nargs='+',default=[4,5,6,7]);args=parser.parse_args()
    _,_,manifest=base.inputs();context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in args.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
            results=list(pool.map(render,manifest['videos']))
    summary=json.loads((DATA/'pkl'/'optimization_summary.json').read_text())
    (DATA/'register-videos-optimized'/'registration_manifest.json').write_text(json.dumps(dict(
        pose_pkl='pkl/result_interpolation_optimized_first101_mesh_9dof.pkl',
        shared_pose='Every camera uses the exact same object/frame matrix; no independent per-camera adjustment.',
        foreground=('Compensated LiDAR occlusion with 10px sample expansion and 2m tolerance; SAM3 narrow pole-like gaps.' if os.environ.get('MULTIVIEW_SCENE_DEPTH_ROOT') else 'SAM3 narrow pole-like gaps only.'),
        optimization=summary,objects=base.asset_rows(),cameras=results),indent=2)+'\n')
    print('OPTIMIZED VIDEOS COMPLETE',flush=True)

if __name__=='__main__':main()
