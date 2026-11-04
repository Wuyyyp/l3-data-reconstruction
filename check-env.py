#!/usr/bin/env python3
"""Check a checkout's required local resources and imports; does not start reconstruction."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

COMPONENTS = {
    'pose': ('POSE_PYTHON', ['DVP_DELIVERY_COLMAP'],
             ['cv2','torch','transformers','dongfeng_registration','loma']),
    'dvp': ('DVP_PYTHON', ['DVP_FULL_BIN','DVP_FAR_BIN','DVP_DELIVERY_COLMAP','DVP_DELIVERY_VISION_PYTHON','OPENMVS_BIN_DIR','DVP_MOGE_WEIGHTS','DVP_MOGE_ROOT'],
            ['cv2','torch','plyfile','scipy','moge']),
    'mesh': ('MESH_PYTHON', ['SAM3_CHECKPOINT','MESH_ASSET_ROOT'], ['torch','sam3','trimesh','pyrender']),
    'streetgs': ('GS_PYTHON', ['GSPLAT_CUDA_SO','ROAD_SEGFORMER_WEIGHTS'],
                ['torch','open3d','plyfile','gsplat','fused_ssim','pypatchworkpp','simple_knn._C','diff_gaussian_rasterization']),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=COMPONENTS)
    args=parser.parse_args()
    if not os.environ.get('CODE_ROOT'):
        parser.error('Source server-env.sh first.')
    failed=False
    for name in ([args.component] if args.component else COMPONENTS):
        key,resources,modules=COMPONENTS[name]
        errors=[]
        for item in [key,*resources]:
            value=os.environ.get(item)
            if not value or not Path(value).exists():
                errors.append(f'{item}: {value or "(unset)"}')
        for folder,files in [('OPENMVS_BIN_DIR',['InterfaceCOLMAP','DensifyPointCloud']),('ROAD_SEGFORMER_WEIGHTS',['config.json','model.safetensors'])]:
            if folder in resources and os.environ.get(folder):
                for file in files:
                    path=Path(os.environ[folder])/file
                    if not path.is_file():errors.append(str(path))
        if name=='mesh' and os.environ.get('MESH_ASSET_ROOT') and not next(Path(os.environ['MESH_ASSET_ROOT']).rglob('*.glb'),None):
            errors.append('MESH_ASSET_ROOT: no .glb asset found')
        py=os.environ.get(key)
        if py and os.access(py,os.X_OK):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            paths={
                'pose':[os.path.join(env['POSE_ROOT'],'src'),env.get('POSE_LOMA_SRC')],
                'mesh':[],
                'dvp':[env['DVP_MOGE_ROOT']],
                'streetgs':[env.get('PREPARATION_DEPS'),env.get('GROUND_PREPROCESS_DEPS'),env.get('STREETGS_EXTRA_PYTHONPATH'),os.path.join(env['STREETGS_ROOT'],'deps/gsplat')]
            }
            env['PYTHONPATH']=os.pathsep.join(p for p in [*paths[name],env.get('PYTHONPATH')] if p)
            cmd=[py,'-c','import '+', '.join(modules)]
            if name=='mesh':
                env.setdefault('PYOPENGL_PLATFORM','egl')
                if env.get('EGL_LIB_DIR'):
                    env['LD_LIBRARY_PATH']=os.pathsep.join(x for x in [env['EGL_LIB_DIR'],env.get('LD_LIBRARY_PATH')] if x)
            if name=='mesh' and env.get('MESH_ENV_ACTIVATE'):
                cmd=['bash','-c','source "$1"; shift; unset PYTHONPATH; exec "$@"','mesh',env['MESH_ENV_ACTIVATE'],*cmd]
            result=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if result.returncode:errors.append(result.stdout[-1600:].strip() or 'import failed')
        elif py:errors.append(f'{key} is not executable')
        print(f'{name}: '+('OK' if not errors else '\n  '+'\n  '.join(errors)))
        failed |= bool(errors)
    return int(failed)

if __name__=='__main__':
    sys.exit(main())
