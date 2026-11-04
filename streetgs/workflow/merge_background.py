from pathlib import Path
import sys,json,shutil
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement
root=Path(sys.argv[1]);g=root/'geometry';target=g/'model/input_ply';report=g/'initialization_merge.json'
assert not report.exists(),'Do not append far points twice'
if '--no-far' in sys.argv:
    source=target/'points3D_bkgd.ply'
    shutil.copy2(source,target/'points3D_lidar.ply')
    report.write_text(json.dumps(dict(lidar_background_points=len(PlyData.read(source)['vertex']),far_points=0,far_ply=None)))
    sys.exit(0)
far=Path(json.loads((root/'dvp_result.json').read_text())['ply'])
a=PlyData.read(target/'points3D_bkgd.ply')['vertex'].data;b=PlyData.read(far)['vertex'].data
origin=np.array(json.loads((g/'input_manifest.json').read_text())['world_origin']);xyz=np.column_stack([b[k] for k in 'xyz']).astype('f8')
egos=np.array([np.loadtxt(g/'converted/ego_pose'/f'{i:06d}.txt')[:2,3] for i in range(101)])
minimum=float(cKDTree(egos).query(xyz[:,:2],workers=4)[0].min()) if len(xyz) else None
assert minimum is None or minimum>49.999
extra=np.zeros(len(b),dtype=a.dtype)
for k in extra.dtype.names:
 if k in b.dtype.names:extra[k]=b[k]
for j,k in enumerate('xyz'):extra[k]=xyz[:,j]-origin[j]
shutil.copy2(target/'points3D_bkgd.ply',target/'points3D_lidar.ply')
PlyData([PlyElement.describe(np.concatenate([a,extra]),'vertex')],text=False).write(target/'points3D_bkgd.ply')
report.write_text(json.dumps(dict(lidar_background_points=len(a),far_points=len(b),far_ply=str(far),far_min_trajectory_xy_m=minimum),indent=2))
