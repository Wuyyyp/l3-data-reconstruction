"""Build native StreetGS initial point clouds using the paired optimized cameras."""
from pathlib import Path
import json,os,sys
ROOT=Path(sys.argv[1]);REPO=Path(os.environ['STREETGS_ROOT'])
os.chdir(REPO);os.environ['PWD']=str(REPO);sys.path.insert(0,str(REPO));sys.argv=['prepare','--config',str(ROOT/'native_config.yaml')]
import numpy as np
from lib.config import cfg
from lib.utils.m18_utils import generate_dataparser_outputs
from plyfile import PlyData
output=generate_dataparser_outputs(str(ROOT/'geometry/converted'),selected_frames=[0,100],build_pointcloud=True,cameras=[0,1,2,3,4,9,10])
counts={p.stem.removeprefix('points3D_'):len(PlyData.read(p)['vertex']) for p in (ROOT/'geometry/model/input_ply').glob('points3D_*.ply')}
expected={'obj_%03d'%int(t) for t in output['obj_info']}
assert expected.issubset(counts),expected-set(counts)
assert all(counts[k]>0 for k in expected)
manifest=json.loads((ROOT/'geometry/input_manifest.json').read_text())
egos=np.stack([np.loadtxt(ROOT/'geometry/converted/ego_pose'/f'{f:06d}.txt') for f in range(101)])
origin=egos[:,:3,3].mean(0)
for name,T in zip(output['image_filenames'],output['c2ws']):
 v=next(v for v in manifest['views'] if v['name']==Path(name).stem)
 expected_T=np.array(v['c2w']);expected_T[:3,3]-=origin
 np.testing.assert_allclose(T,expected_T,atol=1e-7)
manifest.update(world_origin=origin.tolist(),initial_points=counts['bkgd'])
(ROOT/'geometry/input_manifest.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'geometry/reader_validation.json').write_text(json.dumps(dict(status='PASS',point_counts=counts,images=707,world_origin=origin.tolist()),indent=2))
print('NATIVE_INPUTS_READY',counts,flush=True)
