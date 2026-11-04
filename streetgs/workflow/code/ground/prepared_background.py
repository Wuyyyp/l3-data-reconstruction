"""Reuse protected LiDAR+DVP background; only subsequent ground partition is new."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from plyfile import PlyData,PlyElement
from common import CFG,O
from mesh_inputs import readable
start=time.time();root=Path(CFG['mesh_source']);manifest=json.loads(readable(root/'input_manifest.json').read_text())
src=readable(root/'model/input_ply/points3D_bkgd.ply');v=PlyData.read(src)['vertex'].data
origin=np.asarray(manifest['world_origin']);xyz=np.column_stack([v[k] for k in 'xyz']).astype('f8')+origin
assert len(v) and np.isfinite(xyz).all()
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('frame_id','<u2'),('point_id','<u4'),('track_id','<i4'),('is_dynamic','u1')])
a=np.zeros(len(v),dtype);a['track_id']=-1;a['point_id']=np.arange(len(v),dtype='u4')
for i,k in enumerate('xyz'):a[k]=xyz[:,i]
for k in ['red','green','blue']:a[k]=v[k]
O.mkdir(exist_ok=False)
PlyData([PlyElement.describe(a,'vertex')]).write(O/'static_no_ground_world_5cm.ply')
PlyData([PlyElement.describe(a[:0],'vertex')]).write(O/'dynamic_frame_world.ply')
(O/'report.json').write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),points=len(v),world_origin=origin.tolist(),initialization_reused=True,upstream_rerun=False,policy='Original protected background including DVP, no vehicle deletion or far reconstruction rerun. Only subsequent ground partition. point_id refers to prepared PLY row.',seconds=time.time()-start),indent=2))
