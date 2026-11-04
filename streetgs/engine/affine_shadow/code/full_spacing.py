import numpy as np,torch,json,time
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
def serialized_neighbors(tree, queries, bad, prepared, reference, source_origin, train_origin):
    if len(prepared) != len(queries) or 'point_id' not in prepared.dtype.names:
        raise ValueError('Coordinate mismatch without prepared point IDs')
    ids = prepared['point_id'][bad]
    if ids.dtype.kind not in 'ui' or np.any(ids < 0) or np.any(ids >= len(reference)):
        raise ValueError('Invalid prepared point IDs')
    source = np.column_stack([reference[k][ids] for k in 'xyz']).astype('float64')
    world = source + np.asarray(source_origin)
    # Replay the two float32 PLY writes in prepared_background.py and ground/prepare.py.
    serialized = world.astype('float32')
    for axis, origin in enumerate(np.asarray(train_origin)):
        serialized[:, axis] -= origin
    if not np.array_equal(serialized, queries[bad]):
        raise ValueError('Coordinate mismatch is not explained by PLY serialization')
    exact = world - np.asarray(train_origin)
    return tree.query(exact, k=4, workers=4)[0]

@torch.no_grad()
def init_full_spacing(bg,out):
 t=time.time();import os
 root=Path(os.environ['FULL_BACKGROUND_SOURCE'])
 m=json.loads((root/'input_manifest.json').read_text());v=PlyData.read(root/'model/input_ply/points3D_bkgd.ply')['vertex'].data
 from lib.config import cfg
 train_manifest=json.loads((Path(cfg.source_path).parent/'input_manifest.json').read_text())
 xyz=np.column_stack([v[k] for k in 'xyz']).astype('float64')+np.asarray(m['world_origin'])-np.asarray(train_manifest['world_origin'])
 q=bg.get_xyz.detach().cpu().numpy().astype('float64');tree=cKDTree(xyz);d,idx=tree.query(q,k=4,workers=4)
 raw_max=float(d[:,0].max());bad=np.flatnonzero(d[:,0]>=.002)
 if len(bad):
  prepared=PlyData.read(out/'input_ply/points3D_bkgd.ply')['vertex'].data
  d[bad]=serialized_neighbors(tree,q,bad,prepared,v,m['world_origin'],train_manifest['world_origin'])
 assert np.max(d[:,0])<.002,('coordinate mismatch',np.max(d[:,0]))
 scale=np.sqrt(np.maximum(np.mean(d[:,1:]**2,axis=1),1e-7)).astype('float32');old=bg.get_scaling[:,0].detach().cpu().numpy()
 bg._scaling.copy_(torch.from_numpy(np.log(scale)).to(bg._scaling)[:,None].expand_as(bg._scaling))
 (out/'full_spacing_audit.json').write_text(json.dumps(dict(full_points=len(xyz),background_points=len(q),serialized_rows=len(bad),unadjusted_match_max=raw_max,nearest_match_max=float(d[:,0].max()),old_scale_quantiles=np.quantile(old,[.5,.9,.99]).tolist(),new_scale_quantiles=np.quantile(scale,[.5,.9,.99]).tolist(),seconds=time.time()-t),indent=2))
 print('FULL_SPACING_COMPLETE',len(q),flush=True)
