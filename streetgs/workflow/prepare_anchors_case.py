from pathlib import Path
import json,numpy as np,time,hashlib,sys
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from plyfile import PlyData
O=Path(sys.argv[1]);c=json.loads((O/'config.draft.json').read_text());S=Path(c['source_path']).parent
actors=json.loads((S/'actors.json').read_text());(O/'anchors').mkdir(exist_ok=True)
rows=[]
for a in actors:
 tid=a['id'];p=Path(c['model']['ground']['actor_ply_dir'])/('points3D_obj_%03d.ply'%tid);d=PlyData.read(str(p))['vertex'];xyz=np.column_stack([d[k] for k in 'xyz']).astype('float32');assert np.isfinite(xyz).all() and len(xyz)>0
 if len(xyz)>=3:
  _,ix=cKDTree(xyz).query(xyz,k=min(12,len(xyz)),workers=4);near=xyz[ix];v=near-near.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',v,v);val,vec=np.linalg.eigh(cov);normal=vec[:,:,0];normal*=np.where((normal*xyz).sum(1)<0,-1,1)[:,None];t=vec[:,:,2];b=np.cross(normal,t);basis=np.stack([t,b,normal],axis=2)
 else:
  basis=np.tile(np.eye(3,dtype='float32'),(len(xyz),1,1));normal=basis[:,:,2]
 assert np.allclose(np.linalg.det(basis),1,atol=1e-4)
 q=Rotation.from_matrix(basis).as_quat()[:,[3,0,1,2]].astype('float32');np.savez_compressed(O/'anchors'/('obj_%03d.npz'%tid),xyz=xyz,normal=normal.astype('float32'),rotation=q,basis=basis.astype('float32'))
 rows.append(dict(id=tid,source=a.get('source'),points=len(xyz),sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),normal_method='PCA min(12,N)' if len(xyz)>=3 else 'degenerate identity fallback'))
 print('ANCHOR',rows[-1],flush=True)

(O/'anchors_complete.json').write_text(json.dumps(rows,indent=2))
