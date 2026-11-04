from pathlib import Path
import numpy as np,json,time,shutil,hashlib
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement
from common import CFG,S,ROOT,O,G,B
O.mkdir(exist_ok=False);shutil.copy2(__file__,O/'refine.py');t=time.time()
bg=PlyData.read(B/'static_no_ground_world_5cm.ply')['vertex'].data;g=PlyData.read(G/'ground_world_5cm.ply')['vertex'].data
xyz=lambda a:np.column_stack([a[k] for k in 'xyz']).astype('f8')
gp=xyz(g);idx=np.unique(np.floor(gp/.15).astype('int64'),axis=0,return_index=True)[1];gp=gp[idx];tree=cKDTree(gp[:,:2]);p=xyz(bg)
remove=np.zeros(len(p),bool);stats=dict(near_10cm=0,near_15cm=0,near_20cm=0,unsupported_within_2m=0);samples=[]
for start in range(0,len(p),100000):
 q=p[start:start+100000];dist,ix=tree.query(q[:,:2],k=12,workers=CFG['workers']);neigh=gp[ix];mean=neigh.mean(1);v=neigh-mean[:,None,:];cov=np.einsum('nki,nkj->nij',v,v)/12;vals,vecs=np.linalg.eigh(cov);normal=vecs[:,:,0];normal*=np.where(normal[:,2]<0,-1,1)[:,None];res=np.einsum('ni,ni->n',q-mean,normal);rms=np.sqrt(np.maximum(vals[:,0],0));support=(dist[:,-1]<CFG['support_radius'])&(normal[:,2]>CFG['normal_z_min'])&(rms<CFG['plane_rms_max'])&(vals[:,1]>.0025)
 for tol,key in [(.1,'near_10cm'),(.15,'near_15cm'),(.2,'near_20cm')]:stats[key]+=int((support&(abs(res)<tol)).sum())
 stats['unsupported_within_2m']+=int(((dist[:,0]<2)&~support).sum());rm=support&(abs(res)<CFG['ground_tolerance']);remove[start:start+len(q)]=rm
 # Store sampled signed distances for auditable cross sections, not labels of true ground.
 samples.append(np.c_[q[::100],res[::100],support[::100],rm[::100]])
 if start%2000000==0:print('points',start,'removed',int(remove.sum()),flush=True)
 (O/'status.json').write_text(json.dumps(dict(stage='classifying',points_done=start+len(q),total=len(p),seconds=time.time()-t)))
dyn=PlyData.read(B/'dynamic_frame_world.ply')['vertex'].data;assert np.all(dyn['frame_id']==CFG['dynamic_frame'])
def write(name,a):PlyData([PlyElement.describe(a,'vertex')]).write(O/name)
write('static_no_ground_world_5cm.ply',bg[~remove]);write('additional_removed_ground_world.ply',bg[remove]);write('scene_no_ground_dynamic_world.ply',np.concatenate([bg[~remove],dyn]));shutil.copy2(B/'dynamic_frame_world.ply',O/'dynamic_frame_world.ply');np.savez_compressed(O/'classification.npz',removed_input_rows=np.flatnonzero(remove),sampled_diagnostics=np.concatenate(samples))
checks=[dict(file=f.name,points=len(PlyData.read(f)['vertex']),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(O.glob('*.ply'))]
report=dict(input=str(B),pipeline_config=CFG,ground_reference=str(G/'ground_world_5cm.ply'),input_static=len(bg),additional_removed=int(remove.sum()),remaining_static=int((~remove).sum()),dynamic_unchanged=len(dyn),parameter_comparison=stats,method='World ground support + 12-neighbor local PCA; configurable radius, normal, RMS and residual thresholds in pipeline_config; no extrapolation outside supported area.',limitations='No semantic ground truth; low curbs/objects within the configured residual tolerance can be removed; unsupported distant/disconnected ground can remain. Dynamic file preserved byte-for-byte.',outputs=checks,seconds=time.time()-t);(O/'report.json').write_text(json.dumps(report,indent=2))
(O/'status.json').write_text(json.dumps(dict(stage='complete',seconds=time.time()-t)));print(json.dumps(report,indent=2),flush=True)
