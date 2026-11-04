from pathlib import Path
import os,json,subprocess,sys,time,numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
R=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');H=R.parent/'HUGSIM';O=R/'output/ground_ablation_v2_20261025';O.mkdir(exist_ok=True)
S=Path('/mnt/nuplan/l3data-reconstruction-bingxing/tem-test/streetGS-l3-car-mesh_denoised_v3/sample_01020_clip_M18-2_07_20260111130529_DF_f0_100_right/mesh_vehicle_depth_far_lidar')
origin=np.array(json.loads((S/'input_manifest.json').read_text())['world_origin'])
shared=O/'prepared';shared.mkdir(exist_ok=True)
def read(p):return PlyData.read(p)['vertex'].data
def xyz(a):return np.column_stack([a[k] for k in 'xyz'])
def save(p,a):PlyData([PlyElement.describe(a,'vertex')]).write(p)
if not (shared/'ground.npz').exists():
 g=read(H/'exports/mesh_car_newsource_frame50_v2/ground_background_world.ply');v=read(H/'exports/mesh_car_newsource_frame50_v3_roadclean/frame_000050_scene_roadclean_world.ply');v=v[v['source_kind']==0];template=read(S/'model/input_ply/points3D_bkgd.ply');base=np.zeros(len(v),dtype=template.dtype)
 for k in ['x','y','z','red','green','blue']:base[k]=v[k]
 for j,k in enumerate('xyz'):base[k]-=origin[j]
 save(shared/'nonground.ply',base)
 x=xyz(g).astype('f8')-origin;tree=cKDTree(x);normals=[]
 for st in range(0,len(x),20000):
  dist,ix=tree.query(x[st:st+20000],k=16,workers=4);q=x[ix];q=q-q.mean(1,keepdims=True);_,vec=np.linalg.eigh(q.transpose(0,2,1)@q);n=vec[:,:,0];n[n[:,2]<0]*=-1;n[(dist[:,-1]>2)|(n[:,2]<.8)]=[0,0,1];normals.append(n)
 n=np.concatenate(normals);u=np.cross(np.tile([0,1,0],(len(n),1)),n);u/=np.linalg.norm(u,axis=1,keepdims=True);vaxis=np.cross(n,u);rot=Rotation.from_matrix(np.stack([u,vaxis,n],axis=2)).as_quat()[:,[3,0,1,2]]
 np.savez(shared/'ground.npz',xyz=x.astype('f4'),normals=n.astype('f4'),rotation=rot.astype('f4'),rgb=np.column_stack([g[k] for k in ['red','green','blue']]).astype('f4')/255)
print('PREPARED',flush=True)
engine=R/'lib/trainers/improvedgs_dynamic';py=R.parent/'Improved-GS/.venv/bin/python';base_args=json.loads((R/'lib/trainers/improvedgs_static/base_args.json').read_text());seedcfg=json.loads((R/'configs/experiments_m18/dynamic_async_improvedgs.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(R/'deps/improvedgs_dynamic')+':'+str(R.parent/'gsplat_shape_recovery_20261013/deps'),CUDA_HOME='/usr/local/cuda',TORCH_EXTENSIONS_DIR=str(R/'cache/improvedgs_static/torch_extensions'),TMPDIR=str(R/'cache/improvedgs_static'),XDG_CACHE_HOME=str(R/'cache/improvedgs_static'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='4',GROUND_INIT=str(shared/'ground.npz'),GROUND_MASKS=str(H/'exports/v2_ground_mask_strict/ground_masks'))
def run(mode,phase,steps,resume=None):
 out=O/mode/phase;out.mkdir(parents=True,exist_ok=True);inp=out/'input_ply';inp.mkdir(exist_ok=True)
 for p in (S/'model/input_ply').iterdir():
  q=inp/p.name
  if not q.exists():q.symlink_to(shared/'nonground.ply' if p.name=='points3D_bkgd.ply' and mode!='baseline' else p.resolve())
 c=json.loads(json.dumps(seedcfg));c.update(source_path=str(S/'converted'),model_path=str(out),point_cloud_dir=str(out/'point_cloud'),record_dir=str(out/'record'),trained_model_dir=str(out/'trained_model'),workspace=str(R));c['data'].update(type='PreparedGround',regenerate_pcd=False,load_pcd_from=False,streaming_load=False);c['model']['nsg']['opt_track']=False;c['model']['gaussian']['flip_prob']=0.;c['optim']['merge_actor_adam']=True
 config=out/'config.yaml';config.write_text(json.dumps(c,indent=2));args=base_args.copy()
 for flag,value in [('--use_bilateral_grid','false'),('-s',str(S/'converted')),('-m',str(out)),('--budget','4000000' if mode=='baseline' else str(4000000-691624))]:args[args.index(flag)+1]=value
 e=env.copy();e.update(DYNAMIC_CONFIG=str(config),STOP_AFTER=str(steps),GROUND_MODE=mode,GROUND_WARMUP='5' if phase=='smoke' else '3000')
 if phase=='smoke':e['SMOKE']='1'
 if resume:e['RESUME']=str(resume)
 else:e.pop('RESUME',None)
 tag='resume' if resume else 'train';log=out/(tag+'.log');status=dict(mode=mode,phase=phase,stage=tag,gpu=1,steps=steps,started=time.time());(O/'status.json').write_text(json.dumps(status));print(status,flush=True)
 with log.open('w') as f:rc=subprocess.call([str(py),str(engine/'train_grounded.py')]+args,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT)
 if rc:raise RuntimeError(f'{mode}/{phase} {tag} failed {rc}: {log}')
 return out/f'checkpoint_{steps:06d}.pth'
for phase in ['smoke','full']:
 for mode in ['baseline','fixed','anchored','two_stage']:
  marker=O/mode/phase/'passed.json'
  if marker.exists():continue
  if phase=='smoke':ck=run(mode,phase,10);ck=run(mode,phase,20,ck)
  else:ck=run(mode,phase,25000)
  marker.write_text(json.dumps(dict(status='trained',checkpoint=str(ck))))
(O/'status.json').write_text(json.dumps(dict(stage='training_complete',gpu=1)))


