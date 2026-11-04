"""Freeze a fresh StreetGS initialization for a future paired short training.

No render, backward, optimizer.step, densification or training loop is executed.
"""
import sys, os, importlib, random, json, hashlib, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
for name in ['multiarray','numeric','umath','_multiarray_umath']:importlib.import_module('numpy.core.'+name)
for name,module in list(sys.modules.items()):
    if name=='numpy.core' or name.startswith('numpy.core.'):
        sys.modules[name.replace('numpy.core','numpy._core',1)]=module
from lib.config import cfg
from lib.utils.general_utils import safe_state
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.scene import Scene

original=Path(cfg.model_path)
out=ROOT/'output/gsplat_stage123/initialization_baseline'
out.mkdir(parents=True,exist_ok=True)
snapshot=out/'initial_state.pth'
if snapshot.exists():raise FileExistsError('Preserving existing initialization: '+str(snapshot))
cfg.mode='train';cfg.render.backend='legacy';cfg.model_path=str(out)
cfg.data.load_pcd_from=False;cfg.data.regenerate_pcd=False
cfg.to_cuda=False;cfg.data_device='cpu'
shutil.copytree(original/'input_ply',out/'input_ply',dirs_exist_ok=True)
safe_state(False)
with torch.no_grad():
    dataset=Dataset();model=StreetGaussianModel(dataset.scene_info.metadata);scene=Scene(model,dataset)
    model.training_setup()
    state=model.save_state_dict(is_final=False)
    state['iter']=0
    torch.save(state,snapshot)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    torch.save(rng,out/'rng_state.pth')
    names=[c.image_name for c in scene.getTrainCameras()]
    sampler=random.Random();sampler.setstate(rng['python']);stack=[];sequence=[]
    for _ in range(1000):
        if not stack:stack=names.copy()
        sequence.append(stack.pop(sampler.randint(0,len(stack)-1)))
    (out/'camera_sequence_1000.json').write_text(json.dumps(sequence,indent=2))
    (out/'config.yaml').write_text(cfg.dump())
    info=dict(seed=0,optimizer_steps=0,training_iterations_executed=0,
        initial_state_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        camera_count=len(names),camera_sequence_steps=len(sequence),
        model_point_counts={name:int(getattr(model,name).get_xyz.shape[0]) for name in model.model_name_id},
        initial_ply_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'input_ply').glob('*.ply')},
        use='Future paired training must restore this exact state + RNG and replay camera_sequence_1000.json; no paired training has been run.')
    (out/'manifest.json').write_text(json.dumps(info,indent=2))
    print('INITIALIZATION_BASELINE_COMPLETE',json.dumps(info),flush=True)
