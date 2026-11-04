"""Raw MoGe predictions shared by the CPU BA sidecar and calibrated DVP."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('DVP_RUNTIME',REPO/'runtime'))


def cache_path(folder,rgb,weights,level,steps):
    stat=weights.stat()
    version=(str(weights.resolve()),stat.st_size,stat.st_mtime_ns,level,steps,rgb.shape,'moge3-fp16-v1')
    key=hashlib.sha256(repr(version).encode())
    key.update(rgb)
    return folder/(key.hexdigest()+'.npz')


def raw_prediction(model,rgb,weights,level,steps,folder,produce=False):
    path=cache_path(folder,rgb,weights,level,steps) if folder and folder.is_dir() else None
    if path is not None and path.exists():
        with np.load(path,allow_pickle=False) as saved:
            depth,mask=saved['depth'],saved['mask']
        if depth.ndim!=2 or mask.shape!=depth.shape or depth.dtype!=np.float32 or mask.dtype!=bool:
            raise ValueError('Invalid raw MoGe cache '+str(path))
        return model,depth,mask,True
    import torch
    if model is None:
        from moge.model import import_model_class_by_version
        model=import_model_class_by_version('v3').from_pretrained(str(weights)).cuda().eval()
    tensor=torch.from_numpy(rgb).cuda().permute(2,0,1).float().div_(255.0)
    output=model.infer(tensor,resolution_level=level,refine_steps=steps,use_fp16=True)
    depth=output['depth'].detach().cpu().numpy().astype(np.float32)
    mask=output['mask'].detach().cpu().numpy().astype(bool)
    if produce:
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.part',delete=False) as f:
            np.savez(f,depth=depth,mask=mask)
        os.replace(f.name,path)
    return model,depth,mask,False


def precompute(work):
    import torch
    sys.path.insert(0,str(REPO/'mvs_process'))
    from prepare_dvp_scene import geometry
    if torch.cuda.mem_get_info()[0]<38*1024**3:
        print('Yielding GPU headroom',flush=True);return
    torch.set_num_threads(2);cv2.setNumThreads(1)
    folder=work/'.moge_overlap_cache';folder.mkdir(exist_ok=True)
    weights=Path(os.environ.get('DVP_MOGE_WEIGHTS',RUNTIME/'weights/moge-3-vitl/model.pt'));model=None
    for i,row in enumerate(json.loads((work/'pose/input_audit.json').read_text())['mapping']):
        if torch.cuda.mem_get_info()[0]<18*1024**3 or shutil.disk_usage(work).free<500*1024**3:return
        bgr=cv2.imread(row['image']);h,w=bgr.shape[:2]
        _,cw,ch,_,_=geometry(w,h)
        rgb=cv2.cvtColor(cv2.resize(bgr,(cw,ch),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        model,_,_,_=raw_prediction(model,rgb,weights,9,3,folder,produce=True)
        if i%25==0:print('READY MoGe views',i+1,flush=True)


if __name__=='__main__':precompute(Path(sys.argv[1]))
