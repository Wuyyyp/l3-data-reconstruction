import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,math,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
root=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians')
base=root/'output/gsplat_stage67_fixed15k_20261006'
previous=json.loads((root/'output/sample_00380_pose_ba_25000_20261001_172042/pose_comparison/results.json').read_text())
out=base/'old50k_comparison';out.mkdir(exist_ok=True)
names=['old50k','legacy','gsplat','packed','radius1']
reference={n:json.loads((base/'workers'/n/n/'eval_25000.json').read_text()) for n in names[1:]}
lookup={n:{r['image']:r['psnr'] for r in reference[n]['rows']} for n in names[1:]}
def run(row):
    filename=row['image'];stem=Path(filename).stem;cam=int(row['camera'])
    with Image.open(base/'workers/gsplat/gsplat/images_25000'/(stem+'.png')) as f:size=f.size
    with Image.open(Path(previous['gt_dir'])/filename) as f:gt_old=np.array(f.convert('RGB'))
    with Image.open(Path(previous['new_source'])/'images'/filename) as f:gt_new=f.convert('RGB');assert np.array_equal(gt_old,np.array(gt_new)),('GT differs',filename);gt=np.array(gt_new.resize(size,Image.BILINEAR),dtype=np.float32)/255.
    w,h=size;valid=np.ones((h,w),bool)
    if cam==3:valid[:int(250/1280*h),:int(550/1920*w)]=False;valid[:,:int(80/1920*w)]=False
    if cam==4:valid[:int(250/1280*h),int((1920-550)/1920*w):]=False;valid[:,int((1920-80)/1920*w):]=False
    if cam==10:valid[int(1500/2160*h):,:]=False
    result=dict(image=stem,camera=f'{cam:02d}',width=w,height=h,valid_pixels=int(valid.sum()),psnr={},quantization_delta={})
    for n in names:
        path=Path(previous['old_render_dir'])/filename if n=='old50k' else base/'workers'/n/n/'images_25000'/(stem+'.png')
        with Image.open(path) as f:
            im=f.convert('RGB')
            if n=='old50k':im=im.resize(size,Image.BILINEAR)
            else:assert im.size==size
            pred=np.array(im,dtype=np.float32)/255.
        error=(pred-gt)[valid].astype(np.float64)
        mse=float(np.mean(error*error));assert mse>0 and math.isfinite(mse)
        result['psnr'][n]=-10*math.log10(mse)
        if n!='old50k':result['quantization_delta'][n]=result['psnr'][n]-lookup[n][stem]
    return result
start=time.time();rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    with (out/'per_image.jsonl').open('w') as stream:
        for r in pool.map(run,previous['rows']):
            rows.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
            if len(rows)%100==0:print('PROGRESS',len(rows),round(time.time()-start,1),flush=True)
assert len(rows)==707
def avg(rs):return {n:float(np.mean([r['psnr'][n] for r in rs])) for n in names}
summary=avg(rows);delta={n:summary[n]-summary['old50k'] for n in names[1:]}
quant={n:dict(mean=float(np.mean([r['quantization_delta'][n] for r in rows])),max_abs=max(abs(r['quantization_delta'][n]) for r in rows)) for n in names[1:]}
assert all(v['max_abs']<.05 for v in quant.values()),quant
result=dict(summary_psnr=summary,delta_vs_old50k=delta,cameras={c:avg([r for r in rows if r['camera']==c]) for c in sorted({r['camera'] for r in rows})},rows=rows,quantization_vs_float_report=quant,
    image_count=707,gt_pixels_identical=True,elapsed_seconds=time.time()-start,
    protocol='Same current training masks and 1600-pixel-wide dimensions for all 5 variants. PIL bilinear resize of GT and old archived JPEG; new saved PNG at native evaluation size. RGB PSNR arithmetic mean of per-image dB; float64 MSE, no post-hoc color correction. Dynamic pixels retained, production cam03/04 edge and cam10 hood exclusions.',
    caveats=['Old50k has different poses, initialization point cloud, supervision and code pipeline; not a controlled 25k versus 50k training-length ablation.','Archived old images are JPEG; new images are quantized PNG.','Training-view reconstruction only.'])
(out/'results.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
