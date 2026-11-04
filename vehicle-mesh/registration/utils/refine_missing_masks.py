"""Recover SAM3 video misses using cropped vehicle text and geometric box prompts.

Arguments: --gpus lists physical GPU IDs (default 4 5 6 7); --overwrite repeats
recovery for cameras with a completed recovery marker. Existing video matches stay.
Example: python utils/refine_missing_masks.py --gpus 4 5 6 7
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml
import register_baseline as base
from segment_masks import projected_boxes,video_writer
ROOT=Path(__file__).resolve().parents[1]
DATA=base.INPUT
PROCESSOR=None
GPU=None


def initialize(queue):
    global PROCESSOR,GPU
    GPU=int(queue.get());os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if os.environ.get('MULTIVIEW_PERF_MODE')!='optimized':load_processor()
    if os.environ.get('MULTIVIEW_PERF_ROOT'):
        import benchmark_runtime; benchmark_runtime.install('recover')


def load_processor():
    global PROCESSOR
    if PROCESSOR is not None:return
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3_video import checkpoint as _checkpoint
    model=build_sam3_image_model(checkpoint_path=str(_checkpoint()),load_from_HF=False)
    PROCESSOR=Sam3Processor(model,confidence_threshold=.35)


def run_camera(payload):
    item,overwrite=payload;name=item['camera'];directory=DATA/'mask-videos'/'masks'/name
    marker=directory/'assignments.json'
    assignments=json.loads(marker.read_text())
    if assignments.get('recovery_complete') and not overwrite:return assignments
    infos,_,_=base.inputs();selected={int(x) for x in (DATA/'obj_selected.txt').read_text().split()}
    cap=cv2.VideoCapture(str(DATA/item['file']));count=0
    import torch
    for fi,info in enumerate(infos):
        ok,frame=cap.read();assert ok
        boxes=dict(projected_boxes(info,item))
        missing=[identity for identity in selected if str(identity) in boxes and str(identity) not in assignments['frames'][fi]['objects']]
        if not missing:continue
        with np.load(directory/f'{fi:06d}.npz') as old:arrays={key:old[key] for key in old.files}
        for identity in missing:
            bx,by,bw,bh=boxes[str(identity)]
            if bw*bh<64 or min(bw,bh)<5:continue
            margin=max(20,round(max(bw,bh)*.25))
            x0=max(0,bx-margin);y0=max(0,by-margin);x1=min(item['width'],bx+bw+margin);y1=min(item['height'],by+bh+margin)
            rgb=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
            width,height=rgb.shape[1],rgb.shape[0]
            prompt_box=[(bx+bw/2-x0)/width,(by+bh/2-y0)/height,bw/width,bh/height]
            load_processor()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=PROCESSOR.set_image(Image.fromarray(rgb))
                PROCESSOR.set_text_prompt('vehicle',state)
                state=PROCESSOR.add_geometric_prompt(prompt_box,True,state)
            masks=state['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
            scores=state['scores'].detach().float().cpu().numpy()
            best=None
            for mask,confidence in zip(masks,scores):
                intersection=int(mask[by-y0:by-y0+bh,bx-x0:bx-x0+bw].sum())
                union=int(mask.sum())+bw*bh-intersection
                iou=intersection/union if union else 0.
                if iou<.15 or int(mask.sum())<64:continue
                # Compare to competing projected tracks to avoid borrowing a neighbor.
                competitive=False
                for other,(ox,oy,ow,oh) in boxes.items():
                    if int(other)==identity:continue
                    xa=max(0,ox-x0);ya=max(0,oy-y0);xb=min(width,ox+ow-x0);yb=min(height,oy+oh-y0)
                    inter=int(mask[ya:yb,xa:xb].sum()) if xb>xa and yb>ya else 0
                    denom=int(mask.sum())+ow*oh-inter
                    if denom and inter/denom > iou+.03:competitive=True;break
                if competitive:continue
                if best is None or iou>best[0]:best=(iou,float(confidence),mask)
            if best is None:continue
            iou,confidence,mask=best
            yy,xx=np.nonzero(mask);mx,my=int(xx.min()),int(yy.min());mw,mh=int(xx.max()+1-mx),int(yy.max()+1-my)
            crop=mask[my:my+mh,mx:mx+mw]
            key=f'obj_{identity:03d}'
            arrays[key+'_bbox']=np.array([x0+mx,y0+my,mw,mh],np.int32)
            arrays[key+'_bits']=np.packbits(crop,axis=None)
            assignments['frames'][fi]['objects'][str(identity)]=dict(sam3_id=None,confidence=confidence,
                box_iou=iou,pixels=int(mask.sum()),bbox_xywh=[x0+mx,y0+my,mw,mh],
                source='SAM3 image vehicle + projected box prompt')
            count+=1
        np.savez_compressed(directory/f'{fi:06d}.npz',**arrays)
        if fi%20==0:print('RECOVERY GPU',GPU,name,fi,'recovered',count,flush=True)
    cap.release()
    log_root=Path(os.environ.get('MULTIVIEW_LOG_ROOT',DATA/'logs'))
    if count==0 and os.environ.get('MULTIVIEW_PERF_MODE')=='optimized':
        assignments['recovery_complete']=True;assignments['recovered_masks']=0
        marker.write_text(json.dumps(assignments,indent=2)+'\n')
        return assignments
    with (log_root/f'multiview-sam3-mask-{name}.log').open('a') as log:
        writer=video_writer(DATA/'mask-videos'/f'{name}.mp4',(item['width'],item['height']),log)
        for fi in range(101):
            union=np.zeros((item['height'],item['width']),np.uint8)
            with np.load(directory/f'{fi:06d}.npz') as data:
                for identity in selected:
                    key=f'obj_{identity:03d}'
                    if key+'_bbox' not in data:continue
                    x,y,w,h=map(int,data[key+'_bbox']);crop=np.unpackbits(data[key+'_bits'],count=w*h).reshape(h,w)
                    union[y:y+h,x:x+w] |= crop*255
            writer.stdin.write(union.tobytes())
        writer.stdin.close();assert writer.wait()==0
    assignments['matched_frames']=dict(Counter(key for f in assignments['frames'] for key in f['objects']))
    assignments['recovery_complete']=True;assignments['recovered_masks']=count
    marker.write_text(json.dumps(assignments,indent=2)+'\n')
    print('RECOVERY DONE',name,assignments['matched_frames'],flush=True)
    return assignments


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',type=int,nargs='+',default=[4,5,6,7]);p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    _,_,manifest=base.inputs();context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in args.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
            results=list(pool.map(run_camera,[(item,args.overwrite) for item in manifest['videos']]))
    (DATA/'mask-videos'/'mask_manifest.json').write_text(json.dumps(dict(objects=list(base.asset_rows()),frames=101,fps=10,
        cameras=[dict(camera=r['camera'],matched_frames=r['matched_frames'],recovered_masks=r['recovered_masks']) for r in results]),indent=2)+'\n')

if __name__=='__main__':main()
