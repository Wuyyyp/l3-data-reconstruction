"""Segment selected vehicles with SAM3 video tracking and projected-box matching.

Arguments: --gpus lists physical GPU IDs (default 4 5 6 7); --overwrite reruns
completed cameras. Inputs and outputs are relative to this experiment directory.
Example: python utils/segment_masks.py --gpus 4 5 6 7
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import cv2
import imageio_ffmpeg
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MULTIVIEW_RUN_ROOT', str(Path(__file__).resolve().parents[1] / 'input'))).resolve()
import register_baseline as base
DATA=base.INPUT
from box_projection import project_track_box
from sam3_video import Sam3VideoTracker, checkpoint as _checkpoint, match_mask_indices as _match_mask_indices

TRACKER = None
GPU = None

def initialize(gpu_queue):
    global TRACKER, GPU
    GPU = int(gpu_queue.get())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    TRACKER = Sam3VideoTracker()
    if os.environ.get('MULTIVIEW_PERF_ROOT'):
        import benchmark_runtime; benchmark_runtime.install('segment')


def projected_boxes(info, item):
    intrinsic, extrinsic = base.camera_for(info, item['camera'])
    boxes = []
    for identity, box, name in zip(info['track_id'], info['gt_boxes'], info['gt_names']):
        if not (name.startswith('VEHICLE_') or name.startswith('BIKE_')):
            continue
        box = np.array(box)
        center = (base.CANONICAL_FROM_SOURCE @ np.r_[box[:3], 1.])[:3]
        projection = project_track_box(length=box[3], width=box[4], height=box[5],
            center_xyz=tuple(center), heading=box[6] + np.pi/2,
            intrinsic=intrinsic, extrinsic=extrinsic, image_width=item['width'], image_height=item['height'])
        if projection.visible_bbox_xywh is not None:
            x,y,w,h = projection.visible_bbox_xywh
            if w*h >= 25:
                boxes.append((str(identity), (x,y,w,h)))
    return boxes


def video_writer(path, size, log):
    import subprocess
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-f', 'rawvideo',
        '-pix_fmt', 'gray', '-s', f'{size[0]}x{size[1]}', '-r', '10', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-crf', '18',
        '-threads', '4', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE,
        stdout=log, stderr=log)


def run_camera(payload):
    item, overwrite = payload
    name, width, height = item['camera'], item['width'], item['height']
    out = DATA / 'mask-videos'
    masks_dir = out / 'masks' / name
    marker = masks_dir / 'assignments.json'
    if marker.exists() and not overwrite:
        print('REUSE', name, flush=True)
        return json.loads(marker.read_text())
    masks_dir.mkdir(parents=True, exist_ok=True)
    infos, _, _ = base.inputs()
    selected = {int(x) for x in (DATA/'obj_selected.txt').read_text().split()}
    config = dict(vehicle_prompt=os.environ.get('SAM3_VEHICLE_PROMPT','car'),confidence_threshold=float(os.environ.get('SAM3_CONFIDENCE_THRESHOLD','.35')),vehicle_match_min_iou=float(os.environ.get('SAM3_BOX_MATCH_MIN_IOU','.025')))
    records = [projected_boxes(info, item) for info in infos]
    settings = dict(prompt=config['vehicle_prompt'], confidence=float(config['confidence_threshold']),
                    min_iou=float(config['vehicle_match_min_iou']))
    frames = []
    print('SAM3 START', name, 'GPU', GPU, flush=True)

    def consume(index, outputs):
        candidates = []
        for sid, score, mask in zip(outputs['out_obj_ids'], outputs['out_probs'], outputs['out_binary_masks']):
            if float(score) >= settings['confidence']:
                pixels = np.asarray(mask, dtype=bool)
                assert pixels.shape == (height, width)
                if pixels.any():
                    candidates.append((int(sid), float(score), pixels))
        mask_inputs=[x[2] for x in candidates]
        if candidates and os.environ.get('MULTIVIEW_PERF_MODE')!='optimized':mask_inputs=np.stack(mask_inputs)
        matches = _match_mask_indices(mask_inputs, records[index], settings['min_iou']) if candidates else {}
        arrays = {}
        assignments = {}
        for identity, candidate_index in matches.items():
            identity = int(identity)
            if identity not in selected:
                continue
            sid, score, pixels = candidates[candidate_index]
            yy,xx = np.nonzero(pixels)
            x0,y0,x1,y1 = int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
            crop = pixels[y0:y1,x0:x1]
            key = f'obj_{identity:03d}'
            arrays[key+'_bbox'] = np.array([x0,y0,x1-x0,y1-y0],dtype=np.int32)
            arrays[key+'_bits'] = np.packbits(crop,axis=None)
            box = dict(records[index])[str(identity)]
            bx,by,bw,bh = box
            intersection = int(pixels[by:by+bh,bx:bx+bw].sum())
            box_iou = intersection / (int(pixels.sum())+bw*bh-intersection)
            assignments[str(identity)] = dict(sam3_id=sid, confidence=score, box_iou=box_iou,
                                             pixels=int(pixels.sum()), bbox_xywh=[x0,y0,x1-x0,y1-y0])
        np.savez_compressed(masks_dir/f'{index:06d}.npz',**arrays)
        frames.append(dict(frame=index, objects=assignments))
        if index % 20 == 0:
            print('SAM3',name,index,'matched',list(assignments),flush=True)

    TRACKER.infer(DATA/item['file'], _checkpoint(), settings['prompt'], len(infos), (height,width), frame_consumer=consume)
    assert len(frames)==101
    frames.sort(key=lambda x:x['frame'])
    log_root = Path(os.environ.get('MULTIVIEW_LOG_ROOT', DATA/'logs'))
    log_root.mkdir(parents=True,exist_ok=True)
    with (log_root/f'multiview-sam3-mask-{name}.log').open('w') as log:
        writer = video_writer(out/f'{name}.mp4',(width,height),log)
        try:
            for index in range(len(infos)):
                union = np.zeros((height,width),np.uint8)
                with np.load(masks_dir/f'{index:06d}.npz') as data:
                    for identity in selected:
                        key=f'obj_{identity:03d}'
                        if key+'_bbox' not in data:continue
                        x,y,w,h=map(int,data[key+'_bbox'])
                        mask=np.unpackbits(data[key+'_bits'],count=w*h).reshape(h,w)
                        union[y:y+h,x:x+w] |= mask*255
                assert np.isin(union,[0,255]).all()
                writer.stdin.write(union.tobytes())
            writer.stdin.close()
            assert writer.wait()==0
        finally:
            if writer.poll() is None:writer.terminate();writer.wait()
    counts=Counter(identity for frame in frames for identity in frame['objects'])
    result=dict(camera=name,settings=settings,frames=frames,matched_frames=dict(counts),
        matching='All vehicle/bicycle tracks participate in per-frame Hungarian assignment; only selected IDs are saved.',
        mask_storage='Native-resolution exact binary instance crops, bbox_xywh and flattened packbits in frame NPZ files.',
        presentation='Union of selected vehicle body masks only; no shadow, dilation or bbox filling.')
    marker.write_text(json.dumps(result,indent=2)+'\n')
    print('SAM3 DONE',name,dict(counts),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[4,5,6,7])
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    _,_,manifest=base.inputs()
    context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in args.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(args.gpus),mp_context=context,
                initializer=initialize,initargs=(queue,)) as pool:
            results=list(pool.map(run_camera,[(item,args.overwrite) for item in manifest['videos']]))
    (DATA/'mask-videos'/'mask_manifest.json').write_text(json.dumps(dict(
        objects=[int(x) for x in (DATA/'obj_selected.txt').read_text().split()],
        frames=101,fps=10,cameras=[dict(camera=x['camera'],matched_frames=x['matched_frames']) for x in results]),indent=2)+'\n')

if __name__=='__main__':main()
