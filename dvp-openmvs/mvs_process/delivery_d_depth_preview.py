#!/usr/bin/env python3
"""Save a fixed 0–100 m color scale preview of untouched DVP depth maps."""
import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-root',type=Path,required=True)
    args=parser.parse_args();root=args.result_root;scene=root/'09_dvp/scene'
    output=root/'09_dvp/APD_before_prior_filter/previews';output.mkdir(parents=True,exist_ok=True)
    count=0
    for meta_path in sorted((scene/'metadata').glob('*.json')):
        meta=json.loads(meta_path.read_text());label=meta['source_label']
        target=output/(label+'.png')
        if target.exists():count+=1;continue
        backup=root/'09_dvp/APD_before_prior_filter'/meta_path.stem/'depths.dmb'
        path=backup if backup.exists() else scene/'APD'/meta_path.stem/'depths.dmb'
        with path.open('rb') as source:
            version,height,width,kind=struct.unpack('<4i',source.read(16))
            if (version,height,width,kind)!=(1,640,960,5):raise ValueError(f'Bad DVP depth header: {path}')
            depth=np.frombuffer(source.read(),'<f4')
        if depth.size!=height*width:raise ValueError(f'Bad DVP depth payload: {path}')
        depth=depth.reshape(height,width)
        x,y=meta['padding_xy'];w,h=meta['content_size'];depth=depth[y:y+h,x:x+w]
        level=np.uint8(np.rint(255*(1-np.clip(np.nan_to_num(depth,nan=0,posinf=100,neginf=0),0,100)/100)))
        color=cv2.applyColorMap(level,cv2.COLORMAP_TURBO);color[~np.isfinite(depth)|(depth<=0)]=0
        if not cv2.imwrite(str(target),color):raise OSError(target)
        count+=1
    if count!=len(list((scene/'metadata').glob('*.json'))):raise ValueError('Missing DVP previews')
    (output/'README.txt').write_text('Color preview: TURBO(1 - clip(camera-Z,0,100)/100); invalid=black. Original float depth is in APD_before_prior_filter.\n')
    print('Saved raw DVP previews',count)


if __name__=='__main__':main()
