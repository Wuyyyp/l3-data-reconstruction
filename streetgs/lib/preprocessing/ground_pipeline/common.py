"""Shared input configuration for CPU preprocessing stages."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parents[3]

def settings():
    return json.loads(Path(os.environ['ROAD_PIPELINE_CONFIG']).read_text())

CFG=settings()
S=Path(CFG['source'])
ROOT=Path(CFG['output'])
O=Path(os.environ['ROAD_STAGE_OUTPUT'])
G=ROOT/'preprocess/extraction'
B=ROOT/'preprocess/separation'
V=ROOT/'preprocess/refined'

def selected_frames(pointcloud):
    frames=sorted(pointcloud)
    if CFG['frames'] is not None:frames=[f for f in frames if f in CFG['frames']]
    if not frames:raise ValueError('No matching LiDAR frames')
    return frames

def mask_path(directory,name):
    directory=Path(directory)
    for ext in ['.png','.jpg','.jpeg']:
        p=directory/(name+ext)
        if p.exists():return p
    raise FileNotFoundError(directory/name)

def ego_path(name):
    root=Path(CFG['ego_mask'])
    for stem in [name,'000000_'+name.split('_')[1]]:
        for ext in ['.png','.jpg','.jpeg']:
            p=root/(stem+ext)
            if p.exists():return p
    raise FileNotFoundError('No ego mask for '+name+' in '+str(root))

def binary_mask(path,size,threshold=0):
    with Image.open(path) as image:
        a=np.asarray(image.convert('L').resize(size,Image.Resampling.NEAREST))
    # Accept either 0/1 masks or 0/255 masks.
    return a>(0 if a.max()<=1 else threshold)
