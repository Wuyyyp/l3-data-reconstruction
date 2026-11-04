"""Shared lossless masks: OpenMVS 255=ignore; DVP 255=keep."""
from pathlib import Path
import cv2
import numpy as np

VERSION = "dense-mask-v2"

def resize_ignore(mask, size):
    # Any ignored source area invalidates a downsampled pixel.
    return cv2.resize((mask >= 128).astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0

def block_path(scene, image_id):
    p = Path(scene) / "blocks" / f"mask_{int(image_id)}.png"
    return p if p.exists() else p.with_suffix(".jpg")

def rig_mask(shape, top):
    h,w=shape
    a=np.asarray(top, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a)<2 or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1) or a[0,0]!=0 or a[-1,0]!=1 or np.any(np.diff(a[:,0])<=0):
        raise ValueError("Rig topline must span x=0..1 with increasing normalized coordinates")
    points=np.rint(a*np.array([w-1,h-1])).astype(np.int32)
    polygon=np.vstack([points,[[w-1,h-1],[0,h-1]]])
    result=np.zeros((h,w),np.uint8);cv2.fillPoly(result,[polygon],255)
    return result
