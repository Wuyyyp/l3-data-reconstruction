"""Project a 3D cuboid into an image with near-plane clipping."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

BOX_EDGES=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))

@dataclass(frozen=True)
class BoxProjection:
    projected_bbox_xywh: tuple[float,float,float,float] | None
    visible_bbox_xywh: tuple[int,int,int,int] | None
    rejection_code: str | None

def project_track_box(*,length,width,height,center_xyz,heading,intrinsic,extrinsic,image_width,image_height,near_plane=.05):
    signs=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],float)
    local=signs*np.array([length,width,height],float)*.5
    c,s=math.cos(heading),math.sin(heading)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
    corners=local@rotation.T+np.asarray(center_xyz,float)
    camera=(extrinsic@np.column_stack((corners,np.ones(8))).T).T[:,:3]
    if not np.isfinite(camera).all():return BoxProjection(None,None,'invalid_projection')
    points=[p for p in camera if p[2]>=near_plane]
    for a,b in BOX_EDGES:
        first,second=camera[a],camera[b]
        if (first[2]>=near_plane)==(second[2]>=near_plane):continue
        points.append(first+(near_plane-first[2])/(second[2]-first[2])*(second-first))
    if not points:return BoxProjection(None,None,'behind_camera')
    p=np.asarray(points);hom=(intrinsic@p.T).T;pixels=hom[:,:2]/hom[:,2:3]
    if not np.isfinite(pixels).all():return BoxProjection(None,None,'invalid_projection')
    lo,hi=pixels.min(0),pixels.max(0);w,h=hi-lo
    if w<=0 or h<=0:return BoxProjection(None,None,'invalid_projection')
    projected=(float(lo[0]),float(lo[1]),float(w),float(h))
    x0=max(0,min(image_width,math.floor(lo[0])));y0=max(0,min(image_height,math.floor(lo[1])))
    x1=max(0,min(image_width,math.ceil(hi[0])));y1=max(0,min(image_height,math.ceil(hi[1])))
    return BoxProjection(projected,(x0,y0,x1-x0,y1-y0),None) if x1>x0 and y1>y0 else BoxProjection(projected,None,'outside_image')
