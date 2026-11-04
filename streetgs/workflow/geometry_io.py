"""COLMAP and PCD readers shared by the WYS import (from delivery_d_prepare)."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import cv2
def rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def model_images(path, rename=None):
    """Read/optionally rename COLMAP binary images without changing any poses or tracks."""
    records = {}
    temp = path.with_suffix('.renamed') if rename else None
    with path.open('rb') as source:
        out = temp.open('wb') if temp else None
        try:
            count_raw = source.read(8)
            count = struct.unpack('<Q', count_raw)[0]
            if out: out.write(count_raw)
            for _ in range(count):
                raw = source.read(64)
                image_id, *rest = struct.unpack('<i7di', raw)
                name = bytearray()
                while (char := source.read(1)) != b'\0':
                    if not char: raise EOFError(path)
                    name.extend(char)
                name = name.decode()
                nraw = source.read(8)
                observations = source.read(24 * struct.unpack('<Q', nraw)[0])
                if len(observations) != 24 * struct.unpack('<Q', nraw)[0]: raise EOFError(path)
                T = np.eye(4); T[:3, :3] = rot(rest[:4]); T[:3, 3] = rest[4:7]
                if name in records: raise ValueError('Duplicate BA image ' + name)
                records[name] = dict(id=image_id, camera_id=rest[7], w2c=T)
                if out: out.write(raw + rename[name].encode() + b'\0' + nraw + observations)
            if source.read(1): raise ValueError('Unexpected images.bin data')
        finally:
            if out: out.close()
    if temp: temp.replace(path)
    return records

def box_polygon(box, ego_to_camera, K, width, height):
    x, y, z, length, width_box, height_box, yaw = box
    local = np.array([[sx*length/2, sy*width_box/2, sz*height_box/2] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    c, s = np.cos(yaw), np.sin(yaw)
    local[:, :2] = local[:, :2] @ np.array([[c, s], [-s, c]])
    xyz = local + [x, y, z]
    xyz = xyz @ ego_to_camera[:3,:3].T + ego_to_camera[:3,3]
    near=.1;visible=[point for point in xyz if point[2]>=near]
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if i<j and (xyz[i,2]<near)!=(xyz[j,2]<near):
                visible.append(xyz[i]+(xyz[j]-xyz[i])*(near-xyz[i,2])/(xyz[j,2]-xyz[i,2]))
    if len(visible)<3:return None
    uvz = np.asarray(visible) @ K.T; uv = (uvz[:,:2] / uvz[:,2:]).astype(np.float32)
    hull = cv2.convexHull(uv)
    screen = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    area, clipped = cv2.intersectConvexConvex(hull, screen)
    return None if area <= 0 else np.rint(clipped).astype(np.int32)

def read_pcd(path):
    with path.open('rb') as source:
        header={}
        while True:
            line=source.readline().decode('ascii').strip()
            if not line: raise ValueError(f'Invalid PCD header: {path}')
            if line.startswith('#'): continue
            key,*values=line.split();header[key]=values
            if key=='DATA': break
        if header['DATA']!=['binary']: raise ValueError(f'Only binary PCD is supported: {path}')
        names=header['FIELDS'];sizes=list(map(int,header['SIZE']));types=header['TYPE'];counts=list(map(int,header.get('COUNT',['1']*len(names))))
        formats=[]
        for size,kind,count in zip(sizes,types,counts):
            if kind=='F' and size in (4,8): code=f'<f{size}'
            elif kind in ('U','I') and size in (1,2,4,8): code=f'<{kind.lower()}{size}'
            else: raise ValueError(f'Unsupported PCD field: {path} {kind}{size}')
            formats.append((code,(count,)) if count>1 else code)
        dtype=np.dtype({'names':names,'formats':formats})
        points=int(header['POINTS'][0]);raw=source.read()
        if len(raw)!=points*dtype.itemsize: raise ValueError(f'PCD length mismatch: {path}')
        cloud=np.frombuffer(raw,dtype=dtype,count=points)
        return np.column_stack([cloud[name] for name in ('x','y','z')]).astype(np.float64)
