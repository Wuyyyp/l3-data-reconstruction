#!/usr/bin/env python3
import argparse, json, os, re, shutil
from pathlib import Path
import numpy as np

FRAME_RE = re.compile(r'^(\d{6})(.*)$')
DIRS = ['images', 'intrinsics', 'extrinsics', 'ego_pose', 'lidar_depth', 'intensity', 'dynamic_mask']

def new_name(name, offset):
    m = FRAME_RE.match(name)
    if not m:
        return name
    frame = int(m.group(1)) - offset
    if frame < 0:
        raise ValueError(name)
    return f'{frame:06d}{m.group(2)}'

def copy_tree(src, dst, offset):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() or p.is_symlink():
            out = dst / new_name(p.name, offset)
            if p.is_symlink():
                target = os.readlink(p)
                if out.exists() or out.is_symlink():
                    out.unlink()
                os.symlink(target, out)
            else:
                shutil.copy2(p, out)

def remap_timestamps(src_file, dst_file, offset):
    data = json.load(open(src_file))
    if isinstance(data, list):
        out = []
        for item in data:
            item = dict(item)
            if 'FRAME_IDX' in item:
                item['FRAME_IDX'] = int(item['FRAME_IDX']) - offset
            if 'FRAME_NAME' in item:
                item['FRAME_NAME'] = f"{int(item['FRAME_NAME']) - offset:06d}"
            out.append(item)
    else:
        out = {}
        for key, value in data.items():
            if isinstance(value, dict):
                mapped = {}
                for frame, ts in value.items():
                    mapped[f'{int(frame)-offset:06d}'] = ts
                out[key] = mapped
            else:
                out[key] = value
    with open(dst_file, 'w') as f:
        json.dump(out, f, indent=2)

def remap_pointcloud(src_file, dst_file, offset):
    data = np.load(src_file, allow_pickle=True)
    out = {}
    for name in data.files:
        item = data[name]
        if item.shape == () and isinstance(item.item(), dict):
            src_dict = item.item()
            out[name] = {int(k) - offset: v for k, v in src_dict.items()}
        else:
            out[name] = item
    np.savez_compressed(dst_file, **out)

def remap_track_info(src_file, dst_file, offset):
    if not src_file.exists():
        dst_file.write_text('frame_id track_id class_name alpha height width length x y z heading speed\n')
        return
    lines = src_file.read_text().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] == 'frame_id':
            out.append(line)
            continue
        parts[0] = str(int(parts[0]) - offset)
        out.append(' '.join(parts))
    dst_file.write_text('\n'.join(out) + ('\n' if out else ''))

def remap_track_camera_vis(src_file, dst_file, offset):
    if not src_file.exists():
        dst_file.write_text('{}\n')
        return
    data = json.load(open(src_file))
    out = {}

    # Expected shape is track_id -> frame_id -> visible camera list.
    # Keep track ids unchanged and only remap frame ids to 0-based.
    for track_id, frames in data.items():
        if not isinstance(frames, dict):
            continue
        mapped_frames = {}
        for frame, value in frames.items():
            mapped = int(frame) - offset
            if mapped >= 0:
                mapped_frames[str(mapped)] = value
        if mapped_frames:
            out[str(track_id)] = mapped_frames

    with open(dst_file, 'w') as f:
        json.dump(out, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--offset', type=int, required=True)
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for d in DIRS:
        copy_tree(src / d, dst / d, args.offset)
    policy = src / 'dynamic_mask_policy.json'
    if policy.is_file():
        shutil.copy2(policy, dst / policy.name)
    (dst / 'track').mkdir(exist_ok=True)
    remap_track_info(src / 'track' / 'track_info.txt', dst / 'track' / 'track_info.txt', args.offset)
    remap_track_camera_vis(src / 'track' / 'track_camera_vis.json', dst / 'track' / 'track_camera_vis.json', args.offset)
    for name in ['timestamps.json', 'timestamps_specific.json']:
        remap_timestamps(src / name, dst / name, args.offset)
    remap_pointcloud(src / 'pointcloud.npz', dst / 'pointcloud.npz', args.offset)
    print('normalized', src, '->', dst)

if __name__ == '__main__':
    main()
