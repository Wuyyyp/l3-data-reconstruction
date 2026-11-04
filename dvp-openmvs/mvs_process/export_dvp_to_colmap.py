#!/usr/bin/env python3
"""Export DVP depths into a COLMAP StereoFusion workspace (no re-estimation).

Uses the same masks/ranges and depth-derived camera normals as the OpenMVS
adapter. Camera poses and cropped intrinsics come directly from DVP; sparse
tracks are retained only for COLMAP's overlap selection. No NCC prefilter is
claimed: native StereoFusion performs geometric/normal/support filtering.
"""
from mask_policy import block_path

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from convert_dvp_to_openmvs_dmap import read_dmb, camera_normals


def write_array(path, array):
    array = np.asarray(array, dtype='<f4')
    h, w = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    with Path(path).open('wb') as f:
        f.write(f'{w}&{h}&{channels}&'.encode('ascii'))
        # COLMAP stores each channel as a contiguous row-major image plane.
        f.write(array.tobytes() if channels == 1 else array.transpose(2, 0, 1).tobytes())


def read_array(path):
    with Path(path).open('rb') as f:
        header = bytearray()
        while header.count(b'&') < 3:
            c = f.read(1)
            if not c:
                raise ValueError('Truncated COLMAP array header')
            header.extend(c)
        w, h, c = map(int, header.decode().split('&')[:3])
        data = np.frombuffer(f.read(), dtype='<f4').reshape(c, h, w)
        return data[0] if c == 1 else data.transpose(1, 2, 0)


def model_images(path):
    rows = {}
    with path.open() as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split()
            observations = next(f).split()
            rows[Path(fields[9]).stem] = (int(fields[0]), fields[9], observations)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--depth-ranges-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = json.loads((args.run / 'manifest.json').read_text())['images']
    ranges = json.loads(args.depth_ranges_json.read_text())['ranges']
    images = model_images(args.model / 'images.txt')
    labels = {r['source_label'] for r in records}
    if labels != set(images) or labels != set(ranges):
        raise ValueError('DVP, sparse model and range report image sets differ')
    if args.output.exists():
        raise FileExistsError('Use a fresh COLMAP workspace: ' + str(args.output))
    for folder in ['images', 'sparse', 'stereo/depth_maps', 'stereo/normal_maps', 'stereo/consistency_graphs']:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    camera_rows, image_rows, names, counts = [], [], [], []
    max_pose_delta = max_intrinsic_delta = 0.0
    for index, r in enumerate(records, 1):
        label = r['source_label']
        image_id, name, observations = images[label]
        if Path(name).name != name:
            raise ValueError('Nested image paths are not supported by this adapter')
        dvp_id = int(r['id'])
        w, h = r['content_size']
        sw, sh = r['source_size']
        px, py = r['padding_xy']
        lines = (args.run / 'scene/cams' / f'{dvp_id:08d}_cam.txt').read_text().splitlines()
        E = np.loadtxt(lines[1:5])
        K = np.loadtxt(lines[7:10])
        K[0, 2] -= px
        K[1, 2] -= py
        expected_K = np.loadtxt(args.source / 'intrinsics' / f'{label}.txt')
        expected_K[0] *= w / sw
        expected_K[1] *= h / sh
        expected_E = np.linalg.inv(np.loadtxt(args.source / 'ego_pose' / f'{label}.txt'))
        kd, ed = float(np.max(np.abs(K - expected_K))), float(np.max(np.abs(E - expected_E)))
        max_intrinsic_delta, max_pose_delta = max(max_intrinsic_delta, kd), max(max_pose_delta, ed)
        if kd > 1e-4 or ed > 1e-4:
            raise ValueError(f'Camera conversion mismatch {label}: K={kd}, E={ed}')
        q = Rotation.from_matrix(E[:3, :3]).as_quat()[[3, 0, 1, 2]]
        camera_rows.append(f'{image_id} PINHOLE {w} {h} ' + ' '.join(f'{v:.17g}' for v in [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        image_rows.append(f'{image_id} ' + ' '.join(f'{v:.17g}' for v in [*q, *E[:3, 3]]) + f' {image_id} {name}')
        obs = []
        for j in range(0, len(observations), 3):
            obs += [f'{float(observations[j]) * w / sw:.12g}', f'{float(observations[j+1]) * h / sh:.12g}', observations[j+2]]
        image_rows.append(' '.join(obs))
        depth = read_dmb(args.run / 'scene/APD' / f'{dvp_id:08d}' / 'depths.dmb')
        mask = cv2.imread(str(block_path(args.run / 'scene', dvp_id)), 0)
        if mask is None or mask.shape != depth.shape:
            raise ValueError('Depth/mask size mismatch: ' + label)
        depth = depth[py:py+h, px:px+w].copy()
        mask = mask[py:py+h, px:px+w]
        bounds = ranges[label]
        valid = np.isfinite(depth) & (depth >= bounds['depth_min_m']) & (depth <= bounds['depth_max_m']) & (mask >= 128)
        depth[~valid] = 0
        normal = camera_normals(depth, K)
        depth_path = args.output / 'stereo/depth_maps' / (name + '.geometric.bin')
        normal_path = args.output / 'stereo/normal_maps' / (name + '.geometric.bin')
        write_array(depth_path, depth)
        write_array(normal_path, normal)
        if index in (1, len(records)):
            assert np.array_equal(read_array(depth_path), depth)
            assert np.array_equal(read_array(normal_path), normal)
        rgb = cv2.imread(str(args.source / 'images' / name))
        if rgb is None or rgb.shape[1::-1] != (sw, sh):
            raise ValueError('Source RGB mismatch: ' + label)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(args.output / 'images' / name), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(name)
        names.append(name)
        counts.append(int(valid.sum()))
        if index % 50 == 0 or index == len(records):
            print(f'COLMAP exported {index}/{len(records)}; {name}; valid={counts[-1]}', flush=True)
    (args.output / 'sparse/cameras.txt').write_text('\n'.join(camera_rows) + '\n')
    (args.output / 'sparse/images.txt').write_text('\n'.join(image_rows) + '\n')
    shutil.copy2(args.model / 'points3D.txt', args.output / 'sparse/points3D.txt')
    (args.output / 'stereo/fusion.cfg').write_text('\n'.join(names) + '\n')
    stats = dict(images=len(names), valid_depth_pixels=sum(counts), max_camera_pose_delta=max_pose_delta,
                 max_intrinsic_delta=max_intrinsic_delta, normal_source='depth-derived camera normals, same method as OpenMVS adapter',
                 filtering='finite + sparse range + DVP mask before native StereoFusion geometric/normal/support filtering; no PatchMatch NCC filter',
                 depth_source=str(args.run), fixed_100m_clip=False)
    (args.output / 'export_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
