"""Optional method-D far region: conservative compute masks and an exact PLY crop."""
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

from delivery_d_calibrate import read

POLICY = 'trajectory-xy-gt50-near40-support2-halo16-v1'
DISTANCE_M = 50.0


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def trajectory(clip, start, end):
    with (clip / 'result_interpolation_optimized.pkl').open('rb') as source:
        infos = pickle.load(source)['infos']
    if not 0 <= start <= end < len(infos):
        raise ValueError('Invalid far-region frame window')
    transforms = np.asarray([
        info['ego2global_transformation_matrix_camera_reference_offset_optimized']
        for info in infos[start:end + 1]
    ], dtype=np.float64)
    if transforms.shape != (end - start + 1, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError('Far-only mode requires finite optimized ego poses for every window frame')
    return transforms[:, :3, 3].copy()


def compute_mask(depth, support, valid, E, K, threshold, tree):
    if depth.ndim != 2 or depth.shape != support.shape or depth.shape != valid.shape:
        raise ValueError('Prior, support and image mask dimensions differ')
    if E.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(E).all() or not np.isfinite(K).all():
        raise ValueError('Invalid camera matrices for far-region mask')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid calibrated prior uncertainty')
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K).T
    with np.errstate(invalid='ignore', over='ignore'):
        world = (rays * depth[..., None] - E[:3, 3]) @ E[:3, :3]
        allowance = np.linalg.norm((rays @ E[:3, :3])[..., :2], axis=-1) * depth * np.expm1(threshold)
    finite = np.isfinite(world).all(-1) & np.isfinite(depth) & (depth > 0)
    distance = np.full(depth.shape, np.inf)
    distance[finite] = tree.query(world[finite, :2], workers=2)[0]
    # This allowance is heuristic; uncertain pixels remain available to DVP.
    near = (support >= 2) & finite & (distance + allowance < DISTANCE_M - 10)
    near = cv2.erode(near.astype(np.uint8), np.ones((33, 33), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return (valid & ~near).astype(np.uint8) * 255


def prepare_masks(root, clip, start, end):
    cv2.setNumThreads(1)
    scene = root / '09_dvp/scene'
    directory = root / 'far_only'
    masks = directory / 'masks'
    masks.mkdir(parents=True, exist_ok=True)
    xyz = trajectory(clip, start, end)
    expected = dict(policy=POLICY, frames=[start, end],
                    trajectory_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
                    calibration_sha256=hashlib.sha256((root / 'calibration_report.json').read_bytes()).hexdigest())
    metadata = sorted((scene / 'metadata').glob('*.json'))
    if not metadata:
        raise ValueError('No calibrated D metadata for far-only mode')
    report = directory / 'mask_report.json'
    if report.exists():
        saved = json.loads(report.read_text())
        if saved.get('inputs') != expected:
            raise ValueError('Far-region inputs changed; use a fresh result directory')
        if all((masks / f'mask_{int(p.stem)}.png').is_file() for p in metadata) and (directory / 'trajectory.npy').is_file():
            return masks
    tree = cKDTree(xyz[:, :2])
    rows = []
    for mp in metadata:
        meta = json.loads(mp.read_text())
        iid = int(mp.stem)
        depth = read(scene / 'metric_prior_unbounded' / f'{iid:08d}.dmb')
        support = cv2.imread(str(scene / 'prior_support' / f'{iid:08d}.png'), 0)
        block = cv2.imread(str(scene / 'blocks' / f'mask_{iid}.png'), 0)
        if support is None or block is None:
            raise ValueError(f'Missing prior support or valid mask for image {iid}')
        lines = (scene / 'cams' / f'{iid:08d}_cam.txt').read_text().splitlines()
        E = np.array([list(map(float, line.split())) for line in lines[1:5]])
        K = np.array([list(map(float, line.split())) for line in lines[7:10]])
        valid = block >= 128
        mask = compute_mask(depth, support, valid, E, K, meta['prior_log_deviation_threshold'], tree)
        if not cv2.imwrite(str(masks / f'mask_{iid}.png'), mask):
            raise OSError(f'Cannot save far-region mask {iid}')
        rows.append(dict(id=iid, label=meta['source_label'], original_active=int(valid.sum()),
                         active=int(np.count_nonzero(mask)), newly_skipped=int(valid.sum() - np.count_nonzero(mask))))
    with (directory / 'trajectory.tmp').open('wb') as output:
        np.save(output, xyz)
    (directory / 'trajectory.tmp').replace(directory / 'trajectory.npy')
    save_json(report, dict(inputs=expected, distance_m=DISTANCE_M, images=len(rows),
                          original_active=sum(r['original_active'] for r in rows),
                          active=sum(r['active'] for r in rows), rows=rows))
    return masks


def crop_ply(path, xyz, report_path):
    """Keep attributes, permit an empty far region, publish the cropped PLY atomically."""
    vertices = PlyData.read(str(path))['vertex'].data
    points = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    valid = np.isfinite(points).all(1)
    distance = np.zeros(len(points))
    tree = cKDTree(xyz[:, :2])
    distance[valid] = tree.query(points[valid, :2], workers=2)[0]
    keep = valid & (distance > DISTANCE_M)
    raw = path.with_name(path.stem + '_uncropped.ply')
    raw_tmp = raw.with_suffix('.tmp')
    shutil.copy2(path, raw_tmp)
    raw_tmp.replace(raw)
    temporary = path.with_suffix('.tmp')
    PlyData([PlyElement.describe(vertices[keep], 'vertex')], text=False, byte_order='<').write(str(temporary))
    temporary.replace(path)
    result = dict(policy=POLICY, distance_m=DISTANCE_M, input_points=len(points),
                  far_points=int(keep.sum()), min_ego_xy_distance_m=float(distance[keep].min()) if keep.any() else None,
                  uncropped_ply=str(raw), ply=str(path))
    save_json(report_path, result)
    return result['far_points']
