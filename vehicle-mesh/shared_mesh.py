"""Opt-in prep/export for seven-camera shared vehicle registration; writes to a fresh directory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

import os
PACKAGE = Path(__file__).resolve().parent / 'registration'
ASSETS = Path(os.environ.get('MESH_ASSET_ROOT','assets')).resolve()
PKL = 'result_interpolation_optimized_first101.pkl'
FITTED = 'result_interpolation_optimized_first101_mesh_9dof.pkl'
CAMS = {'left_front_camera': 0, 'right_front_camera': 1, 'rear_camera': 2,
        'left_rear_camera': 3, 'right_rear_camera': 4, 'front_camera_fov30': 9, 'front_camera_fov120': 10}
VEHICLES = {'Car', 'Suv', 'Truck', 'Huge_vehicle', 'Vehicle_else', 'Bus', 'Firetruck', 'Ambulance'}
EGO = 'ego2global_transformation_matrix_camera_reference_offset_optimized'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def observations(infos):
    tracks = {}
    if len(infos) != 101:
        raise ValueError('Expected a prepared 101-frame window')
    for frame, info in enumerate(infos):
        ids = [int(obj['id']) for obj in info['objects']]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate annotation ID at frame {frame}')
        for obj in info['objects']:
            if obj['type'] in VEHICLES:
                tracks.setdefault(int(obj['id']), []).append((frame, obj))
    return tracks


def prepare(geometry, dest):
    import sys
    import cv2
    import yaml
    from scipy.spatial.distance import pdist
    sys.path.insert(0, str(PACKAGE.parent))
    from l3_fast_data import collect_lidar, load_asset, match_assets

    seq = geometry / 'mesh_input'
    data = pickle.loads((seq / PKL).read_bytes())
    infos = data['infos']
    tracks = observations(infos)
    if not tracks:
        raise ValueError('No registered four-wheel vehicles in this window; use the native-only branch')
    manifest = read(seq / 'manifest.json')
    if Counter(v['camera'] for v in manifest['videos']) != Counter(CAMS.keys()):
        raise ValueError('Expected all seven camera videos, exactly once')
    views = {(v['frame'], v['cam']): v for v in read(geometry / 'input_manifest.json')['views']}
    if len(views) != 707:
        raise ValueError('Expected 707 camera records')
    inp = dest / 'input'
    inp.mkdir()
    (inp / 'videos').mkdir()
    manifest.update(frames=101, fps=10)
    for v in manifest['videos']:
        video = (seq / v['file']).resolve(strict=True)
        local = inp / 'videos' / (v['camera'] + '.mp4')
        local.symlink_to(video)
        cap = cv2.VideoCapture(str(video))
        try:
            w,h,frames,fps = [cap.get(k) for k in (3,4,7,5)]
        finally:
            cap.release()
        if frames != 101 or abs(fps-10) > .01:
            raise ValueError(f'Wrong video length or frame rate: {video}')
        for f, info in enumerate(infos):
            rec = views[f,CAMS[v['camera']]]
            cam = info['sensors']['cams'][v['camera']]
            if (rec['width'],rec['height']) != (w,h):
                raise ValueError(f'Wrong raster dimensions: {video}')
            np.testing.assert_allclose(cam['cam_intrinsic'],rec['K'],atol=1e-6)
            np.testing.assert_allclose(cam['camera_to_world_optimized'],rec['c2w'],atol=1e-6)
        v.update(file=str(local.relative_to(inp)),width=int(w),height=int(h),fps=10,frames=101)
    for info in infos:
        objects = info['objects']
        ids = [int(o['id']) for o in objects]
        if ids != list(map(int,info['track_id'])) or len(ids) != len(info['gt_boxes']):
            raise ValueError('Annotation rows and track IDs disagree')
        info['gt_names'] = ['VEHICLE_'+o['type'].upper() if o['type'] in VEHICLES else o['type'] for o in objects]
        info['vels'] = [o['velocity'] for o in objects]
    with (inp/PKL).open('wb') as out:
        pickle.dump(data,out,protocol=4)
    write(inp/'manifest.json',manifest)
    points = collect_lidar(infos,tracks,geometry/'converted/pointcloud.npz')
    with tempfile.TemporaryDirectory() as tmp:
        catalog = [load_asset(p,ASSETS,Path(tmp)) for p in sorted(ASSETS.rglob('*.glb'))]
        matches,audit = match_assets(tracks,points,catalog)
    ids = sorted(tracks)
    if set(matches) != set(ids):
        raise ValueError('Incomplete vehicle-to-asset mapping')
    moving = []
    for tid, obs in tracks.items():
        centers = [(np.asarray(infos[f][EGO]) @ np.r_[o['location'],1])[:3] for f,o in obs]
        if len(centers)>1 and pdist(centers).max()>1.0:
            moving.append(tid)
    objects = {str(t):dict(category=matches[t]['category'],model_path=matches[t]['path']) for t in ids}
    (dest/'registration.yaml').write_text(yaml.safe_dump(dict(selected_ids=ids, dynamic_ids=sorted(moving), objects=objects)))
    (dest/'class_whitelist.txt').write_text('\n'.join(sorted({a['category'] for a in catalog}))+'\n')
    write(dest/'selection_statistics.json',dict(policy='all_annotated_four_wheel_vehicles',selected_ids=ids,dynamic_ids=sorted(moving)))
    write(dest/'asset_matches.json',audit)


def export(source, registration, dest):
    # Merge only vehicle poses into the unmodified geometry-stage PKL.
    data = pickle.loads((source/PKL).read_bytes())
    fit = pickle.loads((registration/'pkl'/FITTED).read_bytes())
    infos = data['infos']
    tracks = observations(infos)
    report = read(registration/'pkl/validation_report.json')
    if not report.get('shared_pose_matrices_valid') or not report.get('source_fields_preserved'):
        raise ValueError('Shared pose/source preservation validation missing')
    masks = (registration/'mask-videos/masks').resolve(strict=True)
    assets = read(registration/'asset_manifest.json')['objects']
    if set(map(int,assets)) != set(tracks) or len(fit['infos']) != len(infos):
        raise ValueError('Different vehicle inventory or frame count')
    per_view = {str(t):{} for t in tracks}
    frames = []
    for f,(original,fitted) in enumerate(zip(infos,fit['infos'])):
        if int(original['timestamp']) != int(fitted['timestamp']):
            raise ValueError(f'Frame {f} timestamp mismatch')
        np.testing.assert_allclose(original[EGO],fitted[EGO],atol=1e-8)
        poses = {str(k):p for k,p in fitted['mesh_poses_9dof'].items()}
        expected = {str(o['id']) for o in original['objects'] if int(o['id']) in tracks}
        if expected != poses.keys():
            raise ValueError(f'Missing/extra vehicle poses at frame {f}')
        entries=[]
        for tid,pose in poses.items():
            asset = Path(assets[tid]['asset_path']).resolve(strict=True)
            asset.relative_to(ASSETS.resolve())
            T = np.asarray(pose['world_from_unit_mesh'])
            G = np.asarray(pose['world_from_gltf_scene'])
            if T.shape!=(4,4) or not np.isfinite(T).all() or np.linalg.det(T[:3,:3])<=0:
                raise ValueError(f'Invalid shared transform: {f}/{tid}')
            np.testing.assert_allclose(T[3],[0,0,0,1],atol=1e-8)
            np.testing.assert_allclose(G,np.asarray(original[EGO])@pose['source_ego_from_gltf_scene'],atol=1e-6)
            for cid in CAMS.values():
                per_view[tid][f'{f:06}_{cid:02}']=T.tolist()
            entries.append(dict(track_id=int(tid),asset_path=str(asset),world_from_gltf_scene=G.tolist(),
                                source_ego_from_gltf_scene=pose['source_ego_from_gltf_scene'],status=pose.get('status')))
        original['mesh_poses_9dof']=poses
        frames.append(dict(frame=f,objects=entries))
    (dest/'registration/mask-videos').mkdir(parents=True)
    (dest/'registration/mask-videos/masks').symlink_to(masks)
    with (dest/'result_interpolation_optimized_first101_with_meshes.pkl').open('wb') as out:
        pickle.dump(data,out,protocol=4)
    write(dest/'placements.json',dict(frames=frames))
    write(dest/'view_transforms.json',per_view)
    write(dest/'asset_matches.json',{t:dict(asset_path=v['asset_path']) for t,v in assets.items()})
    write(dest/'contract.json',dict(backend=str(PACKAGE),frames=len(frames),vehicles=len(tracks),
                                    track_frames=sum(map(len,tracks.values())),shared_camera_pose=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('prepare');a.add_argument('--geometry',type=Path,required=True)
    b=sub.add_parser('export');b.add_argument('--source',type=Path,required=True);b.add_argument('--registration',type=Path,required=True)
    for cli in (a,b):cli.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    dest=args.output.absolute()
    if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent,prefix='.shared-mesh-') as tmp:
        stage=Path(tmp)/'ready';stage.mkdir()
        if args.stage=='prepare':prepare(args.geometry.resolve(),stage)
        else:export(args.source.resolve(),args.registration.resolve(),stage)
        stage.rename(dest)
