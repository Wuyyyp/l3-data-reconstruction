"""One-off trimesh registration for the authorized multiview experiment only.
Run from repository root: python utils/register_baseline.py [--preview]
--preview renders one calibrated frame before launching parallel camera jobs.
"""
from __future__ import annotations
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ.setdefault('EGL_DEVICE_ID', '0')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg
import trimesh
import pyglet
pyglet.options['shadow_window'] = False
import pyrender
import yaml

INPUT = Path(os.environ.get('MULTIVIEW_RUN_ROOT', str(Path(__file__).resolve().parents[1] / 'input'))).resolve()
OUTPUT = INPUT / 'register-videos'
CANONICAL_FROM_SOURCE = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
SOURCE_FROM_CANONICAL = np.linalg.inv(CANONICAL_FROM_SOURCE)
CV_FROM_GL = np.diag([1., -1., -1., 1.])
BLENDER_FROM_GLTF = trimesh.transformations.rotation_matrix(np.pi / 2., [1., 0., 0.])


def inputs():
    infos = pickle.load((INPUT / 'result_interpolation_optimized_first101.pkl').open('rb'))['infos']
    selected = [int(x) for x in (INPUT / 'obj_selected.txt').read_text().split()]
    if not selected or len(selected)!=len(set(selected)) or not set(selected).issubset(asset_rows()):
        raise ValueError('Selected IDs must be nonempty, unique and have a manually specified model path')
    video_manifest = json.loads((INPUT / 'manifest.json').read_text())
    assert len(infos) == video_manifest['frames'] == 101
    tracks = []
    for info in infos:
        assert len(info['gt_boxes']) == len(info['track_id']) == len(info['gt_names'])
        frame = {}
        for identity, box in zip(info['track_id'], info['gt_boxes']):
            if identity not in selected:
                continue
            box = np.array(box, dtype=np.float64)
            assert box.shape == (7,) and np.isfinite(box).all() and min(box[3:6]) > 0
            center = (CANONICAL_FROM_SOURCE @ np.r_[box[:3], 1.])[:3]
            yaw = box[6] + np.pi / 2.
            pose = trimesh.transformations.rotation_matrix(yaw, [0., 0., 1.])
            pose[:3, :3] = pose[:3, :3] @ np.diag(box[3:6])
            pose[:3, 3] = center
            frame[int(identity)] = pose
        tracks.append(frame)
    return infos, tracks, video_manifest


def asset_rows():
    manifest_path=os.environ.get('MULTIVIEW_ASSET_MANIFEST')
    if not manifest_path and (INPUT/'asset_manifest.json').is_file():manifest_path=str(INPUT/'asset_manifest.json')
    if not manifest_path:raise FileNotFoundError('asset_manifest.json is required; assign each selected ID a model path')
    assets=json.loads(Path(manifest_path).read_text())['objects'];selected={int(x) for x in (INPUT/'obj_selected.txt').read_text().split()};result={}
    for identity in selected:
        row=dict(assets[str(identity)]);path=Path(row['asset_path']).expanduser()
        if not path.is_file() or path.stem!=row['model_id']:raise ValueError(f'Invalid model path for selected ID {identity}: {path}')
        result[identity]=row
    return result


def load_unit_model(path):
    scene = trimesh.load(str(path), force='scene', process=False)
    geometries = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(BLENDER_FROM_GLTF @ transform)
        geometries.append(mesh)
    all_bounds = np.array([mesh.bounds for mesh in geometries])
    lo = all_bounds[:, 0].min(axis=0)
    hi = all_bounds[:, 1].max(axis=0)
    size = hi - lo
    assert min(size) > 0 and np.isfinite(size).all()
    normalization = np.eye(4)
    normalization[:3, :3] = np.diag(1. / size)
    normalization[:3, 3] = -(lo + hi) / 2. / size
    render_meshes = []
    for mesh in geometries:
        mesh.apply_transform(normalization)
        render_meshes.append(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    return render_meshes, dict(bounds_lwh=size.tolist(), bounds_center=((lo + hi) / 2.).tolist())


def camera_for(info, name):
    cam = info['sensors']['cams'][name]
    reference = np.array(info['ego2global_transformation_matrix_camera_reference_offset_optimized'])
    optimized = cam.get('camera_to_world_optimized')
    if optimized is None:
        external = np.array(cam['extrinsic'])
    else:
        external = np.linalg.inv(np.array(optimized)) @ reference
    return np.array(cam['cam_intrinsic']), external @ SOURCE_FROM_CANONICAL


def restore_vertical_foreground(name,frame_id,render_mask):
    """Keep narrow pole-like gaps through the observed vehicle mask convex hull.

    Only internal, tall missing strips are restored. This does not clip the
    rendered silhouette to the target contour and is not used for IoU scoring.
    """
    path=INPUT/'mask-videos'/'masks'/name/f'{frame_id:06d}.npz'
    if not path.exists():return
    with np.load(path) as data:
        for key in data.files:
            if not key.endswith('_bbox'):continue
            x,y,w,h=map(int,data[key]);bits=data[key[:-5]+'_bits']
            observed=np.unpackbits(bits,count=w*h).reshape(h,w).astype(np.uint8)
            contours,_=cv2.findContours(observed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:continue
            hull=cv2.convexHull(np.concatenate(contours,axis=0))
            filled=np.zeros_like(observed);cv2.fillConvexPoly(filled,hull,1)
            missing=(filled>0)&(observed==0)
            columns=missing.sum(axis=0)>.5*h
            edges=np.diff(np.r_[False,columns,False].astype(np.int8))
            for left,right in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                if right-left>max(3,round(w*.2)):continue
                left=max(0,left-1);right=min(w,right+1)
                render_mask[y:y+h,x+left:x+right]&=~missing[:,left:right]


from functools import lru_cache
SCENE_CAMERA_IDS = {'left_front_camera':0,'right_front_camera':1,'rear_camera':2,'left_rear_camera':3,'right_rear_camera':4,'front_camera_fov30':9,'front_camera_fov120':10}

@lru_cache(maxsize=3)
def scene_depth(name, frame):
    root = os.environ.get('MULTIVIEW_SCENE_DEPTH_ROOT')
    if not root:return None
    stem = f'{frame:06d}_{SCENE_CAMERA_IDS[name]:02d}'
    data = np.load(Path(root)/'lidar_depth'/f'{stem}.npy',allow_pickle=True).item()
    depth = np.full(data['mask'].shape,1e6,np.float32)
    depth[data['mask']] = data['value']
    # Expand sparse samples by 10 px; 2 m tolerance prevents vehicle self-occlusion.
    return cv2.erode(depth,np.ones((21,21),np.uint8))

def scene_visible_mask(name, frame, rendered_depth):
    observed = scene_depth(name,frame)
    return (rendered_depth>0) if observed is None else (rendered_depth>0)&(rendered_depth<=observed+2.)


def render_camera(item, preview=False):
    cv2.setNumThreads(1)
    infos, tracks, manifest = inputs()
    assets = asset_rows()
    name, width, height = item['camera'], item['width'], item['height']
    print('START', name, flush=True)
    scene = pyrender.Scene(bg_color=[0., 0., 0., 0.], ambient_light=[0.72, 0.72, 0.72])
    render_meshes = {}
    normalization = {}
    asset_cache = {}
    for identity, asset in assets.items():
        path = str(Path(asset['asset_path']).expanduser())
        if path not in asset_cache:asset_cache[path] = load_unit_model(path)
        render_meshes[identity], normalization[identity] = asset_cache[path]
    renderer = pyrender.OffscreenRenderer(width, height)
    cap = cv2.VideoCapture(str(INPUT / item['file']))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(infos)
    assert cap.get(cv2.CAP_PROP_FPS) == manifest['fps'] == 10
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
    log_path = Path(os.environ.get('MULTIVIEW_LOG_ROOT', INPUT/'logs'))
    log_path.mkdir(parents=True, exist_ok=True)
    active_nodes = []
    camera_node = None
    light_node = None
    coverage = []
    process = None
    output = OUTPUT / (name + '.mp4')
    if not preview:
        log = (log_path / ('multiview-trimesh-register-' + name + '.log')).open('w')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{width}x{height}', '-r', '10', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-crf', '18', '-threads', '4',
                   '-movflags', '+faststart', str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    try:
        frame_ids = [0] if preview else range(len(infos))
        for frame_id in frame_ids:
            ok, frame = cap.read()
            assert ok and frame.shape[:2] == (height, width)
            for node in active_nodes:
                scene.remove_node(node)
            active_nodes = []
            if camera_node is not None:
                scene.remove_node(camera_node)
                scene.remove_node(light_node)
            intrinsic, external = camera_for(infos[frame_id], name)
            camera_pose = np.linalg.inv(external) @ CV_FROM_GL
            camera = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2],
                                               znear=.1, zfar=300.)
            camera_node = scene.add(camera, pose=camera_pose)
            light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=camera_pose)
            for identity, pose in tracks[frame_id].items():
                corners = np.array([[x, y, z, 1.] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
                points = (external @ pose @ corners.T).T[:, :3]
                valid = points[:, 2] > .1
                if not valid.any():
                    continue
                uv = (intrinsic @ points[valid].T).T
                uv = uv[:, :2] / uv[:, 2:]
                if (uv[:, 0].max() < 0 or uv[:, 0].min() >= width
                        or uv[:, 1].max() < 0 or uv[:, 1].min() >= height):
                    continue
                for mesh in render_meshes[identity]:
                    active_nodes.append(scene.add(mesh, pose=pose))
            if active_nodes:
                rgb, depth = renderer.render(scene, flags=flags)
                mask = scene_visible_mask(name,frame_id,depth)
                if globals().get('RESTORE_VERTICAL_FOREGROUND',False):
                    restore_vertical_foreground(name,frame_id,mask)
                frame[mask] = rgb[:, :, :3][:, :, ::-1][mask]
                coverage.append(int(mask.sum()))
            else:
                coverage.append(0)
            if frame_id in [0, 25, 50, 75, 100]:
                cv2.imwrite(str(OUTPUT / (name + f'_frame{frame_id:03d}.jpg')), frame)
            if preview:
                assert cv2.imwrite('/tmp/trimesh-registration-preview.jpg', cv2.resize(frame, (1280, 720)))
            else:
                process.stdin.write(frame.tobytes())
            if frame_id % 25 == 0:
                print(name, frame_id, 'mesh_pixels', coverage[-1], flush=True)
        if not preview:
            assert not cap.read()[0]
            process.stdin.close()
            assert process.wait() == 0, str(log_path)
    finally:
        cap.release()
        renderer.delete()
        if process is not None and process.poll() is None:
            process.stdin.close()
            process.terminate()
            process.wait()
        if not preview:
            log.close()
    print('DONE', name, flush=True)
    return dict(camera=name, frames=len(coverage), frame_ids=list(frame_ids),
                processed_track_ids_per_frame=[sorted(tracks[fi]) for fi in frame_ids],
                frames_with_mesh=sum(v > 0 for v in coverage),
                mesh_pixels_per_frame=coverage, model_normalization=normalization)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    infos, tracks, manifest = inputs()
    if args.preview:
        render_camera(manifest['videos'][0], preview=True)
        return
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(render_camera, manifest['videos']))
    fallback = [dict(camera=item['camera'], frame=fi) for item in manifest['videos'] for fi, info in enumerate(infos)
                if 'camera_to_world_optimized' not in info['sensors']['cams'][item['camera']]]
    (OUTPUT / 'registration_manifest.json').write_text(json.dumps(dict(
        renderer='trimesh + pyrender, one-off test only', objects=asset_rows(),
        fitting='Each model centered in its measured AABB, mapped along local front/left/up to per-frame box length/width/height.',
        camera='Per-image optimized camera pose relative to optimized reference ego; mounting extrinsic fallback where unavailable.',
        mounting_extrinsic_fallback=fallback, frames=101, fps=10, cameras=results,
        compositing='Opaque textured meshes with inter-model depth; no estimated scene occlusion or new cast shadows.'),indent=2)+'\n')
    print('ALL COMPLETE', OUTPUT, flush=True)

if __name__ == '__main__':
    main()
