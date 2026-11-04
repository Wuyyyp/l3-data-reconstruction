"""Convert a complete Dongfeng delivery to method-D input without changing source files."""
import argparse
import csv
import json
import pickle
import tempfile
from pathlib import Path

class Loader(pickle.Unpickler):
    def find_class(self,module,name):
        return super().find_class(module.replace('numpy._core','numpy.core'),name)

def convert(source,out,pose_config=None):
    source=source.resolve(strict=True)
    out=out.resolve()
    if out.exists() or source in out.parents or out in source.parents:
        raise ValueError('Choose a new directory outside the source delivery')
    report=json.loads((source/'colmap-ba/report.json').read_text())
    if report['coverage']!='complete':
        raise ValueError('A complete BA model is required')
    with (source/'colmap-ba/image_camera_to_world_after_ba.csv').open() as f:ba=list(csv.DictReader(f))
    with (source/'image_extrinsics/image_camera_to_world_after.csv').open() as f:base=list(csv.DictReader(f))
    lookup={(r['frame'],r['camera'],r['image_filename']):r for r in base}
    frames=sorted({int(r['frame']) for r in ba})
    if not frames or len(ba)!=7*len(frames) or frames!=list(range(frames[0],frames[-1]+1)):
        raise ValueError('Expected a contiguous seven-camera BA delivery')
    with (source/'result_interpolation_optimized.pkl').open('rb') as f:data=Loader(f).load()
    front='front_camera_fov30'
    by_file={str(int(i['sensors']['cams'][front]['timestamp']))+'.jpg':i for i in data['infos']}
    front_names={}
    for r in ba:
        peer=lookup[r['frame'],r['camera'],r['image_filename']]
        path=Path(peer['image_path'])
        path=(source/path).resolve(strict=True)
        if r['camera']==front:front_names[int(r['frame'])]=r['image_filename']
        r.update(image_path=str(path),timestamp_ns=peer['timestamp_ns'],frame=str(int(r['frame'])-frames[0]))
    data['infos']=[by_file[front_names[f]] for f in frames]
    # Some pose exports keep the raw lens coefficients but record the JPEG convention
    # in the clip's pose YAML instead of repeating it in every camera dictionary.
    canonical=None
    if pose_config is not None:
        import yaml
        pose_config=pose_config.resolve(strict=True)
        cfg=yaml.safe_load(pose_config.read_text())
        if cfg.get('assumptions',{}).get('result_interpolation_images_undistorted') is not True:
            raise ValueError('Pose config does not identify the images as undistorted')
        canonical=(pose_config.parent/Path(cfg['dataset_root'])).resolve(strict=True)
    import numpy as np
    from PIL import Image
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent/'mvs_process'))
    from delivery_d_prepare import model_cameras,model_images
    cams=model_cameras(source/'colmap-ba/sparse_after/cameras.bin')
    views=model_images(source/'colmap-ba/sparse_after/images.bin')
    for r in ba:
        cam=r['camera'];path=Path(r['image_path'])
        sensor=data['infos'][int(r['frame'])]['sensors']['cams'][cam]
        spec=cams[views[cam+'/'+r['image_filename']]['camera_id']]
        if path.name!=r['image_filename'] or path.name!=str(int(sensor['timestamp']))+'.jpg':
            raise ValueError(f'Wrong image identity: {path}')
        with Image.open(path) as image:
            if image.size!=tuple(spec['size']):raise ValueError(f'Image size disagrees with BA: {path}')
        np.testing.assert_allclose(sensor['cam_intrinsic'],spec['K'],atol=1e-6,rtol=0)
        if not sensor.get('images_undistorted'):
            expected=canonical/'images'/cam/path.name if canonical else None
            if expected is None or expected.resolve()!=path:
                raise ValueError(f'Provide --pose-config for this canonical image: {path}')
        sensor['images_undistorted']=True
        sensor['data_path']=sensor['aws_path']=str(path)
    for i in data['infos']:
        lid=i['sensors']['lidar']['perception']
        p=Path('compensated_perception')/Path(lid['data_path']).name
        if not (source/p).is_file() or not lid.get('dynamic_compensated'):
            raise ValueError(f'Expected compensated perception point cloud: {p}')
        lid['data_path']=lid['aws_path']=str(p)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        dst=Path(temp)/'ready'
        (dst/'colmap/poses').mkdir(parents=True)
        (dst/'colmap/sparse_ba').symlink_to((source/'colmap-ba/sparse_after').resolve(strict=True))
        (dst/'compensated_perception').symlink_to(source/'compensated_perception')
        with (dst/'colmap/poses/image_camera_to_world_after_ba.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(ba[0]));writer.writeheader();writer.writerows(ba)
        with (dst/'result_interpolation_optimized.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
        dst.rename(out)
    print(f'{out}: {len(frames)} frames, {len(ba)} images')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--pose-config',type=Path,help='YAML used for these canonical images, required if the input PKL has no undistortion flag')
    a=p.parse_args();convert(a.source,a.output,a.pose_config)
