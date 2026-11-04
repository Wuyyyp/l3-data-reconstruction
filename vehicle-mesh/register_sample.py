"""Bridge the seven-view registration package to the batch's per-sample mesh inputs."""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import shared_mesh as adapter

HERE = Path(__file__).resolve().parent
VERSION = 'sam3-shared-vehicle-pose-1102'
FITTED = 'result_interpolation_optimized_first101_with_meshes.pkl'


def run(stage, *args):
    subprocess.run([sys.executable, str(HERE/'shared_mesh.py'), stage, *map(str, args)], check=True)


def native_only(source, dest):
    data = pickle.loads((source/adapter.PKL).read_bytes())
    assert not adapter.observations(data['infos'])
    rows = []
    for i, frame in enumerate(data['infos']):
        frame['mesh_poses_9dof'] = {}
        rows.append(dict(frame=i, objects=[]))
    (dest/'registration/mask-videos/masks').mkdir(parents=True)
    (dest/FITTED).write_bytes(pickle.dumps(data, protocol=4))
    for name, value in [('placements', dict(frames=rows)), ('view_transforms', {}), ('asset_matches', {}),
                        ('contract', dict(frames=len(rows), vehicles=0, track_frames=0, shared_camera_pose=True, native_only=True))]:
        adapter.write(dest/(name+'.json'), value)


def verify(source, dest):
    src = pickle.loads((source/adapter.PKL).read_bytes())['infos']
    out = pickle.loads((dest/FITTED).read_bytes())['infos']
    tracks = adapter.observations(src)
    views = adapter.read(dest/'view_transforms.json')
    contract = adapter.read(dest/'contract.json')
    assert len(src)==len(out)==contract['frames']==101 and contract['shared_camera_pose']
    assert set(views)==set(map(str,tracks)) and contract['vehicles']==len(tracks)
    expected = {tid:set() for tid in views}
    total = 0
    for i,(a,b) in enumerate(zip(src,out)):
        assert a.keys()-{'mesh_poses_9dof'} == b.keys()-{'mesh_poses_9dof'}
        for k in a.keys()-{'mesh_poses_9dof'}:
            assert pickle.dumps(a[k],protocol=4)==pickle.dumps(b[k],protocol=4), (i,k)
        ids={str(o['id']) for o in a['objects'] if o['type'] in adapter.VEHICLES}
        assert ids==b['mesh_poses_9dof'].keys()
        for tid in ids:
            matrix=b['mesh_poses_9dof'][tid]['world_from_unit_mesh']
            for cam in adapter.CAMS.values():
                key=f'{i:06d}_{cam:02d}'
                np.testing.assert_array_equal(views[tid][key],matrix)
                expected[tid].add(key)
            total+=1
    assert total==contract['track_frames'] and all(views[k].keys()==keys for k,keys in expected.items())
    return dict(mesh_version=VERSION, vehicles=len(tracks), track_frames=total, identical_camera_keys=total*7,
                source_fields_preserved=True, shared_camera_pose=True, native_only=not bool(tracks))


def main(work,gpu):
    geometry=work/'geometry'
    source=geometry/'mesh_input'
    prep=work/'shared_input'
    fit=work/'shared_registration'
    dest=work/'mesh_inputs/mesh_registration'
    has_cars=bool(adapter.observations(pickle.loads((source/adapter.PKL).read_bytes())['infos']))
    if has_cars and not dest.exists():
        if not prep.exists():
            run('prepare','--geometry',geometry,'--output',prep)
        selection=adapter.read(prep/'selection_statistics.json')
        assert selection['policy']=='all_annotated_four_wheel_vehicles' and selection['selected_ids']
        cmd=[sys.executable,str(HERE/'registration/run_registration.py'),'--input',str(prep/'input'),
             '--config',str(prep/'registration.yaml'),'--output',str(fit),'--jobs','1','--gpus',str(gpu),
             '--include-static','--selection-evidence',str(prep/'selection_statistics.json'),
             '--scene-depth-root',str(geometry/'converted')]
        if fit.exists():
            if not (fit/'pkl'/adapter.FITTED).is_file():
                raise RuntimeError('Incomplete registration initialization; inspect '+str(fit))
            cmd.append('--resume')
        subprocess.run(cmd,check=True)
        run('export','--source',source,'--registration',fit,'--output',dest)
    elif not has_cars and not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            stage=Path(tmp)/'ready'
            native_only(source,stage)
            stage.rename(dest)
    report=verify(source,dest)
    cache=work/'.mesh_overlap_cache'
    if cache.is_dir():
        import shutil
        shutil.rmtree(cache)
    adapter.write(work/'shared_mesh_acceptance.json',report)
    print('SHARED_MESH_ACCEPTED',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('work',type=Path)
    p.add_argument('gpu',type=int,choices=range(8))
    a=p.parse_args()
    main(a.work.absolute(),a.gpu)
