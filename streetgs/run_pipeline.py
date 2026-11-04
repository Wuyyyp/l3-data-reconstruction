#!/usr/bin/env python3
"""Run one L3 original-view window. All output goes under --work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--source-start',type=int,default=0,help='Original frame index for a rebased 101-frame delivery without window_verification.json')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--geometry-only',action='store_true',help='Prepare WYS images, LiDAR and mesh geometry inputs, then exit')
    p.add_argument('--prepare-only',action='store_true')
    far=p.add_mutually_exclusive_group()
    far.add_argument('--far-ply',type=Path,help='Reuse a world-coordinate far-only PLY for this exact window')
    far.add_argument('--no-far',action='store_true')
    mesh=p.add_mutually_exclusive_group()
    mesh.add_argument('--registered-mesh',type=Path,help='Reuse a complete mesh_registration directory and verify against this window')
    mesh.add_argument('--no-mesh',action='store_true',help='Use native actors for all vehicles')
    a=p.parse_args()
    g=Path(__file__).resolve().parent
    w=a.work.resolve()
    src=a.delivery.resolve(strict=True)
    if w==src or w in src.parents or src in w.parents:
        p.error('Use separate input and work directories')
    meta=src/'window_source.json'
    clip=json.loads(meta.read_text())['source']['clip'] if meta.is_file() else w.name.rsplit('_f',1)[0]
    report=json.loads((src/'colmap-ba/report.json').read_text())
    if report.get('coverage')!='complete' or len(report.get('selected_frames',[]))!=101:
        p.error('Expected complete BA over this 101-frame delivery')
    required=['POSE_PYTHON']
    if not a.geometry_only:required.append('ROAD_SEGFORMER_WEIGHTS')
    if not a.geometry_only and not a.no_mesh:required+=['MESH_PYTHON','MESH_ASSET_ROOT']
    if not a.geometry_only and not a.far_ply and not a.no_far:required+=['DVP_PYTHON','DVP_DELIVERY_COLMAP']
    missing=[k for k in required if not os.environ.get(k)]
    if missing:p.error('Set environment variables: '+', '.join(missing))
    for key,folder in [('DVP_ROOT','dvp-openmvs'),('MESH_ROOT','vehicle-mesh'),('POSE_ROOT','pose-repair')]:
        os.environ.setdefault(key,str(g.parent/folder))
    os.environ.update(STREETGS_ROOT=str(g),SOURCE_START=str(a.source_start))
    sys.path.insert(0,str(g/'workflow'))
    from task_queue import write_json
    from vla_job import Job
    w.mkdir(parents=True,exist_ok=True)
    with (w/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        row=dict(sample_name=w.name,clip_name=clip,source=str(src),source_start=a.source_start,no_far=a.no_far,no_mesh=a.no_mesh,
                 far_ply=str(a.far_ply.resolve(strict=True)) if a.far_ply else None,registered_mesh=str(a.registered_mesh.resolve(strict=True)) if a.registered_mesh else None)
        spec=w/'run_inputs.json'
        if spec.exists() and json.loads(spec.read_text())!=row:
            p.error('Work directory belongs to different inputs or options')
        if not a.geometry_only:write_json(spec,row)
        geometry_spec=w/'geometry_inputs.json'
        geo={'source':str(src),'source_start':a.source_start}
        if geometry_spec.exists() and json.loads(geometry_spec.read_text())!=geo:p.error('Geometry belongs to a different window')
        write_json(geometry_spec,geo)
        write_json(w/'accepted_delivery.json',{'status':'PASS','frames':101,'source':str(src)})
        if a.registered_mesh:
            dest=w/'mesh_inputs/mesh_registration'
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():dest.symlink_to(row['registered_mesh'],target_is_directory=True)
            if dest.resolve()!=Path(row['registered_mesh']):p.error('Different mesh already present')
        if (w/'complete.json').exists():
            print((w/'complete.json').read_text())
            return
        job=Job(w,row,a.gpu)
        if a.geometry_only:job.prepare_geometry()
        else:job.execute(prepare_only=a.prepare_only)
if __name__=='__main__':main()
