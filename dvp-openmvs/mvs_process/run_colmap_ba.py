#!/usr/bin/env python3
"""Optional BA stage; publish matching poses/models for DVP and OpenMVS."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from derive_sparse_depth_ranges import quaternion_to_rotation


def image_records(path):
    lines = [line.rstrip('\n') for line in path.open() if not line.lstrip().startswith('#')]
    if len(lines) % 2:
        raise ValueError(f'Malformed COLMAP image records: {path}')
    records = []
    for header, observations in zip(lines[::2], lines[1::2]):
        fields = header.split()
        if len(fields) != 10 or len(observations.split()) % 3:
            raise ValueError(f'Malformed COLMAP image: {header}')
        records.append((fields, observations))
    if not records or len({r[0][9] for r in records}) != len(records):
        raise ValueError('Empty or duplicate COLMAP images')
    return records


def camera_to_world(fields):
    q = np.array(fields[1:5], dtype=float)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError('Invalid camera quaternion')
    R = quaternion_to_rotation(q)
    t = np.array(fields[5:8], dtype=float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ t
    if not np.isfinite(T).all():
        raise ValueError('Nonfinite camera pose')
    return T


def write_images(path, records):
    path.write_text(''.join(' '.join(f) + '\n' + obs + '\n' for f, obs in records))


def publish_poses(source, model, output):
    records = image_records(model / 'images.txt')
    cameras = {}
    for line in (model / 'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        if f[1] != 'PINHOLE' or len(f) != 8:
            raise ValueError('BA adapter requires PINHOLE cameras')
        fx, fy, cx, cy = map(float, f[4:])
        cameras[f[0]] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    labels = {f[9] for f, _ in records}
    if labels != {p.name for p in (source / 'images').glob('*.jpg')}:
        raise ValueError('BA/source image sets differ')
    output.mkdir()
    for p in source.iterdir():
        if p.name not in {'ego_pose', 'input_manifest.json'}:
            (output / p.name).symlink_to(p.resolve(), target_is_directory=p.is_dir())
    (output / 'ego_pose').mkdir()
    deltas, poses = [], {}
    for f, _ in records:
        name = f[9]
        if Path(name).name != name:
            raise ValueError('Expected flat source image names after BA')
        stem = Path(name).stem
        K = np.loadtxt(source / 'intrinsics' / (stem + '.txt'))
        if not np.allclose(K, cameras[f[8]], atol=1e-7):
            raise ValueError('BA unexpectedly changed intrinsics: ' + name)
        old = np.loadtxt(source / 'ego_pose' / (stem + '.txt'))
        T = camera_to_world(f)
        np.savetxt(output / 'ego_pose' / (stem + '.txt'), T, fmt='%.17g')
        angle = math.degrees(math.acos(float(np.clip((np.trace(T[:3, :3] @ old[:3, :3].T) - 1) / 2, -1, 1))))
        deltas.append(dict(image=name, translation_m=float(np.linalg.norm(T[:3, 3] - old[:3, 3])), rotation_deg=angle))
        poses[stem] = T.tolist()
    manifest_path = source / 'input_manifest.json'
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text())
        for r in m.get('records', []):
            label = Path(r.get('label', r.get('image', ''))).stem
            if label in poses:
                r['camera_to_world'] = poses[label]
                r['image'] = str(output / 'images' / (label + '.jpg'))
        m.update(bundle_adjustment=True, pose_formula='COLMAP BA camera-to-world')
        (output / 'input_manifest.json').write_text(json.dumps(m, indent=2))
    return deltas


def validate_tools(args):
    if sys.version_info < (3, 11):
        raise RuntimeError('BA requires Python >=3.11')
    sys.path.insert(0, str(args.dongfeng_root / 'src'))
    from dongfeng_registration.colmap_reconstruction import _build_bundle_adjustment_command
    executable = args.prior_colmap if args.mode == 'absolute_pose_prior' else args.colmap
    command = 'absolute_pose_prior_bundle_adjuster' if args.mode == 'absolute_pose_prior' else 'bundle_adjuster'
    result = subprocess.run([executable, command, '-h'], capture_output=True, text=True)
    if result.returncode or '--BundleAdjustment.refine_rig_from_world' not in result.stdout + result.stderr:
        raise RuntimeError(f'{executable} lacks {command}; use COLMAP 4.2 and build Dongfeng_Pose/scripts/build_colmap_absolute_pose_prior.sh for pose priors')
    return _build_bundle_adjustment_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--image-folder', type=Path)
    p.add_argument('--neighbors-name')
    p.add_argument('--mode', choices=['standard', 'absolute_pose_prior'], default='absolute_pose_prior')
    p.add_argument('--colmap', required=True)
    p.add_argument('--prior-colmap', required=True)
    p.add_argument('--interface-colmap', required=True)
    p.add_argument('--dongfeng-root', type=Path, required=True)
    p.add_argument('--position-sigma-m', type=float, default=.1)
    p.add_argument('--rotation-sigma-deg', type=float, default=.5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume-only', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in [a.position_sigma_m, a.rotation_sigma_deg]):
        p.error('Prior sigmas must be finite and positive')
    build_command = validate_tools(a)
    if a.check:
        print('PASS: COLMAP BA mode=' + a.mode)
        return
    if not all([a.sample_root, a.output, a.image_folder, a.neighbors_name]):
        p.error('Sample, output, image folder and neighbor filename are required')
    a.sample_root, a.output = a.sample_root.resolve(), a.output.absolute()
    source = a.sample_root / '03_converted'
    if not a.image_folder.is_dir() or a.image_folder.resolve() != (source / 'images').resolve():
        raise ValueError('Fusion image folder must point to the same source images')
    if source.resolve() in a.output.resolve().parents:
        raise ValueError('BA output must be outside the original converted input')
    input_model = a.sample_root / '07_sparse/triangulated_text'
    for n in ['cameras.txt', 'images.txt', 'points3D.txt']:
        if not (input_model / n).is_file():
            raise FileNotFoundError(input_model / n)
    paths = list(input_model.glob('*.txt')) + list((source / 'ego_pose').glob('*.txt')) + list((source / 'intrinsics').glob('*.txt'))
    digest = hashlib.sha256()
    for file in sorted(paths):
        digest.update(str(file).encode()); digest.update(file.read_bytes())
    settings = dict(mode=a.mode, position_sigma_m=a.position_sigma_m, rotation_sigma_deg=a.rotation_sigma_deg,
                    colmap=str(a.colmap), prior_colmap=str(a.prior_colmap), sample=str(a.sample_root),
                    image_folder=str(a.image_folder.absolute()), neighbors_name=a.neighbors_name, input_sha256=digest.hexdigest())
    for executable in [a.colmap, a.prior_colmap if a.mode == 'absolute_pose_prior' else a.colmap, a.interface_colmap]:
        path = Path(shutil.which(executable) or executable)
        settings[str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
    settings['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings['dongfeng_ba_sha256'] = hashlib.sha256((a.dongfeng_root / 'src/dongfeng_registration/colmap_reconstruction.py').read_bytes()).hexdigest()
    request = a.output / 'request.json'
    if a.output.exists():
        if not request.is_file() or json.loads(request.read_text()) != settings:
            raise RuntimeError('BA inputs/settings changed; use a fresh --result-root')
        report = a.output / 'report.json'
        if (a.resume or a.resume_only) and report.is_file() and json.loads(report.read_text())['status'] == 'PASS':
            print('Reusing complete COLMAP BA: ' + str(a.output))
            return
        raise RuntimeError('Existing/incomplete BA output; use --resume for a completed run or a fresh result root')
    if a.resume_only:
        raise RuntimeError('BA has not completed; start with --from-stage 1')
    a.output.mkdir(parents=True)
    request.write_text(json.dumps(settings, indent=2))
    text = a.output / 'input_text'
    text.mkdir()
    for f in input_model.glob('*.txt'):
        shutil.copy2(f, text / f.name)
    records = image_records(text / 'images.txt')
    csv_rows, rename = [], {}
    for fields, _ in records:
        name = fields[9]
        if Path(name).name != name or ',' in name:
            raise ValueError('Expected flat image names without commas')
        stem = Path(name).stem
        T = np.loadtxt(source / 'ego_pose' / (stem + '.txt'))
        if not np.allclose(camera_to_world(fields), T, atol=1e-5):
            raise ValueError('Sparse/input poses disagree: ' + name)
        camera = stem.rsplit('_', 1)[1]
        row = dict(camera=camera, image_filename=name)
        row.update({f'T_cw_{i}{j}': float(T[i, j]) for i in range(4) for j in range(4)})
        csv_rows.append(row)
        if a.mode == 'absolute_pose_prior':
            # Dongfeng's prior reader keys rows by camera/image_filename.
            fields[9] = camera + '/' + name
            rename[fields[9]] = name
    write_images(text / 'images.txt', records)
    pose_csv = a.output / 'pose_priors.csv'
    with pose_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    binary = a.output / 'sparse_binary'
    binary.mkdir()
    config = SimpleNamespace(colmap_executable=a.colmap, absolute_pose_prior_colmap_executable=a.prior_colmap,
                             ba_mode=a.mode, ba_refine_points3d=True, pose_csv=pose_csv,
                             ba_position_prior_sigma_m=a.position_sigma_m, ba_rotation_prior_sigma_deg=a.rotation_sigma_deg)
    subprocess.run(build_command(config, input_path=text, output_path=binary), check=True)
    final_text = a.output / 'sparse_text'
    final_text.mkdir()
    subprocess.run([a.colmap, 'model_converter', '--input_path', str(binary), '--output_path', str(final_text), '--output_type', 'TXT'], check=True)
    final_records = image_records(final_text / 'images.txt')
    if rename:
        if set(rename) != {f[9] for f, _ in final_records}:
            raise ValueError('BA lost or renamed images')
        for fields, _ in final_records:
            fields[9] = rename[fields[9]]
        write_images(final_text / 'images.txt', final_records)
        subprocess.run([a.colmap, 'model_converter', '--input_path', str(final_text), '--output_path', str(binary), '--output_type', 'BIN'], check=True)
    deltas = publish_poses(source, final_text, a.output / 'converted')
    interface = a.output / 'colmap_input'
    interface.mkdir(); (interface / 'sparse').symlink_to(binary, target_is_directory=True)
    global_dir = a.output / 'global'
    global_dir.mkdir()
    scene = global_dir / 'scene.mvs'
    subprocess.run([a.interface_colmap, '-w', str(global_dir), '-i', str(interface), '-o', str(scene),
                    '--image-folder', str(a.image_folder.absolute()), '--binary', '1', '--archive-type', '2', '-v', '2'], check=True)
    import re
    match = re.fullmatch(r'neighbors_diverse_pm(\d+)_top20.txt', a.neighbors_name)
    if not match:
        raise ValueError('Expected a pm<N>_top20 neighbor file')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'data_preprocess/steps/08_build_diverse_neighbors.py'),
                    '--scene', str(scene), '--images-txt', str(final_text / 'images.txt'), '--output', str(global_dir / a.neighbors_name),
                    '--audit', str(global_dir / 'neighbors.json'), '--radius', match[1], '--sources', '20', '--cross-camera-quota', '6'], check=True)
    report = dict(status='PASS', mode=a.mode, image_count=len(deltas), position_sigma_m=a.position_sigma_m,
                  rotation_sigma_deg=a.rotation_sigma_deg, prior_is_soft=True, images=deltas,
                  translation_max_m=max(r['translation_m'] for r in deltas), rotation_max_deg=max(r['rotation_deg'] for r in deltas),
                  source=str(a.output / 'converted'), scene=str(scene))
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}, indent=2))


if __name__ == '__main__':
    main()
