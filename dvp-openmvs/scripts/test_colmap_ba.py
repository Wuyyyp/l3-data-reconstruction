#!/usr/bin/env python3
"""Check pose conversion, input immutability, and the BA switch contract."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mvs_process'))
from run_colmap_ba import camera_to_world, image_records, publish_poses, main as ba_main


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, model, output = root/'source', root/'model', root/'optimized'
        for p in [source/'images', source/'ego_pose', source/'intrinsics', model]:
            p.mkdir(parents=True, exist_ok=True)
        (source/'images/000000_00.jpg').write_bytes(b'original image')
        original = np.eye(4)
        np.savetxt(source/'ego_pose/000000_00.txt', original)
        K = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        np.savetxt(source/'intrinsics/000000_00.txt', K)
        (model/'cameras.txt').write_text('1 PINHOLE 100 80 100 100 50 40\n')
        # 90 degree Z rotation and a camera center away from the origin.
        q = np.sqrt(.5)
        (model/'images.txt').write_text(f'# images\n1 {q} 0 0 {q} 2 -1 -3 1 000000_00.jpg\n\n')
        records = image_records(model/'images.txt')
        assert len(records) == 1 and records[0][1] == ''
        expected = np.eye(4)
        expected[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        expected[:3, 3] = [1, 2, 3]
        assert np.allclose(camera_to_world(records[0][0]), expected)
        deltas = publish_poses(source, model, output)
        assert np.allclose(np.loadtxt(output/'ego_pose/000000_00.txt'), expected)
        assert np.allclose(np.loadtxt(source/'ego_pose/000000_00.txt'), original)
        assert (output/'images').is_symlink()
        assert np.isclose(deltas[0]['rotation_deg'], 90)
        assert np.isclose(deltas[0]['translation_m'], np.sqrt(14))
        # Changed intrinsics must never silently reach DVP/OpenMVS.
        (model/'cameras.txt').write_text('1 PINHOLE 100 80 101 100 50 40\n')
        try:
            publish_poses(source, model, root/'invalid')
        except ValueError as error:
            assert 'intrinsics' in str(error)
        else:
            raise AssertionError('Changed intrinsics were accepted')
    script = ROOT/'mvs_process/run_batch_mvsnet_to_openmvs_ply.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    help_text = subprocess.check_output(['bash', str(script), '--help'], text=True)
    assert '--no-colmap-ba' in help_text and 'disabled by default' in help_text
    text = script.read_text()
    assert 'BA_MODE=${DVP_BA_MODE:-absolute_pose_prior}' in text
    # Explicit enable must require BA tools, even when the environment disables BA.
    result = subprocess.run(['bash', str(script), '--colmap-ba', '--check'], capture_output=True, text=True,
                            env=dict(os.environ, DVP_COLMAP_BA='0', DVP_BA_PYTHON=sys.executable, DVP_COLMAP_BIN='/missing/colmap'))
    assert result.returncode != 0 and '/missing/colmap' in result.stderr
    with tempfile.TemporaryDirectory() as directory:
        sample = Path(directory)/'sample'
        (sample/'03_converted/images').mkdir(parents=True)
        wrong_images = Path(directory)/'wrong_images'
        wrong_images.mkdir()
        argv = ['run_colmap_ba.py', '--sample-root', str(sample), '--output', str(Path(directory)/'ba'),
                '--image-folder', str(wrong_images), '--neighbors-name', 'neighbors_diverse_pm20_top20.txt',
                '--colmap', 'unused', '--prior-colmap', 'unused', '--interface-colmap', 'unused', '--dongfeng-root', str(ROOT)]
        with patch('run_colmap_ba.validate_tools'), patch.object(sys, 'argv', argv):
            try:
                ba_main()
            except ValueError as error:
                assert 'same source images' in str(error)
            else:
                raise AssertionError('Unrelated fusion images were accepted')
        fake = Path(directory)/'APD'
        fake.write_text('#!/bin/sh\n# DVP_SKIP_FUSION DVP_DENSE_MASK_V2\nexit 0\n')
        fake.chmod(0o755)
        disabled = subprocess.run(['bash', str(script), '--no-colmap-ba', '--check', '--from-stage', '2', '--to-stage', '2'],
                                  capture_output=True, text=True, env=dict(os.environ, PYTHON_BIN=sys.executable,
                                  DVP_BIN=str(fake), DVP_BA_COLMAP='/missing/prior', DVP_ITERATION_PROFILE='legacy',
                                  DVP_GPU_IDS='0', DVP_CPU_PIPELINE='0', DVP_COLMAP_BA='1'))
        assert disabled.returncode == 0, disabled.stderr
        assert 'COLMAP BA:      0' in disabled.stdout
        # The default must succeed without any BA tools; the env override enables BA.
        env = dict(os.environ, PYTHON_BIN=sys.executable, DVP_BIN=str(fake),
                   DVP_BA_PYTHON=sys.executable, DVP_COLMAP_BIN='/missing/colmap',
                   DVP_BA_COLMAP='/missing/prior', DVP_ITERATION_PROFILE='legacy',
                   DVP_GPU_IDS='0', DVP_CPU_PIPELINE='0')
        env.pop('DVP_COLMAP_BA', None)
        check = ['bash', str(script), '--check', '--from-stage', '2', '--to-stage', '2']
        default = subprocess.run(check, capture_output=True, text=True, env=env)
        assert default.returncode == 0, default.stderr
        assert 'COLMAP BA:      0' in default.stdout
        enabled = subprocess.run(check, capture_output=True, text=True,
                                 env=dict(env, DVP_COLMAP_BA='1'))
        assert enabled.returncode != 0 and '/missing/colmap' in enabled.stderr
    print('PASS: BA pose propagation, original input preservation, intrinsics guard, default-off switch and fail-closed preflight')


if __name__ == '__main__':
    main()
