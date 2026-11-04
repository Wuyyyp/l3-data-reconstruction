#!/usr/bin/env python3
"""CPU boundary tests and mocked command-routing tests; no reconstruction launched."""
import contextlib
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mvs_process'))
import delivery_d_far as far
import run_delivery_d as runner


def cloud(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('nx', 'f4')])
    for i, xyz in enumerate(points):
        vertices[i] = (*xyz, i + 1, i / 10)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))
    return vertices


class GeometryTests(unittest.TestCase):
    def test_mask_keeps_uncertain_far_and_boundary_support(self):
        shape = (65, 65)
        depth = np.full(shape, 10.)
        support = np.full(shape, 2, dtype=np.uint8)
        valid = np.ones(shape, bool)
        E = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
        K = np.array([[10000, 0, 32], [0, 10000, 32], [0, 0, 1.]])
        tree = cKDTree([[0, 0]])
        def mask(d=depth, s=support, v=valid, threshold=0., t=tree):
            return far.compute_mask(d, s, v, E, K, threshold, t)
        m = mask()
        self.assertEqual(m[32, 32], 0)
        self.assertTrue((m[:16] == 255).all())
        self.assertTrue((mask(np.full(shape, 80.)) == 255).all())
        self.assertTrue((mask(s=np.ones(shape, np.uint8)) == 255).all())
        self.assertTrue((mask(np.full(shape, np.nan)) == 255).all())
        self.assertTrue((mask(np.full(shape, 30.), threshold=np.log(1.5)) == 255).all())
        self.assertEqual(mask(np.full(shape, 80.), t=cKDTree([[0, 0], [70, 0]]))[32, 32], 0)
        invalid = valid.copy(); invalid[32, 32] = False
        self.assertEqual(mask(np.full(shape, 80.), v=invalid)[32, 32], 0)

    def test_crop_strict_xy_all_positions_and_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloud.ply'
            original = cloud(path, [(50, 0, 1000), (50.1, 0, 0), (200, 0, 0),
                                    (0, 51, -1000), (100, 0, 0), (200, 0, np.nan)])
            raw = path.read_bytes()
            count = far.crop_ply(path, np.array([[0, 0, 0], [100, 0, 0]]), Path(tmp) / 'report.json')
            self.assertEqual(count, 2)
            np.testing.assert_array_equal(PlyData.read(path)['vertex'].data, original[[2, 3]])
            self.assertEqual(path.with_name('cloud_uncropped.ply').read_bytes(), raw)

    def test_trajectory_is_window_local_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp)
            infos = []
            for x in [0, 100, 200]:
                pose = np.eye(4); pose[0, 3] = x
                infos.append(dict(ego2global_transformation_matrix_camera_reference_offset_optimized=pose))
            (clip / 'result_interpolation_optimized.pkl').write_bytes(pickle.dumps(dict(infos=infos)))
            np.testing.assert_array_equal(far.trajectory(clip, 1, 2), [[100, 0, 0], [200, 0, 0]])
            with self.assertRaises(ValueError): far.trajectory(clip, 0, 3)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.clip = self.base / 'clip'; self.clip.mkdir()
        self.output = self.base / 'output'
        poses = []
        for x in [0, 100]:
            pose = np.eye(4); pose[0, 3] = x
            poses.append(dict(ego2global_transformation_matrix_camera_reference_offset_optimized=pose))
        (self.clip / 'result_interpolation_optimized.pkl').write_bytes(pickle.dumps(dict(infos=poses)))
        paths = dict(csv='colmap/poses/image_camera_to_world_after_ba.csv', images='colmap/sparse_ba/images.bin',
                     cameras='colmap/sparse_ba/cameras.bin', points3D='colmap/sparse_ba/points3D.bin', pkl='result_interpolation_optimized.pkl')
        for key, name in paths.items():
            p = self.clip / name; p.parent.mkdir(parents=True, exist_ok=True)
            if key != 'pkl': p.write_bytes(b'fixture')
        self.hashes = {key: hashlib.sha256((self.clip / name).read_bytes()).hexdigest() for key, name in paths.items()}
        self.binary = self.base / 'far_APD'; self.binary.touch()
        self.calls = []
        self.points = [(0, 0, 0), (50, 0, 1000), (100, 0, 0), (200, 0, 0)]

    def prepare(self, enabled):
        root = self.output / self.clip.name / ('f0_1_far50' if enabled else 'f0_1')
        (root / 'prepared/audit').mkdir(parents=True)
        (root / 'prepared/audit/preparation.json').write_text(json.dumps(dict(
            status='PASS', frames=[0, 1], delivery=str(self.clip), ba_model_sha256=self.hashes)))
        scene = root / '09_dvp/scene'; scene.mkdir(parents=True)
        for n in [10, 20]:
            (scene / f'pair{n}.txt').write_text('1\n0\n' + str(n) + ' ' + ' '.join(f'{i} 1' for i in range(1, n + 1)) + '\n')
        (scene / 'pair.txt').write_bytes((scene / 'pair10.txt').read_bytes())
        (root / 'calibration_report.json').write_text('{}')
        return root

    def fake_call(self, command, env, log):
        command = list(map(str, command)); self.calls.append((command, dict(env)))
        root = log.parents[1]
        if command[0] in [str(self.binary), str(Path(os.environ.get('DVP_FULL_BIN',runner.REPO / 'build-cpu-pipeline/APD')))]:
            p = root / '09_dvp/scene/APD/00000000/depths.dmb'; p.parent.mkdir(parents=True); p.touch()
        elif str(runner.PREVIEW) in command:
            p = root / '09_dvp/APD_before_prior_filter/previews/README.txt'; p.parent.mkdir(parents=True); p.touch()
        elif str(runner.FILTER) in command: (root / 'prior_filter_report.json').write_text('{}')
        elif str(runner.CONVERT) in command: (root / '10_openmvs_fusion/conversion_stats.json').write_text('{}')
        elif command[0] == str(runner.MVS / 'DensifyPointCloud'):
            cloud(root / '10_openmvs_fusion/openmvs_filtered_fused.ply', self.points)
        else: self.fail('Unexpected command: ' + str(command))

    def run_case(self, enabled):
        root = self.prepare(enabled)
        with patch.object(runner, 'FAR_BIN', self.binary), patch.object(runner, 'call', self.fake_call), \
             patch.object(far, 'prepare_masks', return_value=root / 'far_only/masks') as masks, \
             patch.dict(os.environ, {'DVP_FAR_MASK_DIR': '/must-not-leak'}):
            result = Path(runner.run_window(self.clip, self.output, 0, 1, 2, far_only=enabled))
            self.assertEqual(result.parent.parent, root)
            self.assertEqual(masks.call_count, int(enabled))
            native = [row for row in self.calls if row[0][0] in [str(self.binary), str(Path(os.environ.get('DVP_FULL_BIN',runner.REPO / 'build-cpu-pipeline/APD')))]]
            self.assertEqual(len(native), 1)
            self.assertEqual(native[0][0][0], str(self.binary if enabled else Path(os.environ.get('DVP_FULL_BIN',runner.REPO / 'build-cpu-pipeline/APD'))))
            self.assertEqual('DVP_FAR_MASK_DIR' in native[0][1], enabled)
            self.assertNotIn('DVP_FAR_MASK_DIR', self.calls[-1][1])
            count = len(self.calls)
            self.assertEqual(runner.run_window(self.clip, self.output, 0, 1, 2, far_only=enabled), str(result))
            self.assertEqual(len(self.calls), count, 'Completed result should resume without commands')
        return root, result

    def test_default_pipeline_and_resume_unchanged(self):
        root, result = self.run_case(False)
        self.assertEqual(runner.ply_count(result), 4)
        self.assertFalse((root / 'far_only').exists())

    def test_enabled_pipeline_crops_and_resumes(self):
        root, result = self.run_case(True)
        self.assertEqual(runner.ply_count(result), 1)
        self.assertEqual(json.loads((root / 'audit/D.complete').read_text())['far_policy'], far.POLICY)

    def test_empty_far_output_is_completed_and_reusable(self):
        self.points = self.points[:3]
        _, result = self.run_case(True)
        self.assertEqual(runner.ply_count(result), 0)

    def test_mixed_mode_refused_before_any_commands(self):
        root = self.prepare(True)
        (root / 'audit').mkdir()
        (root / 'audit/run_config.json').write_text(json.dumps(dict(far_only=False)))
        with patch.object(runner, 'FAR_BIN', self.binary), patch.object(runner, 'call', self.fake_call):
            with self.assertRaisesRegex(ValueError, 'mode/policy'):
                runner.run_window(self.clip, self.output, 0, 1, 0, far_only=True)
        self.assertEqual(self.calls, [])

    def test_changed_input_rejected_before_far_work(self):
        self.prepare(True)
        (self.clip / 'colmap/sparse_ba/images.bin').write_bytes(b'changed')
        with patch.object(runner, 'FAR_BIN', self.binary), patch.object(runner, 'call', self.fake_call):
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                runner.run_window(self.clip, self.output, 0, 1, 0, far_only=True)
        self.assertEqual(self.calls, [])

    def test_cli_default_and_far_switch_reach_every_window(self):
        pkl = self.clip / 'result_interpolation_optimized.pkl'
        info = pickle.loads(pkl.read_bytes())['infos'][0]
        pkl.write_bytes(pickle.dumps(dict(infos=[info] * 203)))
        for enabled, gpus in [(False, '0'), (True, '0'), (True, '0,1')]:
            argv = ['run_delivery_d.py', '--clip-root', str(self.clip),
                    '--output-root', str(self.output), '--gpu-ids', gpus]
            if enabled:
                argv.append('--far-only')
            with patch.object(sys, 'argv', argv), patch.object(Path, 'is_file', return_value=True), \
                 patch.object(runner, 'run_window', return_value='result.ply') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sorted((c.args[2], c.args[3]) for c in run.call_args_list),
                             [(0, 100), (101, 201), (202, 202)])
            self.assertTrue(all(c.args[-1] == enabled for c in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
