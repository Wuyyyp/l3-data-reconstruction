import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dongfeng_registration import colmap_reconstruction as colmap
from dongfeng_registration.deliverables import export_project_deliverables


def test_ba_csv_contains_absolute_camera_to_world_not_deltas(tmp_path):
    records = [dict(camera='camera_b', frame=5, name='camera_b/200.jpg'),
               dict(camera='camera_a', frame=4, name='camera_a/100.jpg')]
    poses = {r['name']: np.eye(4) for r in records}
    poses['camera_a/100.jpg'][:3, 3] = [123, -45, 6]
    poses['camera_b/200.jpg'][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    path = tmp_path/'poses.csv'
    colmap._export_ba_pose_csv(path, records, poses)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['camera'], int(r['frame'])) for r in rows] == [('camera_a', 4), ('camera_b', 5)]
    for row in rows:
        actual = np.array([[float(row[f'T_cw_{i}{j}']) for j in range(4)] for i in range(4)])
        np.testing.assert_allclose(actual, poses[f"{row['camera']}/{row['image_filename']}"])


def test_export_keeps_ba_index_and_original_pkl_on_repeat(tmp_path):
    source = tmp_path/'source.pkl'; source.write_bytes(b'base-pose-pkl')
    output = tmp_path/'delivery'
    ba = output/'colmap-ba'; ba.mkdir(parents=True)
    (ba/'STATUS.json').write_text('{"status":"SUCCESS"}')
    (ba/'report.json').write_text('{"coverage":"partial","selected_frames":[0,1,2]}')
    for _ in range(2):
        export_project_deliverables(output_root=output, optimized_pkl=source)
        manifest = json.loads((output/'manifest.json').read_text())
        assert manifest['colmap_ba']['pose_csv'] == 'colmap-ba/image_camera_to_world_after_ba.csv'
        assert manifest['colmap_ba']['applied_to_pkl'] is False
        assert (output/'result_interpolation_optimized.pkl').read_bytes() == source.read_bytes()
        assert (output/'README_delivery.md').read_text().count('## COLMAP BA') == 1
    assert json.loads((ba/'report.json').read_text())['selected_frames'] == [0,1,2]


def test_failed_rerun_does_not_keep_success_report(tmp_path, monkeypatch):
    delivery = tmp_path/'delivery'; output = delivery/'colmap-ba'; output.mkdir(parents=True)
    for name in ('report.json','colmap_reconstruction_report.json'):
        (output/name).write_text('{"status":"SUCCESS"}')
    cfg = SimpleNamespace(output_root=output, delivery_root=delivery, run_bundle_adjustment=True)
    monkeypatch.setattr(colmap, 'load_colmap_config', lambda *a, **kw: cfg)
    def fail(config):
        raise ValueError('bad input')
    monkeypatch.setattr(colmap, '_run_colmap_reconstruction', fail)
    with pytest.raises(ValueError, match='bad input'):
        colmap.run_colmap_reconstruction(tmp_path/'config.yaml')
    assert not (output/'report.json').exists()
    assert not (output/'colmap_reconstruction_report.json').exists()
    assert json.loads((output/'STATUS.json').read_text())['status'] == 'FAILED'
    assert json.loads((output/'timing_summary.json').read_text())['status'] == 'FAILED'


def test_failed_command_keeps_log_returncode_and_timing(tmp_path):
    with pytest.raises(RuntimeError, match='exited 7'):
        colmap._run_logged_command([sys.executable, '-c', 'print("solver failed"); raise SystemExit(7)'], output_root=tmp_path, stage='colmap_ba')
    assert 'solver failed' in (tmp_path/'logs/colmap_ba.log').read_text()
    stage = json.loads((tmp_path/'timing_summary.json').read_text())['stages']['colmap_ba']
    assert stage['returncode'] == 7 and stage['seconds'] > 0


def test_standalone_config_defaults_to_colmap_ba_inside_delivery(tmp_path):
    config = tmp_path/'config.yaml'
    config.write_text('canonical_root: canonical\nsemantic_cache_root: masks\ndelivery_root: delivery\ncolmap_executable: colmap\n')
    loaded = colmap.load_colmap_config(config)
    assert loaded.output_root == tmp_path/'delivery/colmap-ba'
    overridden = colmap.load_colmap_config(config, output_root=tmp_path/'explicit')
    assert overridden.output_root == tmp_path/'explicit'
