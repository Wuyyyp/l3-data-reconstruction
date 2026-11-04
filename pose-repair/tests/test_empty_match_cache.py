from pathlib import Path
import pandas as pd
import pytest
from dongfeng_registration.extrinsic_optimization import _collect_multicamera_match_cache


def collect(root):
    return _collect_multicamera_match_cache(root, ['cam'], None, [2, 4], [], None)


def write(root, baseline, frame):
    path = root / 'intra_camera' / 'cam' / f'baselines_{baseline}m' / 'retained_matches.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_empty_relation_does_not_block_valid_matches(tmp_path):
    write(tmp_path, 2, pd.DataFrame())
    write(tmp_path, 4, pd.DataFrame({'camera': ['cam'], 'value': [123]}))
    result = collect(tmp_path)
    assert result['value'].tolist() == [123]
    assert result['camera1'].tolist() == ['cam']


def test_all_empty_relations_raise_no_matches(tmp_path):
    write(tmp_path, 2, pd.DataFrame())
    with pytest.raises(FileNotFoundError, match='no retained matches'):
        collect(tmp_path)


@pytest.mark.parametrize('frame, message', [
    (pd.DataFrame({'value': [123]}), 'matches must contain'),
    (pd.DataFrame({'camera': ['wrong']}), 'outside directory pair'),
])
def test_nonempty_invalid_matches_are_still_rejected(tmp_path, frame, message):
    write(tmp_path, 2, frame)
    with pytest.raises(ValueError, match=message):
        collect(tmp_path)
