from pathlib import Path

import pandas as pd

from scripts.generate_reprojection_qc import _load_reprojection_qc_matches


def test_compact_reprojection_qc_matches_are_loaded_as_preselected(tmp_path: Path):
    observations = tmp_path / "observations"
    observations.mkdir(parents=True)
    compact = pd.DataFrame(
        {
            "match_scope": ["intra_camera"],
            "camera1": ["camera_a"],
            "camera2": ["camera_a"],
            "pair_relation": ["baseline_2m"],
            "frame1": [1],
            "frame2": [2],
            "timestamp1_ns": [100],
            "timestamp2_ns": [200],
            "frame_offset": [1],
            "baseline_target_m": [2.0],
            "x1": [10.0],
            "y1": [20.0],
            "x2": [30.0],
            "y2": [40.0],
            "before_reprojected1_x": [11.0],
            "before_reprojected1_y": [21.0],
            "before_reprojected2_x": [31.0],
            "before_reprojected2_y": [41.0],
            "before_reprojection_rms_px": [2.0],
            "after_reprojected1_x": [10.5],
            "after_reprojected1_y": [20.5],
            "after_reprojected2_x": [30.5],
            "after_reprojected2_y": [40.5],
            "after_reprojection_rms_px": [1.0],
        }
    )
    compact.to_parquet(observations / "qc_matches.parquet", index=False)

    loaded, preselected = _load_reprojection_qc_matches(tmp_path)

    assert preselected is True
    pd.testing.assert_frame_equal(loaded, compact)
