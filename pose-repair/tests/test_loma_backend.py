import numpy as np
import pandas as pd

from dongfeng_registration.deep_matchers import loma_matches_from_points


class Calibration:
    model = "rational"
    images_undistorted = True
    intrinsic = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros(8, dtype=np.float64)


class Config:
    deep_score_threshold = 0.0
    lk_mode = "off"


def test_loma_matches_from_points_preserves_schema_and_semantic_filtering():
    first_points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    second_points = np.array([[11.0, 10.0], [21.0, 20.0], [31.0, 30.0]])
    first_mask = np.zeros((50, 50), dtype=np.uint8)
    second_mask = np.zeros((50, 50), dtype=np.uint8)
    second_mask[20, 21] = 255

    matches, diagnostics = loma_matches_from_points(
        first_points,
        second_points,
        Calibration(),
        Config(),
        (50, 50),
        (50, 50),
        first_dynamic_mask=first_mask,
        second_dynamic_mask=second_mask,
    )

    assert isinstance(matches, pd.DataFrame)
    assert matches["backend"].tolist() == ["loma", "loma"]
    assert matches["backend_score"].tolist() == [1.0, 1.0]
    assert diagnostics["semantic_destination_rejected"] == 1
    assert diagnostics["post_semantic_matches"] == 2
    assert np.array_equal(diagnostics["raw_first_points"], first_points)
    assert np.array_equal(diagnostics["raw_second_points"], second_points)


def test_loma_matches_batch_preserves_pair_order(monkeypatch):
    from dongfeng_registration import deep_matchers

    class Config:
        deep_score_threshold = 0.0
        lk_mode = "off"
        model_device = "cpu"
        deep_image_size = 1600

    specs = [
        (
            np.array([[10.0, 10.0]]),
            np.array([[11.0, 10.0]]),
        ),
        (
            np.array([[20.0, 20.0], [30.0, 30.0]]),
            np.array([[21.0, 20.0], [31.0, 30.0]]),
        ),
    ]
    call_index = {"value": 0}

    monkeypatch.setattr(
        deep_matchers,
        "_get_loma_backend_bundle",
        lambda *_args, **_kwargs: {
            "model": object(),
            "model_name": "loma_B",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "device_fallback": False,
        },
    )

    def fake_run(
        bundle,
        first_image,
        second_image,
        scale_first,
        scale_second,
        score_threshold=0.0,
        max_keypoints=None,
    ):
        index = call_index["value"]
        call_index["value"] += 1
        return specs[index]

    monkeypatch.setattr(deep_matchers, "_run_loma_matching", fake_run)

    results = deep_matchers.loma_matches_batch(
        [
            {
                "first_image": np.zeros((50, 50, 3), dtype=np.uint8),
                "second_image": np.zeros((50, 50, 3), dtype=np.uint8),
                "calibration": Calibration(),
                "config": Config(),
            },
            {
                "first_image": np.zeros((50, 50, 3), dtype=np.uint8),
                "second_image": np.zeros((50, 50, 3), dtype=np.uint8),
                "calibration": Calibration(),
                "config": Config(),
            },
        ]
    )

    assert len(results) == 2
    assert results[0][0]["backend"].tolist() == ["loma"]
    assert results[1][0]["backend"].tolist() == ["loma", "loma"]
