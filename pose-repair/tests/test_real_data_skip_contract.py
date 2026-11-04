from pathlib import Path

from tests.conftest import real_t3_data_available


def test_real_t3_data_available_requires_core_files(tmp_path: Path):
    root = tmp_path / "dataset"

    assert real_t3_data_available(root) is False

    (root / "calib" / "center_camera_fov30").mkdir(parents=True)
    (root / "calib" / "center_camera_fov30" / "center_camera_fov30-intrinsic.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (root / "camera" / "center_camera_fov30").mkdir(parents=True)
    (root / "camera" / "center_camera_fov30.txt").write_text("", encoding="utf-8")
    (root / "ved").mkdir()
    (root / "ved" / "LocalLocalization.csv").write_text("", encoding="utf-8")

    assert real_t3_data_available(root) is True
