import os
from pathlib import Path


os.environ.setdefault("COLUMNS", "240")


def real_t3_data_available(root: Path) -> bool:
    """Return True only when the optional local T3 real dataset is present."""

    root = Path(root)
    return all(
        path.exists()
        for path in (
            root
            / "calib"
            / "center_camera_fov30"
            / "center_camera_fov30-intrinsic.json",
            root / "camera" / "center_camera_fov30.txt",
            root / "ved" / "LocalLocalization.csv",
        )
    )
