import importlib.util
from pathlib import Path
import pickle
import sys


_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "compensate_dynamic_perception_pcd.py"
)
_SCRIPTS_DIR = _MODULE_PATH.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "dongfeng_registration_compensate_dynamic_perception_pcd",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
compensation_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compensation_script)

_SWEEP_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "visualize_dynamic_offset_compensation_sweep.py"
)
_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "dongfeng_registration_visualize_dynamic_offset_compensation_sweep",
    _SWEEP_MODULE_PATH,
)
assert _SWEEP_SPEC is not None and _SWEEP_SPEC.loader is not None
offset_sweep_script = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(offset_sweep_script)

_RAW_COMP_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "compensate_dynamic_raw_lidars.py"
)
_RAW_COMP_SPEC = importlib.util.spec_from_file_location(
    "dongfeng_registration_compensate_dynamic_raw_lidars",
    _RAW_COMP_MODULE_PATH,
)
assert _RAW_COMP_SPEC is not None and _RAW_COMP_SPEC.loader is not None
raw_compensation_script = importlib.util.module_from_spec(_RAW_COMP_SPEC)
_RAW_COMP_SPEC.loader.exec_module(raw_compensation_script)


def test_resolve_sensor_file_accepts_canonical_relative_paths(tmp_path: Path):
    dataset_root = tmp_path / "canonical"
    pcd = dataset_root / "lidar" / "perception" / "000001.pcd"
    image = dataset_root / "images" / "front_camera_fov30" / "000001.jpg"
    pcd.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    pcd.write_text("pcd", encoding="utf-8")
    image.write_bytes(b"jpg")

    assert compensation_script.resolve_sensor_file(
        dataset_root,
        "lidar/perception/000001.pcd",
        "perception",
    ) == pcd
    assert compensation_script.resolve_sensor_file(
        dataset_root,
        "images/front_camera_fov30/000001.jpg",
        "front_camera_fov30",
    ) == image


def test_parse_offset_output_specs_sorts_offsets(tmp_path: Path):
    specs = offset_sweep_script.parse_offset_output_specs(
        [
            f"10={tmp_path / 'offset10'}",
            f"0={tmp_path / 'offset0'}",
            f"7={tmp_path / 'offset7'}",
        ]
    )

    assert [row[0] for row in specs] == [0.0, 7.0, 10.0]
    assert [row[1].name for row in specs] == ["offset0", "offset7", "offset10"]


def test_offset_sweep_uses_raw_lidar_source_points_for_same_order_qc():
    assert hasattr(offset_sweep_script, "original_xyz_for_offset_qc")


def test_raw_lidar_compensation_resolver_accepts_canonical_raw_lidar_paths(
    tmp_path: Path,
):
    dataset_root = tmp_path / "canonical"
    lidar_path = (
        dataset_root
        / "lidar"
        / "raw_lidars"
        / "front_lidar"
        / "000001.pcd"
    )
    lidar_path.parent.mkdir(parents=True)
    lidar_path.write_text("pcd", encoding="utf-8")

    resolved = raw_compensation_script.resolve_lidar_file(
        dataset_root,
        "front_lidar",
        {"data_path": "lidar/raw_lidars/front_lidar/000001.pcd"},
    )

    assert resolved == lidar_path


def test_raw_lidar_compensation_requires_canonical_standard_pkl(tmp_path: Path):
    raw_pkl = tmp_path / "result_interpolation.pkl"
    with raw_pkl.open("wb") as handle:
        pickle.dump({"infos": []}, handle)

    try:
        raw_compensation_script.load_canonical_infos(raw_pkl)
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("raw result_interpolation.pkl should be rejected")

    standard_pkl = tmp_path / "result_interpolation_standard.pkl"
    with standard_pkl.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "canonical_dataset": True,
                    "standardized_source": {
                        "pose_backend": "result_interpolation_pkl",
                    },
                },
                "infos": [],
            },
            handle,
        )

    assert raw_compensation_script.load_canonical_infos(standard_pkl) == []
