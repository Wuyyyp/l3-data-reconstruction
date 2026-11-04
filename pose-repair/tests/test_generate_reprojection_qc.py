from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_qc_module():
    script = Path(__file__).parents[1] / "scripts" / "generate_reprojection_qc.py"
    spec = spec_from_file_location("generate_reprojection_qc", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_image_paths_prefers_microsecond_filename_for_ns_timestamp():
    qc = _load_qc_module()

    paths = qc._candidate_image_paths(
        Path("images/front_camera_fov120"), 1_766_546_037_201_104_000
    )

    assert paths[0].name == "1766546037201104.jpg"


def test_qc_cli_defaults_to_bounded_tiles_and_parallel_writers(tmp_path: Path):
    qc = _load_qc_module()

    args = qc.build_parser().parse_args(
        [
            "--optimization-output",
            str(tmp_path / "optimization"),
            "--dataset-root",
            str(tmp_path / "dataset"),
        ]
    )

    assert args.tile_width == 1280
    assert args.tile_height == 720
    assert args.workers == 4
    assert args.jpeg_quality == 90
