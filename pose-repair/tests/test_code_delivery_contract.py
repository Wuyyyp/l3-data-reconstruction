from pathlib import Path

from typer.testing import CliRunner

from dongfeng_registration.cli import app
from dongfeng_registration.deliverables import export_project_deliverables


runner = CliRunner()


def test_gitignore_excludes_non_code_delivery_directories():
    ignored = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())

    assert "data/" in ignored
    assert "data_cache/" in ignored
    assert "outputs/" in ignored
    assert "deliverables/" in ignored
    assert ".models/" in ignored


def test_readme_is_runnable_code_delivery_guide():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "\u4ee3\u7801\u4ea4\u4ed8" in text
    assert "canonical dataset" in text
    assert "run-delivery" in text
    assert "normalize-dataset" in text
    assert "ego2global_transformation_matrix_camera_reference_offset_optimized" in text
    assert "compensated_perception/*.pcd" in text
    assert "data_cache/" in text
    assert "outputs/" in text
    assert "deliverables/" in text
    assert "generate_reprojection_qc.py" in text
    assert "D:\\\\DongFeng\\\\outputs" not in text
    assert "\u7532\u65b9" not in text
    assert "\u5ba2\u6237" not in text
    assert "\u6d60\uff43\uff54" not in text


def test_m18_delivery_config_dry_run_loads_mainline():
    result = runner.invoke(
        app,
        [
            "run-delivery",
            "--config",
            "configs/m18.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "normalize-dataset" in result.output
    assert "compensate_dynamic_raw_lidars.py" in result.output
    assert "segment-dataset" not in result.output
    assert "match-delivery" in result.output
    assert "optimize-multicamera-extrinsics" in result.output
    assert "export-deliverables" in result.output


def test_delivery_readme_uses_readable_chinese_and_projection_contract(tmp_path: Path):
    optimized_pkl = tmp_path / "result_interpolation_optimized.pkl"
    optimized_pkl.write_bytes(b"fake-pkl")

    result = export_project_deliverables(
        output_root=tmp_path / "delivery",
        vehicle_model="M18",
        optimized_pkl=optimized_pkl,
    )

    readme = result.readme_path.read_text(encoding="utf-8")

    assert "\u4ea4\u4ed8\u6570\u636e\u8bf4\u660e" in readme
    assert "\u6295\u5f71\u4f7f\u7528\u7ea6\u5b9a" in readme
    assert "ego2global_transformation_matrix_camera_reference_offset_optimized" in readme
    assert "compensated_perception/*.pcd" in readme
    assert "\u7532\u65b9" not in readme
    assert "\u5ba2\u6237" not in readme
    assert "\u6d60\uff43\uff54" not in readme
