import json
import os
from pathlib import Path

from typer.testing import CliRunner

from dongfeng_registration.cli import app
from dongfeng_registration.deliverables import (
    _copy_file,
    _link_or_copy_file,
    export_project_deliverables,
)


runner = CliRunner()


def test_bulk_delivery_file_uses_hardlink_when_supported(tmp_path: Path):
    source = tmp_path / "source.jpg"
    destination = tmp_path / "delivery" / "source.jpg"
    source.write_bytes(b"bulk-artifact")

    method = _link_or_copy_file(source, destination)

    assert method == "hardlink"
    assert destination.stat().st_ino == source.stat().st_ino


def test_copy_file_supports_long_destination_path(tmp_path: Path):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"long-path-copy")
    destination = tmp_path
    for index in range(5):
        destination /= f"relation_{index}_" + ("x" * 32)
    destination /= ("camera_pair_" + ("y" * 72) + ".jpg")
    assert len(str(destination.resolve())) > 260

    _copy_file(source, destination)

    readable_destination = destination
    if os.name == "nt":
        readable_destination = Path("\\\\?\\" + str(destination.resolve()))
    assert readable_destination.read_bytes() == b"long-path-copy"


def test_export_project_deliverables_packages_standard_outputs(tmp_path: Path):
    source = tmp_path / "source"
    optimized_pkl = source / "result_interpolation_optimized.pkl"
    compensated = source / "compensated_perception"
    optimization_output = source / "optimization_output"
    projection_qc = source / "projection_qc"
    deliverable_dir = tmp_path / "deliverables" / "M18_delivery"

    optimized_pkl.parent.mkdir(parents=True)
    optimized_pkl.write_bytes(b"fake-pkl")
    compensated.mkdir()
    (compensated / "000001.pcd").write_text("pcd-1", encoding="utf-8")
    (compensated / "000002.pcd").write_text("pcd-2", encoding="utf-8")
    (compensated / "notes.txt").write_text("not a point cloud", encoding="utf-8")
    (optimization_output / "metrics").mkdir(parents=True)
    (optimization_output / "metrics" / "summary.json").write_text(
        '{"after": {"reprojection_median_px": 0.3}}',
        encoding="utf-8",
    )
    (optimization_output / "observations").mkdir(parents=True)
    (optimization_output / "observations" / "reprojection_outliers.parquet").write_bytes(
        b"outliers"
    )
    (optimization_output / "observations" / "qc_matches.parquet").write_bytes(
        b"qc"
    )
    (optimization_output / "observations" / "qc_sample_manifest.json").write_text(
        '{"sample_ratio": 0.01}',
        encoding="utf-8",
    )
    (optimization_output / "deliverables" / "image_extrinsics").mkdir(parents=True)
    (
        optimization_output
        / "deliverables"
        / "image_extrinsics"
        / "image_camera_to_world_after.csv"
    ).write_text("camera,frame_index\nfront,0\n", encoding="utf-8")
    projection_qc.mkdir()
    (projection_qc / "front_0000.jpg").write_bytes(b"jpg")

    result = export_project_deliverables(
        output_root=deliverable_dir,
        vehicle_model="M18",
        optimized_pkl=optimized_pkl,
        compensated_perception_dir=compensated,
        optimization_output_dir=optimization_output,
        projection_qc_dir=projection_qc,
    )

    assert result.output_root == deliverable_dir
    assert (deliverable_dir / "result_interpolation_optimized.pkl").read_bytes() == b"fake-pkl"
    assert sorted(path.name for path in (deliverable_dir / "compensated_perception").glob("*.pcd")) == [
        "000001.pcd",
        "000002.pcd",
    ]
    assert (deliverable_dir / "image_extrinsics" / "image_camera_to_world_after.csv").exists()
    assert (deliverable_dir / "metrics" / "summary.json").exists()
    assert (
        deliverable_dir / "optimization_diagnostics" / "reprojection_outliers.parquet"
    ).exists()
    assert (deliverable_dir / "optimization_diagnostics" / "qc_matches.parquet").exists()
    assert (
        deliverable_dir / "optimization_diagnostics" / "qc_sample_manifest.json"
    ).exists()
    assert (deliverable_dir / "projection_qc" / "front_0000.jpg").exists()

    manifest = json.loads((deliverable_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vehicle_model"] == "M18"
    assert manifest["pkl"]["file"] == "result_interpolation_optimized.pkl"
    assert manifest["pkl"]["ego_field_for_projection"] == (
        "ego2global_transformation_matrix_camera_reference_offset_optimized"
    )
    assert manifest["copied_counts"]["compensated_perception_pcd"] == 2
    assert manifest["copied_counts"]["optimization_diagnostic_files"] == 3
    assert "optimized pkl" in manifest["required_for_projection"]
    assert "compensated point clouds" in manifest["required_for_projection"]
    assert not manifest["missing_optional_items"]

    readme = (deliverable_dir / "README_delivery.md").read_text(encoding="utf-8")
    assert "ego2global_transformation_matrix_camera_reference_offset_optimized" in readme
    assert "compensated_perception" in readme


def test_export_project_deliverables_cli(tmp_path: Path):
    optimized_pkl = tmp_path / "result_interpolation_optimized.pkl"
    optimized_pkl.write_bytes(b"fake-pkl")

    result = runner.invoke(
        app,
        [
            "export-deliverables",
            "--output-root",
            str(tmp_path / "delivery"),
            "--vehicle-model",
            "L29",
            "--optimized-pkl",
            str(optimized_pkl),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (tmp_path / "delivery" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["vehicle_model"] == "L29"
    assert manifest["pkl"]["file"] == "result_interpolation_optimized.pkl"


def test_export_project_deliverables_removes_source_point_clouds_when_requested(
    tmp_path: Path,
):
    optimized_pkl = tmp_path / "result_interpolation_optimized.pkl"
    compensated = tmp_path / "dynamic" / "compensated_perception"
    optimized_pkl.write_bytes(b"fake-pkl")
    compensated.mkdir(parents=True)
    source_cloud = compensated / "000001.pcd"
    source_cloud.write_text("pcd", encoding="utf-8")
    source_inode = source_cloud.stat().st_ino

    export_project_deliverables(
        output_root=tmp_path / "deliverables",
        optimized_pkl=optimized_pkl,
        compensated_perception_dir=compensated,
        remove_compensated_perception_source=True,
    )

    assert not compensated.exists()
    assert (
        tmp_path / "deliverables" / "compensated_perception" / "000001.pcd"
    ).read_text(encoding="utf-8") == "pcd"
    assert (
        tmp_path / "deliverables" / "compensated_perception" / "000001.pcd"
    ).stat().st_ino == source_inode


def test_export_project_deliverables_reuses_existing_point_clouds_on_repeat_export(
    tmp_path: Path,
):
    optimized_pkl = tmp_path / "result_interpolation_optimized.pkl"
    output_root = tmp_path / "deliverables"
    compensated = output_root / "compensated_perception"
    optimized_pkl.write_bytes(b"updated-pkl")
    compensated.mkdir(parents=True)
    (compensated / "000001.pcd").write_text("pcd", encoding="utf-8")

    export_project_deliverables(
        output_root=output_root,
        optimized_pkl=optimized_pkl,
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied_counts"]["compensated_perception_pcd"] == 1
    assert "compensated_perception/*.pcd" not in manifest["missing_optional_items"]
    assert (compensated / "000001.pcd").read_text(encoding="utf-8") == "pcd"


def test_export_project_deliverables_cli_accepts_removed_source_on_repeat_export(
    tmp_path: Path,
):
    optimized_pkl = tmp_path / "result_interpolation_optimized.pkl"
    output_root = tmp_path / "deliverables"
    compensated = output_root / "compensated_perception"
    missing_source = tmp_path / "dynamic" / "compensated_perception"
    optimized_pkl.write_bytes(b"updated-pkl")
    compensated.mkdir(parents=True)
    (compensated / "000001.pcd").write_text("pcd", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "export-deliverables",
            "--output-root",
            str(output_root),
            "--optimized-pkl",
            str(optimized_pkl),
            "--compensated-perception-dir",
            str(missing_source),
            "--remove-compensated-perception-source",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["copied_counts"]["compensated_perception_pcd"] == 1
    assert (compensated / "000001.pcd").exists()
