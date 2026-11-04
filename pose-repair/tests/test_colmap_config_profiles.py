from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = tuple(sorted((ROOT / "configs").glob("*.yaml")))
WINDOWS_EXECUTABLE = "D:/DongFeng/Code/tools/colmap_pose_prior/bin/colmap.exe"
LINUX_EXECUTABLE = "../tools/colmap_pose_prior/bin/colmap"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda path: path.name)
def test_shipped_colmap_profiles_default_to_absolute_pose_prior(
    config_path: Path,
) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    colmap = payload.get("colmap", payload)

    assert colmap["run_bundle_adjustment"] is True
    assert colmap["ba_mode"] == "absolute_pose_prior"
    assert colmap["ba_position_prior_sigma_m"] == pytest.approx(0.1)
    assert colmap["ba_rotation_prior_sigma_deg"] == pytest.approx(0.5)
    expected_executable = (
        WINDOWS_EXECUTABLE if "windows" in config_path.stem else LINUX_EXECUTABLE
    )
    assert colmap["absolute_pose_prior_colmap_executable"] == expected_executable


@pytest.mark.parametrize(
    "config_path",
    [
        path
        for path in CONFIGS
        if path.name not in {"t3_openmvs_smoke.yaml", "t3_openmvs_cuda_smoke.yaml"}
    ],
    ids=lambda path: path.name,
)
def test_shipped_vehicle_profiles_default_to_colmap_ba_without_dense(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["colmap"]["mvs_backend"] == "colmap"
    assert payload["colmap"]["enabled"] is True
    assert payload["colmap"]["run_bundle_adjustment"] is True
    assert payload["colmap"]["run_dense"] is False


def test_t3_openmvs_smoke_profile_is_bounded_and_project_local() -> None:
    config_path = ROOT / "configs/t3_openmvs_smoke.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["mvs_backend"] == "openmvs"
    assert 0 < payload["frame_count"] <= 10
    assert payload["run_dense"] is True
    assert payload["run_bundle_adjustment"] is True
    assert payload["absolute_pose_prior_colmap_executable"] == LINUX_EXECUTABLE
    assert payload["openmvs_interface_colmap_executable"] == "../tools/openmvs/bin/InterfaceCOLMAP"
    assert payload["openmvs_densify_point_cloud_executable"] == (
        "../tools/openmvs/bin/DensifyPointCloud"
    )
    assert "sky" in payload["colmap_dense_mask_classes"]
    assert "car" in payload["colmap_dense_mask_classes"]


def test_t3_openmvs_cuda_smoke_profile_uses_isolated_gpu_tools() -> None:
    config_path = ROOT / "configs/t3_openmvs_cuda_smoke.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["mvs_backend"] == "openmvs"
    assert 0 < payload["frame_count"] <= 10
    assert payload["openmvs_gpu_device"] == "0"
    assert payload["openmvs_interface_colmap_executable"] == (
        "../tools/openmvs-cuda/bin/InterfaceCOLMAP"
    )
    assert payload["openmvs_densify_point_cloud_executable"] == (
        "../tools/openmvs-cuda/bin/DensifyPointCloud"
    )
