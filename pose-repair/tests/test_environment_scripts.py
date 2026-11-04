from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("script_name", ["bootstrap.ps1", "install_env.ps1"])
def test_environment_script_installs_local_loma(script_name: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name

    assert ".tools\\loma" in script.read_text(encoding="utf-8")


def test_openmvs_build_script_is_pinned_headless_and_self_verifying() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_openmvs.sh"

    text = script.read_text(encoding="utf-8")

    assert "v2.4.0" in text
    assert "OPENMVS_ENABLE_CUDA:-OFF" in text
    assert "OpenMVS_BUILD_VIEWER=OFF" in text
    assert "OpenMVS_ENABLE_TESTS=OFF" in text
    assert "--target InterfaceCOLMAP DensifyPointCloud" in text
    assert "tools/openmvs" in text
    assert 'interface_help=$("$install_dir/bin/InterfaceCOLMAP" --help 2>&1 || true)' in text
    assert 'densify_help=$("$install_dir/bin/DensifyPointCloud" --help 2>&1 || true)' in text
    assert "grep -E 'input-file|input path|COLMAP'" in text
    assert "grep -E 'resolution-level|number-views'" in text
    assert 'bash "$repo_root/scripts/build_eigen.sh"' in text
    assert 'bash "$repo_root/scripts/build_cgal.sh"' in text
    assert '-DEigen3_DIR="$eigen3_dir"' in text
    assert '-DCGAL_DIR="$cgal_dir"' in text
    assert 'bash "$repo_root/scripts/build_nanoflann.sh"' in text
    assert 'bash "$repo_root/scripts/build_vcg.sh"' in text
    assert '-Dnanoflann_DIR="$nanoflann_dir"' in text
    assert '-DVCG_ROOT="$vcg_install_dir"' in text


def test_openmvs_build_script_supports_isolated_cuda_builds() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/build_openmvs.sh").read_text(encoding="utf-8")

    assert 'enable_cuda=${OPENMVS_ENABLE_CUDA:-OFF}' in text
    assert 'cuda_architectures=${OPENMVS_CUDA_ARCHITECTURES:-native}' in text
    assert 'cuda_home=${CUDA_HOME:-/usr/local/cuda}' in text
    assert '-DOpenMVS_USE_CUDA="$enable_cuda"' in text
    assert '-DCMAKE_CUDA_ARCHITECTURES="$cuda_architectures"' in text
    assert '-DCMAKE_CUDA_COMPILER="$cuda_compiler"' in text
    assert "CUDA build requested but nvcc was not found" in text
    assert 'install_dir=${OPENMVS_INSTALL_DIR:-"$repo_root/tools/openmvs"}' in text


def test_openmvs_source_dependencies_are_pinned_and_project_local() -> None:
    root = Path(__file__).resolve().parents[1]
    nanoflann_script = root / "scripts/build_nanoflann.sh"
    vcg_script = root / "scripts/build_vcg.sh"

    nanoflann_text = nanoflann_script.read_text(encoding="utf-8")
    vcg_text = vcg_script.read_text(encoding="utf-8")

    assert "v1.6.3" in nanoflann_text
    assert "tools/nanoflann" in nanoflann_text
    assert "nanoflannConfig.cmake" in nanoflann_text
    assert "658ba36d0a5666650da6e066b4794efc5a463407" in vcg_text
    assert "tools/vcg" in vcg_text
    assert "vcg/complex/complex.h" in vcg_text
    ignore_text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/tools/nanoflann/" in ignore_text
    assert "/tools/vcg/" in ignore_text


def test_cgal_build_script_is_pinned_and_used_by_openmvs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_cgal.sh"
    openmvs_script = root / "scripts/build_openmvs.sh"

    text = script.read_text(encoding="utf-8")
    openmvs_text = openmvs_script.read_text(encoding="utf-8")

    assert "v6.0.1" in text
    assert "tools/cgal" in text
    assert "CGALConfig.cmake" in text
    assert 'bash "$repo_root/scripts/build_cgal.sh"' in openmvs_text
    assert '-DCGAL_DIR="$cgal_dir"' in openmvs_text
    assert "/tools/cgal/" in (root / ".gitignore").read_text(encoding="utf-8")


def test_boost_build_script_is_pinned_verified_and_used_by_openmvs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_boost.sh"
    cgal_script = root / "scripts/build_cgal.sh"
    openmvs_script = root / "scripts/build_openmvs.sh"

    text = script.read_text(encoding="utf-8")
    cgal_text = cgal_script.read_text(encoding="utf-8")
    openmvs_text = openmvs_script.read_text(encoding="utf-8")

    assert "1.84.0" in text
    assert "cc4b893acf645c9d4b698e9a0f08ca8846aa5d6c68275c14c3e7949c24109454" in text
    assert "tools/boost" in text
    assert "sha256sum --check" in text
    assert 'partial_archive="${archive}.partial"' in text
    assert "--continue-at -" in text
    assert "--retry-all-errors" not in text
    assert "for attempt in $(seq 1 10)" in text
    assert 'mv "$partial_archive" "$archive"' in text
    assert '-sZSTD_INCLUDE="$zstd_include_dir"' in text
    assert '-sZSTD_LIBPATH="$zstd_lib_dir"' in text
    assert "zstd_base" in text
    assert "  -a \\\n" in text
    assert "grep -q 'zstd_base'" not in text
    assert "grep 'zstd_base' >/dev/null" in text
    assert 'bash "$repo_root/scripts/build_boost.sh"' in cgal_text
    assert '-DBOOST_ROOT="$boost_install_dir"' in cgal_text
    assert 'bash "$repo_root/scripts/build_boost.sh"' in openmvs_text
    assert '-DBOOST_ROOT="$boost_install_dir"' in openmvs_text
    assert "/tools/boost/" in (root / ".gitignore").read_text(encoding="utf-8")


def test_openmvs_build_applies_optional_jpegxl_patch() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/build_openmvs.sh").read_text(encoding="utf-8")
    patch = root / "third_party/openmvs_optional_jpegxl_2.4.0.patch"

    patch_text = patch.read_text(encoding="utf-8")

    assert "pkg_check_modules(${PREFIX} QUIET IMPORTED_TARGET" in patch_text
    assert "openmvs_optional_jpegxl_2.4.0.patch" in script
    assert 'git -C "$source_dir" apply "$jpegxl_patch_file"' in script


def test_openmvs_optional_jpegxl_patch_has_valid_git_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = root / "third_party/openmvs_optional_jpegxl_2.4.0.patch"

    result = subprocess.run(
        ["git", "apply", "--numstat", str(patch)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "libs/IO/CMakeLists.txt" in result.stdout


def test_openmvs_build_applies_ubuntu_20_04_source_compat_patch() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/build_openmvs.sh").read_text(encoding="utf-8")
    patch = root / "third_party/openmvs_ubuntu_20_04_compat_2.4.0.patch"

    patch_text = patch.read_text(encoding="utf-8")

    assert "#ifndef __has_builtin" in patch_text
    assert "CV_VERSION_MINOR >= 12" in patch_text
    assert "openmvs_ubuntu_20_04_compat_2.4.0.patch" in script
    assert 'git -C "$source_dir" apply "$ubuntu_compat_patch_file"' in script


def test_openmvs_ubuntu_20_04_compat_patch_has_valid_git_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = root / "third_party/openmvs_ubuntu_20_04_compat_2.4.0.patch"

    result = subprocess.run(
        ["git", "apply", "--numstat", str(patch)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "libs/Common/Config.h" in result.stdout
    assert "libs/Common/Types.inl" in result.stdout


def test_openmvs_ubuntu_dependency_installer_lists_required_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/install_openmvs_ubuntu_dependencies.sh"

    text = script.read_text(encoding="utf-8")

    for package in (
        "libboost-iostreams-dev",
        "libboost-program-options-dev",
        "libboost-serialization-dev",
        "libboost-system-dev",
        "libcgal-dev",
        "libjpeg-dev",
        "libopencv-dev",
        "libpng-dev",
        "libtiff-dev",
        "libzstd-dev",
    ):
        assert package in text


def test_absolute_pose_prior_colmap_build_is_cpu_headless_and_self_verifying() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_colmap_absolute_pose_prior.sh"

    text = script.read_text(encoding="utf-8")

    assert "be5e29168d4aff238409d60424812df66aac919f" in text
    assert "-DCUDA_ENABLED=OFF" in text
    assert "-DGUI_ENABLED=OFF" in text
    assert "-DTESTS_ENABLED=OFF" in text
    assert "-DCGAL_ENABLED=OFF" in text
    assert "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF" in text
    assert "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF" in text
    assert (
        'install -m 0644 "$repo_root/cmake/FindOpenImageIO.cmake" '
        '"$source_dir/cmake/FindOpenImageIO.cmake"'
    ) in text
    assert 'bash "$repo_root/scripts/build_eigen.sh"' in text
    assert '-DEigen3_DIR="$eigen3_dir"' in text
    assert "absolute_pose_prior_bundle_adjuster" in text


def test_openimageio_compat_module_defines_expected_imported_target() -> None:
    module = Path(__file__).resolve().parents[1] / "cmake/FindOpenImageIO.cmake"

    text = module.read_text(encoding="utf-8")

    assert "find_path(OpenImageIO_INCLUDE_DIR" in text
    assert "find_library(OpenImageIO_LIBRARY" in text
    assert "OpenImageIO::OpenImageIO" in text


def test_colmap_build_applies_openimageio_2_1_compat_patch() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/build_colmap_absolute_pose_prior.sh").read_text(
        encoding="utf-8"
    )
    patch = root / "third_party/colmap_openimageio_2_1_compat_4.2.0.patch"

    patch_text = patch.read_text(encoding="utf-8")

    assert "OIIO_MAKE_VERSION" in patch_text
    assert "10000" in patch_text
    assert "src/colmap/util/oiio_utils.h" in patch_text
    assert "colmap_openimageio_2_1_compat_4.2.0.patch" in script
    assert 'git -C "$source_dir" apply "$oiio_compat_patch_file"' in script
    assert '"$source_dir/src/colmap/util/oiio_utils.h"' in script


def test_colmap_openimageio_compat_patch_has_valid_git_syntax() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = root / "third_party/colmap_openimageio_2_1_compat_4.2.0.patch"

    result = subprocess.run(
        ["git", "apply", "--numstat", str(patch)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "src/colmap/util/oiio_utils.h" in result.stdout


def test_generated_openmvs_install_is_ignored() -> None:
    ignore_file = Path(__file__).resolve().parents[1] / ".gitignore"

    assert "/tools/openmvs/" in ignore_file.read_text(encoding="utf-8")


def test_eigen_build_script_is_pinned_and_project_local() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_eigen.sh"

    text = script.read_text(encoding="utf-8")

    assert "3.4.0" in text
    assert "tools/eigen" in text
    assert "Eigen3Config.cmake" in text
    assert "/tools/eigen/" in (root / ".gitignore").read_text(encoding="utf-8")


def test_ceres_build_script_is_pinned_and_used_by_colmap() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/build_ceres.sh"
    colmap_script = root / "scripts/build_colmap_absolute_pose_prior.sh"

    text = script.read_text(encoding="utf-8")
    colmap_text = colmap_script.read_text(encoding="utf-8")

    assert "2.2.0" in text
    assert "tools/ceres" in text
    assert 'bash "$repo_root/scripts/build_eigen.sh"' in text
    assert '-DEigen3_DIR="$eigen3_dir"' in text
    assert "-DBUILD_TESTING=OFF" in text
    assert "-DBUILD_EXAMPLES=OFF" in text
    assert "-DUSE_CUDA=OFF" in text
    assert "CeresConfig.cmake" in text
    assert "/tools/ceres/" in (root / ".gitignore").read_text(encoding="utf-8")
    assert 'bash "$repo_root/scripts/build_ceres.sh"' in colmap_text
    assert '-DCeres_DIR="$ceres_dir"' in colmap_text


@pytest.mark.parametrize(
    "script_name",
    [
        "build_colmap_absolute_pose_prior.sh",
        "build_openmvs.sh",
        "build_eigen.sh",
        "build_ceres.sh",
        "build_boost.sh",
        "build_cgal.sh",
        "build_nanoflann.sh",
        "build_vcg.sh",
        "install_openmvs_ubuntu_dependencies.sh",
    ],
)
def test_linux_build_scripts_use_lf_line_endings(script_name: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name

    assert b"\r\n" not in script.read_bytes()


@pytest.mark.parametrize(
    "script_name",
    [
        "build_colmap_absolute_pose_prior.sh",
        "build_openmvs.sh",
        "build_eigen.sh",
        "build_ceres.sh",
        "build_boost.sh",
        "build_cgal.sh",
        "build_nanoflann.sh",
        "build_vcg.sh",
    ],
)
def test_linux_build_scripts_exclude_inherited_windows_path(script_name: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name

    text = script.read_text(encoding="utf-8")

    assert "LINUX_BUILD_PATH" in text
    assert "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in text


def test_colmap_source_patch_uses_lf_line_endings() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = root / "third_party/colmap_absolute_pose_prior_4.2.0.patch"

    assert b"\r\n" not in patch.read_bytes()
    assert "*.patch text eol=lf" in (root / ".gitattributes").read_text(encoding="utf-8")
