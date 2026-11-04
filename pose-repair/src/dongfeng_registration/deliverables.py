from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any


DEFAULT_OPTIMIZED_EGO_FIELD = (
    "ego2global_transformation_matrix_camera_reference_offset_optimized"
)


@dataclass(frozen=True)
class DeliveryExportResult:
    output_root: Path
    manifest_path: Path
    readme_path: Path


def _extended_filesystem_path(path: Path) -> Path:
    path = Path(path)
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return Path(resolved)
    if resolved.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + resolved[2:])
    return Path("\\\\?\\" + resolved)


def _copy_file(src: Path, dst: Path) -> None:
    source_path = _extended_filesystem_path(src)
    destination_path = _extended_filesystem_path(dst)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _link_or_copy_file(src: Path, dst: Path) -> str:
    source_path = _extended_filesystem_path(src)
    destination_path = _extended_filesystem_path(dst)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        try:
            if os.path.samefile(source_path, destination_path):
                return "existing"
        except OSError:
            pass
        source_stat = source_path.stat()
        destination_stat = destination_path.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return "existing"
        destination_path.unlink()
    try:
        os.link(source_path, destination_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "copy"


def _copy_tree_files(src_dir: Path, dst_dir: Path, *, pattern: str = "*") -> int:
    count = 0
    if not src_dir.exists():
        return count
    for src in src_dir.rglob(pattern):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        _link_or_copy_file(src, dst_dir / rel)
        count += 1
    return count


def _move_tree_if_possible(
    src_dir: Path,
    dst_dir: Path,
    *,
    pattern: str,
) -> tuple[int, bool]:
    source_files = [path for path in src_dir.rglob(pattern) if path.is_file()]
    all_files = [path for path in src_dir.rglob("*") if path.is_file()]
    if not source_files or len(source_files) != len(all_files) or dst_dir.exists():
        return 0, False
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(_extended_filesystem_path(src_dir), _extended_filesystem_path(dst_dir))
    except OSError:
        return 0, False
    return len(source_files), True


def _optional_source_record(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _write_manifest(
    *,
    output_root: Path,
    vehicle_model: str,
    pkl_name: str,
    pkl_kind: str,
    ego_field: str,
    sources: dict[str, str | None],
    copied_counts: dict[str, int],
    missing_optional_items: list[str],
) -> Path:
    manifest: dict[str, Any] = {
        "vehicle_model": vehicle_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        "sources": sources,
        "pkl": {
            "kind": pkl_kind,
            "file": pkl_name,
            "ego_field_for_projection": ego_field,
            "camera_extrinsic_path": "infos[*].sensors.cams.<camera>.extrinsic",
        },
        "point_clouds": {
            "directory": "compensated_perception",
            "pattern": "*.pcd",
        },
        "copied_counts": copied_counts,
        "missing_optional_items": missing_optional_items,
        "required_for_projection": [
            "optimized pkl" if pkl_kind == "optimized" else "standard pkl",
            "compensated point clouds",
            "use pkl camera extrinsics",
            f"use ego field: {ego_field}",
        ],
    }
    path = output_root / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_readme(
    *,
    output_root: Path,
    vehicle_model: str,
    pkl_name: str,
    pkl_kind: str,
    ego_field: str,
    has_compensated_point_clouds: bool,
    has_image_extrinsics: bool,
    has_metrics: bool,
    has_optimization_diagnostics: bool,
    has_projection_qc: bool,
) -> Path:
    pkl_label = "优化后的 pkl" if pkl_kind == "optimized" else "标准化 pkl"
    point_cloud_line = (
        "- `compensated_perception/*.pcd`：动态目标补偿后的 perception 融合点云。\n"
        if has_compensated_point_clouds
        else "- 未随包提供 `compensated_perception/*.pcd`；如需点云投影，请另行指定补偿后的点云目录。\n"
    )
    image_extrinsics_line = (
        "- `image_extrinsics/`：逐图像 camera-to-world 外参 CSV。\n"
        if has_image_extrinsics
        else ""
    )
    metrics_line = "- `metrics/`：优化前后重投影误差等评估指标。\n" if has_metrics else ""
    diagnostics_line = (
        "- `optimization_diagnostics/`：异常匹配明细和 1% 重投影 QC 样本。\n"
        if has_optimization_diagnostics
        else ""
    )
    qc_line = "- `projection_qc/`：点云投影或重投影 QC 可视化。\n" if has_projection_qc else ""
    text = f"""# {vehicle_model} 交付数据说明

本目录由项目标准输出整理生成，用于交付核验。

## 文件内容

- `{pkl_name}`：{pkl_label}。
{point_cloud_line}{image_extrinsics_line}{metrics_line}{diagnostics_line}{qc_line}- `manifest.json`：交付物来源、字段约定和文件数量清单。

## 投影使用约定

做点云投影到图像时，建议使用：

- ego 位姿字段：`{ego_field}`
- 相机安装外参：`infos[*].sensors.cams.<camera>.extrinsic`
- 点云：`compensated_perception/*.pcd`

不要覆盖原始 `ego2global_transformation_matrix` 字段；该字段用于回溯原始输入。
"""
    path = output_root / "README_delivery.md"
    path.write_text(text, encoding="utf-8")
    return path

def export_project_deliverables(
    *,
    output_root: Path,
    vehicle_model: str = "unknown",
    optimized_pkl: Path | None = None,
    standard_pkl: Path | None = None,
    compensated_perception_dir: Path | None = None,
    optimization_output_dir: Path | None = None,
    projection_qc_dir: Path | None = None,
    image_extrinsics_dir: Path | None = None,
    metrics_dir: Path | None = None,
    ego_field: str = DEFAULT_OPTIMIZED_EGO_FIELD,
    remove_compensated_perception_source: bool = False,
) -> DeliveryExportResult:
    """Package standardized project outputs into a stable delivery tree.

    When requested, remove the dynamic compensation point-cloud source after its
    contents have been copied successfully.  This keeps the final deliverable as
    the sole retained copy in an end-to-end delivery run.
    """

    if optimized_pkl is None and standard_pkl is None:
        raise ValueError("optimized_pkl or standard_pkl must be provided")
    if optimized_pkl is not None and standard_pkl is not None:
        raise ValueError("provide only one of optimized_pkl or standard_pkl")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pkl_source = Path(optimized_pkl if optimized_pkl is not None else standard_pkl)
    if not pkl_source.exists():
        raise FileNotFoundError(pkl_source)
    pkl_kind = "optimized" if optimized_pkl is not None else "standard"
    pkl_name = (
        "result_interpolation_optimized.pkl"
        if pkl_kind == "optimized"
        else "result_interpolation_standard.pkl"
    )
    _copy_file(pkl_source, output_root / pkl_name)

    optimization_output_dir = (
        Path(optimization_output_dir) if optimization_output_dir is not None else None
    )
    image_extrinsics_source = (
        Path(image_extrinsics_dir)
        if image_extrinsics_dir is not None
        else (
            optimization_output_dir / "deliverables" / "image_extrinsics"
            if optimization_output_dir is not None
            else None
        )
    )
    metrics_source = (
        Path(metrics_dir)
        if metrics_dir is not None
        else (optimization_output_dir / "metrics" if optimization_output_dir is not None else None)
    )
    optimization_observations_source = (
        optimization_output_dir / "observations"
        if optimization_output_dir is not None
        else None
    )

    copied_counts = {
        "pkl": 1,
        "compensated_perception_pcd": 0,
        "image_extrinsics_files": 0,
        "metrics_files": 0,
        "optimization_diagnostic_files": 0,
        "projection_qc_files": 0,
    }
    missing_optional_items: list[str] = []

    compensated_destination = output_root / "compensated_perception"
    compensated_source_moved = False
    if compensated_perception_dir is not None and Path(compensated_perception_dir).exists():
        compensated_source = Path(compensated_perception_dir)
        if remove_compensated_perception_source:
            (
                copied_counts["compensated_perception_pcd"],
                compensated_source_moved,
            ) = _move_tree_if_possible(
                compensated_source,
                compensated_destination,
                pattern="*.pcd",
            )
        if not compensated_source_moved:
            copied_counts["compensated_perception_pcd"] = _copy_tree_files(
                compensated_source,
                compensated_destination,
                pattern="*.pcd",
            )
    else:
        copied_counts["compensated_perception_pcd"] = sum(
            1 for path in compensated_destination.rglob("*.pcd") if path.is_file()
        )
    if copied_counts["compensated_perception_pcd"] == 0:
        missing_optional_items.append("compensated_perception/*.pcd")

    if image_extrinsics_source is not None:
        copied_counts["image_extrinsics_files"] = _copy_tree_files(
            image_extrinsics_source,
            output_root / "image_extrinsics",
        )
        if copied_counts["image_extrinsics_files"] == 0:
            missing_optional_items.append("image_extrinsics")
    else:
        missing_optional_items.append("image_extrinsics")

    if metrics_source is not None:
        copied_counts["metrics_files"] = _copy_tree_files(
            metrics_source,
            output_root / "metrics",
        )
        if copied_counts["metrics_files"] == 0:
            missing_optional_items.append("metrics")
    else:
        missing_optional_items.append("metrics")

    if optimization_observations_source is not None:
        diagnostic_names = (
            "reprojection_outliers.parquet",
            "qc_matches.parquet",
            "qc_sample_manifest.json",
        )
        for name in diagnostic_names:
            source = optimization_observations_source / name
            if not source.exists():
                continue
            _copy_file(source, output_root / "optimization_diagnostics" / name)
            copied_counts["optimization_diagnostic_files"] += 1
        if copied_counts["optimization_diagnostic_files"] == 0:
            missing_optional_items.append("optimization_diagnostics")
    elif optimization_output_dir is not None:
        missing_optional_items.append("optimization_diagnostics")

    if projection_qc_dir is not None:
        copied_counts["projection_qc_files"] = _copy_tree_files(
            Path(projection_qc_dir),
            output_root / "projection_qc",
        )
        if copied_counts["projection_qc_files"] == 0:
            missing_optional_items.append("projection_qc")
    else:
        missing_optional_items.append("projection_qc")

    if (
        remove_compensated_perception_source
        and not compensated_source_moved
        and compensated_perception_dir is not None
        and Path(compensated_perception_dir).exists()
        and copied_counts["compensated_perception_pcd"] > 0
    ):
        compensated_source = Path(compensated_perception_dir)
        compensated_destination = output_root / "compensated_perception"
        if compensated_source.resolve() == compensated_destination.resolve():
            raise ValueError(
                "compensated_perception source and delivery destination must differ"
            )
        shutil.rmtree(compensated_source)

    manifest_path = _write_manifest(
        output_root=output_root,
        vehicle_model=vehicle_model,
        pkl_name=pkl_name,
        pkl_kind=pkl_kind,
        ego_field=ego_field,
        sources={
            "optimized_pkl": _optional_source_record(optimized_pkl),
            "standard_pkl": _optional_source_record(standard_pkl),
            "compensated_perception_dir": _optional_source_record(
                compensated_perception_dir
            ),
            "optimization_output_dir": _optional_source_record(optimization_output_dir),
            "image_extrinsics_dir": _optional_source_record(image_extrinsics_source),
            "metrics_dir": _optional_source_record(metrics_source),
            "optimization_observations_dir": _optional_source_record(
                optimization_observations_source
            ),
            "projection_qc_dir": _optional_source_record(projection_qc_dir),
        },
        copied_counts=copied_counts,
        missing_optional_items=missing_optional_items,
    )
    readme_path = _write_readme(
        output_root=output_root,
        vehicle_model=vehicle_model,
        pkl_name=pkl_name,
        pkl_kind=pkl_kind,
        ego_field=ego_field,
        has_compensated_point_clouds=copied_counts["compensated_perception_pcd"] > 0,
        has_image_extrinsics=copied_counts["image_extrinsics_files"] > 0,
        has_metrics=copied_counts["metrics_files"] > 0,
        has_optimization_diagnostics=(
            copied_counts["optimization_diagnostic_files"] > 0
        ),
        has_projection_qc=copied_counts["projection_qc_files"] > 0,
    )
    update_colmap_delivery_index(output_root)
    return DeliveryExportResult(
        output_root=output_root,
        manifest_path=manifest_path,
        readme_path=readme_path,
    )



def update_colmap_delivery_index(output_root: Path) -> None:
    """Index the colocated BA result without changing the base PKL or pose CSVs."""
    output_root = Path(output_root)
    ba_root = output_root / "colmap-ba"
    if not (ba_root / "STATUS.json").is_file():
        return
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["colmap_ba"] = {
            "directory": "colmap-ba",
            "status_file": "colmap-ba/STATUS.json",
            "report": "colmap-ba/report.json",
            "pose_csv": "colmap-ba/image_camera_to_world_after_ba.csv",
            "applied_to_pkl": False,
            "scope_source": "colmap-ba/report.json:selected_frames",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path = output_root / "README_delivery.md"
    if readme_path.is_file():
        text = readme_path.read_text(encoding="utf-8")
        if "## COLMAP BA" not in text:
            readme_path.write_text(text + "\n## COLMAP BA\n\n"
                "`colmap-ba/` 保存 BA 模型、BA 后外参、位姿改变量、日志和耗时。\n"
                "先检查 `STATUS.json`；成功后按 `report.json` 中的 selected_frames 确认覆盖范围。\n"
                "基础 PKL 和 image_extrinsics/ 不回写 BA；BA 后位姿见 "
                "`colmap-ba/image_camera_to_world_after_ba.csv`。\n", encoding="utf-8")
