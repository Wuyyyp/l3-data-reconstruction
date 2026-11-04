from __future__ import annotations

import json
import shutil
from pathlib import Path


KNOWN_CAMERA_NAMES = {
    "center_camera_fov30",
    "center_camera_fov120",
    "front_camera_fov120",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_front_camera",
    "right_rear_camera",
}


def migrate_semantic_cache_layout(
    cache_root: Path,
    *,
    remove_legacy: bool = False,
) -> dict[str, int]:
    cache_root = Path(cache_root)
    moved_masks = 0
    moved_reports = 0
    removed_dirs = 0

    for metadata_path in cache_root.glob("*/*/*.json"):
        if "backends" in metadata_path.parts or "reports" in metadata_path.parts:
            continue
        camera = metadata_path.parent.name
        cache_key = metadata_path.parent.parent.name
        if camera not in KNOWN_CAMERA_NAMES:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        backend = str(metadata.get("backend", "mask2former"))
        target_base = cache_root / "backends" / backend / cache_key / camera
        target_mask = target_base / "masks" / f"{metadata_path.stem}.png"
        target_metadata = target_base / "metadata" / metadata_path.name
        target_preview = target_base / "previews" / f"{metadata_path.stem}.jpg"
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        target_metadata.parent.mkdir(parents=True, exist_ok=True)
        target_preview.parent.mkdir(parents=True, exist_ok=True)
        legacy_mask = metadata_path.with_suffix(".png")
        legacy_preview = metadata_path.parent / "previews" / f"{metadata_path.stem}.jpg"
        if legacy_mask.exists() and not target_mask.exists():
            shutil.move(str(legacy_mask), str(target_mask))
            moved_masks += 1
        if not target_metadata.exists():
            shutil.move(str(metadata_path), str(target_metadata))
        if legacy_preview.exists() and not target_preview.exists():
            shutil.move(str(legacy_preview), str(target_preview))

    legacy_reports = list((cache_root / "reports").glob("*.csv"))
    for report_csv in legacy_reports:
        camera = report_csv.stem
        target_root = cache_root / "reports" / camera
        target_root.mkdir(parents=True, exist_ok=True)
        target_csv = target_root / "segment_report.csv"
        if not target_csv.exists():
            shutil.move(str(report_csv), str(target_csv))
            moved_reports += 1
        config_json = report_csv.with_suffix(".config.json")
        target_json = target_root / "config.json"
        if config_json.exists() and not target_json.exists():
            shutil.move(str(config_json), str(target_json))

    for camera in KNOWN_CAMERA_NAMES:
        legacy_stage_report = cache_root / camera / "semantic" / "segment_report.csv"
        target_root = cache_root / "reports" / camera
        target_root.mkdir(parents=True, exist_ok=True)
        target_csv = target_root / "segment_report.csv"
        if legacy_stage_report.exists() and not target_csv.exists():
            shutil.move(str(legacy_stage_report), str(target_csv))
            moved_reports += 1

    if remove_legacy:
        for candidate in cache_root.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name in {"backends", "reports"}:
                continue
            try:
                shutil.rmtree(candidate)
                removed_dirs += 1
            except OSError:
                pass
        reports_root = cache_root / "reports"
        for legacy_file in list(reports_root.glob("*.csv")) + list(
            reports_root.glob("*.config.json")
        ):
            legacy_file.unlink(missing_ok=True)

    return {
        "moved_masks": moved_masks,
        "moved_reports": moved_reports,
        "removed_dirs": removed_dirs,
    }
