from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dongfeng_registration.visualization import draw_reprojection_qc_overview


def _find_camera_image_dirs(dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    image_root = dataset_root / "images"
    if image_root.exists():
        for path in sorted(image_root.iterdir()):
            if path.is_dir() and any(path.glob("*.jpg")) and path.name not in result:
                result[path.name] = path
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir() and any(path.glob("*.jpg")) and path.name not in result:
            result[path.name] = path
    if not result:
        raise FileNotFoundError(
            f"no canonical images/<camera> or direct camera image directories found under {dataset_root}"
        )
    return result


def _candidate_image_paths(image_dir: Path, timestamp_ns: int) -> list[Path]:
    timestamp = int(timestamp_ns)
    candidates: list[Path] = []
    if timestamp % 1000 == 0:
        candidates.append(image_dir / f"{timestamp // 1000}.jpg")
    candidates.append(image_dir / f"{timestamp}.jpg")
    return candidates


def _load_image(
    image_dirs: dict[str, Path],
    cache: dict[tuple[str, int], np.ndarray],
    camera: str,
    timestamp_ns: int,
) -> np.ndarray:
    key = (camera, int(timestamp_ns))
    cached = cache.get(key)
    if cached is not None:
        return cached.copy()
    paths = _candidate_image_paths(image_dirs[camera], int(timestamp_ns))
    image = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            break
    if image is None:
        raise FileNotFoundError(paths[0])
    if len(cache) > 32:
        cache.clear()
    cache[key] = image
    return image.copy()


def _safe_relation(value: object) -> str:
    if pd.isna(value):
        return "none"
    return str(value)


def _format_distance_folder(value: object, prefix: str) -> str:
    if pd.isna(value):
        return f"{prefix}_unknown"
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        text = str(int(round(number)))
    else:
        text = f"{number:.3f}".rstrip("0").rstrip(".")
    return f"{prefix}s_{text}m" if prefix == "baseline" else f"{prefix}_{text}m"


def _format_offset_folder(frame_offset: object) -> str:
    value = int(frame_offset)
    return f"offset_{abs(value):03d}"


def _qc_subdir(
    output_dir: Path,
    *,
    scope: str,
    camera1: str,
    camera2: str,
    relation: object,
    baseline_target_m: object,
    frame_offset: object,
) -> Path:
    pair_name = f"{camera1}__{camera2}"
    if scope == "intra_camera":
        baseline_dir = _format_distance_folder(baseline_target_m, "baseline")
        return output_dir / scope / camera1 / baseline_dir
    relation_name = _safe_relation(relation)
    return output_dir / scope / pair_name / relation_name


def _sample_group_keys(groups: pd.core.groupby.DataFrameGroupBy, ratio: float) -> list[tuple]:
    keys = list(groups.groups.keys())
    if not keys:
        return []
    if ratio >= 1.0:
        return keys
    count = max(1, int(math.ceil(len(keys) * max(0.0, ratio))))
    indices = np.linspace(0, len(keys) - 1, count, dtype=int)
    return [keys[int(index)] for index in indices]


def _load_reprojection_qc_matches(
    optimization_output: Path,
) -> tuple[pd.DataFrame, bool]:
    observations = Path(optimization_output) / "observations"
    compact_path = observations / "qc_matches.parquet"
    if compact_path.exists():
        return pd.read_parquet(compact_path), True

    before_path = observations / "before_matches.parquet"
    after_path = observations / "after_matches.parquet"
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(
            "reprojection QC requires observations/qc_matches.parquet or the legacy "
            "before_matches.parquet and after_matches.parquet pair"
        )
    before = pd.read_parquet(before_path)
    after = pd.read_parquet(after_path)
    if len(before) != len(after):
        raise ValueError(
            f"before/after row count mismatch: {len(before)} vs {len(after)}"
        )

    base_columns = [
        "match_scope",
        "camera1",
        "camera2",
        "frame1",
        "frame2",
        "timestamp1_ns",
        "timestamp2_ns",
        "frame_offset",
        "baseline_target_m",
        "pair_relation",
        "x1",
        "y1",
        "x2",
        "y2",
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "reprojection_rms_px",
    ]
    missing = [column for column in base_columns if column not in before.columns]
    if missing:
        raise KeyError(f"before_matches missing columns: {missing}")
    projection_columns = [
        "reprojected1_x",
        "reprojected1_y",
        "reprojected2_x",
        "reprojected2_y",
        "reprojection_rms_px",
    ]
    missing = [column for column in projection_columns if column not in after.columns]
    if missing:
        raise KeyError(f"after_matches missing columns: {missing}")

    frame = before[base_columns].copy()
    frame.rename(
        columns={column: f"before_{column}" for column in projection_columns},
        inplace=True,
    )
    for column in projection_columns:
        frame[f"after_{column}"] = after[column].to_numpy()
    return frame, False


def generate_reprojection_qc(
    *,
    optimization_output: Path,
    dataset_root: Path,
    output_dir: Path | None = None,
    sample_ratio: float = 0.1,
    tile_width: int | None = 1280,
    tile_height: int | None = 720,
    max_matches_per_image: int | None = None,
    workers: int = 4,
    jpeg_quality: int = 90,
) -> Path:
    output_dir = output_dir or (optimization_output / "reprojection_qc")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, preselected = _load_reprojection_qc_matches(optimization_output)
    effective_sample_ratio = 1.0 if preselected else float(sample_ratio)

    image_dirs = _find_camera_image_dirs(dataset_root)
    image_cache: dict[tuple[str, int], np.ndarray] = {}
    group_keys = [
        "match_scope",
        "camera1",
        "camera2",
        "pair_relation",
        "frame1",
        "frame2",
        "timestamp1_ns",
        "timestamp2_ns",
        "frame_offset",
        "baseline_target_m",
    ]
    rows: list[dict[str, object]] = []
    pending: list[tuple[Future, dict[str, object]]] = []
    worker_count = max(1, int(workers))

    def collect_oldest() -> None:
        future, row = pending.pop(0)
        row.update(future.result())
        rows.append(row)

    top_groups = [
        "match_scope",
        "camera1",
        "camera2",
        "pair_relation",
        "baseline_target_m",
    ]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _group_name, relation_frame in frame.groupby(
            top_groups, dropna=False, sort=True
        ):
            relation_grouped = relation_frame.groupby(
                group_keys, dropna=False, sort=True
            )
            relation_groups = list(relation_grouped)
            if effective_sample_ratio < 1.0:
                count = max(
                    1,
                    int(math.ceil(len(relation_groups) * effective_sample_ratio)),
                )
                selected = np.linspace(
                    0, len(relation_groups) - 1, count, dtype=int
                )
                relation_groups = [
                    relation_groups[int(index)] for index in selected
                ]
            for key, group in relation_groups:
                while len(pending) >= worker_count:
                    collect_oldest()
                (
                    scope,
                    camera1,
                    camera2,
                    relation,
                    frame1,
                    frame2,
                    timestamp1_ns,
                    timestamp2_ns,
                    frame_offset,
                    baseline_target_m,
                ) = key
                image1 = _load_image(
                    image_dirs, image_cache, camera1, int(timestamp1_ns)
                )
                image2 = _load_image(
                    image_dirs, image_cache, camera2, int(timestamp2_ns)
                )
                relation_name = _safe_relation(relation)
                pair_name = f"{camera1}__{camera2}"
                baseline_name = (
                    _format_distance_folder(baseline_target_m, "baseline")
                    if scope == "intra_camera"
                    else relation_name
                )
                offset_name = _format_offset_folder(frame_offset)
                title = (
                    f"{scope}/{baseline_name}/{offset_name} {pair_name} "
                    f"{int(frame1):04d}->{int(frame2):04d} "
                    f"offset={int(frame_offset):+d}"
                )
                subdir = _qc_subdir(
                    output_dir,
                    scope=str(scope),
                    camera1=str(camera1),
                    camera2=str(camera2),
                    relation=relation,
                    baseline_target_m=baseline_target_m,
                    frame_offset=frame_offset,
                )
                stem = (
                    f"{pair_name}_{int(frame1):04d}_{int(frame2):04d}"
                    f"_offset_{int(frame_offset):+04d}_qc"
                )
                path = subdir / f"{stem}.jpg"
                future = executor.submit(
                    draw_reprojection_qc_overview,
                    image1,
                    image2,
                    group[["x1", "y1"]].to_numpy(dtype=np.float32),
                    group[["x2", "y2"]].to_numpy(dtype=np.float32),
                    group[
                        ["before_reprojected1_x", "before_reprojected1_y"]
                    ].to_numpy(dtype=np.float32),
                    group[
                        ["after_reprojected1_x", "after_reprojected1_y"]
                    ].to_numpy(dtype=np.float32),
                    group[
                        ["before_reprojected2_x", "before_reprojected2_y"]
                    ].to_numpy(dtype=np.float32),
                    group[
                        ["after_reprojected2_x", "after_reprojected2_y"]
                    ].to_numpy(dtype=np.float32),
                    path,
                    tile_size=(
                        None
                        if tile_width is None or tile_height is None
                        else (int(tile_width), int(tile_height))
                    ),
                    title=title,
                    max_matches=max_matches_per_image,
                    jpeg_quality=int(jpeg_quality),
                )
                pending.append(
                    (
                        future,
                        {
                            "path": str(path),
                            "match_scope": scope,
                            "pair_relation": relation_name,
                            "camera1": camera1,
                            "camera2": camera2,
                            "frame1": int(frame1),
                            "frame2": int(frame2),
                            "frame_offset": int(frame_offset),
                            "baseline_target_m": (
                                None
                                if pd.isna(baseline_target_m)
                                else float(baseline_target_m)
                            ),
                            "qc_group": str(subdir.relative_to(output_dir)),
                            "match_count": int(len(group)),
                            "before_median_px": float(
                                group["before_reprojection_rms_px"].median()
                            ),
                            "after_median_px": float(
                                group["after_reprojection_rms_px"].median()
                            ),
                            "before_p95_px": float(
                                group["before_reprojection_rms_px"].quantile(0.95)
                            ),
                            "after_p95_px": float(
                                group["after_reprojection_rms_px"].quantile(0.95)
                            ),
                        },
                    )
                )
        while pending:
            collect_oldest()

    index = pd.DataFrame(rows)
    index_path = output_dir / "reprojection_qc_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    return index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sampled reprojection QC images.")
    parser.add_argument("--optimization-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\DongFeng"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--tile-width", type=int, default=1280)
    parser.add_argument("--tile-height", type=int, default=720)
    parser.add_argument("--max-matches-per-image", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index_path = generate_reprojection_qc(
        optimization_output=args.optimization_output,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        sample_ratio=args.sample_ratio,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        max_matches_per_image=args.max_matches_per_image,
        workers=args.workers,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"reprojection_qc_index={index_path}")


if __name__ == "__main__":
    main()
