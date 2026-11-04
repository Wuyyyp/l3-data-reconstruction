from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .camera_models import CameraCalibration
from .config import load_config
from .evaluation import summarize_errors
from .match_paths import resolve_retained_matches_path
from .navigation import NavigationTrajectory
from .pipeline import _evaluate_navigation


def _retained_subset(matches: pd.DataFrame) -> pd.DataFrame:
    mask = matches["pose_quality_valid"].astype(bool) & matches[
        "reprojection_threshold_inlier"
    ].astype(bool)
    return matches.loc[mask].copy()


def _evaluate_matches_for_time_offset(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    time_offset_ns: int,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for (_, _), group in matches.groupby(["timestamp1_ns", "timestamp2_ns"], sort=False):
        evaluated, _ = _evaluate_navigation(
            group,
            int(group["timestamp1_ns"].iloc[0]) + int(time_offset_ns),
            int(group["timestamp2_ns"].iloc[0]) + int(time_offset_ns),
            calibration,
            navigation,
        )
        evaluated = evaluated.copy()
        evaluated["navigation_time_offset_ns"] = int(time_offset_ns)
        groups.append(evaluated)
    return pd.concat(groups, ignore_index=True) if groups else matches.iloc[0:0].copy()


def scan_navigation_time_offsets(
    matches: pd.DataFrame,
    navigation: NavigationTrajectory,
    calibration: CameraCalibration,
    offsets_ns: np.ndarray,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    retained = _retained_subset(matches)
    for offset_ns in np.asarray(offsets_ns, dtype=np.int64):
        evaluated = _evaluate_matches_for_time_offset(
            retained,
            navigation,
            calibration,
            int(offset_ns),
        )
        results.append(evaluated)
    return pd.concat(results, ignore_index=True) if results else retained.iloc[0:0].copy()


def summarize_time_offset_scan(scan_matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for offset_ns, group in scan_matches.groupby("navigation_time_offset_ns", sort=True):
        sampson = summarize_errors(
            np.sqrt(group["nav_sampson_px2"].to_numpy(dtype=np.float64))
        )
        reprojection = summarize_errors(
            group["reprojection_rms_px"].to_numpy(dtype=np.float64)
        )
        rows.append(
            {
                "time_offset_ns": int(offset_ns),
                "time_offset_ms": int(round(int(offset_ns) / 1_000_000)),
                "count": int(reprojection.get("count", 0)),
                "sampson_mean": float(sampson.get("mean", np.nan)),
                "sampson_median": float(sampson.get("median", np.nan)),
                "sampson_p95": float(sampson.get("p95", np.nan)),
                "reprojection_mean_px": float(reprojection.get("mean", np.nan)),
                "reprojection_median_px": float(reprojection.get("median", np.nan)),
                "reprojection_p95_px": float(reprojection.get("p95", np.nan)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _summarize_by_frame_offset(scan_matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for (offset_ns, frame_offset), group in scan_matches.groupby(
        ["navigation_time_offset_ns", "frame_offset"], sort=True
    ):
        reprojection = summarize_errors(group["reprojection_rms_px"].to_numpy(dtype=np.float64))
        rows.append(
            {
                "time_offset_ns": int(offset_ns),
                "time_offset_ms": int(round(int(offset_ns) / 1_000_000)),
                "frame_offset": int(frame_offset),
                "count": int(reprojection.get("count", 0)),
                "reprojection_median_px": float(reprojection.get("median", np.nan)),
                "reprojection_p95_px": float(reprojection.get("p95", np.nan)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_scan(summary: pd.DataFrame, output: Path) -> None:
    visual_root = output / "visualizations"
    visual_root.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["time_offset_ms"], summary["reprojection_median_px"], marker="o")
    axis.set_xlabel("Navigation time offset (ms)")
    axis.set_ylabel("Median reprojection error (px)")
    axis.set_title("Time offset scan: reprojection median")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(visual_root / "reprojection_median_vs_time_offset.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(summary["time_offset_ms"], summary["sampson_mean"], marker="o")
    axis.set_xlabel("Navigation time offset (ms)")
    axis.set_ylabel("Mean sqrt(Sampson)")
    axis.set_title("Time offset scan: Sampson mean")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(visual_root / "sampson_mean_vs_time_offset.png", dpi=160)
    plt.close(figure)


def _write_report(
    output: Path,
    camera: str,
    summary: pd.DataFrame,
    per_offset: pd.DataFrame,
    offset_values_ms: list[int],
) -> Path:
    best = summary.sort_values("reprojection_median_px", ascending=True).iloc[0]
    path = output / f"{camera}_time_offset_scan_report.md"
    lines = [
        f"# {camera} Navigation time offset scan",
        "",
        f"- Scanned offsets (ms): {offset_values_ms}",
        f"- Best offset by reprojection median: {int(best['time_offset_ms'])} ms",
        f"- Best reprojection median: {float(best['reprojection_median_px']):.6f} px",
        f"- Best reprojection P95: {float(best['reprojection_p95_px']):.6f} px",
        f"- Best Sampson mean: {float(best['sampson_mean']):.6f}",
        "",
        "## Overall summary",
        "",
        "| offset ms | count | reprojection median px | reprojection p95 px | Sampson mean |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.time_offset_ms} | {row.count} | {row.reprojection_median_px:.6f} | "
            f"{row.reprojection_p95_px:.6f} | {row.sampson_mean:.6f} |"
        )
    if not per_offset.empty:
        lines += [
            "",
            "## By frame offset",
            "",
            "| nav offset ms | frame offset | count | reprojection median px | reprojection p95 px |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in per_offset.itertuples(index=False):
            lines.append(
                f"| {row.time_offset_ms} | {row.frame_offset} | {row.count} | "
                f"{row.reprojection_median_px:.6f} | {row.reprojection_p95_px:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_time_offset_scan_from_retained_matches(
    config_path: Path,
    retained_matches_input: Path,
    output: Path,
    camera: str = "center_camera_fov30",
    scan_from_ms: int = -300,
    scan_to_ms: int = 300,
    step_ms: int = 10,
) -> Path:
    config = load_config(Path(config_path))
    calibration = CameraCalibration.from_dataset(config.dataset_root, camera)
    navigation = NavigationTrajectory.from_csv(
        config.dataset_root / config.assumptions["navigation_source"]
    )
    matches = pd.read_parquet(
        resolve_retained_matches_path(Path(retained_matches_input))
    )
    output = Path(output)
    for relative in ("metrics", "observations", "visualizations"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    offsets_ms = list(range(int(scan_from_ms), int(scan_to_ms) + 1, int(step_ms)))
    offsets_ns = np.array(offsets_ms, dtype=np.int64) * 1_000_000
    scan_matches = scan_navigation_time_offsets(matches, navigation, calibration, offsets_ns)
    summary = summarize_time_offset_scan(scan_matches)
    per_offset = _summarize_by_frame_offset(scan_matches)

    scan_matches.to_parquet(output / "observations" / "time_offset_scan_matches.parquet", index=False)
    summary.to_csv(output / "metrics" / "time_offset_summary.csv", index=False)
    per_offset.to_csv(output / "metrics" / "time_offset_by_frame_offset.csv", index=False)
    (output / "metrics" / "time_offset_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    (output / "resolved_time_offset_scan.yaml").write_text(
        yaml.safe_dump(
            {
                "camera": camera,
                "scan_from_ms": int(scan_from_ms),
                "scan_to_ms": int(scan_to_ms),
                "step_ms": int(step_ms),
                "offsets_ms": offsets_ms,
                "retained_matches_input": str(retained_matches_input),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _plot_scan(summary, output)
    _write_report(output, camera, summary, per_offset, offsets_ms)
    return output

