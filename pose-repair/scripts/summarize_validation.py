from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = root / "outputs" / "real-data-validation"
    rows = []
    for summary_path in validation.glob("window-*/metrics/camera_summary.csv"):
        rows.append(pd.read_csv(summary_path).iloc[0].to_dict())
    summary = pd.DataFrame(rows).sort_values("camera")
    metrics_dir = validation / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / "camera_summary.csv", index=False)
    payload = {
        "camera_count": int(len(summary)),
        "all_have_matches": bool((summary["match_count"] > 0).all()),
        "total_matches": int(summary["match_count"].sum()),
        "total_tracks": int(summary["track_count"].sum()),
        "cameras": summary.to_dict(orient="records"),
    }
    (metrics_dir / "global_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(summary["camera"], summary["epipolar_median"])
    axes[0].set_ylabel("Median symmetric epipolar error (px)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(summary["camera"], summary["reprojection_median"])
    axes[1].set_ylabel("Median reprojection error (px)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=40)
    figure.tight_layout()
    figure.savefig(validation / "camera_comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
