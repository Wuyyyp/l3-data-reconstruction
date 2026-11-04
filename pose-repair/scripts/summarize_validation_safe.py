from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _draw_chart(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1800, 900
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cameras = summary["camera"].tolist()
    epipolar = pd.to_numeric(summary["epipolar_median"], errors="coerce").fillna(0).to_numpy()
    reprojection = pd.to_numeric(summary["reprojection_median"], errors="coerce").fillna(0).to_numpy()
    for row, (title, values, color) in enumerate(
        [
            ("Median symmetric epipolar error (px)", epipolar, (230, 120, 40)),
            ("Median two-view reprojection error (px)", reprojection, (50, 160, 70)),
        ]
    ):
        top, bottom = 80 + row * 420, 390 + row * 420
        cv2.putText(canvas, title, (80, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        maximum = max(float(np.max(values)), 1e-9)
        slot = (width - 160) / len(cameras)
        for index, (camera, value) in enumerate(zip(cameras, values)):
            x0 = int(80 + index * slot + slot * 0.15)
            x1 = int(80 + (index + 1) * slot - slot * 0.15)
            y1 = bottom
            y0 = int(bottom - (value / maximum) * 270)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
            cv2.putText(
                canvas, f"{value:.2f}", (x0, max(top, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1,
            )
            label = camera.replace("_camera", "")
            cv2.putText(
                canvas, label[:18], (x0, bottom + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
            )
    cv2.imwrite(str(path), canvas)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = root / "outputs" / "real-data-validation"
    rows = [
        pd.read_csv(path).iloc[0].to_dict()
        for path in validation.glob("window-*/metrics/camera_summary.csv")
    ]
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
    _draw_chart(summary, validation / "camera_comparison.png")


if __name__ == "__main__":
    main()
