from __future__ import annotations

import numpy as np


def summarize_errors(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "over_1px_ratio": float(np.mean(array > 1)),
        "over_2px_ratio": float(np.mean(array > 2)),
        "over_5px_ratio": float(np.mean(array > 5)),
        "over_10px_ratio": float(np.mean(array > 10)),
    }
