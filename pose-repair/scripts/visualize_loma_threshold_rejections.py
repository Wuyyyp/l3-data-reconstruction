from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _load_image_dirs(dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in dataset_root.glob("**/camera/*"):
        if path.is_dir() and any(path.glob("*.jpg")):
            result[path.name] = path
    return result


def _read_image(image_dirs: dict[str, Path], camera: str, timestamp_ns: int) -> np.ndarray:
    path = image_dirs[camera] / f"{int(timestamp_ns)}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _resize_to_common_height(
    first: np.ndarray,
    second: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_height = max(first.shape[0], second.shape[0])

    def resize(image: np.ndarray, points: np.ndarray):
        scale = target_height / image.shape[0]
        target_width = int(round(image.shape[1] * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        transformed = np.asarray(points, dtype=np.float32).copy()
        if len(transformed):
            transformed[:, 0] *= scale
            transformed[:, 1] *= scale
        return resized, transformed

    first_resized, first_transformed = resize(first, first_points)
    second_resized, second_transformed = resize(second, second_points)
    return first_resized, second_resized, first_transformed, second_transformed


def _draw_pair_row(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    *,
    title: str,
    line_color: tuple[int, int, int] | None,
) -> np.ndarray:
    first, second, p1s, p2s = _resize_to_common_height(
        first_image, second_image, first_points, second_points
    )
    canvas = np.hstack([first, second])
    offset = first.shape[1]
    for index, (p1, p2) in enumerate(zip(p1s, p2s)):
        color = line_color
        if color is None:
            value = np.array([[index % 256]], dtype=np.uint8)
            color = tuple(int(x) for x in cv2.applyColorMap(value, cv2.COLORMAP_TURBO)[0, 0])
        a = tuple(np.round(p1).astype(int))
        b = tuple((np.round(p2) + [offset, 0]).astype(int))
        cv2.circle(canvas, a, 1, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 1, color, -1, cv2.LINE_AA)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title[:180],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _match_keep_mask(
    base_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    tolerance_px: float,
) -> np.ndarray:
    if len(base_points) == 0:
        return np.zeros((0,), dtype=bool)
    if len(candidate_points) == 0:
        return np.zeros((len(base_points),), dtype=bool)
    # Four-dimensional coordinate distance: x1, y1, x2, y2.
    diff = base_points[:, None, :] - candidate_points[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    return distance.min(axis=1) <= float(tolerance_px)


def _read_raw(cache_root: Path, label: str, camera: str, baseline: str) -> pd.DataFrame:
    dirs = list(
        (cache_root / f"cache_{label}").glob(
            f"loma__mask2former__*/intra_camera/{camera}/baselines_{baseline}/raw_matches.parquet"
        )
    )
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected one raw cache for {label}, got {dirs}")
    return pd.read_parquet(dirs[0])


def generate_visualizations(
    *,
    cache_root: Path,
    dataset_root: Path,
    output_dir: Path,
    camera: str,
    baseline: str,
    sample_ratio: float,
    tolerance_px: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _read_raw(cache_root, "thr01", camera, baseline)
    higher = {
        "0.3": _read_raw(cache_root, "thr03", camera, baseline),
        "0.5": _read_raw(cache_root, "thr05", camera, baseline),
        "0.7": _read_raw(cache_root, "thr07", camera, baseline),
    }
    image_dirs = _load_image_dirs(dataset_root)
    group_keys = ["frame1", "frame2", "timestamp1_ns", "timestamp2_ns"]
    base_groups = list(base.groupby(group_keys, sort=True))
    if sample_ratio < 1.0:
        count = max(1, int(np.ceil(len(base_groups) * sample_ratio)))
        selected_indices = np.linspace(0, len(base_groups) - 1, count, dtype=int)
        base_groups = [base_groups[int(i)] for i in selected_indices]
    rows: list[dict[str, object]] = []
    for key, group in base_groups:
        frame1, frame2, timestamp1_ns, timestamp2_ns = key
        first_image = _read_image(image_dirs, camera, int(timestamp1_ns))
        second_image = _read_image(image_dirs, camera, int(timestamp2_ns))
        base_points = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        row_images = [
            _draw_pair_row(
                first_image,
                second_image,
                group[["x1", "y1"]].to_numpy(dtype=np.float32),
                group[["x2", "y2"]].to_numpy(dtype=np.float32),
                title=f"threshold 0.1 all matches | {camera} {int(frame1):04d}->{int(frame2):04d} n={len(group)}",
                line_color=None,
            )
        ]
        summary = {
            "frame1": int(frame1),
            "frame2": int(frame2),
            "timestamp1_ns": int(timestamp1_ns),
            "timestamp2_ns": int(timestamp2_ns),
            "base_count": int(len(group)),
        }
        for threshold, candidate in higher.items():
            candidate_group = candidate.loc[
                (candidate["frame1"].astype(int) == int(frame1))
                & (candidate["frame2"].astype(int) == int(frame2))
            ]
            candidate_points = candidate_group[["x1", "y1", "x2", "y2"]].to_numpy(
                dtype=np.float32
            )
            kept = _match_keep_mask(
                base_points,
                candidate_points,
                tolerance_px=tolerance_px,
            )
            rejected = group.loc[~kept]
            summary[f"rejected_by_{threshold}"] = int(len(rejected))
            summary[f"kept_by_{threshold}"] = int(np.count_nonzero(kept))
            row_images.append(
                _draw_pair_row(
                    first_image,
                    second_image,
                    rejected[["x1", "y1"]].to_numpy(dtype=np.float32),
                    rejected[["x2", "y2"]].to_numpy(dtype=np.float32),
                    title=(
                        f"removed by threshold {threshold} vs 0.1 | "
                        f"red rejected n={len(rejected)} kept={np.count_nonzero(kept)}"
                    ),
                    line_color=(0, 0, 255),
                )
            )
        width = max(row.shape[1] for row in row_images)
        padded = []
        for row in row_images:
            if row.shape[1] == width:
                padded.append(row)
            else:
                canvas = np.zeros((row.shape[0], width, 3), dtype=np.uint8)
                canvas[:, : row.shape[1]] = row
                padded.append(canvas)
        rendered = np.vstack(padded)
        path = output_dir / f"{camera}_baseline_{baseline}_{int(frame1):04d}_{int(frame2):04d}_threshold_rejections.jpg"
        cv2.imwrite(str(path), rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        summary["path"] = str(path)
        rows.append(summary)
    index = pd.DataFrame(rows)
    index_path = output_dir / "threshold_rejection_visualization_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="left_front_camera")
    parser.add_argument("--baseline", default="6m")
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--tolerance-px", type=float, default=2.0)
    args = parser.parse_args()
    index = generate_visualizations(
        cache_root=args.cache_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        camera=args.camera,
        baseline=args.baseline,
        sample_ratio=args.sample_ratio,
        tolerance_px=args.tolerance_px,
    )
    print(f"threshold_rejection_index={index}")


if __name__ == "__main__":
    main()
