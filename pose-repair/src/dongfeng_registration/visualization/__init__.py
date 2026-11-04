from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "draw_match_overlay",
    "draw_tiled_match_overlay",
    "draw_epipolar_overlay",
    "draw_reprojection_overlay",
    "draw_tiled_reprojection_overlay",
    "draw_tiled_bidirectional_reprojection_overlay",
    "draw_tiled_before_after_reprojection_overlay",
    "draw_ransac_retained_reprojection_qc",
    "resize_with_padding",
    "draw_reprojection_qc_overview",
    "plot_error_over_time",
]


def _save(
    path: Path,
    image: np.ndarray,
    *,
    jpeg_quality: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path
    if os.name == "nt":
        output_path = Path("\\\\?\\" + str(path.resolve()))
    suffix = path.suffix or ".jpg"
    encode_params: list[int] = []
    if suffix.lower() in {".jpg", ".jpeg"} and jpeg_quality is not None:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(suffix, image, encode_params)
    if not ok:
        raise OSError(f"Failed to encode visualization image as {suffix}: {path}")
    output_path.write_bytes(encoded.tobytes())


def _color_from_index(index: int) -> tuple[int, int, int]:
    return tuple(
        int(x)
        for x in cv2.applyColorMap(
            np.array([[index % 256]], dtype=np.uint8), cv2.COLORMAP_TURBO
        )[0, 0]
    )


def _cv_point_or_none(
    point: np.ndarray | list[float] | tuple[float, float],
    image_shape: tuple[int, ...],
    *,
    limit_factor: float = 100.0,
) -> tuple[int, int] | None:
    values = np.asarray(point, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        return None
    height, width = image_shape[:2]
    limit = max(float(width), float(height), 1.0) * float(limit_factor)
    if np.max(np.abs(values[:2])) > limit:
        return None
    rounded = np.round(values[:2])
    return (int(rounded[0]), int(rounded[1]))


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] == width:
            padded.append(row)
            continue
        canvas = np.zeros((row.shape[0], width, 3), dtype=np.uint8)
        canvas[:, : row.shape[1]] = row
        padded.append(canvas)
    return np.vstack(padded)


def resize_with_padding(
    image: np.ndarray,
    points: np.ndarray | None = None,
    *,
    target_size: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int]]:
    """Resize an image into a fixed canvas while preserving aspect ratio.

    ``target_size`` is ``(width, height)``. Point coordinates are transformed
    with the same scale and padding, so overlays stay geometrically aligned.
    """

    target_width, target_height = target_size
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive width and height")
    scale = min(target_width / width, target_height / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    pad_x = int((target_width - resized_width) // 2)
    pad_y = int((target_height - resized_height) // 2)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    transformed = None
    if points is not None:
        transformed = np.asarray(points, dtype=np.float32).copy()
        transformed[:, 0] = transformed[:, 0] * scale + pad_x
        transformed[:, 1] = transformed[:, 1] * scale + pad_y
    return canvas, transformed, {
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
    }


def resize_to_height(
    image: np.ndarray,
    points: np.ndarray | None = None,
    *,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int]]:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive width and height")
    scale = target_height / height
    target_width = int(round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    transformed = None
    if points is not None:
        transformed = np.asarray(points, dtype=np.float32).copy()
        transformed[:, 0] *= scale
        transformed[:, 1] *= scale
    return resized, transformed, {
        "scale": float(scale),
        "target_width": int(target_width),
        "target_height": int(target_height),
    }


def _draw_title(canvas: np.ndarray, text: str, *, y: int = 28) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text[:180],
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_reprojection_delta(
    canvas: np.ndarray,
    observed: np.ndarray,
    before_reprojected: np.ndarray,
    after_reprojected: np.ndarray,
) -> None:
    before_color = (0, 0, 255)
    after_color = (0, 255, 0)
    observed_color = (255, 255, 255)
    coordinate_limit = 1_000_000.0
    valid = (
        np.isfinite(observed).all(axis=1)
        & np.isfinite(before_reprojected).all(axis=1)
        & np.isfinite(after_reprojected).all(axis=1)
        & (np.abs(observed) < coordinate_limit).all(axis=1)
        & (np.abs(before_reprojected) < coordinate_limit).all(axis=1)
        & (np.abs(after_reprojected) < coordinate_limit).all(axis=1)
    )
    for obs, before, after in zip(
        observed[valid], before_reprojected[valid], after_reprojected[valid]
    ):
        obs_point = tuple(np.round(obs).astype(int))
        before_point = tuple(np.round(before).astype(int))
        after_point = tuple(np.round(after).astype(int))
        cv2.line(canvas, obs_point, before_point, before_color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas, before_point, before_color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_8
        )
        cv2.line(canvas, obs_point, after_point, after_color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            after_point,
            after_color,
            cv2.MARKER_TILTED_CROSS,
            7,
            1,
            cv2.LINE_8,
        )
        cv2.circle(canvas, obs_point, 3, observed_color, 1, cv2.LINE_8)


def draw_reprojection_qc_overview(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    before_reprojected_first: np.ndarray,
    after_reprojected_first: np.ndarray,
    before_reprojected_second: np.ndarray,
    after_reprojected_second: np.ndarray,
    path: Path,
    *,
    tile_size: tuple[int, int] | None = (1280, 720),
    title: str = "reprojection QC",
    max_matches: int | None = None,
    jpeg_quality: int = 90,
) -> dict[str, int | float]:
    """Draw one QC image containing matches and bidirectional reprojection.

    Layout:
    top-left A image, top-right B image: retained match lines;
    bottom-left A image: B->A reprojection red(before)/green(after);
    bottom-right B image: A->B reprojection red(before)/green(after).
    """

    first_points = np.asarray(first_points, dtype=np.float32)
    second_points = np.asarray(second_points, dtype=np.float32)
    before_reprojected_first = np.asarray(before_reprojected_first, dtype=np.float32)
    after_reprojected_first = np.asarray(after_reprojected_first, dtype=np.float32)
    before_reprojected_second = np.asarray(before_reprojected_second, dtype=np.float32)
    after_reprojected_second = np.asarray(after_reprojected_second, dtype=np.float32)
    total = len(first_points)
    if max_matches is not None and total > max_matches:
        selected = np.linspace(0, total - 1, max_matches, dtype=int)
        first_points = first_points[selected]
        second_points = second_points[selected]
        before_reprojected_first = before_reprojected_first[selected]
        after_reprojected_first = after_reprojected_first[selected]
        before_reprojected_second = before_reprojected_second[selected]
        after_reprojected_second = after_reprojected_second[selected]

    all_first = np.vstack([first_points, before_reprojected_first, after_reprojected_first])
    all_second = np.vstack([second_points, before_reprojected_second, after_reprojected_second])
    if tile_size is None:
        target_height = max(first_image.shape[0], second_image.shape[0])
        first_tile, first_all, first_meta = resize_to_height(
            first_image, all_first, target_height=target_height
        )
        second_tile, second_all, second_meta = resize_to_height(
            second_image, all_second, target_height=target_height
        )
    else:
        first_tile, first_all, first_meta = resize_with_padding(
            first_image, all_first, target_size=tile_size
        )
        second_tile, second_all, second_meta = resize_with_padding(
            second_image, all_second, target_size=tile_size
        )
    n = len(first_points)
    first_obs = first_all[:n]
    first_before = first_all[n : 2 * n]
    first_after = first_all[2 * n :]
    second_obs = second_all[:n]
    second_before = second_all[n : 2 * n]
    second_after = second_all[2 * n :]

    top = np.hstack([first_tile.copy(), second_tile.copy()])
    for index, (first, second) in enumerate(zip(first_obs, second_obs)):
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            continue
        color = _color_from_index(index)
        p1 = tuple(np.round(first).astype(int))
        p2 = tuple((np.round(second) + [first_tile.shape[1], 0]).astype(int))
        cv2.circle(top, p1, 1, color, -1, cv2.LINE_AA)
        cv2.circle(top, p2, 1, color, -1, cv2.LINE_AA)
        cv2.line(top, p1, p2, color, 1, cv2.LINE_AA)
    _draw_title(top, f"{title} | matches n={n} | red=before green=after")

    bottom_left = first_tile.copy()
    _draw_reprojection_delta(bottom_left, first_obs, first_before, first_after)
    _draw_title(bottom_left, "B -> A reprojection on image A")

    bottom_right = second_tile.copy()
    _draw_reprojection_delta(bottom_right, second_obs, second_before, second_after)
    _draw_title(bottom_right, "A -> B reprojection on image B")
    bottom = np.hstack([bottom_left, bottom_right])

    rendered = np.vstack([top, bottom])
    _save(path, rendered, jpeg_quality=jpeg_quality)
    return {
        "tile_width": int(max(first_tile.shape[1], second_tile.shape[1])),
        "tile_height": int(max(first_tile.shape[0], second_tile.shape[0])),
        "first_width": int(first_tile.shape[1]),
        "first_height": int(first_tile.shape[0]),
        "second_width": int(second_tile.shape[1]),
        "second_height": int(second_tile.shape[0]),
        "canvas_width": int(rendered.shape[1]),
        "canvas_height": int(rendered.shape[0]),
        "match_count": int(n),
        "first_scale": float(first_meta["scale"]),
        "second_scale": float(second_meta["scale"]),
    }


def draw_match_overlay(first_image, second_image, first_points, second_points, path: Path):
    height = max(first_image.shape[0], second_image.shape[0])
    width = first_image.shape[1] + second_image.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: first_image.shape[0], : first_image.shape[1]] = first_image
    canvas[: second_image.shape[0], first_image.shape[1] :] = second_image
    offset = first_image.shape[1]
    for index, (first, second) in enumerate(zip(first_points[:150], second_points[:150])):
        color = _color_from_index(index)
        p1 = tuple(np.round(first).astype(int))
        p2 = tuple((np.round(second) + [offset, 0]).astype(int))
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.circle(canvas, p2, 4, color, -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
    _save(path, canvas)


def draw_tiled_match_overlay(
    first_image,
    second_image,
    first_points,
    second_points,
    path: Path,
    max_per_row: int = 150,
    line_color: tuple[int, int, int] | None = None,
    title_prefix: str = "matches",
):
    first_points = np.asarray(first_points, dtype=np.float32)
    second_points = np.asarray(second_points, dtype=np.float32)
    layouts: list[dict[str, int]] = []
    rows: list[np.ndarray] = []
    total = len(first_points)
    for row_index, start in enumerate(range(0, total, max_per_row)):
        end = min(total, start + max_per_row)
        canvas_height = max(first_image.shape[0], second_image.shape[0])
        canvas_width = first_image.shape[1] + second_image.shape[1]
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[: first_image.shape[0], : first_image.shape[1]] = first_image
        canvas[: second_image.shape[0], first_image.shape[1] :] = second_image
        offset = first_image.shape[1]
        for index, (first, second) in enumerate(
            zip(first_points[start:end], second_points[start:end]), start=start
        ):
            color = line_color if line_color is not None else _color_from_index(index)
            p1 = tuple(np.round(first).astype(int))
            p2 = tuple((np.round(second) + [offset, 0]).astype(int))
            cv2.circle(canvas, p1, 4, color, -1)
            cv2.circle(canvas, p2, 4, color, -1)
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"row {row_index + 1}: {title_prefix} {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(canvas)
        layouts.append({"row": row_index, "start_index": start, "end_index": end})
    rendered = _stack_rows(rows)
    _save(path, rendered)
    return layouts




def draw_epipolar_overlay(first_image, second_image, first_points, second_points, fundamental, path):
    canvas = second_image.copy()
    lines = cv2.computeCorrespondEpilines(
        np.asarray(first_points, dtype=np.float32).reshape(-1, 1, 2), 1, fundamental
    ).reshape(-1, 3)
    width = canvas.shape[1]
    for index, (line, point) in enumerate(zip(lines[:100], second_points[:100])):
        color = _color_from_index(index)
        a, b, c = line
        if abs(b) > 1e-12:
            y0, y1 = int(round(-c / b)), int(round(-(c + a * (width - 1)) / b))
            cv2.line(canvas, (0, y0), (width - 1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, color, -1)
    _save(path, canvas)


def draw_reprojection_overlay(image, observed, reprojected, path):
    canvas = image.copy()
    errors = np.linalg.norm(np.asarray(reprojected) - np.asarray(observed), axis=1)
    maximum = max(float(np.percentile(errors, 95)) if len(errors) else 1.0, 1e-9)
    for obs, rep, error in zip(observed[:150], reprojected[:150], errors[:150]):
        color_value = np.uint8(np.clip(error / maximum * 255, 0, 255))
        color = tuple(
            int(x)
            for x in cv2.applyColorMap(
                np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_TURBO
            )[0, 0]
        )
        p1, p2 = tuple(np.round(obs).astype(int)), tuple(np.round(rep).astype(int))
        cv2.circle(canvas, p1, 5, color, 1)
        cv2.drawMarker(canvas, p2, color, cv2.MARKER_CROSS, 8, 1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
    _save(path, canvas)


def draw_tiled_reprojection_overlay(
    image,
    observed,
    reprojected,
    path: Path,
    max_per_row: int = 150,
):
    observed = np.asarray(observed, dtype=np.float32)
    reprojected = np.asarray(reprojected, dtype=np.float32)
    errors = np.linalg.norm(reprojected - observed, axis=1) if len(observed) else np.array([])
    maximum = max(float(np.percentile(errors, 95)) if len(errors) else 1.0, 1e-9)
    layouts: list[dict[str, int]] = []
    rows: list[np.ndarray] = []
    total = len(observed)
    for row_index, start in enumerate(range(0, total, max_per_row)):
        end = min(total, start + max_per_row)
        canvas = image.copy()
        for obs, rep, error in zip(
            observed[start:end], reprojected[start:end], errors[start:end]
        ):
            color_value = np.uint8(np.clip(error / maximum * 255, 0, 255))
            color = tuple(
                int(x)
                for x in cv2.applyColorMap(
                    np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_TURBO
                )[0, 0]
            )
            p1 = tuple(np.round(obs).astype(int))
            p2 = tuple(np.round(rep).astype(int))
            cv2.circle(canvas, p1, 5, color, 1)
            cv2.drawMarker(canvas, p2, color, cv2.MARKER_CROSS, 8, 1)
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"row {row_index + 1}: reprojection {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(canvas)
        layouts.append({"row": row_index, "start_index": start, "end_index": end})
    rendered = _stack_rows(rows)
    _save(path, rendered)
    return layouts


def draw_tiled_bidirectional_reprojection_overlay(
    first_image,
    second_image,
    observed_first,
    reprojected_first,
    observed_second,
    reprojected_second,
    path: Path,
    max_per_row: int = 150,
):
    observed_first = np.asarray(observed_first, dtype=np.float32)
    reprojected_first = np.asarray(reprojected_first, dtype=np.float32)
    observed_second = np.asarray(observed_second, dtype=np.float32)
    reprojected_second = np.asarray(reprojected_second, dtype=np.float32)
    # pad images to same width for stacking
    max_width = max(first_image.shape[1], second_image.shape[1])
    _pad = lambda img: np.pad(
        img,
        ((0, 0), (0, max_width - img.shape[1]), (0, 0)),
        mode="constant",
        constant_values=0,
    ) if img.shape[1] < max_width else img
    first_image = _pad(first_image)
    second_image = _pad(second_image)
    first_errors = (
        np.linalg.norm(reprojected_first - observed_first, axis=1)
        if len(observed_first)
        else np.array([])
    )
    second_errors = (
        np.linalg.norm(reprojected_second - observed_second, axis=1)
        if len(observed_second)
        else np.array([])
    )
    error_samples = (
        np.concatenate([first_errors, second_errors])
        if len(first_errors) or len(second_errors)
        else np.array([])
    )
    maximum = max(
        float(np.percentile(error_samples, 95)) if len(error_samples) else 1.0,
        1e-9,
    )
    layouts: list[dict[str, int]] = []
    rows: list[np.ndarray] = []
    total = len(observed_first)
    for row_index, start in enumerate(range(0, total, max_per_row)):
        end = min(total, start + max_per_row)
        top = second_image.copy()
        bottom = first_image.copy()
        for obs, rep, error in zip(
            observed_second[start:end],
            reprojected_second[start:end],
            second_errors[start:end],
        ):
            color_value = np.uint8(np.clip(error / maximum * 255, 0, 255))
            color = tuple(
                int(x)
                for x in cv2.applyColorMap(
                    np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_TURBO
                )[0, 0]
            )
            p1 = tuple(np.round(obs).astype(int))
            p2 = tuple(np.round(rep).astype(int))
            cv2.circle(top, p1, 5, color, 1)
            cv2.drawMarker(top, p2, color, cv2.MARKER_CROSS, 8, 1)
            cv2.line(top, p1, p2, color, 1, cv2.LINE_AA)
        for obs, rep, error in zip(
            observed_first[start:end],
            reprojected_first[start:end],
            first_errors[start:end],
        ):
            color_value = np.uint8(np.clip(error / maximum * 255, 0, 255))
            color = tuple(
                int(x)
                for x in cv2.applyColorMap(
                    np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_TURBO
                )[0, 0]
            )
            p1 = tuple(np.round(obs).astype(int))
            p2 = tuple(np.round(rep).astype(int))
            cv2.circle(bottom, p1, 5, color, 1)
            cv2.drawMarker(bottom, p2, color, cv2.MARKER_CROSS, 8, 1)
            cv2.line(bottom, p1, p2, color, 1, cv2.LINE_AA)
        cv2.putText(
            top,
            f"row {row_index + 1}: A->B reprojection {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bottom,
            f"row {row_index + 1}: B->A reprojection {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([top, bottom]))
        layouts.append({"row": row_index, "start_index": start, "end_index": end})
    rendered = _stack_rows(rows)
    _save(path, rendered)
    return layouts


def _draw_text_panel(
    canvas: np.ndarray,
    lines: list[str],
    *,
    top_right: bool = True,
    scale_multiplier: float = 1.0,
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55 * float(scale_multiplier)
    thickness = max(1, int(round(1 * float(scale_multiplier))))
    line_height = int(round(22 * float(scale_multiplier)))
    text_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    panel_width = text_width + 22
    panel_height = line_height * len(lines) + 14
    x0 = canvas.shape[1] - panel_width - 10 if top_right else 10
    y0 = 10
    x0 = max(0, x0)
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (min(canvas.shape[1] - 1, x0 + panel_width), min(canvas.shape[0] - 1, y0 + panel_height)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x0 + 10, y0 + 22 + i * line_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _format_optional_float(value, precision: int) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{precision}f}"


def draw_ransac_retained_reprojection_qc(
    first_image,
    second_image,
    first_points,
    second_points,
    *,
    ransac_inlier,
    retained_mask,
    reprojected_first,
    reprojected_second,
    metadata: dict[str, object] | None,
    path: Path,
    max_per_row: int = 150,
    include_rejected_ransac: bool = True,
    reprojection_layout: str = "vertical",
    reprojection_points: str = "retained",
    retained_match_color_mode: str = "green",
    text_scale_multiplier: float = 1.0,
):
    first_points = np.asarray(first_points, dtype=np.float32)
    second_points = np.asarray(second_points, dtype=np.float32)
    reprojected_first = np.asarray(reprojected_first, dtype=np.float32)
    reprojected_second = np.asarray(reprojected_second, dtype=np.float32)
    ransac_inlier = np.asarray(ransac_inlier, dtype=bool)
    retained_mask = np.asarray(retained_mask, dtype=bool)

    total = len(first_points)
    if total == 0:
        _save(path, np.zeros((64, 64, 3), dtype=np.uint8))
        return []

    max_height = max(first_image.shape[0], second_image.shape[0])
    first_image, first_points_scaled, first_resize = resize_to_height(
        first_image, first_points, target_height=max_height
    )
    second_image, second_points_scaled, _second_resize = resize_to_height(
        second_image, second_points, target_height=max_height
    )
    reprojected_first_scaled = np.asarray(reprojected_first, dtype=np.float32).copy()
    reprojected_first_scaled *= float(first_resize["scale"])
    reprojected_second_scaled = np.asarray(reprojected_second, dtype=np.float32).copy()
    reprojected_second_scaled *= float(_second_resize["scale"])

    rows: list[np.ndarray] = []
    layouts: list[dict[str, int]] = []
    green = (0, 220, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)

    metadata = metadata or {}
    pose_degraded = bool(metadata.get("pose_degraded", False))
    metadata_lines = [
        f"pose_degraded = {pose_degraded}",
        "reprojection_median_px = "
        f"{_format_optional_float(metadata.get('reprojection_median_px', np.nan), 2)}",
        "reprojection_p75_px = "
        f"{_format_optional_float(metadata.get('reprojection_p75_px', np.nan), 2)}",
        "ray_gap_median_m = "
        f"{_format_optional_float(metadata.get('ray_gap_median_m', np.nan), 3)}",
        "ray_gap_p75_m = "
        f"{_format_optional_float(metadata.get('ray_gap_p75_m', np.nan), 3)}",
    ]
    if not pose_degraded:
        metadata_lines.extend(
            [
                "reproj adaptive median/K/MAD = "
                f"{_format_optional_float(metadata.get('reprojection_threshold_median_px', np.nan), 2)}"
                "/"
                f"{_format_optional_float(metadata.get('reprojection_threshold_k', 2.0), 2)}"
                "/"
                f"{_format_optional_float(metadata.get('reprojection_threshold_mad_px', np.nan), 2)}",
                "reprojection_threshold_px = "
                f"{_format_optional_float(metadata.get('reprojection_threshold_px', np.nan), 2)}"
                " (calc="
                f"{_format_optional_float(metadata.get('reprojection_threshold_calculated_px', np.nan), 2)}"
                ")",
                "ray adaptive median/K/MAD = "
                f"{_format_optional_float(metadata.get('ray_gap_threshold_median_m', np.nan), 3)}"
                "/"
                f"{_format_optional_float(metadata.get('ray_gap_threshold_k', 2.0), 2)}"
                "/"
                f"{_format_optional_float(metadata.get('ray_gap_threshold_mad_m', np.nan), 3)}",
                "ray_gap_threshold_m = "
                f"{_format_optional_float(metadata.get('ray_gap_threshold_m', np.nan), 3)}"
                " (calc="
                f"{_format_optional_float(metadata.get('ray_gap_threshold_calculated_m', np.nan), 3)}"
                ")",
            ]
        )
    filter_steps = metadata.get("filter_steps", [])
    if isinstance(filter_steps, list):
        for step in filter_steps:
            if not isinstance(step, dict):
                continue
            name = str(step.get("stage", ""))
            deleted = step.get("deleted_this_step", "")
            remaining = step.get("remaining_after", "")
            metadata_lines.append(f"{name}: -{deleted}, left={remaining}")

    for row_index, start in enumerate(range(0, total, max_per_row)):
        end = min(total, start + max_per_row)
        canvas_height = max(first_image.shape[0], second_image.shape[0])
        canvas_width = first_image.shape[1] + second_image.shape[1]
        match_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        match_canvas[: first_image.shape[0], : first_image.shape[1]] = first_image
        match_canvas[: second_image.shape[0], first_image.shape[1] :] = second_image
        offset = first_image.shape[1]

        for local_index, (first, second, is_ransac, is_retained) in enumerate(
            zip(
                first_points_scaled[start:end],
                second_points_scaled[start:end],
                ransac_inlier[start:end],
                retained_mask[start:end],
            ),
            start=start,
        ):
            if not is_ransac:
                continue
            if not is_retained and not include_rejected_ransac:
                continue
            if is_retained and str(retained_match_color_mode) == "index":
                color = _color_from_index(local_index)
            else:
                color = green if is_retained else red
            p1 = _cv_point_or_none(first, match_canvas.shape)
            p2 = _cv_point_or_none(np.asarray(second) + [offset, 0], match_canvas.shape)
            if p1 is None or p2 is None:
                continue
            cv2.circle(match_canvas, p1, 2, color, -1)
            cv2.circle(match_canvas, p2, 2, color, -1)
            cv2.line(match_canvas, p1, p2, color, 1, cv2.LINE_AA)

        title = (
            "green=retained red=RANSAC-inlier rejected"
            if include_rejected_ransac
            else "green=retained only"
        )
        cv2.putText(
            match_canvas,
            f"row {row_index + 1}: {title} {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        _draw_text_panel(
            match_canvas,
            metadata_lines,
            top_right=True,
            scale_multiplier=text_scale_multiplier,
        )

        left = first_image.copy()
        right = second_image.copy()
        row_retained = retained_mask[start:end]
        if str(reprojection_points) == "retained":
            row_reprojection_mask = row_retained
        elif str(reprojection_points) == "ransac_inliers":
            row_reprojection_mask = ransac_inlier[start:end]
        else:
            raise ValueError("reprojection_points must be 'retained' or 'ransac_inliers'")
        row_reprojection_retained = retained_mask[start:end][row_reprojection_mask]
        for obs, rep, is_retained in zip(
            first_points_scaled[start:end][row_reprojection_mask],
            reprojected_first_scaled[start:end][row_reprojection_mask],
            row_reprojection_retained,
        ):
            if not (np.isfinite(obs).all() and np.isfinite(rep).all()):
                continue
            color = green if is_retained else red
            p1 = _cv_point_or_none(obs, left.shape)
            p2 = _cv_point_or_none(rep, left.shape)
            if p1 is None or p2 is None:
                continue
            cv2.circle(left, p1, 3, white, 1, cv2.LINE_AA)
            cv2.drawMarker(left, p2, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            cv2.line(left, p1, p2, color, 1, cv2.LINE_AA)
        for obs, rep, is_retained in zip(
            second_points_scaled[start:end][row_reprojection_mask],
            reprojected_second_scaled[start:end][row_reprojection_mask],
            row_reprojection_retained,
        ):
            if not (np.isfinite(obs).all() and np.isfinite(rep).all()):
                continue
            color = green if is_retained else red
            p1 = _cv_point_or_none(obs, right.shape)
            p2 = _cv_point_or_none(rep, right.shape)
            if p1 is None or p2 is None:
                continue
            cv2.circle(right, p1, 3, white, 1, cv2.LINE_AA)
            cv2.drawMarker(right, p2, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            cv2.line(right, p1, p2, color, 1, cv2.LINE_AA)
        cv2.putText(
            left,
            f"row {row_index + 1}: B->A reprojection on A, retained only",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            right,
            f"row {row_index + 1}: A->B reprojection on B, retained only",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if str(reprojection_layout) == "side_by_side":
            reproj_height = max(left.shape[0], right.shape[0])
            if left.shape[0] < reproj_height:
                left = np.pad(left, ((0, reproj_height - left.shape[0]), (0, 0), (0, 0)))
            if right.shape[0] < reproj_height:
                right = np.pad(
                    right, ((0, reproj_height - right.shape[0]), (0, 0), (0, 0))
                )
            reprojection_canvas = np.hstack([left, right])
            row = _stack_rows([match_canvas, reprojection_canvas])
        elif str(reprojection_layout) == "vertical":
            reproj_width = max(left.shape[1], right.shape[1])
            if left.shape[1] < reproj_width:
                left = np.pad(left, ((0, 0), (0, reproj_width - left.shape[1]), (0, 0)))
            if right.shape[1] < reproj_width:
                right = np.pad(right, ((0, 0), (0, reproj_width - right.shape[1]), (0, 0)))
            row = _stack_rows([match_canvas, left, right])
        else:
            raise ValueError("reprojection_layout must be 'vertical' or 'side_by_side'")
        rows.append(row)
        layouts.append({"row": row_index, "start_index": start, "end_index": end})

    rendered = _stack_rows(rows)
    _save(path, rendered)
    return layouts


def draw_tiled_before_after_reprojection_overlay(
    image,
    observed,
    before_reprojected,
    after_reprojected,
    path: Path,
    max_per_row: int = 150,
):
    observed = np.asarray(observed, dtype=np.float32)
    before_reprojected = np.asarray(before_reprojected, dtype=np.float32)
    after_reprojected = np.asarray(after_reprojected, dtype=np.float32)
    layouts: list[dict[str, int]] = []
    rows: list[np.ndarray] = []
    total = len(observed)
    before_color = (0, 0, 255)
    after_color = (0, 255, 0)
    observed_color = (255, 255, 255)
    for row_index, start in enumerate(range(0, total, max_per_row)):
        end = min(total, start + max_per_row)
        canvas = image.copy()
        for obs, before, after in zip(
            observed[start:end],
            before_reprojected[start:end],
            after_reprojected[start:end],
        ):
            obs_point = tuple(np.round(obs).astype(int))
            before_point = tuple(np.round(before).astype(int))
            after_point = tuple(np.round(after).astype(int))
            cv2.circle(canvas, obs_point, 5, observed_color, 1)
            cv2.drawMarker(canvas, before_point, before_color, cv2.MARKER_CROSS, 8, 1)
            cv2.line(canvas, obs_point, before_point, before_color, 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, after_point, after_color, cv2.MARKER_TILTED_CROSS, 8, 1)
            cv2.line(canvas, obs_point, after_point, after_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"row {row_index + 1}: red=before green=after {start}..{end - 1}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            observed_color,
            2,
            cv2.LINE_AA,
        )
        rows.append(canvas)
        layouts.append({"row": row_index, "start_index": start, "end_index": end})
    rendered = _stack_rows(rows)
    _save(path, rendered)
    return layouts


def plot_error_over_time(pair_metrics: pd.DataFrame, path: Path, camera: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    if not pair_metrics.empty:
        time_seconds = pair_metrics["timestamp1_ns"] / 1e9
        axis.plot(time_seconds, pair_metrics["nav_symmetric_median_px"], label="epipolar")
        if "reprojection_median_px" in pair_metrics:
            axis.plot(time_seconds, pair_metrics["reprojection_median_px"], label="reprojection")
        axis.legend()
    axis.set_title(f"{camera}: geometry error over time")
    axis.set_xlabel("timestamp (s)")
    axis.set_ylabel("median error (px)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
