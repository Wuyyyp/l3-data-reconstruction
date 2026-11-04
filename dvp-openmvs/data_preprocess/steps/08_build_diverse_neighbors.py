#!/usr/bin/env python3
"""Build OpenMVS neighbors with explicit temporal-band and cross-camera quotas."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TEMPORAL_BANDS = (
    ("past_far", -20, -11),
    ("past_mid", -10, -4),
    ("past_near", -3, -1),
    ("future_near", 1, 3),
    ("future_mid", 4, 10),
    ("future_far", 11, 20),
)


def read_tracks(images_txt: Path) -> dict[str, set[int]]:
    # Keep empty observation rows: COLMAP still writes one after an image with
    # no triangulated points, and dropping it shifts every later header/row pair.
    lines = [line.strip() for line in images_txt.open() if not line.lstrip().startswith("#")]
    if len(lines) % 2:
        raise ValueError(f"malformed COLMAP images file: {images_txt}")
    tracks: dict[str, set[int]] = {}
    for header, observations in zip(lines[0::2], lines[1::2]):
        header_fields = header.split()
        fields = observations.split()
        if len(header_fields) < 10 or len(fields) % 3:
            raise ValueError(f"malformed COLMAP image record: {header!r}")
        stem = Path(header_fields[9]).stem
        tracks[stem] = {int(fields[i]) for i in range(2, len(fields), 3) if int(fields[i]) >= 0}
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--images-txt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--sources", type=int, default=20)
    parser.add_argument("--cross-camera-quota", type=int, default=6)
    parser.add_argument("--reference-frame", type=int)
    args = parser.parse_args()

    names = [value.decode() for value in re.findall(rb"([0-9]{6}_(?:00|01|02|03|04|09|10)\.jpg)", args.scene.read_bytes())]
    if len(names) != len(set(names)):
        raise RuntimeError(f"scene image names are not unique: {len(names)}/{len(set(names))}")
    stems = [Path(name).stem for name in names]
    scene_index = {stem: index for index, stem in enumerate(stems)}
    tracks = read_tracks(args.images_txt)
    if set(stems) != set(tracks):
        raise RuntimeError(f"scene/model mismatch scene={len(stems)} model={len(tracks)}")
    point_views: dict[int, list[str]] = defaultdict(list)
    for stem, point_ids in tracks.items():
        for point_id in point_ids:
            point_views[point_id].append(stem)

    rows: list[dict[str, object]] = []
    output_lines: list[str] = []
    summaries: list[dict[str, object]] = []
    references = stems
    if args.reference_frame is not None:
        references = [stem for stem in stems if int(stem.split("_")[0]) == args.reference_frame]
    for reference in references:
        ref_frame, ref_camera = reference.split("_")
        ref_frame_i = int(ref_frame)
        shared_counts: Counter[str] = Counter()
        for point_id in tracks[reference]:
            shared_counts.update(point_views[point_id])
        candidates: list[dict[str, object]] = []
        for source in stems:
            if source == reference:
                continue
            src_frame, src_camera = source.split("_")
            delta = int(src_frame) - ref_frame_i
            if abs(delta) > args.radius:
                continue
            shared = shared_counts[source]
            candidates.append({
                "source": source,
                "delta": delta,
                "source_camera": src_camera,
                "same_camera": src_camera == ref_camera,
                "shared": shared,
            })

        selected: list[tuple[dict[str, object], str]] = []
        used: set[str] = set()

        def add(item: dict[str, object] | None, reason: str) -> None:
            if item is not None and str(item["source"]) not in used:
                selected.append((item, reason))
                used.add(str(item["source"]))

        # Exactly one same-camera source from each available signed time band.
        for band, low, high in TEMPORAL_BANDS:
            pool = [c for c in candidates if c["same_camera"] and int(c["shared"]) > 0 and low <= int(c["delta"]) <= high]
            pool.sort(key=lambda c: (int(c["shared"]), -abs(int(c["delta"]))), reverse=True)
            add(pool[0] if pool else None, f"temporal:{band}")

        # Cross-camera quota: first spread across distinct cameras, then fill by shared points.
        cross = [c for c in candidates if not c["same_camera"] and int(c["shared"]) > 0 and abs(int(c["delta"])) <= 3]
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in cross:
            grouped[str(item["source_camera"])].append(item)
        group_best = []
        for camera, pool in grouped.items():
            pool.sort(key=lambda c: (int(c["shared"]), -abs(int(c["delta"]))), reverse=True)
            group_best.append(pool[0])
        group_best.sort(key=lambda c: int(c["shared"]), reverse=True)
        for item in group_best:
            if sum(not bool(c["same_camera"]) for c, _ in selected) >= args.cross_camera_quota:
                break
            add(item, "cross_camera:distinct")
        cross.sort(key=lambda c: (int(c["shared"]), -abs(int(c["delta"]))), reverse=True)
        for item in cross:
            if sum(not bool(c["same_camera"]) for c, _ in selected) >= args.cross_camera_quota:
                break
            add(item, "cross_camera:fill")
        if sum(not bool(c["same_camera"]) for c, _ in selected) < args.cross_camera_quota:
            wider_cross = [c for c in candidates if not c["same_camera"] and int(c["shared"]) > 0 and str(c["source"]) not in used]
            wider_cross.sort(key=lambda c: (int(c["shared"]), -abs(int(c["delta"]))), reverse=True)
            for item in wider_cross:
                if sum(not bool(c["same_camera"]) for c, _ in selected) >= args.cross_camera_quota:
                    break
                add(item, "cross_camera:wider_time_fallback")

        # Fill remaining slots by evidence, preferring cross-camera and non-near temporal sources on ties.
        remaining = [c for c in candidates if int(c["shared"]) > 0 and str(c["source"]) not in used]
        remaining.sort(
            key=lambda c: (
                int(c["shared"]),
                not bool(c["same_camera"]),
                min(abs(int(c["delta"])), 10),
            ),
            reverse=True,
        )
        for item in remaining:
            if len(selected) >= args.sources:
                break
            add(item, "evidence_fill")
        # Keep the L3 top-20 contract on unusually weak frames, but only after
        # every evidence-backed candidate has been used. Prefer same-camera
        # temporal support over unrelated lateral cameras for this fallback.
        fallback = [c for c in candidates if int(c["shared"]) == 0 and str(c["source"]) not in used]
        fallback.sort(key=lambda c: (bool(c["same_camera"]), -abs(int(c["delta"]))), reverse=True)
        for item in fallback:
            if len(selected) >= args.sources:
                break
            add(item, "zero_evidence_last_resort")
        if len(selected) != args.sources:
            raise RuntimeError(f"{reference}: selected {len(selected)}, expected {args.sources}")

        output_lines.append(" ".join(str(scene_index[name]) for name in [reference] + [str(c["source"]) for c, _ in selected]))
        cross_count = sum(not bool(c["same_camera"]) for c, _ in selected)
        source_cameras = len({str(c["source_camera"]) for c, _ in selected if not bool(c["same_camera"])})
        bands_present = set()
        for c, reason in selected:
            if reason.startswith("temporal:"):
                bands_present.add(reason.split(":", 1)[1])
        summaries.append({
            "reference": reference,
            "sources": len(selected),
            "cross_camera_sources": cross_count,
            "distinct_cross_cameras": source_cameras,
            "zero_evidence_sources": sum(int(c["shared"]) == 0 for c, _ in selected),
            "temporal_bands": sorted(bands_present),
        })
        for rank, (item, reason) in enumerate(selected, 1):
            rows.append({
                "reference": reference,
                "rank": rank,
                "source": item["source"],
                "frame_delta": item["delta"],
                "same_camera": int(bool(item["same_camera"])),
                "shared_sparse_points": item["shared"],
                "selection_reason": reason,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n")
    with args.audit.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "PASS",
        "scene_images": len(stems),
        "reference_images": len(references),
        "sources_per_reference": args.sources,
        "candidate_radius_frames": args.radius,
        "selection_policy": "positive sparse evidence is required whenever available; zero-evidence sources are used only as the final fallback needed to keep exactly 20 sources, preferring same-camera temporal views",
        "cross_camera_sources_min": min(int(s["cross_camera_sources"]) for s in summaries),
        "cross_camera_sources_mean": sum(int(s["cross_camera_sources"]) for s in summaries) / len(summaries),
        "distinct_cross_cameras_min": min(int(s["distinct_cross_cameras"]) for s in summaries),
        "zero_evidence_sources_total": sum(int(s["zero_evidence_sources"]) for s in summaries),
        "zero_evidence_sources_max_per_reference": max(int(s["zero_evidence_sources"]) for s in summaries),
        "references_with_all_six_temporal_bands": sum(len(s["temporal_bands"]) == 6 for s in summaries),
        "references": summaries,
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "references"}, indent=2))


if __name__ == "__main__":
    main()
