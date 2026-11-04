from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dongfeng_registration.dynamic_object_compensation import write_polylinez_shapefile


def _features(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in values:
        out.append(
            {
                "frame": item.get("frame", 0),
                "object_id": item.get("object_id", ""),
                "object_type": item.get("object_type", ""),
                "timestamp_s": item.get("timestamp_s", 0.0),
                "corners": item["corners"],
                "length": item.get("length", 0.0),
                "width": item.get("width", 0.0),
                "height": item.get("height", 0.0),
                "yaw": item.get("yaw", 0.0),
                "status": item.get("status", ""),
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    box_dir = Path(args.box_dir)
    output_dir = Path(args.output_dir)
    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    json_paths = sorted(box_dir.glob("frame_*_boxes.json"))
    written = 0
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        frame_index = int(payload.get("frame_index", -1))
        if frame_index < int(args.start) or frame_index >= int(args.end):
            continue
        stem = json_path.stem
        write_polylinez_shapefile(before_dir / f"{stem}.shp", _features(payload.get("before", [])))
        write_polylinez_shapefile(after_dir / f"{stem}.shp", _features(payload.get("after", [])))
        written += 2
        print(f"[{frame_index:04d}] {json_path.name} -> before/after shp")
    print(f"[done] shapefiles={written} -> {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export saved dynamic before/after box JSON to SHP.")
    parser.add_argument("--box-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000_000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
