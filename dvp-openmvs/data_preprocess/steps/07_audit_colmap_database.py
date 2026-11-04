#!/usr/bin/env python3
"""Record the COLMAP image/feature/match inventory."""

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--expected-images", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as connection:
        report = {
            "images": connection.execute("select count(*) from images").fetchone()[0],
            "images_with_features": connection.execute(
                "select count(*) from keypoints where rows > 0"
            ).fetchone()[0],
            "verified_pairs": connection.execute(
                "select count(*) from two_view_geometries where rows > 0"
            ).fetchone()[0],
        }
    report["requested_pairs"] = sum(1 for _ in args.pairs.open())
    report["status"] = "PASS" if report["images"] == args.expected_images else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
