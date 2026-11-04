#!/usr/bin/env python3
"""Print COLMAP PINHOLE parameters for one camera code."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--camera", required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    calibration = next(
        value for value in data["calibrations"].values() if value["camera_code"] == args.camera
    )
    matrix = calibration["K"]
    print(",".join(map(str, (matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]))))


if __name__ == "__main__":
    main()
