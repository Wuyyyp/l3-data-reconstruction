#!/usr/bin/env python3
"""Select exactly one segment row from the scanner TSV."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--clip-name", required=True)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-end", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [
        row
        for row in rows
        if row["clip_name"] == args.clip_name
        and row["side"] == args.side
        and int(row["frame_start"]) == args.frame_start
        and int(row["frame_end"]) == args.frame_end
    ]
    if len(matches) != 1:
        available = [
            {
                "clip_name": row["clip_name"],
                "side": row["side"],
                "frame_start": int(row["frame_start"]),
                "frame_end": int(row["frame_end"]),
            }
            for row in rows
        ]
        raise RuntimeError(f"expected one matching segment, found {len(matches)}; available={available}")
    row = matches[0]
    local_pkl = Path(row["local_pkl_path"])
    if not local_pkl.is_file() and args.data_root:
        candidates = sorted(
            (args.data_root / "Result").glob(
                f"batch_*/V5_*/{args.clip_name}/result_pkl_all/{args.clip_name}.pkl"
            )
        )
        if len(candidates) == 1:
            row["local_pkl_path"] = str(candidates[0])
    normalized = {
        **row,
        "segment_index": int(row["segment_index"]),
        "clip_index_0based": int(row["clip_index_0based"]),
        "frame_start": int(row["frame_start"]),
        "frame_end": int(row["frame_end"]),
        "frame_count": int(row["frame_count"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(normalized, ensure_ascii=False))


if __name__ == "__main__":
    main()
