from __future__ import annotations

from pathlib import Path


def resolve_retained_matches_path(
    source: Path,
) -> Path:
    source = Path(source)
    candidates: list[Path] = []
    if source.is_file():
        candidates.append(source)
    else:
        candidates.extend(
            [
                source / "retained_matches.parquet",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "unable to resolve retained matches parquet from "
        f"{source}; checked: {', '.join(str(path) for path in candidates)}"
    )
