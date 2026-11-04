from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ns: int
    path: Path


class Dataset:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def frames(self, camera: str) -> list[Frame]:
        timestamp_path = self.root / "camera" / f"{camera}.txt"
        image_dir = self.root / "camera" / camera
        if not timestamp_path.exists():
            raise FileNotFoundError(timestamp_path)
        frames: list[Frame] = []
        for line in timestamp_path.read_text(encoding="utf-8").splitlines():
            index_text, timestamp_text = line.split(",", maxsplit=1)
            index = int(index_text.strip())
            timestamp_ns = int(timestamp_text.strip())
            frames.append(
                Frame(index=index, timestamp_ns=timestamp_ns, path=image_dir / f"{timestamp_ns}.jpg")
            )
        return frames
