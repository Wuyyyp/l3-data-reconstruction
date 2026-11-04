from __future__ import annotations

import pandas as pd


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def build_tracks(matches: pd.DataFrame, camera: str) -> pd.DataFrame:
    union_find = _UnionFind()
    coordinates = {}
    for row in matches.itertuples(index=False):
        first, second = (int(row.frame1), int(row.kp1)), (int(row.frame2), int(row.kp2))
        union_find.union(first, second)
        coordinates.setdefault(first, (float(row.x1), float(row.y1)))
        coordinates.setdefault(second, (float(row.x2), float(row.y2)))
    roots = {node: union_find.find(node) for node in coordinates}
    ids = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    records = [
        {"camera": camera, "track_id": ids[root], "frame": node[0], "keypoint_id": node[1],
         "x": coordinates[node][0], "y": coordinates[node][1]}
        for node, root in roots.items()
    ]
    if not records:
        return pd.DataFrame(columns=["camera", "track_id", "frame", "keypoint_id", "x", "y"])
    return pd.DataFrame(records).sort_values(["track_id", "frame"], ignore_index=True)
