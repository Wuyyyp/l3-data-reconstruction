"""Classify motion only by map XY displacement in the reconstruction window."""

import numpy as np

VERSION = "map-xy-window-displacement-gt3m-v1"
THRESHOLD_M = 3.0


def object_id(obj):
    value = obj.get("id")
    if value is None or str(value).strip().lower() in ("", "-1", "none", "nan"):
        return None
    return str(value)


def policy_version(threshold_m=THRESHOLD_M):
    threshold_m = float(threshold_m)
    if not np.isfinite(threshold_m) or threshold_m < 0:
        raise ValueError("displacement threshold must be finite and non-negative")
    return "map-xy-window-displacement-gt%sm-v1" % format(threshold_m, "g")


def select_tracks(frames, threshold_m=THRESHOLD_M):
    """frames yields (original_frame_id, objects, ego_to_map_4x4).

    Objects contain id/location/type. Positions without a finite map transform
    do not count as evidence of motion. Unknown tracks remain static.
    """
    version = policy_version(threshold_m)
    threshold_m = float(threshold_m)
    tracks, points = {}, {}
    missing_ids = 0
    frame_ids = []
    for frame_id, objects, pose in frames:
        frame_ids.append(int(frame_id))
        pose = np.asarray(pose, dtype=np.float64) if pose is not None else None
        valid_pose = pose is not None and pose.shape == (4, 4) and np.isfinite(pose).all()
        seen = set()
        for obj in objects:
            key = object_id(obj)
            if key is None:
                missing_ids += 1
                continue
            if key in seen:
                raise ValueError("duplicate track ID in frame %s: %s" % (frame_id, key))
            seen.add(key)
            row = tracks.setdefault(key, {
                "type": str(obj.get("type", "")), "observations": 0,
                "valid_map_positions": 0, "unknown_map_positions": 0,
            })
            row["observations"] += 1
            try:
                center = np.asarray(obj.get("location"), dtype=np.float64)
                valid_center = center.shape == (3,) and np.isfinite(center).all()
            except (TypeError, ValueError):
                valid_center = False
            if not valid_pose or not valid_center:
                row["unknown_map_positions"] += 1
                continue
            xy = (pose[:3, :3] @ center + pose[:3, 3])[:2]
            if not np.isfinite(xy).all():
                row["unknown_map_positions"] += 1
                continue
            points.setdefault(key, []).append((int(frame_id), xy))
            row["valid_map_positions"] += 1

    selected = set()
    for key, row in tracks.items():
        observations = points.get(key, [])
        maximum_sq, farthest_frames = 0.0, None
        if len(observations) >= 2:
            xy = np.asarray([point for _, point in observations], dtype=np.float64)
            for i in range(len(xy) - 1):
                delta = xy[i + 1:] - xy[i]
                distances_sq = np.einsum("ij,ij->i", delta, delta)
                j = int(np.argmax(distances_sq))
                value = float(distances_sq[j])
                if value > maximum_sq:
                    maximum_sq = value
                    farthest_frames = [observations[i][0], observations[i + 1 + j][0]]
        row["max_displacement_m"] = float(np.sqrt(maximum_sq))
        row["farthest_frames"] = farthest_frames
        row["selected"] = maximum_sq > threshold_m ** 2
        if row["selected"]:
            selected.add(key)
    report = {
        "version": version, "threshold_m": threshold_m,
        "comparison": "strictly greater", "space": "map XY in metres",
        "metric": "maximum pairwise distance; not accumulated path length or speed",
        "scope": "reconstruction_window", "frame_ids": frame_ids,
        "selected_track_ids": sorted(selected), "missing_id_observations": missing_ids,
        "unknown_policy": "missing ID, insufficient or invalid map positions remain static",
        "tracks": tracks,
    }
    return selected, report
