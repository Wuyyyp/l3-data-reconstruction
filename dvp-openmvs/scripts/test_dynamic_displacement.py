from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preprocess"))
from M18proc.dynamic_policy import select_tracks, VERSION


def frame(index, points, translation=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, 3] = translation
    objects = [dict(id=key, location=xyz, velocity=[100, 0, 0]) for key, xyz in points.items()]
    return index, objects, pose


# Speed is irrelevant, and exactly 3 m is static.
selected, report = select_tracks([
    frame(5, {"still": [0,0,0], "boundary": [0,0,0], "moving": [0,0,0]}),
    frame(105, {"still": [0,0,0], "boundary": [3,0,0], "moving": [3.001,0,0]}),
])
assert selected == {"moving"} and report["version"] == VERSION
assert report["frame_ids"] == [5, 105]

# Ego motion must not make a static world object dynamic.
selected, report = select_tracks([
    frame(0, {"static_world": [10,0,0]}),
    frame(1, {"static_world": [0,0,0]}, translation=(10,0,0)),
])
assert not selected and report["tracks"]["static_world"]["max_displacement_m"] == 0

# A return to the start is still motion; altitude-only changes are static.
selected, _ = select_tracks([
    frame(0, {"return": [0,0,0], "vertical": [0,0,0]}),
    frame(1, {"return": [4,0,0], "vertical": [0,0,20]}),
    frame(2, {"return": [0,0,0], "vertical": [0,0,0]}),
])
assert selected == {"return"}

# Do not accumulate jitter, and do not use observations outside the window.
jitter = [frame(i, {"jitter": [(-1)**i,0,0]}) for i in range(100)]
assert not select_tracks(jitter)[0]
outside = frame(101, {"jitter": [100,0,0]})
assert select_tracks(jitter + [outside])[0] == {"jitter"}

# Unknown/single observations remain static. Invalid positions do not select.
selected, report = select_tracks([
    frame(0, {None: [0,0,0], "single": [0,0,0], "bad": [float("nan"),0,0]}),
    frame(1, {"bad": [10,0,0]}),
])
assert not selected and report["missing_id_observations"] == 1
assert report["tracks"]["bad"]["unknown_map_positions"] == 1

# Large map coordinates and rotated ego frames retain correct distances.
pose = np.eye(4)
pose[:2, :2] = [[0,-1],[1,0]]
pose[:3, 3] = [21738,-1577,5182]
assert select_tracks([(0,[dict(id=7,location=[0,0,0])],pose),
                      (1,[dict(id=7,location=[4,0,0])],pose)])[0] == {"7"}

try:
    select_tracks([(0,[dict(id=1,location=[0,0,0]),dict(id=1,location=[1,0,0])],np.eye(4))])
except ValueError:
    pass
else:
    raise AssertionError("duplicate IDs must fail")
print("PASS: map displacement, threshold, window, ego motion, return, jitter, unknowns, rotation")

selected, report = select_tracks([
    frame(0, {"boundary": [0,0,0], "moving": [0,0,0]}),
    frame(100, {"boundary": [1,0,0], "moving": [1.001,0,0]}),
], threshold_m=1)
assert selected == {"moving"}
assert report["threshold_m"] == 1 and report["version"] == "map-xy-window-displacement-gt1m-v1"
assert not select_tracks([
    frame(0, {"static": [10,0,0]}),
    frame(100, {"static": [0,0,0]}, translation=(10,0,0)),
], threshold_m=1)[0]
for invalid in (-1, float('nan'), float('inf')):
    try:
        select_tracks([], threshold_m=invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid threshold must fail")
print("PASS: configurable 1m threshold, strict boundary, ego cancellation and invalid inputs")
