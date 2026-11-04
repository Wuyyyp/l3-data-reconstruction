from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class NavigationTrajectory:
    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    allow_end_extrapolation: bool = False

    @classmethod
    def from_csv(cls, path: Path) -> "NavigationTrajectory":
        frame = pd.read_csv(path)
        timestamps = (
            frame["timestamp.sec"].to_numpy(dtype=np.int64) * 1_000_000_000
            + frame["timestamp.nsec"].to_numpy(dtype=np.int64)
        )
        positions = frame[
            [
                "local_pose.translation.x",
                "local_pose.translation.y",
                "local_pose.translation.z",
            ]
        ].to_numpy(dtype=np.float64)
        quaternions = frame[
            [
                "local_pose.rotation.x",
                "local_pose.rotation.y",
                "local_pose.rotation.z",
                "local_pose.rotation.w",
            ]
        ].to_numpy(dtype=np.float64)
        return cls(timestamps, positions, quaternions)

    def pose_at(self, timestamp_ns: int) -> np.ndarray:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < self.timestamps_ns[0]:
            raise ValueError("timestamp outside navigation range")
        if timestamp_ns > self.timestamps_ns[-1]:
            if not self.allow_end_extrapolation or len(self.timestamps_ns) < 2:
                raise ValueError("timestamp outside navigation range")
            end = int(self.timestamps_ns[-1])
            previous = int(self.timestamps_ns[-2])
            interval = end - previous
            if interval <= 0 or timestamp_ns > end + interval:
                raise ValueError("timestamp outside one-frame end extrapolation range")
            alpha = (timestamp_ns - end) / interval
            position = self.positions[-1] + alpha * (self.positions[-1] - self.positions[-2])
            relative = Rotation.from_quat(self.quaternions_xyzw[-2]).inv() * Rotation.from_quat(self.quaternions_xyzw[-1])
            rotation = (Rotation.from_quat(self.quaternions_xyzw[-1]) * Rotation.from_rotvec(alpha * relative.as_rotvec())).as_matrix()
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            return pose
        right = int(np.searchsorted(self.timestamps_ns, timestamp_ns, side="left"))
        if right < len(self.timestamps_ns) and self.timestamps_ns[right] == timestamp_ns:
            position = self.positions[right]
            rotation = Rotation.from_quat(self.quaternions_xyzw[right]).as_matrix()
        else:
            left = right - 1
            times = self.timestamps_ns[[left, right]].astype(np.float64)
            alpha = (timestamp_ns - times[0]) / (times[1] - times[0])
            position = self.positions[left] * (1 - alpha) + self.positions[right] * alpha
            rotations = Rotation.from_quat(self.quaternions_xyzw[[left, right]])
            rotation = Slerp(times, rotations)(float(timestamp_ns)).as_matrix()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose


def navigation_time_offset_ns(assumptions: dict[str, object] | None) -> int:
    assumptions = assumptions or {}
    return int(round(float(assumptions.get("navigation_time_offset_ms", 0.0)) * 1_000_000.0))
