from __future__ import annotations

import cv2
import numpy as np


def relative_pose(first_world_pose: np.ndarray, second_world_pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(first_world_pose) @ second_world_pose


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def essential_from_relative_pose(transform: np.ndarray) -> np.ndarray:
    return skew(transform[:3, 3]) @ transform[:3, :3]


def fundamental_from_essential(essential, first_intrinsic, second_intrinsic):
    return np.linalg.inv(second_intrinsic).T @ essential @ np.linalg.inv(first_intrinsic)


def epipolar_errors(first_points, second_points, matrix):
    first_points = np.asarray(first_points, dtype=np.float64)
    second_points = np.asarray(second_points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    first = np.column_stack([first_points, np.ones(len(first_points), dtype=np.float64)])
    second = np.column_stack([second_points, np.ones(len(second_points), dtype=np.float64)])
    lines_second = (matrix @ first.T).T
    lines_first = (matrix.T @ second.T).T
    residual = np.einsum("ij,ij->i", second, lines_second)
    norm_second = np.linalg.norm(lines_second[:, :2], axis=1)
    norm_first = np.linalg.norm(lines_first[:, :2], axis=1)
    eps = np.finfo(np.float64).eps
    distance_second = np.abs(residual) / np.maximum(norm_second, eps)
    distance_first = np.abs(residual) / np.maximum(norm_first, eps)
    return {
        "point_to_epiline": distance_second,
        "symmetric_epipolar": np.sqrt(distance_first**2 + distance_second**2),
        "sampson": residual**2 / np.maximum(norm_first**2 + norm_second**2, eps),
        "algebraic": residual,
    }


def triangulate_and_reproject(first_points, second_points, first_world_to_camera, second_world_to_camera):
    homogeneous = cv2.triangulatePoints(
        first_world_to_camera[:3],
        second_world_to_camera[:3],
        np.asarray(first_points, dtype=np.float64).T,
        np.asarray(second_points, dtype=np.float64).T,
    )
    world = (homogeneous[:3] / homogeneous[3]).T
    world_h = np.column_stack([world, np.ones(len(world))])
    first_camera = (first_world_to_camera @ world_h.T).T[:, :3]
    second_camera = (second_world_to_camera @ world_h.T).T[:, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        first_reprojected = first_camera[:, :2] / first_camera[:, 2, None]
        second_reprojected = second_camera[:, :2] / second_camera[:, 2, None]
    first_error = np.linalg.norm(first_reprojected - first_points, axis=1)
    second_error = np.linalg.norm(second_reprojected - second_points, axis=1)
    return {
        "world_points": world,
        "first_camera_points": first_camera,
        "second_camera_points": second_camera,
        "first_reprojected_normalized": first_reprojected,
        "second_reprojected_normalized": second_reprojected,
        "valid_depth": (first_camera[:, 2] > 0) & (second_camera[:, 2] > 0),
        "reprojection_error_normalized": np.sqrt((first_error**2 + second_error**2) / 2),
    }


def ray_geometry_metrics(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_camera_to_world: np.ndarray,
    second_camera_to_world: np.ndarray,
) -> dict[str, np.ndarray]:
    first_points = np.asarray(first_points, dtype=np.float64)
    second_points = np.asarray(second_points, dtype=np.float64)
    first_dirs_camera = np.column_stack(
        [first_points, np.ones(len(first_points), dtype=np.float64)]
    )
    second_dirs_camera = np.column_stack(
        [second_points, np.ones(len(second_points), dtype=np.float64)]
    )
    first_dirs_camera /= np.linalg.norm(first_dirs_camera, axis=1, keepdims=True)
    second_dirs_camera /= np.linalg.norm(second_dirs_camera, axis=1, keepdims=True)
    first_dirs_world = (first_camera_to_world[:3, :3] @ first_dirs_camera.T).T
    second_dirs_world = (second_camera_to_world[:3, :3] @ second_dirs_camera.T).T
    first_dirs_world /= np.linalg.norm(first_dirs_world, axis=1, keepdims=True)
    second_dirs_world /= np.linalg.norm(second_dirs_world, axis=1, keepdims=True)

    dot = np.sum(first_dirs_world * second_dirs_world, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    triangulation_angle_deg = np.degrees(np.arccos(dot))

    first_center = first_camera_to_world[:3, 3]
    second_center = second_camera_to_world[:3, 3]
    baseline = second_center - first_center
    cross = np.cross(first_dirs_world, second_dirs_world)
    cross_norm = np.linalg.norm(cross, axis=1)
    baseline_dot_cross = np.abs(cross @ baseline)
    ray_gap = np.empty((len(first_points),), dtype=np.float64)
    non_parallel = cross_norm > np.finfo(np.float64).eps
    ray_gap[non_parallel] = baseline_dot_cross[non_parallel] / cross_norm[non_parallel]
    ray_gap[~non_parallel] = np.linalg.norm(
        np.cross(np.tile(baseline, (np.sum(~non_parallel), 1)), first_dirs_world[~non_parallel]),
        axis=1,
    )
    return {
        "triangulation_angle_deg": triangulation_angle_deg,
        "ray_closest_distance_m": ray_gap,
    }
