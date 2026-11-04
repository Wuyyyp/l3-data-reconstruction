"""Camera and coordinate constants required by the bundled M18 PKL converter."""

import numpy as np

cameras_type = {
    "left_front_camera": 0,
    "right_front_camera": 1,
    "rear_camera": 2,
    "left_rear_camera": 3,
    "right_rear_camera": 4,
    "front_camera_fov200": 5,
    "rear_camera_fov200": 6,
    "left_camera_fov200": 7,
    "right_camera_fov200": 8,
    "center_camera_fov30": 9,
    "center_camera_fov120": 10,
}

lidar_type = {
    "perception": 0,
    "front_lidar": 1,
    "rear_lidar": 2,
    "left_lidar": 3,
    "right_lidar": 4,
    "top_center_lidar": 5,
    "front_left_lidar": 6,
    "front_right_lidar": 7,
}

# M18/Waymo axes [forward, left, up] to OpenCV axes [right, down, forward].
POSE_WAYMO2CV = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
POSE_WAYMO2GL = POSE_WAYMO2CV.copy()

image_heights = [1280, 1280, 1280, 1280, 1280, 1536, 1536, 1536, 1536, 2160, 2160]
image_widths = [1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 3840, 3840]
downsample_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
