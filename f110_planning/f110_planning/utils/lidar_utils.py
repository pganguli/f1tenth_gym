"""
LiDAR and path tracking utility functions.
"""

import numpy as np
from numba import njit

from .geometry_utils import pi_2_pi
from .pure_pursuit_utils import nearest_point


@njit(cache=True)
def get_side_distances(scan: np.ndarray) -> tuple[float, float]:
    """
    Calculate the minimum distance to the left and right walls from a lidar scan.

    The lidar has a FOV of 4.7 radians (approx 270 degrees) and 1080 beams.
    Indices:
    - 0: -fov/2 (-2.35 rad, approx -135 degrees)
    - 540: 0 rad (forward)
    - 1079: fov/2 (2.35 rad, approx 135 degrees)

    Left side (90 degrees or pi/2 rad) is roughly at index:
    (pi/2 - (-fov/2)) / (fov / 1079)
    = (1.57 + 2.35) / (4.7 / 1079)
    = 3.92 / 0.004356 = 900

    Right side (-90 degrees or -pi/2 rad) is roughly at index:
    (-pi/2 - (-fov/2)) / (fov / 1079)
    = (-1.57 + 2.35) / (4.7 / 1079)
    = 0.78 / 0.004356 = 179

    We'll take a small window around these indices to find the minimum distance.
    """
    fov = 4.7
    num_beams = len(scan)
    angle_increment = fov / (num_beams - 1)

    # Left side (pi/2)
    left_angle = np.pi / 2
    left_idx = int((left_angle + fov / 2) / angle_increment)

    # Right side (-pi/2)
    right_angle = -np.pi / 2
    right_idx = int((right_angle + fov / 2) / angle_increment)

    # Window size (approx 10 degrees)
    window_angle = 10 * np.pi / 180
    window_size = int(window_angle / angle_increment)

    left_min = np.min(
        scan[max(0, left_idx - window_size) : min(num_beams, left_idx + window_size)]
    )
    right_min = np.min(
        scan[max(0, right_idx - window_size) : min(num_beams, right_idx + window_size)]
    )

    return left_min, right_min


@njit(cache=True)
def get_heading_error(
    waypoints: np.ndarray, car_position: np.ndarray, car_theta: float
) -> float:
    """
    Calculate the heading error angle between the car's current orientation and the intended path.

    Args:
        waypoints (np.ndarray): Array of waypoints [N, 2] or [N, 3+] with x, y coordinates
        car_position (np.ndarray): Current car position [x, y]
        car_theta (float): Current car orientation in radians (global frame)

    Returns:
        float: Heading error in radians. Positive means car is turning left of intended path,
               negative means car is turning right of intended path.
    """
    # Get nearest point on the waypoint trajectory
    _, _, _, i = nearest_point(car_position, waypoints[:, 0:2])

    # Get the next waypoint to form a line segment
    # Handle wrap-around for circular tracks
    next_i = (i + 1) % len(waypoints)

    # Calculate the direction vector of the intended path
    direction_vector = waypoints[next_i, 0:2] - waypoints[i, 0:2]

    # Calculate the intended heading angle (slope of the line segment)
    intended_theta = np.arctan2(direction_vector[1], direction_vector[0])

    # Calculate the heading error (angular difference)
    heading_error = pi_2_pi(intended_theta - car_theta)

    return heading_error
