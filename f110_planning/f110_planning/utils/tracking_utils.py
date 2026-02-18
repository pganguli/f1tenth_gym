"""
Utility functions for tracking planners.
"""

from typing import Any

import numpy as np
from numba import njit

from .geometry_utils import pi_2_pi
from .pure_pursuit_utils import nearest_point


def get_vehicle_state(obs: dict[str, Any], ego_idx: int) -> np.ndarray:
    """
    Extracts the vehicle state from the observation dictionary.

    Args:
        obs (dict): The observation dictionary.
        ego_idx (int): The index of the ego vehicle.

    Returns:
        np.ndarray: The vehicle state [x, y, theta, velocity].
    """
    return np.array(
        [
            obs["poses_x"][ego_idx],
            obs["poses_y"][ego_idx],
            obs["poses_theta"][ego_idx],
            obs["linear_vels_x"][ego_idx],
        ]
    )


@njit(cache=True)
def calculate_tracking_errors(
    vehicle_state: np.ndarray, waypoints: np.ndarray, wheelbase: float
) -> tuple[float, float, int, float]:
    """
    Calculates tracking errors (heading and crosstrack) relative to the front axle.

    Args:
        vehicle_state (np.ndarray): [x, y, heading, velocity] of the vehicle.
        waypoints (np.ndarray): waypoints to track [x, y, velocity, heading, ...].
        wheelbase (float): The wheelbase of the vehicle.

    Returns:
        tuple: (theta_e, ef, target_index, goal_velocity)
            theta_e (float): heading error
            ef (float): lateral crosstrack error at the front axle
            target_index (int): index of the nearest waypoint
            goal_velocity (float): target velocity at the nearest waypoint
    """
    # distance to the closest point to the front axle center
    fx = vehicle_state[0] + wheelbase * np.cos(vehicle_state[2])
    fy = vehicle_state[1] + wheelbase * np.sin(vehicle_state[2])
    position_front_axle = np.array([fx, fy])
    nearest_point_front, _, _, target_index = nearest_point(
        position_front_axle, waypoints[:, 0:2]
    )
    vec_dist_nearest_point = position_front_axle - nearest_point_front

    # crosstrack error
    front_axle_vec_rot_90 = np.array(
        [
            [np.cos(vehicle_state[2] - np.pi / 2.0)],
            [np.sin(vehicle_state[2] - np.pi / 2.0)],
        ]
    )
    ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

    # heading error
    theta_raceline = waypoints[target_index, 3]
    theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

    # target velocity
    goal_velocity = waypoints[target_index, 2]

    return theta_e, ef[0], target_index, goal_velocity
