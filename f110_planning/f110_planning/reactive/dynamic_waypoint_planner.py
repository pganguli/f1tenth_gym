"""
Dynamic Waypoint Planner

This planner dynamically computes an imaginary waypoint at each step based on:
- Left and right wall distances (for lateral centering)
- Heading error angle (for angular alignment)
The car follows this continuously updated waypoint to stay centered and aligned.
"""

from typing import Any

import numpy as np

from ..base import Action, BasePlanner
from ..utils import get_actuation, get_heading_error, get_side_distances


class DynamicWaypointPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Dynamic waypoint following planner for F1TENTH

    This planner creates an imaginary waypoint ahead of the car at each timestep,
    positioned to simultaneously:
    1. Center the car between left and right walls
    2. Align the car's heading with the track direction

    By continuously tracking this dynamically computed waypoint, the planner
    achieves behavior similar to Pure Pursuit on actual raceline waypoints.

    Args:
        waypoints (np.ndarray): Reference waypoints for heading error computation
        lookahead_distance (float, default=1.0): Lookahead distance for waypoint placement
        max_speed (float, default=5.0): Maximum speed
        wheelbase (float, default=0.33): Vehicle wheelbase for steering computation
        lateral_gain (float, default=1.0): Gain for lateral centering correction
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        waypoints: np.ndarray,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        wheelbase: float = 0.33,
        lateral_gain: float = 1.0,
    ):
        self.waypoints = waypoints
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.wheelbase = wheelbase
        self.lateral_gain = lateral_gain

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Plan action by computing dynamic waypoint from sensor data.

        Args:
            obs (dict): Observation dictionary containing scans and poses
            ego_idx (int): Index of ego vehicle

        Returns:
            Action: Steering angle and speed command
        """
        scan = obs["scans"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        car_theta = obs["poses_theta"][ego_idx]

        # Get sensor-based metrics
        left_dist, right_dist = get_side_distances(scan)
        heading_error = get_heading_error(self.waypoints, car_position, car_theta)

        # Compute lateral offset to center between walls
        # If right_dist > left_dist, car is closer to left wall, need to move right
        # If left_dist > right_dist, car is closer to right wall, need to move left
        # In vehicle frame: y positive is left
        lateral_error = (left_dist - right_dist) / 2.0

        # Create imaginary waypoint in vehicle frame
        # x: forward, y: left (positive)
        target_x_vehicle = self.lookahead_distance
        target_y_vehicle = self.lateral_gain * lateral_error

        # The waypoint should also account for heading correction
        # heading_error = intended_theta - car_theta
        # If positive: should turn left, place waypoint to the left
        # If negative: should turn right, place waypoint to the right
        target_y_vehicle += 0.5 * heading_error * self.lookahead_distance

        # Create the waypoint position with speed (for compatibility with get_actuation)
        lookahead_point = np.array(
            [
                car_position[0]
                + target_x_vehicle * np.cos(car_theta)
                - target_y_vehicle * np.sin(car_theta),
                car_position[1]
                + target_x_vehicle * np.sin(car_theta)
                + target_y_vehicle * np.cos(car_theta),
                self.max_speed,
            ]
        )

        # Use Pure Pursuit actuation logic
        speed, steering_angle = get_actuation(
            car_theta,
            lookahead_point,
            car_position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return Action(steer=steering_angle, speed=speed)
