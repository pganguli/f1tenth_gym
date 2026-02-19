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
    Adaptive Dynamic waypoint following planner for F1TENTH

    This planner creates an imaginary waypoint ahead of the car at each timestep,
    positioned to simultaneously:
    1. Center the car between left and right walls (Reactive centering)
    2. Align the car's heading with the track direction (Heading alignment)

    The lookahead distance and target speed are adjusted adaptively based on 
    the car's current velocity and the local track curvature. By continuously 
    tracking this dynamically computed waypoint, the planner achieves behavior 
    similar to Pure Pursuit on actual raceline waypoints without needing a 
    pre-planned global path.

    Args:
        waypoints (np.ndarray): Reference waypoints for heading error computation
        lookahead_distance (float, default=1.0): Tuning gain for adaptive lookahead. 
            Scaling this increases/decreases the speed-based lookahead window.
        max_speed (float, default=5.0): Maximum speed on straight sections. 
            Planner will automatically decelerate for turns and centering maneuvers.
        wheelbase (float, default=0.33): Vehicle wheelbase for steering computation.
        lateral_gain (float, default=1.0): Multiplier for lateral centering aggressiveness.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        waypoints: np.ndarray,
        lookahead_distance: float = 1.0,  # Tuning gain for adaptive velocity-based lookahead
        max_speed: float = 5.0,
        wheelbase: float = 0.33,
        lateral_gain: float = 1.0,
    ):
        self.waypoints = waypoints
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.wheelbase = wheelbase
        self.lateral_gain = lateral_gain
        self.last_target_point = None  # To be picked up by renderer

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
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
        current_speed = obs["linear_vels_x"][ego_idx]

        # Get sensor-based metrics
        left_dist, right_dist = get_side_distances(scan)
        heading_error = get_heading_error(self.waypoints, car_position, car_theta)

        # 1. Adaptive Lookahead Optimization
        # Rule: Look further ahead as speed increases to smooth the path,
        # but stay closer in slow/tight sections for precision.
        # We use self.lookahead_distance as a tuning factor for the speed-based gain.
        # Constant '0.8': Minimum lookahead distance (L_min)
        lookahead = max(0.8, (self.lookahead_distance / 5.0) * current_speed + 0.5)

        # 2. Curvature-based Speed Optimization
        # We estimate the 'straightness' of the track ahead.
        # Constant '0.8': Aggressiveness of curvature detection.
        # Large heading errors/lateral offsets indicate a complex section.
        curvature_proxy = np.abs(heading_error) + (
            np.abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
        )

        # Optimize max speed based on curvature
        # If curvature is low (straight), go up to self.max_speed.
        # If curvature is high, drop towards a safe minimum.
        dynamic_limit = self.max_speed * (1.0 / (1.0 + 2.0 * curvature_proxy))
        target_speed = max(2.5, dynamic_limit) # Ensure we never crawl too slowly

        # Compute lateral error (positive: car too far right, needs to move left towards center)
        lateral_error = (left_dist - right_dist) / 2.0

        # Create imaginary waypoint in vehicle frame
        # x: forward, y: left (positive)
        target_x_vehicle = lookahead
        target_y_vehicle = self.lateral_gain * (lateral_error + np.sin(heading_error) * lookahead)

        # Create the waypoint position (global frame)
        self.last_target_point = np.array(
            [
                car_position[0]
                + target_x_vehicle * np.cos(car_theta)
                - target_y_vehicle * np.sin(car_theta),
                car_position[1]
                + target_x_vehicle * np.sin(car_theta)
                + target_y_vehicle * np.cos(car_theta),
                target_speed,
            ]
        )

        # Use Pure Pursuit actuation logic
        _, steering_angle = get_actuation(
            car_theta,
            self.last_target_point,
            car_position,
            lookahead,
            self.wheelbase,
        )

        # 3. Dynamic Slip/Stability Penalty
        # If we are steering hard at high speeds, cap speed to avoid wash out.
        stability_limit = 0.4189 * self.max_speed
        stability_factor = 1.0 - (np.abs(steering_angle) * current_speed / stability_limit)
        speed = target_speed * max(0.4, stability_factor)

        return Action(steer=steering_angle, speed=speed)
