"""
Stanley waypoint tracker
"""

from typing import Any

import numpy as np

from ..base import Action, BasePlanner
from ..utils import calculate_tracking_errors, get_vehicle_state


class StanleyPlanner(BasePlanner):
    """
    Front Wheel Feedback Controller (Stanley) for tracking the vehicle path.
    References:
    - Stanley: http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf
    - Tracking:
      https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

    Args:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 4], optional, default=None): waypoints to track
            columns are [x, y, velocity, heading]

    Attributes:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 4], optional, default=None): waypoints to track
            columns are [x, y, velocity, heading]
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        waypoints: np.ndarray = np.array([]),
        k_path: float = 5.0,
    ):
        self.wheelbase = wheelbase
        self.waypoints = waypoints
        self.k_path = k_path

    def calc_theta_and_ef(  # pylint: disable=too-many-locals
        self, vehicle_state: np.ndarray, waypoints: np.ndarray
    ) -> tuple[float, float, int, float]:
        """
        Calculate the heading and cross-track errors
        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 4]): waypoints to track [x, y, velocity, heading]
        """
        theta_e, ef, target_index, goal_velocity = calculate_tracking_errors(
            vehicle_state, waypoints, self.wheelbase
        )

        return theta_e, ef, target_index, goal_velocity

    def controller(
        self, vehicle_state: np.ndarray, waypoints: np.ndarray, k_path: float
    ) -> tuple[float, float]:
        """
        Front Wheel Feedback Controller to track the path.
        Based on the heading error theta_e and the crosstrack error ef we
        calculate the steering angle.
        Returns the optimal steering angle delta is P-Controller with the
        proportional gain k.

        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 4]): waypoints to track
            k_path (float): proportional gain

        Returns:
            theta_e (float): heading error
            ef (numpy.ndarray [2, ]): crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_velocity (float): target velocity
        """

        theta_e, ef, _, goal_velocity = self.calc_theta_and_ef(
            vehicle_state, waypoints
        )

        # Calculate final steering angle/ control input in [rad]:
        # Steering Angle based on error + heading error
        cte_front = np.atan2(k_path * ef, vehicle_state[3])
        delta = cte_front + theta_e

        return delta, goal_velocity

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Plan function

        Args:
            obs (dict): dictionary of observations
            ego_idx (int): index of the ego vehicle

        Returns:
            Action: steering_angle and speed
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation."
            )
        vehicle_state = get_vehicle_state(obs, ego_idx)
        steering_angle, speed = self.controller(
            vehicle_state, self.waypoints, self.k_path
        )
        return Action(steering_angle, speed)
