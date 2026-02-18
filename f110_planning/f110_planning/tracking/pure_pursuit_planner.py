"""
Pure Pursuit waypoint tracker
"""

import warnings
from typing import Any, Optional

import numpy as np

from ..base import Action, BasePlanner
from ..utils import get_actuation, get_vehicle_state, intersect_point, nearest_point


class PurePursuitPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992,
    https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track

    Attributes:
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 4]): static list of waypoints
            columns are [x, y, velocity, heading]
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        lookahead_distance: float = 0.8,
        max_speed: float = 5.0,
        waypoints: np.ndarray = np.array([]),
    ):
        self.max_reacquire = 20.0
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.waypoints = waypoints

    def _get_current_waypoint(
        self, lookahead_distance: float, position: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity)
                None if no point is found
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation."
            )

        _, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            _, i2, _ = intersect_point(
                position, lookahead_distance, self.waypoints[:, 0:2], i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.array(
                [self.waypoints[i2, 0], self.waypoints[i2, 1], self.waypoints[i, 2]]
            )
            return current_waypoint

        if nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]

        return None

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            obs (dict): dictionary of observations
            ego_idx (int): index of the ego vehicle

        Returns:
            Action: commanded velocity and steering angle
        """
        vehicle_state = get_vehicle_state(obs, ego_idx)
        position = vehicle_state[:2]
        lookahead_point = self._get_current_waypoint(self.lookahead_distance, position)

        if lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return Action(steer=0.0, speed=0.0)

        lookahead_point[2] = self.max_speed

        speed, steering_angle = get_actuation(
            vehicle_state[2],
            lookahead_point,
            position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return Action(steer=steering_angle, speed=speed)
