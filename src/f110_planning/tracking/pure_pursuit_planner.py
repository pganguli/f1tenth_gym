"""
Pure Pursuit waypoint tracker
"""

import warnings

import numpy as np

from .. import Action, BasePlanner
from ..utils import get_actuation, intersect_point, nearest_point


class PurePursuitPlanner(BasePlanner):
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track

    Attributes:
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        lookahead_distance: float = 0.8,
        waypoints: np.ndarray = np.array([]),
    ):
        self.max_reacquire = 20.0
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.waypoints = waypoints

    def _get_current_waypoint(self, lookahead_distance: float, position):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation."
            )

        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(
                position, lookahead_distance, self.waypoints[:, 0:2], i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.array(
                [self.waypoints[i2, 0], self.waypoints[i2, 1], self.waypoints[i, 2]]
            )
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

    def plan(self, obs):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """
        position = np.array([obs["pose_x"][0], obs["pose_y"][0]])
        lookahead_point = self._get_current_waypoint(self.lookahead_distance, position)

        if lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return Action(steer=0.0, speed=0.0)

        speed, steering_angle = get_actuation(
            obs["pose_theta"][0],
            lookahead_point,
            position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return Action(steer=steering_angle, speed=speed)
