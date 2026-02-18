"""
DNN planner
"""

import numpy as np

from .. import Action, BasePlanner


class DNNPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    DNN-based planner for F1TENTH.
    """
    LENGTH = 0.58
    LOOKAHEAD_DIST = 3
    MAX_SPEED = 5
    MIN_SPEED = 0

    def __init__(self):
        self.target_point = np.array([0.0, 0.0])

    def plan(self, obs, ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        # angle = np.arctan2(-self.target_point[0], self.target_point[1])
        lidar_model = obs["lidar_model"][ego_idx]
        w, pos, rot = lidar_model[0], lidar_model[1], lidar_model[2]
        target_point = np.array(
            [w / 2, self.LOOKAHEAD_DIST]
        )  # this is relative to track's coordinate frame
        car_rot_x = np.array([np.cos(rot), np.sin(rot)])
        car_rot_y = np.array([np.cos(rot + np.pi / 2), np.sin(rot + np.pi / 2)])
        car_rot = np.column_stack([car_rot_x, car_rot_y])
        self.target_point = car_rot.transpose().dot(target_point - np.array([pos, 0]))
        target_point = self.target_point + np.array(
            [0, self.LENGTH]
        )  # relative to back of car
        angle = np.arctan2(-target_point[0], target_point[1])
        distance = np.linalg.norm(target_point)
        steering_angle = np.arctan(2 * self.LENGTH * np.sin(angle) / distance)

        speed_interpolation = 2 * abs(steering_angle) / np.pi
        speed = self.MAX_SPEED + speed_interpolation * (self.MIN_SPEED - self.MAX_SPEED)

        return Action(steer=steering_angle, speed=speed)
