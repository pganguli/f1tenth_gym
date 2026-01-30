import numpy as np

from .. import Action, BasePlanner
from ..utils import index2Angle, polar2Rect


class WallFollowingAnglePlanner(BasePlanner):
    def __init__(
        self,
        wall_offset: float = 10,
        v_max: float = 5,
        v_min: float = 0.01,
    ):
        self.wall_offset = wall_offset
        self.v_max = v_max
        self.v_min = v_min

    def plan(self, obs):
        obs = obs["scans"][0]
        octantN = len(obs) // 6

        baseIndex = octantN
        rightSide = polar2Rect(obs[baseIndex], index2Angle(baseIndex))
        rightSideOffset: np.ndarray = polar2Rect(
            obs[baseIndex + self.wall_offset], index2Angle(baseIndex + self.wall_offset)
        )
        wallDirection: np.ndarray = rightSideOffset - rightSide
        angle = np.arctan2(wallDirection[1], wallDirection[0]) - np.pi / 2
        speedInterpolation = 2 * abs(angle) / np.pi
        return Action(
            steer=angle,
            speed=self.v_max * (1 - speedInterpolation)
            + self.v_min * speedInterpolation,
        )
