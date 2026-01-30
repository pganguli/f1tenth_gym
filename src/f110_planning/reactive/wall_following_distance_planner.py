import numpy as np
from simple_pid import PID

from .. import Action, BasePlanner
from ..utils import index2Angle, polar2Rect


class WallFollowingDistancePlanner(BasePlanner):
    def __init__(
        self,
        distance_target: float = 0.31 + 1,
        wall_offset: float = 10,
        v_max: float = 2,
        v_min: float = 2,
    ):
        self.pid = PID(0.1, 0.0, 0.5, setpoint=self.distance_target)
        self.distance_target = distance_target
        self.wall_offset = wall_offset
        self.v_max = v_max
        self.v_min = v_min

    def plan(self, obs):
        obs = obs["scans"][0]
        octantN = len(obs) // 6
        baseIndex = octantN - self.wall_offset
        rightSide = polar2Rect(obs[baseIndex], index2Angle(baseIndex))
        rightSideOffset: np.ndarray = polar2Rect(
            obs[octantN + self.wall_offset], index2Angle(octantN + self.wall_offset)
        )
        print(rightSideOffset)
        wallDirection: np.ndarray = rightSideOffset - rightSide
        wallDistance = np.linalg.norm(
            np.cross(wallDirection, -rightSide)
        ) / np.linalg.norm(wallDirection)

        control = self.pid(float(wallDistance))
        if control is None:
            control = 0
        print("\r\x1b[2K" + f"{wallDistance:.2f}", end="")

        speedInterpolation = 2 * abs(control) / np.pi
        return Action(
            steer=control,
            speed=self.v_max * (1 - speedInterpolation)
            + self.v_min * speedInterpolation,
        )
