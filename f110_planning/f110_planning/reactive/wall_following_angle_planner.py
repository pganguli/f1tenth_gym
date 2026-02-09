import numpy as np
from typing import Any

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

    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action:
        scan_data = obs["scans"][ego_idx]
        octantN = len(scan_data) // 6

        baseIndex = octantN
        rightSide = polar2Rect(scan_data[baseIndex], index2Angle(baseIndex))
        rightSideOffset: np.ndarray = polar2Rect(
            scan_data[int(baseIndex + self.wall_offset)], index2Angle(int(baseIndex + self.wall_offset))
        )
        wallDirection: np.ndarray = rightSideOffset - rightSide
        angle = np.arctan2(wallDirection[1], wallDirection[0]) - np.pi / 2
        speedInterpolation = 2 * abs(angle) / np.pi
        return Action(
            steer=angle,
            speed=self.v_max * (1 - speedInterpolation)
            + self.v_min * speedInterpolation,
        )
