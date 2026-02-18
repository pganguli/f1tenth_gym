"""
Random planner module.
"""

import random
from typing import Any

from ..base import Action, BasePlanner


class RandomPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """A planner for generating random controls."""

    def __init__(
        self,
        s_min: float = -0.4189,
        s_max: float = 0.4189,
        v_min: float = -5.0,
        v_max: float = 20.0,
    ):
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        speed = random.uniform(self.v_min, self.v_max)
        steer = random.uniform(self.s_min, self.s_max)
        return Action(steer=steer, speed=speed)
