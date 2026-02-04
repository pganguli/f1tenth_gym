import numpy as np
from typing import Any

from .. import Action, BasePlanner


class DummyPlanner(BasePlanner):
    def plan(self, obs: dict[str, Any]) -> Action:
        return Action(steer=np.pi / 2, speed=1)
