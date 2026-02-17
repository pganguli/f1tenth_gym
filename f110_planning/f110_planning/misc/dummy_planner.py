from typing import Any

import numpy as np

from .. import Action, BasePlanner


class DummyPlanner(BasePlanner):
    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action:
        return Action(steer=np.pi / 2, speed=1)
