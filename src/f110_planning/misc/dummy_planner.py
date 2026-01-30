import numpy as np

from .. import Action, BasePlanner


class DummyPlanner(BasePlanner):
    def plan(self, obs):
        return Action(steer=np.pi / 2, speed=1)
