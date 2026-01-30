import numpy as np

from .. import Action, BasePlanner


class LongestPathPlanner(BasePlanner):
    def __init__(self, max_speed: float = 2.0):
        self.max_speed = max_speed

    def plan(self, obs):
        obs = obs["scans"][0]
        quadrantN = len(obs) // 4

        frontIndexStart = quadrantN
        frontIndexEnd = 3 * quadrantN
        maxI = np.argmax(obs[frontIndexStart:frontIndexEnd]) + frontIndexStart
        angle = 2 * np.pi / len(obs) * maxI - np.pi

        return Action(angle, self.max_speed)
