import numpy as np

from .. import Action, BasePlanner
from typing import Any


class LongestPathPlanner(BasePlanner):
    def __init__(self, max_speed: float = 2.0):
        self.max_speed = max_speed

    def plan(self, obs: dict[str, Any]) -> Action:
        scan_data = obs["scans"][0]
        quadrantN = len(scan_data) // 4

        frontIndexStart = quadrantN
        frontIndexEnd = 3 * quadrantN
        maxI = np.argmax(scan_data[frontIndexStart:frontIndexEnd]) + frontIndexStart
        angle = 2 * np.pi / len(scan_data) * maxI - np.pi

        return Action(angle, self.max_speed)
