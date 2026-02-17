from typing import Any

import numpy as np

from .. import Action, BasePlanner


class LongestPathPlanner(BasePlanner):
    def __init__(self, max_speed: float = 2.0):
        self.max_speed = max_speed

    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action:
        scan_data = obs["scans"][ego_idx]
        quadrantN = len(scan_data) // 4

        frontIndexStart = quadrantN
        frontIndexEnd = 3 * quadrantN
        maxI = np.argmax(scan_data[frontIndexStart:frontIndexEnd]) + frontIndexStart
        angle = 2 * np.pi / len(scan_data) * maxI - np.pi

        return Action(angle, self.max_speed)
