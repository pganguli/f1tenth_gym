"""
Longest path planner module.
"""

from typing import Any

import numpy as np

from ..base import Action, BasePlanner


class LongestPathPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Reactive planner that steers towards the longest LIDAR range.
    """

    def __init__(self, max_speed: float = 2.0):
        self.max_speed = max_speed

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        scan_data = obs["scans"][ego_idx]
        quadrant_n = len(scan_data) // 4

        front_index_start = quadrant_n
        front_index_end = 3 * quadrant_n
        max_i = np.argmax(scan_data[front_index_start:front_index_end]) + front_index_start
        angle = 2 * np.pi / len(scan_data) * max_i - np.pi

        return Action(angle, self.max_speed)
