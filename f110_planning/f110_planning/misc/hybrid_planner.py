"""
Hybrid planner module combining manual and autonomous control.
"""

from typing import Any

from pyglet import window as pyg_window

from ..base import Action, BasePlanner
from .manual_planner import ManualPlanner


class HybridPlanner(BasePlanner):
    """
    A planner that switches between manual and autonomous control.
    """

    def __init__(self, manual_planner: ManualPlanner, auto_planner: BasePlanner):
        """
        Initialize the HybridPlanner.

        Args:
            manual_planner (ManualPlanner): The manual controller.
            auto_planner (BasePlanner): The autonomous controller (e.g., PurePursuit).
        """
        self.manual = manual_planner
        self.auto = auto_planner

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Plan action by choosing between manual and autonomous input.

        Args:
            obs (dict): Observation dictionary.
            ego_idx (int): Index of the vehicle.

        Returns:
            Action: Computed action.
        """
        keys = self.manual.keys
        if (
            keys[pyg_window.key.W]
            or keys[pyg_window.key.A]
            or keys[pyg_window.key.D]
            or keys[pyg_window.key.S]
        ):
            return self.manual.plan(obs, ego_idx)
        return self.auto.plan(obs, ego_idx)
