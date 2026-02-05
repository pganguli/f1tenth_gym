from pyglet import display as pyg_display
from pyglet import window as pyg_window

from typing import Any
from .. import Action, BasePlanner


class ManualPlanner(BasePlanner):
    """
    Example Planner
    """

    @staticmethod
    def _kbd_init():
        display = pyg_display.get_display()
        keys = pyg_window.key.KeyStateHandler()
        windows = display.get_windows()
        if not windows:
            raise RuntimeError("No pyglet window found")
        windows[0].push_handlers(keys)
        return keys

    def __init__(self, s_min: float, s_max: float, v_max: float):
        self.keys = ManualPlanner._kbd_init()
        self.s_min = s_min
        self.s_max = s_max
        self.v_max = v_max

    def plan(self, obs: dict[str, Any]) -> Action:
        speed = 0.0
        steer = 0.0
        if self.keys[pyg_window.key.W]:
            speed = self.v_max / 8
        if self.keys[pyg_window.key.A]:
            steer = self.s_max / 2
        if self.keys[pyg_window.key.D]:
            steer = self.s_min / 2
        return Action(steer=steer, speed=speed)
