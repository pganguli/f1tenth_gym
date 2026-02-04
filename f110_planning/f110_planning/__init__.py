from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from .misc import FlippyPlanner, ManualPlanner, RandomPlanner
from .tracking import PurePursuitPlanner


class Action(NamedTuple):
    steer: float
    speed: float


class BasePlanner(ABC):
    @abstractmethod
    def plan(self, obs: dict[str, list[Any]]) -> Action: ...


__all__ = [
    "Action",
    "BasePlanner",
    "FlippyPlanner",
    "ManualPlanner",
    "RandomPlanner",
    "PurePursuitPlanner",
]
