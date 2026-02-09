from abc import ABC, abstractmethod
from typing import Any, NamedTuple


class Action(NamedTuple):
    steer: float
    speed: float


class BasePlanner(ABC):
    @abstractmethod
    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action: ...


# Import submodules AFTER defining Action and BasePlanner to avoid circular imports
from .misc import FlippyPlanner, ManualPlanner, RandomPlanner
from .tracking import PurePursuitPlanner

__all__ = [
    "Action",
    "BasePlanner",
    "FlippyPlanner",
    "ManualPlanner",
    "RandomPlanner",
    "PurePursuitPlanner",
]
