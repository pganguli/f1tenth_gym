from .. import Action, BasePlanner


class FlippyPlanner(BasePlanner):
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """

    def __init__(
        self,
        flip_every: int = 1,
        steer: float = 2,
        speed: float = 1,
    ):
        self.flip_every = flip_every
        self.steer = steer
        self.speed = speed
        self.counter = 0

    def plan(self, obs):
        self.counter += 1
        if self.counter % self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return Action(steer=self.steer, speed=self.speed)
