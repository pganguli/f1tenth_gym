from enum import Enum


class Integrator(Enum):
    """
    Integrator Enum for selecting integration method
    """

    RK4 = 1
    Euler = 2
