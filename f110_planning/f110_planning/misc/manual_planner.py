"""
Manual planner module for keyboard control.
"""

from typing import Any

import numpy as np
from pyglet import display as pyg_display
from pyglet import window as pyg_window

from ..base import Action, BasePlanner


class ManualPlanner(BasePlanner):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Physics-aware Manual Planner for keyboard control.

    This planner uses WASD/Arrow keys to control the car with realistic dynamics:
    - Acceleration and braking rates are applied over time.
    - Steering angle changes at a fixed rate (simulating a steering wheel).
    - States are synchronized with the simulation's timestep.
    """

    @staticmethod
    def _kbd_init():
        """Initialize keyboard handlers by searching for the active pyglet window."""
        display = pyg_display.get_display()
        keys = pyg_window.key.KeyStateHandler()
        windows = display.get_windows()
        if not windows:
            msg = "No pyglet window found. Ensure render() has been called or a window matches."
            raise RuntimeError(msg)
        windows[0].push_handlers(keys)
        return keys

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        s_min: float = -0.4189,
        s_max: float = 0.4189,
        v_min: float = -5.0,
        v_max: float = 20.0,
        accel: float = 8.0,
        decel: float = 15.0,
        steer_rate: float = 1.5,
        dt: float = 0.01,
    ):
        """
        Initialize the manual planner with physical constraints.

        Args:
            s_min (float): Minimum steering angle [rad]
            s_max (float): Maximum steering angle [rad]
            v_min (float): Minimum velocity [m/s]
            v_max (float): Maximum velocity [m/s]
            accel (float): Acceleration rate [m/s^2]
            decel (float): Deceleration/Braking rate [m/s^2]
            steer_rate (float): Rate of change for steering [rad/s]
            dt (float): Simulation timestep for integration
        """
        self.keys = ManualPlanner._kbd_init()
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.accel = accel
        self.decel = decel
        self.steer_rate = steer_rate
        self.dt = dt

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Compute the next action based on current keyboard state and ego velocity.

        Logic:
        - W/S: Increment/Decrement velocity based on accel/decel.
        - A/D: Increment/Decrement steering angle based on steer_rate.
        - No keys: Velocity decays towards zero; steering centers.
        """
        current_speed = obs["linear_vels_x"][ego_idx]
        current_steer = obs["steering_angles"][ego_idx]

        # Speed control (Gas/Brake)
        if self.keys[pyg_window.key.W]:
            # Gas: increase target speed
            speed = min(self.v_max, current_speed + self.accel * self.dt)
        elif self.keys[pyg_window.key.S]:
            # Brake/Reverse: decrease target speed
            speed = max(self.v_min, current_speed - self.decel * self.dt)
        else:
            # Coasting: slowly return to zero speed if no input
            if abs(current_speed) < 0.2:
                speed = 0.0
            else:
                # Slight friction/coasting deceleration
                speed = current_speed - np.sign(current_speed) * (self.accel / 4.0) * self.dt

        # Steering control (Wheel)
        if self.keys[pyg_window.key.A]:
            # Turn left
            steer = min(self.s_max, current_steer + self.steer_rate * self.dt)
        elif self.keys[pyg_window.key.D]:
            # Turn right
            steer = max(self.s_min, current_steer - self.steer_rate * self.dt)
        else:
            # Return-to-center logic
            if abs(current_steer) < self.steer_rate * self.dt:
                steer = 0.0
            else:
                steer = current_steer - np.sign(current_steer) * self.steer_rate * self.dt

        return Action(steer=steer, speed=speed)
