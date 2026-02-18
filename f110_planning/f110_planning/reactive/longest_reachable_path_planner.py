"""
Longest reachable path planner module.
"""

from typing import Any

import numpy as np
from simple_pid import PID

from .. import Action, BasePlanner


class LongestReachablePathPlanner(BasePlanner):  # pylint: disable=too-few-public-methods
    """
    Reactive planner that finds the longest reachable path by checking for obstacles
    within the car's width along a candidate direction.
    """

    @staticmethod
    def get_coordinates(r: float, angle: float) -> np.ndarray:
        """Converts polar coordinates to Cartesian coordinates."""
        return np.array([r * np.cos(angle), r * np.sin(angle)])

    @staticmethod
    def index_to_angle(i: int, max_index: int, start_angle: float, end_angle: float) -> float:
        """Converts a LIDAR scan index to an angle in radians."""
        angle_range = end_angle - start_angle
        return i / max_index * angle_range + start_angle

    def __init__(
        self,
        v_max: float = 6,
        v_min: float = 4,
        car_width: float = 0.31 + 0.05,
        adjustment_step: int = 5,
    ):
        self.pid = PID(-1, 0, 0, setpoint=0)
        self.v_max = v_max
        self.v_min = v_min
        self.car_width = car_width
        self.adjustment_step = adjustment_step

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        """
        Plans the next action by finding the longest path that is wide enough for the car.
        """
        scan_data = obs["scans"][ego_idx]
        octant_n = len(scan_data) // 6

        front_index_start = 1 * octant_n
        front_index_end = 5 * octant_n

        max_i = np.argmax(scan_data[front_index_start:front_index_end]) + front_index_start
        original_i = max_i
        lidar_origin = np.array([0, 0])
        car_width_offset = np.array([self.car_width, 0])
        p0 = lidar_origin - car_width_offset
        p2 = lidar_origin + car_width_offset

        # iOffset = 0
        i_direction = (
            1
            if scan_data[original_i + self.adjustment_step]
            > scan_data[original_i - self.adjustment_step]
            else -1
        )
        switched_direction = False
        while True:
            distant_point = LongestReachablePathPlanner.get_coordinates(
                scan_data[max_i],
                LongestReachablePathPlanner.index_to_angle(
                    int(max_i), len(scan_data) - 1, -np.pi / 4, 5 * np.pi / 4
                ),
            )
            p1 = distant_point - car_width_offset
            basis = np.linalg.inv(np.column_stack([p2 - p0, p1 - p0]))
            num_obstacles = 0

            for i in range(0, len(scan_data) - 1):
                dist = scan_data[i]
                angle = LongestReachablePathPlanner.index_to_angle(
                    i, len(scan_data) - 1, -np.pi / 4, 5 * np.pi / 4
                )
                point = LongestReachablePathPlanner.get_coordinates(dist, angle)
                change_of_basis = basis.dot(point - p0)
                in_region = 0 <= change_of_basis[0] <= 1 and 0 <= change_of_basis[1] <= 1
                if in_region and change_of_basis[1] < 0.8:
                    num_obstacles += 1
                    if max_i != original_i:
                        break

            if num_obstacles == 0:
                # we found an angle where there were no other points in the path
                break

            max_i += self.adjustment_step * i_direction
            max_i = np.clip(max_i, 0, len(scan_data) - 1)
            if max_i in (0, len(scan_data) - 1):
                if not switched_direction:
                    i_direction *= -1
                    max_i = original_i + self.adjustment_step * i_direction
                else:
                    print("could not find adjustment")
                    break

        angle = (
            LongestReachablePathPlanner.index_to_angle(
                int(max_i), len(scan_data) - 1, -np.pi / 4, 5 * np.pi / 4
            )
            - np.pi / 2
        )
        control = self.pid(float(angle))
        if control is None:
            control = angle
        speed_interpolation = 2 * abs(control) / np.pi
        return Action(
            steer=control,
            speed=self.v_max * (1 - speed_interpolation)
            + self.v_min * speed_interpolation,
        )
