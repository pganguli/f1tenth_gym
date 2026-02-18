"""
Bubble planner for obstacle avoidance.
"""

from typing import Any

import numpy as np

from .. import Action, BasePlanner
from ..utils import index_to_angle


class BubblePlanner(BasePlanner):
    """
    A reactive planner that creates a 'bubble' around the car and avoids obstacles.
    """

    def __init__(self, safety_radius: float = 1.3, avoidance_speed: float = 0.5):
        self.safety_radius = safety_radius
        self.avoidance_speed = avoidance_speed

    # Detect obstacles within the safety radius
    @staticmethod
    def detect_obstacles(
        radius: float, obs: dict[str, Any], ego_idx: int
    ) -> list[tuple[float, float]]:
        """
        Detect obstacles within a given radius using LIDAR data.

        Args:
            radius: The safety radius to check for obstacles.
            obs: Observations from the environment.
            ego_idx: The index of the ego vehicle.

        Returns:
            A list of (angle, distance) pairs for obstacles within the radius.
        """
        lidar_data = obs["scans"][ego_idx]
        obstacles = []
        for i, distance in enumerate(lidar_data):
            angle = index_to_angle(i)
            if distance <= radius:
                obstacles.append((angle, distance))
        return obstacles

    # Find the closest obstacle from a list of obstacles
    @staticmethod
    def find_closest_obstacle(
        obstacles: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """
        Find the obstacle closest to the vehicle.

        Args:
            obstacles: A list of (angle, distance) pairs.

        Returns:
            The (angle, distance) pair of the closest obstacle.
        """
        return min(obstacles, key=lambda v: v[1])

    # Calculate the direction of an obstacle relative to the car
    @staticmethod
    def calculate_obstacle_direction(obstacle_location: tuple[float, float]) -> float:
        """
        Calculate the direction of an obstacle relative to the vehicle heading.

        Args:
            obstacle_location: The (angle, distance) pair of the obstacle.

        Returns:
            The direction to the obstacle in radians.
        """
        obstacle_angle = obstacle_location[0]
        if obstacle_angle > np.pi:
            obstacle_angle -= 2 * np.pi  # convert angle to range [-pi, pi]
        obstacle_direction = -obstacle_angle  # calculate opposite direction
        return obstacle_direction

    # Function to calculate the avoidance direction based on the obstacle direction
    @staticmethod
    def calculate_avoidance_direction(obstacle_direction: float) -> float:
        """
        Calculate the direction to steer to avoid an obstacle.

        Args:
            obstacle_direction: The direction of the obstacle.

        Returns:
            The avoidance steering direction in radians.
        """
        avoidance_direction = obstacle_direction + np.pi  # add pi radians (180 degrees)
        if avoidance_direction > np.pi:
            avoidance_direction -= 2 * np.pi  # convert direction to range [-pi, pi]
        return avoidance_direction

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        """
        Plan the next action to avoid obstacles.

        Args:
            obs: Observations from the environment.
            ego_idx: The index of the ego vehicle.

        Returns:
            The action to take (steer and speed).
        """
        obstacles = BubblePlanner.detect_obstacles(self.safety_radius, obs, ego_idx)
        # If no obstacles, keep moving forward
        if not obstacles:
            Action(steer=0.0, speed=1.0)
        # Calculate the avoidance direction and set steering command
        avoidance_direction = BubblePlanner.calculate_avoidance_direction(
            BubblePlanner.calculate_obstacle_direction(
                BubblePlanner.find_closest_obstacle(obstacles)
            )
        )
        return Action(steer=avoidance_direction, speed=self.avoidance_speed)
