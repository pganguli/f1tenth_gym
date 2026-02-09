import numpy as np
from typing import Any

from .. import Action, BasePlanner
from ..utils import index2Angle


class BubblePlanner(BasePlanner):
    def __init__(self, safety_radius: float = 1.3, avoidance_speed: float = 0.5):
        self.safety_radius = safety_radius
        self.avoidance_speed = avoidance_speed

    # Detect obstacles within the safety radius
    @staticmethod
    def detect_obstacles(radius: float, obs: dict[str, Any], ego_idx: int) -> list[tuple[float, float]]:
        lidar_data = obs["scans"][ego_idx]
        obstacles = []
        for i, distance in enumerate(lidar_data):
            angle = index2Angle(i)
            if distance <= radius:
                obstacles.append((angle, distance))
        return obstacles

    # Find the closest obstacle from a list of obstacles
    @staticmethod
    def find_closest_obstacle(obstacles: list[tuple[float, float]]) -> tuple[float, float]:
        return min(obstacles, key=lambda v: v[1])

    # Calculate the direction of an obstacle relative to the car
    @staticmethod
    def calculate_obstacle_direction(obstacle_location: tuple[float, float]) -> float:
        obstacle_angle = obstacle_location[0]
        if obstacle_angle > np.pi:
            obstacle_angle -= 2 * np.pi  # convert angle to range [-pi, pi]
        obstacle_direction = -obstacle_angle  # calculate opposite direction
        return obstacle_direction

    # Function to calculate the avoidance direction based on the obstacle direction
    @staticmethod
    def calculate_avoidance_direction(obstacle_direction: float) -> float:
        avoidance_direction = obstacle_direction + np.pi  # add pi radians (180 degrees)
        if avoidance_direction > np.pi:
            avoidance_direction -= 2 * np.pi  # convert direction to range [-pi, pi]
        return avoidance_direction

    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action:
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
