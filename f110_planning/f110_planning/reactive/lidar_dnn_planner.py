"""
PyTorch-based DNN planner for F1TENTH.
Uses trained models to predict wall distances and heading errors for navigation.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base import Action, BasePlanner
from ..utils import F110_MAX_STEER, F110_WHEELBASE, get_actuation


class LidarDNNPlanner(BasePlanner):
    """
    DNN-based planner using PyTorch models for LiDAR-based navigation.
    Matches DynamicWaypointPlanner logic exactly, but uses DNN predictions.
    """

    def __init__(
        self,
        left_model_path: Optional[str] = None,
        right_model_path: Optional[str] = None,
        heading_model_path: Optional[str] = None,
        wall_model_path: Optional[str] = None,
        arch_id: int = 5,
        heading_arch_id: Optional[int] = None,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        lateral_gain: float = 1.0,
    ):
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.lateral_gain = lateral_gain
        self.wheelbase = F110_WHEELBASE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_target_point = None  # To be picked up by renderer

        # Load models
        # Priority: wall_model (dual output) > left/right separate models
        self.wall_model = self._load_model(wall_model_path, arch_id, task="wall")
        
        self.left_model = self._load_model(left_model_path, arch_id, task="heading")
        self.right_model = self._load_model(right_model_path, arch_id, task="heading")
        
        # Heading error always uses heading task
        h_arch = heading_arch_id if heading_arch_id is not None else arch_id
        self.heading_model = self._load_model(heading_model_path, h_arch, task="heading")

    def _load_model(self, path: Optional[str], arch_id: int, task: str = "heading"):
        if not path:
            return None
        from ..utils.nn_models import get_architecture

        model = get_architecture(arch_id, task=task)
        # Use weights_only=True for security and to suppress warnings
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, model, scan):
        if model is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(scan).float().unsqueeze(0).unsqueeze(0).to(self.device)
            # Normalization consistent with training (range is approx 0-10m)
            x = torch.clip(x / 10.0, 0, 1)
            out = model(x)
            if out.shape[1] > 1:
                return out.cpu().numpy().flatten()
            return out.item()

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:
        scan = obs["scans"][ego_idx]
        car_theta = obs["poses_theta"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        current_speed = obs["linear_vels_x"][ego_idx]

        # 1. Predict geometric features using DNNs
        if self.wall_model is not None:
            wall_dists = self.predict(self.wall_model, scan)
            left_dist, right_dist = wall_dists[0], wall_dists[1]
        else:
            left_dist = self.predict(self.left_model, scan) or 0.0
            right_dist = self.predict(self.right_model, scan) or 0.0
            
        heading_error = self.predict(self.heading_model, scan) or 0.0

        # 2. Match DynamicWaypointPlanner Logic Exactly

        # Adaptive Lookahead Optimization
        lookahead = max(0.8, (self.lookahead_distance / 5.0) * current_speed + 0.5)

        # Curvature-based Speed Optimization
        curvature_proxy = np.abs(heading_error) + (
            np.abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
        )

        dynamic_limit = self.max_speed * (1.0 / (1.0 + 2.0 * curvature_proxy))
        target_speed = max(2.5, dynamic_limit)

        lateral_error = (left_dist - right_dist) / 2.0

        # Create imaginary waypoint in vehicle frame
        target_x_vehicle = lookahead
        target_y_vehicle = self.lateral_gain * (
            lateral_error + np.sin(heading_error) * lookahead
        )

        # Create target point in global frame for Pure Pursuit logic
        self.last_target_point = np.array(
            [
                car_position[0]
                + target_x_vehicle * np.cos(car_theta)
                - target_y_vehicle * np.sin(car_theta),
                car_position[1]
                + target_x_vehicle * np.sin(car_theta)
                + target_y_vehicle * np.cos(car_theta),
                target_speed,
            ]
        )

        # Use Pure Pursuit actuation logic
        _, steering_angle = get_actuation(
            car_theta,
            self.last_target_point,
            car_position,
            lookahead,
            self.wheelbase,
        )

        # Dynamic Slip/Stability Penalty
        stability_limit = F110_MAX_STEER * self.max_speed
        stability_factor = 1.0 - (
            np.abs(steering_angle) * current_speed / stability_limit
        )
        speed = target_speed * max(0.4, stability_factor)

        return Action(steer=steering_angle, speed=speed)
