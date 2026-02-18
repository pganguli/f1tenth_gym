#!/usr/bin/env python3
"""
DNN Waypoint Follower

Drives the F1TENTH car using trained DNN models for:
- Heading error (phi) prediction
- Wall distance prediction

Replaces ground truth geometric calculations with model inference.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from f110_gym.envs.base_classes import Integrator
from f110_planning.render_callbacks import (
    camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.utils import get_actuation, load_waypoints

from f110_planning import Action, BasePlanner

# -----------------------------------------------------------------------------
# Model Architectures (Must match training scripts)
# -----------------------------------------------------------------------------


class PhiCNN(nn.Module):
    """1D CNN for heading angle prediction (matches train_phi.py)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


class WallCNN(nn.Module):
    """1D CNN for wall distance prediction (matches train_wall.py)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


# -----------------------------------------------------------------------------
# DNN Waypoint Planner
# -----------------------------------------------------------------------------


class DNNWaypointPlanner(BasePlanner):
    """
    Waypoint planner using DNN predictions instead of ground truth.
    """

    def __init__(
        self,
        phi_model_path: str,
        wall_model_path: str,
        data_path: str,
        waypoints: np.ndarray,
        lookahead_distance: float = 1.0,
        max_speed: float = 5.0,
        wheelbase: float = 0.33,
        lateral_gain: float = 1.0,
        device: str = "cpu",
    ):
        self.waypoints = waypoints
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.wheelbase = wheelbase
        self.lateral_gain = lateral_gain
        self.device = torch.device(device)

        # Load Models
        print(f"Loading Phi Model: {phi_model_path}")
        self.phi_model = PhiCNN().to(self.device)
        self.phi_model.load_state_dict(
            torch.load(phi_model_path, map_location=self.device)
        )
        self.phi_model.eval()

        print(f"Loading Wall Model: {wall_model_path}")
        self.wall_model = WallCNN().to(self.device)
        self.wall_model.load_state_dict(
            torch.load(wall_model_path, map_location=self.device)
        )
        self.wall_model.eval()

        # Load Dataset for Normalization Stats
        print(f"Loading Dataset for Normalization: {data_path}")
        data = np.load(data_path)
        scans = data["scans"]
        self.X_mean = scans.mean()
        self.X_std = scans.std() + 1e-6
        print(f"Normalization Stats - Mean: {self.X_mean:.4f}, Std: {self.X_std:.4f}")

        # Precompute angle channel
        self.angles = np.linspace(-1.0, 1.0, 1080)

    def preprocess_scan(self, scan: np.ndarray) -> torch.Tensor:
        """Normalize scan and add angle channel."""
        # 1. Normalize
        scan_norm = (scan - self.X_mean) / self.X_std

        # 2. Add angle channel
        # Scan shape: (1080,) -> need (1, 2, 1080) for model
        input_data = np.stack([scan_norm, self.angles], axis=0)  # (2, 1080)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(
            0
        )  # (1, 2, 1080)

        return input_tensor.to(self.device)

    def plan(self, obs, ego_idx: int) -> Action:
        scan = obs["scans"][ego_idx]
        car_position = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]])
        car_theta = obs["poses_theta"][ego_idx]

        # ---------------------------------------------------------
        # DNN Inference
        # ---------------------------------------------------------
        input_tensor = self.preprocess_scan(scan)

        with torch.no_grad():
            # Predict Phi (Heading Error)
            pred_phi_norm = self.phi_model(input_tensor).item()
            heading_error = pred_phi_norm * np.pi  # Denormalize

            # Predict Wall Distances
            pred_wall_log = self.wall_model(input_tensor).squeeze().cpu().numpy()
            pred_wall = np.exp(pred_wall_log)  # Inverse log
            left_dist = pred_wall[0]
            right_dist = pred_wall[1]

        # ---------------------------------------------------------
        # Control Logic (Same as DynamicWaypointPlanner)
        # ---------------------------------------------------------

        # Lateral error: + means closer to left wall (move right), - means closer to right (move left)
        # Target: roughly 0 (centered)
        lateral_error = (left_dist - right_dist) / 2.0

        # Create imaginary waypoint in vehicle frame
        target_x_vehicle = self.lookahead_distance
        target_y_vehicle = self.lateral_gain * lateral_error

        # Heading correction
        target_y_vehicle += 0.5 * heading_error * self.lookahead_distance

        # Transform to world frame for Pure Pursuit
        lookahead_point = np.array(
            [
                car_position[0]
                + target_x_vehicle * np.cos(car_theta)
                - target_y_vehicle * np.sin(car_theta),
                car_position[1]
                + target_x_vehicle * np.sin(car_theta)
                + target_y_vehicle * np.cos(car_theta),
                self.max_speed,
            ]
        )

        # Get actuation
        speed, steering_angle = get_actuation(
            car_theta,
            lookahead_point,
            car_position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return Action(steer=steering_angle, speed=speed)


# -----------------------------------------------------------------------------
# Main Simulation Loop
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="DNN Waypoint Follower Simulation")
    parser.add_argument(
        "--map-path", default="data/maps/F1/Oschersleben/Oschersleben_map", help="Map path"
    )
    parser.add_argument("--map-ext", default=".png", help="Map extension")
    parser.add_argument(
        "--data-path",
        default="data/datasets/lidar_tracking_Example_pure_pursuit_n4500.npz",
        help="Dataset path for normalization",
    )

    # Model paths (defaults to what we just trained)
    parser.add_argument(
        "--phi-model",
        default="data/models/phi_model_e300_lr5e-4_bs64.pth",
        help="Phi model path",
    )
    parser.add_argument(
        "--wall-model",
        default="data/models/wall_model_e300_lr5e-4_bs64.pth",
        help="Wall model path",
    )

    parser.add_argument("--render-mode", default="human_fast", help="Render mode")
    parser.add_argument("--max-laps", type=int, default=None, help="Max laps")
    return parser.parse_args()


def main():
    args = parse_args()

    # Config hardcoded matching original script for now
    sx = 0.0
    sy = 0.0
    stheta = 1.37079632679

    # Load waypoints for rendering (still useful to see the track)
    waypoints = load_waypoints("data/maps/F1/Oschersleben/Oschersleben_centerline.tsv")

    # Initialize DNN Planner
    planner = DNNWaypointPlanner(
        phi_model_path=args.phi_model,
        wall_model_path=args.wall_model,
        data_path=args.data_path,
        waypoints=waypoints,
        lookahead_distance=1.5,
        max_speed=5.0,
        lateral_gain=1.0,
    )

    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map_path,
        map_ext=args.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=args.render_mode,
        render_fps=60,
        max_laps=args.max_laps,
    )

    # Add render callbacks
    env.unwrapped.add_render_callback(camera_tracking)
    env.unwrapped.add_render_callback(render_lidar)
    # Note: render_side_distances in f110_planning/render_callbacks.py uses ground truth
    # We could potentially modify it to use predicted values, but for now we leave it
    # to compare visual truth vs planner behavior
    env.unwrapped.add_render_callback(render_side_distances)

    # Heading error renderer uses ground truth vs waypoints
    heading_error_renderer = create_heading_error_renderer(waypoints, agent_idx=0)
    env.unwrapped.add_render_callback(heading_error_renderer)

    # Trace
    trace_renderer = create_trace_renderer(agent_idx=0)
    env.unwrapped.add_render_callback(trace_renderer)

    # Dynamic waypoint renderer
    # This visualizes where the planner *would* aim using ground truth
    # We might want to subclass this to visualize where DNN aims, but for now keep original
    dynamic_waypoint_renderer = create_dynamic_waypoint_renderer(planner, agent_idx=0)
    env.unwrapped.add_render_callback(dynamic_waypoint_renderer)

    obs, info = env.reset(options={"poses": np.array([[sx, sy, stheta]])})
    env.render()

    laptime = 0.0
    start = time.time()

    print("\nStarting simulation with DNN Planner...")
    print("Drive safely!")

    done = False
    while not done:
        action = planner.plan(obs, ego_idx=0)

        obs, step_reward, terminated, truncated, info = env.step(
            np.array([[action.steer, action.speed]])
        )

        done = terminated or truncated
        laptime += float(step_reward)
        env.render()

    print(f"Sim elapsed time: {time.time() - start:.2f}")
    print(f"Sim lap time: {laptime:.2f}")

    env.close()


if __name__ == "__main__":
    main()
