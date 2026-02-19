#!/usr/bin/env python3
"""
Simulation script for waypoint following with Dynamic Waypoint Planner.
Visualizes the planner's dynamic target points and vehicle trace.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.reactive import DynamicWaypointPlanner
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.utils import load_waypoints

# Default configuration
DEFAULT_MAP = "data/maps/F1/Oschersleben/Oschersleben_map"
DEFAULT_MAP_EXT = ".png"
DEFAULT_WAYPOINTS = "data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"
DEFAULT_START_X = 0.0
DEFAULT_START_Y = 0.0
DEFAULT_START_THETA = 2.85
DEFAULT_RENDER_MODE = "human_fast"
DEFAULT_RENDER_FPS = 60
DEFAULT_PLANNER = "pure_pursuit"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Follow waypoints with Dynamic Waypoint Planner."
    )

    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP,
        help=f"Path to map file (without extension). Default: {DEFAULT_MAP}",
    )

    parser.add_argument(
        "--map-ext",
        type=str,
        default=DEFAULT_MAP_EXT,
        help=f"Map file extension. Default: {DEFAULT_MAP_EXT}",
    )

    parser.add_argument(
        "--waypoints",
        type=str,
        default=DEFAULT_WAYPOINTS,
        help=f"Path to waypoints file. Default: {DEFAULT_WAYPOINTS}",
    )

    parser.add_argument(
        "--start-x",
        type=float,
        default=DEFAULT_START_X,
        help=f"Starting X position. Default: {DEFAULT_START_X}",
    )

    parser.add_argument(
        "--start-y",
        type=float,
        default=DEFAULT_START_Y,
        help=f"Starting Y position. Default: {DEFAULT_START_Y}",
    )

    parser.add_argument(
        "--start-theta",
        type=float,
        default=DEFAULT_START_THETA,
        help=f"Starting orientation (radians). Default: {DEFAULT_START_THETA}",
    )

    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "human_fast", None],
        default=DEFAULT_RENDER_MODE,
        help=f"Render mode for visualization. Default: {DEFAULT_RENDER_MODE}",
    )

    parser.add_argument(
        "--render-fps",
        type=int,
        default=DEFAULT_RENDER_FPS,
        help=f"Render FPS. Default: {DEFAULT_RENDER_FPS}",
    )

    return parser.parse_args()


def main():  # pylint: disable=too-many-locals
    """
    Main function to run the dynamic waypoint following simulation.
    """
    args = parse_args()

    waypoints = load_waypoints(args.waypoints)

    planner = DynamicWaypointPlanner(
        waypoints=waypoints, lookahead_distance=1.5, max_speed=5.0, lateral_gain=1.0
    )

    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=args.render_mode if args.render_mode != "None" else None,
        render_fps=args.render_fps,
        max_laps=None,
    )

    env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)

    heading_error_renderer = create_heading_error_renderer(waypoints, agent_idx=0)
    env.unwrapped.add_render_callback(heading_error_renderer)

    dynamic_waypoint_renderer = create_dynamic_waypoint_renderer(planner, agent_idx=0)
    env.unwrapped.add_render_callback(dynamic_waypoint_renderer)

    trace_renderer = create_trace_renderer(agent_idx=0)
    env.unwrapped.add_render_callback(trace_renderer)

    obs, _ = env.reset(
        options={"poses": np.array([[args.start_x, args.start_y, args.start_theta]])}
    )
    env.render()

    laptime = 0.0
    start = time.time()

    done = False
    while not done:
        action = planner.plan(obs, ego_idx=0)
        speed, steer = action.speed, action.steer
        obs, step_reward, terminated, truncated, _ = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += float(step_reward)
        env.render()

    print(f"Sim elapsed time: {time.time() - start:.2f}")
    print(f"Sim lap time: {laptime:.2f}")

    env.close()


if __name__ == "__main__":
    main()
