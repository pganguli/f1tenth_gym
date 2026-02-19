#!/usr/bin/env python3
"""
Simulation script for waypoint tracking with multiple agents.
Supports single-agent hybrid control (default) and multi-agent waypoint following.
"""

import argparse
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.misc import HybridPlanner, ManualPlanner
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_heading_error_renderer,
    create_trace_renderer,
    create_waypoint_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.tracking import PurePursuitPlanner
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

# Predefined color palette for agent traces
COLOR_PALETTE = {
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
}
TRACE_COLORS = ["yellow", "cyan", "magenta", "red", "green", "blue"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified waypoint tracking simulation."
    )

    parser.add_argument(
        "--waypoints",
        type=str,
        nargs="+",
        default=[DEFAULT_WAYPOINTS],
        help=f"One or more waypoint files. Each specifies an agent. Default: {DEFAULT_WAYPOINTS}",
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Enable hybrid control (Manual override) for Agent 0. Default: True",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_false",
        dest="hybrid",
        help="Disable hybrid control for Agent 0.",
    )

    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP,
        help=f"Path to map file. Default: {DEFAULT_MAP}",
    )

    parser.add_argument(
        "--map-ext",
        type=str,
        default=DEFAULT_MAP_EXT,
        help=f"Map file extension. Default: {DEFAULT_MAP_EXT}",
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
        choices=["human", "human_fast", "None"],
        default=DEFAULT_RENDER_MODE,
        help=f"Render mode. Default: {DEFAULT_RENDER_MODE}",
    )

    parser.add_argument(
        "--render-fps",
        type=int,
        default=DEFAULT_RENDER_FPS,
        help=f"Render FPS. Default: {DEFAULT_RENDER_FPS}",
    )

    return parser.parse_args()


def main():  # pylint: disable=too-many-locals, too-many-statements
    """
    Main function to run the waypoint tracking simulation.
    """
    args = parse_args()

    # Convert "None" string to None object
    render_mode = None if args.render_mode == "None" else args.render_mode

    # Load waypoints for all agents
    agent_waypoints = [load_waypoints(w) for w in args.waypoints]
    num_agents = len(agent_waypoints)

    # Hybrid control requires a window, so disable it if headless
    enable_hybrid = args.hybrid and render_mode is not None

    # Initialize environment
    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=num_agents,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=render_mode,
        render_fps=args.render_fps,
        max_laps=None,
    )

    # Reset environment once to initialize window if needed
    poses = np.zeros((num_agents, 3))
    for i in range(num_agents):
        poses[i] = [args.start_x, args.start_y, args.start_theta]
    
    obs, _ = env.reset(options={"poses": poses})
    if render_mode:
        env.render()

    # Setup planners (now that window is potentially ready)
    planners = []
    for i in range(num_agents):
        # Base autonomous planner
        auto_planner = PurePursuitPlanner(waypoints=agent_waypoints[i])
        
        # If hybrid is enabled for Agent 0, wrap it
        if i == 0 and enable_hybrid:
            manual_planner = ManualPlanner()
            planner = HybridPlanner(manual_planner, auto_planner)
        else:
            planner = auto_planner
            
        planners.append(planner)

    # Add render callbacks
    if render_mode:
        env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
        env.unwrapped.add_render_callback(render_lidar)
        env.unwrapped.add_render_callback(render_side_distances)
        
        for i in range(num_agents):
            color_name = TRACE_COLORS[i % len(TRACE_COLORS)]
            color = COLOR_PALETTE[color_name]
            
            # Trace for each agent
            env.unwrapped.add_render_callback(
                create_trace_renderer(agent_idx=i, color=color, max_points=10000)
            )
            
            # Render waypoints for each agent (if not too many)
            if num_agents <= 3 and agent_waypoints[i].size > 0:
                wp_color = color + (64,) # Add transparency
                env.unwrapped.add_render_callback(
                    create_waypoint_renderer(agent_waypoints[i], color=wp_color)
                )

        if agent_waypoints[0].size > 0:
            env.unwrapped.add_render_callback(
                create_heading_error_renderer(agent_waypoints[0], agent_idx=0)
            )

    print(f"Starting simulation with {num_agents} agent(s)...")
    laptimes = np.zeros(num_agents)
    start_time = time.time()

    done = False
    try:
        while not done:
            actions = []
            for i in range(num_agents):
                # Plan for each agent
                action = planners[i].plan(obs, ego_idx=i)
                actions.append([action.steer, action.speed])
            
            obs, rewards, terminated, truncated, _ = env.step(np.array(actions))
            
            # termination is usually per-agent in f110_gym, but simplified here
            # to end when any agent (or specifically agent 0) is done if needed.
            # terminated/truncated are boolean arrays in multi-agent f110_gym
            if isinstance(terminated, np.ndarray):
                done = terminated.any() or truncated.any()
                laptimes += rewards
            else:
                done = terminated or truncated
                laptimes[0] += rewards

            if render_mode:
                env.render()
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    total_real_time = time.time() - start_time
    print(f"\nSimulation Finished:")
    for i in range(num_agents):
        print(f"Agent {i} Laptime: {laptimes[i]:.2f}s")
    print(f"Real Time: {total_real_time:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
