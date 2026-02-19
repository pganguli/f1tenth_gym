#!/usr/bin/env python3
"""
Unified noisy waypoint following simulation.

This script consolidates waypoint_follow_1.py, waypoint_follow_2.py, and waypoint_follow_3.py
into a single parameterized script that supports multiple agents with different waypoint files.
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints

# Default configuration
DEFAULT_MAP = "data/maps/F1/Oschersleben/Oschersleben_map"
DEFAULT_MAP_EXT = ".png"
DEFAULT_WAYPOINTS = ["data/maps/F1/Oschersleben/Oschersleben_centerline.tsv"]
DEFAULT_START_X = 0.0
DEFAULT_START_Y = 0.0
DEFAULT_START_THETA = 1.37079632679
DEFAULT_RENDER_MODE = "human"
DEFAULT_RENDER_FPS = 60
DEFAULT_PLANNER = "pure_pursuit"

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

DEFAULT_TRACE_COLORS = ["yellow", "cyan", "magenta", "red", "green", "blue"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run F1TENTH waypoint following simulation with "
            "configurable agents and noise levels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single agent with noisy waypoints (replicate waypoint_follow_1.py)
  python waypoint_follow_noisy.py

  # Two agents: noisy vs original (replicate waypoint_follow_2.py)
  python waypoint_follow_noisy.py \\
    --waypoints data/maps/Example/Example_raceline_noisy_normal_m0.0_sd0.05.csv \\
    --waypoints data/maps/Example/Example_raceline.csv \\
    --agent-labels "Noisy" "Original"

  # Three agents with different noise levels (replicate waypoint_follow_3.py)
  python waypoint_follow_noisy.py \\
    --waypoints data/maps/Example/Example_raceline_noisy_normal_m0.0_sd0.05.csv \\
    --waypoints data/maps/Example/Example_raceline.csv \\
    --waypoints data/maps/Example/Example_raceline_noisy_normal_m0.0_sd0.06.csv \\
    --agent-labels "Noisy" "Original" "Very Noisy"
        """,
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
        action="append",
        help=(
            "Waypoint CSV file for an agent (can be specified multiple times for multiple agents). "
            f"Default: {DEFAULT_WAYPOINTS[0]}"
        ),
    )

    parser.add_argument(
        "--original-waypoints",
        type=str,
        help=(
            "Original waypoints CSV to display as reference markers. "
            "If not specified, uses first waypoint file."
        ),
    )

    parser.add_argument(
        "--agent-labels",
        type=str,
        nargs="+",
        help="Labels for agents (for terminal output). Number should match waypoint files.",
    )

    parser.add_argument(
        "--trace-colors",
        type=str,
        nargs="+",
        help=(
            f"Trace colors for agents (named colors or 'R,G,B' format). "
            f"Default: {', '.join(DEFAULT_TRACE_COLORS[:3])}"
        ),
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
        help=f"Starting orientation in radians. Default: {DEFAULT_START_THETA}",
    )

    parser.add_argument(
        "--render-mode",
        type=str,
        default=DEFAULT_RENDER_MODE,
        help=f"Render mode for visualization. Default: {DEFAULT_RENDER_MODE}",
    )

    parser.add_argument(
        "--render-fps",
        type=int,
        default=DEFAULT_RENDER_FPS,
        help=f"Render FPS. Default: {DEFAULT_RENDER_FPS}",
    )

    parser.add_argument(
        "--no-inter-agent-collision",
        action="store_true",
        help="Disable collision detection between agents (automatically enabled for multi-agent)",
    )

    parser.add_argument(
        "--planner",
        type=str,
        default=DEFAULT_PLANNER,
        choices=["pure_pursuit"],
        help=f"Planner type to use. Default: {DEFAULT_PLANNER}",
    )

    return parser.parse_args()


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string into RGB tuple.

    Args:
        color_str: Named color or 'R,G,B' format

    Returns:
        RGB tuple (R, G, B)
    """
    # Check if it's a named color
    if color_str.lower() in COLOR_PALETTE:
        return COLOR_PALETTE[color_str.lower()]

    # Try to parse as R,G,B
    try:
        parts = color_str.split(",")
        if len(parts) == 3:
            r, g, b = map(int, parts)
            if all(0 <= c <= 255 for c in [r, g, b]):
                return (r, g, b)
    except ValueError:
        pass

    # Default to yellow if parsing fails
    print(f"Warning: Could not parse color '{color_str}', using yellow")
    return COLOR_PALETTE["yellow"]


def create_planner(planner_type: str, waypoints: np.ndarray):
    """Create a planner instance.

    Args:
        planner_type: Type of planner
        waypoints: Waypoints array

    Returns:
        Planner instance
    """
    if planner_type == "pure_pursuit":
        return PurePursuitPlanner(waypoints=waypoints)

    raise ValueError(f"Unknown planner type: {planner_type}")


def setup_environment(args, num_agents: int):
    """Setup the simulation environment.

    Args:
        args: Parsed command line arguments
        num_agents: Number of agents to simulate

    Returns:
        Gymnasium environment
    """
    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=num_agents,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=args.render_mode,
        render_fps=args.render_fps,
    )

    # Disable inter-agent collisions if requested or if multiple agents
    if args.no_inter_agent_collision or num_agents > 1:

        def ignore_inter_agent_collision():
            env.unwrapped.sim.collisions = np.zeros(env.unwrapped.num_agents)
            env.unwrapped.sim.collision_idx = -np.ones(env.unwrapped.num_agents)

        env.unwrapped.sim.check_collision = ignore_inter_agent_collision

    return env


def setup_render_callbacks(
    env, num_agents: int, trace_colors: List[Tuple[int, int, int]], original_waypoints
):
    """Setup render callbacks for visualization."""
    from f110_planning.render_callbacks import (  # pylint: disable=import-outside-toplevel
        create_camera_tracking,
        create_trace_renderer,
        create_waypoint_renderer,
        render_lidar,
        render_side_distances,
    )

    # 1. Camera Tracking (follow Agent 0)
    env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))

    # 2. LIDAR and Side distances (for Agent 0)
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)

    # 3. Add trace renderer for each agent
    for i in range(num_agents):
        color = trace_colors[i] if i < len(trace_colors) else COLOR_PALETTE["white"]
        env.unwrapped.add_render_callback(
            create_trace_renderer(agent_idx=i, color=color, max_points=10000)
        )

    # 4. Add original waypoints as reference
    if original_waypoints is not None and original_waypoints.size > 0:
        render_waypoints = create_waypoint_renderer(
            original_waypoints, color=(255, 255, 255, 64), name="waypoint_shapes_orig"
        )
        env.unwrapped.add_render_callback(render_waypoints)


def run_simulation(
    env, planners: List, num_agents: int, start_pose: np.ndarray, agent_labels: List[str]
):  # pylint: disable=too-many-locals
    """Run the simulation loop.

    Args:
        env: Gymnasium environment
        planners: List of planner instances
        num_agents: Number of agents
        start_pose: Starting pose [x, y, theta]
        agent_labels: Labels for each agent

    Returns:
        Tuple of (simulation_time, real_time)
    """
    # Create initial poses for all agents (same starting position)
    poses = np.array([start_pose for _ in range(num_agents)])

    obs, _ = env.reset(options={"poses": poses})
    env.render()

    start_time = time.time()
    laptime = 0.0
    done = False

    while not done:
        # Get action from each planner
        actions = []
        for i, planner in enumerate(planners):
            action = planner.plan(obs, ego_idx=i)
            actions.append([action.steer, action.speed])

        actions = np.array(actions)

        # Step simulation
        obs, step_reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        # Print status when simulation ends
        if done:
            print(f"Simulation ended. Terminated: {terminated}, Truncated: {truncated}")
            for i in range(num_agents):
                reason = (
                    "Collision" if obs["collisions"][i] > 0 else "Lap Finished / Other"
                )
                label = agent_labels[i] if i < len(agent_labels) else f"Agent {i}"
                print(f" - {label} (Agent {i}) status: {reason}")

        laptime += float(step_reward)
        env.render()

    real_time = time.time() - start_time
    return laptime, real_time


def main():  # pylint: disable=too-many-locals, too-many-branches
    """Main execution function."""
    args = parse_args()

    # Determine waypoint files to use
    waypoint_files = args.waypoints if args.waypoints else DEFAULT_WAYPOINTS
    num_agents = len(waypoint_files)

    print(f"Setting up simulation with {num_agents} agent(s)")
    print(f"Map: {args.map}{args.map_ext}")

    # Load waypoints for each agent
    waypoints_list = []
    for i, wpt_file in enumerate(waypoint_files):
        wpt_path = Path(wpt_file)
        if not wpt_path.exists():
            print(f"Error: Waypoint file not found: {wpt_file}")
            return
        waypoints = load_waypoints(wpt_file)
        waypoints_list.append(waypoints)
        print(f"Agent {i}: {wpt_file}")

    # Load original waypoints for display
    if args.original_waypoints:
        original_waypoints = load_waypoints(args.original_waypoints)
    elif waypoint_files:
        # Use first waypoint file if it looks like an original (not noisy)
        first_file = waypoint_files[0]
        if "noisy" not in first_file.lower() and len(waypoint_files) > 1:
            original_waypoints = waypoints_list[0]
        else:
            # Try to infer original waypoints path
            original_path = first_file.replace("_noisy", "").split("_normal_")[0]
            if Path(original_path).exists():
                original_waypoints = load_waypoints(original_path)
            else:
                original_waypoints = waypoints_list[0]
    else:
        original_waypoints = None

    # Create planners
    planners = []
    for waypoints in waypoints_list:
        planner = create_planner(args.planner, waypoints)
        planners.append(planner)

    # Setup agent labels
    if args.agent_labels:
        agent_labels = args.agent_labels
    else:
        agent_labels = [f"Agent {i}" for i in range(num_agents)]

    # Setup trace colors
    if args.trace_colors:
        trace_colors = [parse_color(c) for c in args.trace_colors]
    else:
        trace_colors = [
            COLOR_PALETTE[DEFAULT_TRACE_COLORS[i % len(DEFAULT_TRACE_COLORS)]]
            for i in range(num_agents)
        ]

    # Create environment
    env = setup_environment(args, num_agents)

    # Setup visualization
    setup_render_callbacks(env, num_agents, trace_colors, original_waypoints)

    # Run simulation
    start_pose = np.array([args.start_x, args.start_y, args.start_theta])
    print(f"Starting simulation from pose: ({args.start_x}, {args.start_y}, {args.start_theta})")

    laptime, real_time = run_simulation(env, planners, num_agents, start_pose, agent_labels)

    env.close()

    print(f"\nSim elapsed time: {laptime:.3f}s, Real elapsed time: {real_time:.3f}s")


if __name__ == "__main__":
    main()
