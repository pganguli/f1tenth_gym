#!/usr/bin/env python3
"""
Data generation script for F1TENTH simulation.
Collects lidar scans and wall distance data while following waypoints.
Outputs data in NPZ format for efficient storage and loading.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import get_heading_error, get_side_distances, load_waypoints

DEFAULT_MAP_PATH = "data/maps/Example/Example"
DEFAULT_MAP_EXT = ".png"
DEFAULT_WAYPOINT_PATH = "data/maps/Example/Example_raceline.csv"
DEFAULT_OUTPUT_DIR = "data/datasets"
DEFAULT_MAX_STEPS = 4500
DEFAULT_PLANNER = "pure_pursuit"
DEFAULT_START_X = 0.7
DEFAULT_START_Y = 0.0
DEFAULT_START_THETA = 1.37079632679
DEFAULT_RENDER_MODE = "human_fast"
DEFAULT_RENDER_FPS = 60


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate F1TENTH simulation data with lidar scans and wall distances."
    )

    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP_PATH,
        help=f"Path to the map file (without extension). Default: {DEFAULT_MAP_PATH}",
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
        default=DEFAULT_WAYPOINT_PATH,
        help=f"Path to waypoints CSV file. Default: {DEFAULT_WAYPOINT_PATH}",
    )

    parser.add_argument(
        "--planner",
        type=str,
        choices=["pure_pursuit"],
        default=DEFAULT_PLANNER,
        help=f"Planner to use for navigation. Default: {DEFAULT_PLANNER}",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ file path. If not specified, generates filename from parameters.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum number of simulation steps. Default: {DEFAULT_MAX_STEPS}",
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
        help=f"Render FPS when visualizing. Default: {DEFAULT_RENDER_FPS}",
    )

    return parser.parse_args()


def create_planner(planner_type: str, waypoints: np.ndarray):
    """Create a planner instance based on the specified type.

    Args:
        planner_type: Type of planner to create ("pure_pursuit" or "disparity_extender")
        waypoints: Waypoints array for the planner

    Returns:
        Planner instance
    """
    if planner_type == "pure_pursuit":
        return PurePursuitPlanner(waypoints=waypoints)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")


def setup_environment(args):
    """Setup the simulation environment.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (env, planner, waypoints)
    """
    waypoints = load_waypoints(args.waypoints)
    planner = create_planner(args.planner, waypoints)

    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=args.render_mode,
        render_fps=args.render_fps,
    )

    return env, planner, waypoints


def collect_data(env, planner, waypoints, max_steps: int, start_pose: np.ndarray):
    """Collect simulation data by running the planner.

    Args:
        env: Gymnasium environment
        planner: Planner instance
        waypoints: Waypoints array
        max_steps: Maximum number of steps to run
        start_pose: Initial pose [x, y, theta]

    Returns:
        Dictionary containing collected data arrays
    """
    scans_list = []
    left_dists_list = []
    right_dists_list = []
    heading_errors_list = []

    obs, info = env.reset(options={"poses": start_pose.reshape(1, -1)})

    step = 0
    done = False

    while step < max_steps and not done:
        action = planner.plan(obs, ego_idx=0)
        speed, steer = action.speed, action.steer

        obs, step_reward, terminated, truncated, info = env.step(
            np.array([[steer, speed]])
        )

        scan = obs["scans"][0]
        left_dist, right_dist = get_side_distances(scan)
        car_position = np.array([obs["poses_x"][0], obs["poses_y"][0]])
        heading_error = get_heading_error(
            waypoints, car_position, obs["poses_theta"][0]
        )

        scans_list.append(scan)
        left_dists_list.append(left_dist)
        right_dists_list.append(right_dist)
        heading_errors_list.append(heading_error)

        done = terminated or truncated
        step += 1

    data = {
        "scans": np.array(scans_list),
        "left_wall_dist": np.array(left_dists_list),
        "right_wall_dist": np.array(right_dists_list),
        "heading_error": np.array(heading_errors_list),
    }

    return data, step


def save_data(data: dict, output_path: str):
    """Save collected data to NPZ file.

    Args:
        data: Dictionary of data arrays
        output_path: Path to save the NPZ file
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to NPZ format
    np.savez(output_path, **data)


def generate_output_filename(args, num_samples: int) -> str:
    """Generate output filename based on generation parameters.

    Args:
        args: Parsed command line arguments
        num_samples: Number of samples collected

    Returns:
        Filename string

    Format: lidar_tracking_<map>_<planner>_n<samples>.npz
    Example: lidar_tracking_Example_pure_pursuit_n500.npz
    """
    # Extract map name from path
    map_name = Path(args.map).name

    # Create filename
    filename = f"lidar_tracking_{map_name}_{args.planner}_n{num_samples}.npz"

    # Combine with output directory
    output_path = Path(DEFAULT_OUTPUT_DIR) / filename

    return str(output_path)


def main():
    """Main execution function."""
    args = parse_args()

    print(f"Setting up environment with map: {args.map}")
    print(f"Loading waypoints from: {args.waypoints}")
    print(f"Using planner: {args.planner}")

    env, planner, waypoints = setup_environment(args)

    start_pose = np.array([args.start_x, args.start_y, args.start_theta])
    print(f"Collecting data for up to {args.max_steps} steps...")

    data, num_steps = collect_data(env, planner, waypoints, args.max_steps, start_pose)

    env.close()

    # Generate output filename if not specified
    output_path = (
        args.output if args.output else generate_output_filename(args, num_steps)
    )

    save_data(data, output_path)

    print(f"Saved {num_steps} samples to {output_path}")
    print(f"Data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()
