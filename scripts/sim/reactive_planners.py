#!/usr/bin/env python3
"""
Simulation script to test various reactive planners.
Supports: bubble, gap_follower, disparity, and dynamic.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.reactive import (
    BubblePlanner,
    DisparityExtenderPlanner,
    DynamicWaypointPlanner,
    GapFollowerPlanner,
    LidarDNNPlanner,
)
from f110_planning.render_callbacks import (
    create_camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    create_waypoint_renderer,
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
DEFAULT_PLANNER = "dnn"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test various reactive planners in simulation."
    )

    parser.add_argument(
        "--planner",
        type=str,
        choices=["bubble", "gap", "disparity", "dynamic", "dnn"],
        default=DEFAULT_PLANNER,
        help=f"Reactive planner to use. Choices: bubble, gap, disparity, dynamic, dnn. Default: {DEFAULT_PLANNER}",
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
        help=f"Path to waypoints file (used for reference in some planners). Default: {DEFAULT_WAYPOINTS}",
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
        help=f"Render mode for visualization. Default: {DEFAULT_RENDER_MODE}",
    )

    parser.add_argument(
        "--render-fps",
        type=int,
        default=DEFAULT_RENDER_FPS,
        help=f"Render FPS. Default: {DEFAULT_RENDER_FPS}",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Target speed for planners (overrides planner defaults).",
    )

    parser.add_argument(
        "--lookahead",
        type=float,
        default=1.5,
        help="Lookahead distance (gain) for dynamic/tracking planners. Default: 1.5",
    )

    parser.add_argument(
        "--lateral-gain",
        type=float,
        default=1.0,
        help="Lateral gain for dynamic planner. Default: 1.0",
    )

    parser.add_argument(
        "--safety-radius",
        type=float,
        default=1.3,
        help="Safety radius for bubble planner. Default: 1.3",
    )

    parser.add_argument(
        "--bubble-radius",
        type=int,
        default=160,
        help="Bubble radius (beams) for gap follower. Default: 160",
    )

    parser.add_argument(
        "--wall-model",
        type=str,
        default="data/models/left_wall_dist,right_wall_dist_arch5.pth",
        help="Path to dual-head wall distance model (PTH).",
    )

    parser.add_argument(
        "--left-model",
        type=str,
        default=None,
        help="Path to separate left wall distance model (PTH).",
    )

    parser.add_argument(
        "--right-model",
        type=str,
        default=None,
        help="Path to separate right wall distance model (PTH).",
    )

    parser.add_argument(
        "--heading-model",
        type=str,
        default="data/models/heading_error_arch7.pth",
        help="Path to the heading error model (PTH).",
    )

    parser.add_argument(
        "--arch",
        type=int,
        default=5,
        help="Architecture ID for wall model. Default: 5",
    )

    parser.add_argument(
        "--heading-arch",
        type=int,
        default=7,
        help="Architecture ID for heading model. Default: 7",
    )

    return parser.parse_args()


def main():  # pylint: disable=too-many-locals, too-many-statements
    """
    Main function to run the reactive planning simulation.
    """
    args = parse_args()

    # Convert "None" string to None object
    render_mode = None if args.render_mode == "None" else args.render_mode

    # Load waypoints (required by Dynamic Planner, optional for others as reference)
    waypoints = load_waypoints(args.waypoints)

    # Initialize the chosen planner
    if args.planner == "bubble":
        kwargs = {"safety_radius": args.safety_radius}
        if args.speed is not None:
            kwargs["avoidance_speed"] = args.speed
        planner = BubblePlanner(**kwargs)
    elif args.planner == "gap":
        kwargs = {"bubble_radius": args.bubble_radius}
        if args.speed is not None:
            kwargs["corners_speed"] = min(4.0, args.speed)
            kwargs["straights_speed"] = args.speed
        planner = GapFollowerPlanner(**kwargs)
    elif args.planner == "disparity":
        planner = DisparityExtenderPlanner()
        if args.speed is not None:
            planner.absolute_max_speed = args.speed
    elif args.planner == "dynamic":
        kwargs = {
            "waypoints": waypoints,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        planner = DynamicWaypointPlanner(**kwargs)
    elif args.planner == "dnn":
        kwargs = {
            "wall_model_path": args.wall_model,
            "left_model_path": args.left_model,
            "right_model_path": args.right_model,
            "heading_model_path": args.heading_model,
            "arch_id": args.arch,
            "heading_arch_id": args.heading_arch,
            "lookahead_distance": args.lookahead,
            "lateral_gain": args.lateral_gain,
        }
        if args.speed is not None:
            kwargs["max_speed"] = args.speed
        planner = LidarDNNPlanner(**kwargs)
    else:
        raise ValueError(f"Unknown planner: {args.planner}")

    # Create the environment
    env = gym.make(
        "f110_gym:f110-v0",
        map=args.map,
        map_ext=args.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode=render_mode,
        render_fps=args.render_fps,
        max_laps=None,
    )

    # Standard render callbacks
    if render_mode:
        env.unwrapped.add_render_callback(create_camera_tracking(rotate=True))
        env.unwrapped.add_render_callback(render_lidar)
        env.unwrapped.add_render_callback(render_side_distances)
        env.unwrapped.add_render_callback(create_trace_renderer(agent_idx=0))
        
        # Reference waypoints for context
        if waypoints.size > 0:
            env.unwrapped.add_render_callback(create_waypoint_renderer(waypoints))
            env.unwrapped.add_render_callback(create_heading_error_renderer(waypoints, 0))

        # Planner-specific renderers
        if args.planner in ["dynamic", "dnn"]:
            env.unwrapped.add_render_callback(
                create_dynamic_waypoint_renderer(planner, agent_idx=0)
            )

    # Reset environment
    obs, _ = env.reset(
        options={"poses": np.array([[args.start_x, args.start_y, args.start_theta]])}
    )
    
    if render_mode:
        env.render()

    print(f"Starting simulation with {args.planner} planner...")
    laptime = 0.0
    start_time = time.time()

    done = False
    try:
        while not done:
            action = planner.plan(obs, ego_idx=0)
            obs, step_reward, terminated, truncated, _ = env.step(
                np.array([[action.steer, action.speed]])
            )
            done = terminated or truncated
            laptime += float(step_reward)
            
            if render_mode:
                env.render()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    total_real_time = time.time() - start_time
    print(f"\nSimulation Finished:")
    print(f"Planner: {args.planner}")
    print(f"Simulated Time: {laptime:.2f}s")
    print(f"Real Time: {total_real_time:.2f}s")
    print(f"Real-time factor: {laptime / total_real_time:.2f}x")

    env.close()


if __name__ == "__main__":
    main()
