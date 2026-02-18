#!/usr/bin/env python3
"""
Simulation script for waypoint following with Dynamic Waypoint Planner.
Visualizes the planner's dynamic target points and vehicle trace.
"""

import time
from argparse import Namespace

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.reactive import DynamicWaypointPlanner
from f110_planning.render_callbacks import (
    camera_tracking,
    create_dynamic_waypoint_renderer,
    create_heading_error_renderer,
    create_trace_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.utils import load_waypoints


def main():  # pylint: disable=too-many-locals
    """
    Main function to run the dynamic waypoint following simulation.
    """
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints = load_waypoints("data/maps/Example/Example_raceline.csv")

    planner = DynamicWaypointPlanner(
        waypoints=waypoints, lookahead_distance=1.5, max_speed=5.0, lateral_gain=1.0
    )

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode="human_fast",
        render_fps=60,
        max_laps=None,
    )

    env.unwrapped.add_render_callback(camera_tracking)
    env.unwrapped.add_render_callback(render_lidar)
    env.unwrapped.add_render_callback(render_side_distances)

    heading_error_renderer = create_heading_error_renderer(waypoints, agent_idx=0)
    env.unwrapped.add_render_callback(heading_error_renderer)

    dynamic_waypoint_renderer = create_dynamic_waypoint_renderer(
        waypoints, agent_idx=0, lookahead_distance=1.5, lateral_gain=1.0
    )
    env.unwrapped.add_render_callback(dynamic_waypoint_renderer)

    trace_renderer = create_trace_renderer(agent_idx=0)
    env.unwrapped.add_render_callback(trace_renderer)

    obs, _ = env.reset(
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
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
