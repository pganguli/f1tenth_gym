#!/usr/bin/env python3
"""
Manual control simulation script with RANSAC-based assistance.
Allows manual driving using WASD keys while providing RANSAC-based
midline tracking when no keys are pressed.
"""

import time
from argparse import Namespace

import gymnasium as gym
import numpy as np
import pyglet
from f110_gym.envs.base_classes import Integrator
from f110_planning.reactive import RansacMidlinePlanner
from f110_planning.render_callbacks import (
    camera_tracking,
    create_heading_error_renderer,
    create_ransac_walls_renderer,
    create_trace_renderer,
    render_lidar,
    render_side_distances,
)
from f110_planning.utils import load_waypoints


def get_keys():
    """Get keyboard input from pyglet window"""
    display = pyglet.display.get_display()
    keys = pyglet.window.key.KeyStateHandler()
    windows = display.get_windows()
    if not windows:
        raise RuntimeError("No pyglet window found")
    windows[0].push_handlers(keys)
    return keys


def main():  # pylint: disable=too-many-locals
    """
    Main function to run manual simulation with RANSAC assistance.
    """
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints = load_waypoints("data/maps/Example/Example_raceline.csv")

    planner = RansacMidlinePlanner(max_speed=5.0)

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
    env.unwrapped.add_render_callback(
        create_trace_renderer(color=(255, 255, 0), max_points=10000)
    )

    ransac_walls_renderer = create_ransac_walls_renderer(agent_idx=0)
    env.unwrapped.add_render_callback(ransac_walls_renderer)

    heading_error_renderer = create_heading_error_renderer(waypoints, agent_idx=0)
    env.unwrapped.add_render_callback(heading_error_renderer)

    obs, _ = env.reset(
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
    )
    env.render()

    keys = get_keys()

    laptime = 0.0
    start = time.time()

    done = False
    while not done:
        state = env.unwrapped.sim.agents[0].state

        print(
            f"x = {state[0]:.3f}",
            f"y = {state[1]:.3f}",
            f"δ = {state[2]:.3f}",
            f"v = {state[3]:.3f}",
            f"θ = {state[4]:.3f}",
            f"θ′ = {state[5]:.3f}",
            f"β = {state[6]:.3f}",
            sep="\t",
        )

        if keys[pyglet.window.key.Q]:
            break

        ego_steer = 0
        ego_speed = 0

        if keys[pyglet.window.key.W]:
            ego_speed = env.unwrapped.params["v_max"] / 8
        if keys[pyglet.window.key.A]:
            ego_steer = env.unwrapped.params["s_max"] / 2
        if keys[pyglet.window.key.D]:
            ego_steer = env.unwrapped.params["s_min"] / 2

        # Use RANSAC planner if no manual input
        if not (
            keys[pyglet.window.key.W]
            or keys[pyglet.window.key.A]
            or keys[pyglet.window.key.D]
        ):
            action = planner.plan(obs, ego_idx=0)
            ego_steer = action.steer
            ego_speed = action.speed

        # Clip actions to valid ranges
        ego_steer = np.clip(
            ego_steer, env.unwrapped.params["s_min"], env.unwrapped.params["s_max"]
        )
        ego_speed = np.clip(
            ego_speed, env.unwrapped.params["v_min"], env.unwrapped.params["v_max"]
        )

        action = np.array([[ego_steer, ego_speed]])

        obs, step_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        laptime += float(step_reward)
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
