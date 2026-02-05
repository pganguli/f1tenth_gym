#!/usr/bin/env python3

import time
from argparse import Namespace

import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator

import gymnasium as gym
from f110_planning.tracking import PurePursuitPlanner


def main():
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1.375,
    }  # 0.90338203837889}

    # Configuration for Example map
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.0,
        sy=0.0,
        stheta=0.0,
    )

    try:
        waypoints = np.loadtxt("data/maps/Example/Example_raceline.csv", delimiter=";", skiprows=3)
    except Exception:
        print("Could not load waypoints, using empty")
        waypoints = np.array([])

    planner = PurePursuitPlanner(waypoints=waypoints)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        # planner.render_waypoints(env_renderer)

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode="human",
    )
    env.unwrapped.add_render_callback(render_callback)

    obs, info = env.reset(
        options={"poses": np.array([[conf.sx, conf.sy, conf.stheta]])}
    )
    env.render()

    laptime = 0.0
    start = time.time()

    done = False
    while not done:
        action = planner.plan(obs)
        speed, steer = action.speed, action.steer
        obs, step_reward, terminated, truncated, info = env.step(np.array([[steer, speed]]))
        done = terminated or truncated
        laptime += step_reward
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
