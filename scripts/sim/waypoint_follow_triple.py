#!/usr/bin/env python3

import time
from argparse import Namespace

import gymnasium as gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_planning.tracking import PurePursuitPlanner
from f110_planning.utils import load_waypoints


def main():
    # Configuration for Example map
    conf = Namespace(
        map_path="data/maps/Example/Example",
        map_ext=".png",
        sx=0.7,
        sy=0.0,
        stheta=1.37079632679,
    )

    waypoints_orig = load_waypoints("data/maps/Example/Example_raceline.csv")
    waypoints_noisy = load_waypoints(
        "data/maps/Example/Example_raceline_noisy_normal_m0.0_sd0.05.csv"
    )
    waypoints_very_noisy = load_waypoints(
        "data/maps/Example/Example_raceline_noisy_normal_m0.0_sd0.06.csv"
    )

    planner_noisy = PurePursuitPlanner(waypoints=waypoints_noisy)
    planner_orig = PurePursuitPlanner(waypoints=waypoints_orig)
    planner_very_noisy = PurePursuitPlanner(waypoints=waypoints_very_noisy)

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=3,
        timestep=0.01,
        integrator=Integrator.RK4,
        render_mode="human",
        render_fps=60,
    )

    from f110_planning.render_callbacks import (
        create_trace_renderer,
        create_waypoint_renderer,
    )

    # Monkeypatch simulator to ignore inter-agent collisions for this run
    def ignore_inter_agent_collision():
        env.unwrapped.sim.collisions = np.zeros(env.unwrapped.num_agents)
        env.unwrapped.sim.collision_idx = -np.ones(env.unwrapped.num_agents)

    env.unwrapped.sim.check_collision = ignore_inter_agent_collision

    # Trace for Noisy Car (Agent 0)
    env.unwrapped.add_render_callback(
        create_trace_renderer(agent_idx=0, color=(255, 255, 0), max_points=10000)
    )  # Yellow
    
    # Trace for Original Car (Agent 1)
    env.unwrapped.add_render_callback(
        create_trace_renderer(agent_idx=1, color=(0, 255, 255), max_points=10000)
    )  # Cyan
    
    # Trace for Very Noisy Car (Agent 2)
    env.unwrapped.add_render_callback(
        create_trace_renderer(agent_idx=2, color=(255, 0, 255), max_points=10000)
    )  # Magenta

    if waypoints_orig.size > 0:
        render_waypoints_orig = create_waypoint_renderer(
            waypoints_orig, color=(255, 255, 255, 64), name="waypoint_shapes_orig"
        )  # Transparent White for Reference
        env.unwrapped.add_render_callback(render_waypoints_orig)

    # Start all cars at the same position
    poses = np.array([
        [conf.sx, conf.sy, conf.stheta],
        [conf.sx, conf.sy, conf.stheta],
        [conf.sx, conf.sy, conf.stheta]
    ])
    obs, info = env.reset(options={"poses": poses})
    env.render()

    start = time.time()
    laptime = 0.0
    done = False
    
    while not done:
        # Actions for all agents
        action_noisy = planner_noisy.plan(obs, ego_idx=0)
        action_orig = planner_orig.plan(obs, ego_idx=1)
        action_very_noisy = planner_very_noisy.plan(obs, ego_idx=2)

        actions = np.array([
            [action_noisy.steer, action_noisy.speed],
            [action_orig.steer, action_orig.speed],
            [action_very_noisy.steer, action_very_noisy.speed],
        ])

        obs, step_reward, terminated, truncated, info = env.step(actions)

        done = terminated or truncated
        if done:
            print(f"Simulation ended. Terminated: {terminated}, Truncated: {truncated}")
            for i in range(env.unwrapped.num_agents):
                reason = "Collision" if obs["collisions"][i] > 0 else "Lap Finished / Other"
                agent_lbl = "Noisy" if i == 0 else "Original" if i == 1 else "Very Noisy"
                print(f" - Agent {i} ({agent_lbl}) status: {reason}")
        
        laptime += float(step_reward)
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
