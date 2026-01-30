#!/usr/bin/env python3

import numpy as np
from f110_gym.envs.base_classes import Integrator
from pyglet import canvas as pyg_canvas
from pyglet import window as pyg_window
from pyglet.gl import GL_LINES
from sklearn.linear_model import RANSACRegressor

from gym import make as env_make

obs_scan = []


def compute_midline(ranges, fov=4.7, max_range=10.0, ransac_thresh=0.05):
    """
    Compute midline heading and lateral error from 1D LiDAR scan.

    Parameters:
    - ranges: np.array of 1080 LiDAR distances
    - fov: LiDAR field of view in radians
    - max_range: maximum valid LiDAR distance
    - ransac_thresh: inlier distance threshold for RANSAC

    Returns:
    - psi_mid: midline heading (radians) in vehicle frame
    - e_y: lateral error (meters, positive = car left of midline)
    - wall_lines: list of two tuples (slope, intercept)
    """

    N = len(ranges)
    # 1. Convert polar -> Cartesian
    angles = np.linspace(-fov / 2, fov / 2, N)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    # 2. Filter valid points
    mask = (ranges > 0.01) & (ranges < max_range)  # remove zero/inf
    if np.sum(mask) < 10:
        # no valid points
        return 0.0, 0.0, [(0, 0), (0, 0)]
    x = x[mask]
    y = y[mask]
    points = np.vstack([x, y]).T

    # 3. Fit first wall with RANSAC
    ransac1 = RANSACRegressor(residual_threshold=ransac_thresh)
    ransac1.fit(x.reshape(-1, 1), y)
    a1 = ransac1.estimator_.coef_[0]
    b1 = ransac1.estimator_.intercept_

    # 4. Remove inliers of first wall to fit second wall
    inliers1 = ransac1.inlier_mask_
    remaining_points = points[~inliers1]
    if len(remaining_points) >= 2:
        x2 = remaining_points[:, 0]
        y2 = remaining_points[:, 1]
        ransac2 = RANSACRegressor(residual_threshold=ransac_thresh)
        ransac2.fit(x2.reshape(-1, 1), y2)
        a2 = ransac2.estimator_.coef_[0]
        b2 = ransac2.estimator_.intercept_
    else:
        # if second wall is not visible, assume parallel offset
        a2 = a1
        b2 = b1 + 0.31  # use nominal F1Tenth track width in meters
        # optionally you can tune b2 based on expected track width

    # 5. Midline heading = average of slopes
    psi_mid = 0.5 * (np.arctan(a1) + np.arctan(a2))

    # 6. Midline lateral offset at vehicle x=0
    y_mid = 0.5 * (b1 + b2)
    e_y = -y_mid  # positive = car left of midline

    return psi_mid, e_y, [(a1, b1), (a2, b2)]


def camera_tracking_callback(env_renderer):
    """
    Update camera to follow car
    """
    e = env_renderer
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.left = left - 500
    e.right = right + 500
    e.top = top + 500
    e.bottom = bottom - 500


def render_lidar_callback(env_renderer):
    global scans
    """
    Draw LiDAR rays in Pyglet 1.4 using batch.add and GL_LINES.
    Optimized: draws every Nth ray, deletes previous lines each frame.
    """
    fov = 4.7
    e = env_renderer

    # Car pose
    x_vertices = e.cars[0].vertices[::2]
    y_vertices = e.cars[0].vertices[1::2]
    x_car = np.mean(x_vertices)
    y_car = np.mean(y_vertices)
    yaw = e.poses[0][2]

    # Latest scan
    try:
        scan = obs_scan  # shape (1080,)
    except Exception:
        return

    num_rays = len(scan)
    angles = np.linspace(-fov / 2, fov / 2, num_rays)

    # Draw every step-th ray for performance
    step = 30
    scan = scan[::step]
    angles = angles[::step]

    # Clear previous lines
    if hasattr(e, "lidar_lines"):
        for line in e.lidar_lines:
            line.delete()
    e.lidar_lines = []

    # Scale for visualization
    scale = 50

    for r, theta in zip(scan, angles):
        if r <= 0.0:
            continue

        dx = r * np.cos(theta + yaw) * scale
        dy = r * np.sin(theta + yaw) * scale

        # Simple color coding: closer = red, farther = blue
        c = int(min(r / 10.0, 1.0) * 255)
        color = [255, 255 - c, 255 - c, 255, 255 - c, 255 - c]

        vertices = [x_car, y_car, x_car + dx, y_car + dy]

        line = e.batch.add(
            2,
            GL_LINES,
            None,
            ("v2f", vertices),
            ("c3B", color),
        )
        e.lidar_lines.append(line)


def render_fitted_lines_callback(env_renderer):
    """
    Render the fitted walls and midline for F1Tenth in Pyglet with scaling.
    Walls and midline stay aligned with the car as it moves.

    Parameters:
    - env_renderer: the environment renderer object
    - scale: scaling factor to convert meters -> pixels
    """
    global obs_scan
    e = env_renderer
    scale = 50

    # Car pose in pixels (from Pyglet vertices)
    x_vertices = e.cars[0].vertices[::2]
    y_vertices = e.cars[0].vertices[1::2]
    x_car = np.mean(x_vertices)
    y_car = np.mean(y_vertices)
    yaw = e.poses[0][2]

    # Compute fitted lines
    psi_mid, e_y, wall_lines = compute_midline(obs_scan)

    # Forward horizon in vehicle frame (meters)
    x_horizon = np.linspace(0, 5.0, 10)  # 5 meters ahead

    # Clear previous lines
    if hasattr(e, "fitted_lines"):
        for line in e.fitted_lines:
            line.delete()
    e.fitted_lines = []

    # Draw walls (blue/red) and midline (green)
    for idx, (a, b) in enumerate(wall_lines + [(0.0, 0.0)]):
        if idx == 2:  # midline
            a = np.tan(psi_mid)
            b = -e_y
            color = [255, 255, 0] * 2  # yellow
        else:
            color = (
                [0, 255, 255] * 2 if idx == 0 else [0, 255, 255] * 2
            )  # wall1=cyan, wall2=cyan

        # Draw line segments
        for i in range(len(x_horizon) - 1):
            x0, x1 = x_horizon[i], x_horizon[i + 1]
            y0, y1 = a * x0 + b, a * x1 + b

            # Rotate offsets into world frame (vehicle frame to world frame)
            dx0 = np.cos(yaw) * x0 - np.sin(yaw) * y0
            dy0 = np.sin(yaw) * x0 + np.cos(yaw) * y0
            dx1 = np.cos(yaw) * x1 - np.sin(yaw) * y1
            dy1 = np.sin(yaw) * x1 + np.cos(yaw) * y1

            # Add to car position (in pixels) and apply scaling only to offsets
            vertices = [
                x_car + dx0 * scale,
                y_car + dy0 * scale,
                x_car + dx1 * scale,
                y_car + dy1 * scale,
            ]

            # Add line to batch
            line = e.batch.add(
                2,
                GL_LINES,
                None,
                ("v2f", vertices),
                ("c3B", color),
            )
            e.fitted_lines.append(line)


def env_init(init_x=0.0, init_y=0.0, init_theta=1.57):
    env = env_make(
        id="f110_gym:f110-v0",
        params={
            "mu": 1.0489,  # surface friction coefficient [-]
            "C_Sf": 4.718,  # Cornering stiffness coefficient, front [1/rad]
            "C_Sr": 5.4562,  # Cornering stiffness coefficient, rear [1/rad]
            "lf": 0.15875,  # Distance from center of gravity to front axle [m]
            "lr": 0.17145,  # Distance from center of gravity to rear axle [m]
            "h": 0.074,  # Height of center of gravity [m]
            "m": 3.74,  # Total mass of the vehicle [kg]
            "I": 0.04712,  # Moment of inertial of the entire vehicle about the z axis [kgm^2]
            "s_min": -0.4189,  # Minimum steering angle constraint [rad]
            "s_max": 0.4189,  # Maximum steering angle constraint [rad]
            "sv_min": -3.2,  # Minimum steering velocity constraint [rad/s]
            "sv_max": 3.2,  # Maximum steering velocity constraint [rad/s]
            "v_switch": 7.319,  # Switching velocity (velocity at which the acceleration is no longer able to create wheel spin) [m/s]
            "a_max": 9.51,  # Maximum longitudinal acceleration [m/s^2]
            "v_min": -5.0,  # Minimum longitudinal velocity [m/s]
            "v_max": 20.0,  # Maximum longitudinal velocity [m/s]
            "width": 0.31,  # width of the vehicle [m]
            "length": 0.58,  # length of the vehicle [m]
        },
        map="data/map",
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    env.add_render_callback(camera_tracking_callback)
    env.add_render_callback(render_lidar_callback)
    env.add_render_callback(render_fitted_lines_callback)
    obs, step_reward, done, info = env.reset(
        poses=np.array([[init_x, init_y, init_theta]])
    )
    return env, obs, step_reward, done, info


def kbd_init():
    display = pyg_canvas.get_display()
    keys = pyg_window.key.KeyStateHandler()
    windows = display.get_windows()
    if not windows:
        raise RuntimeError("No pyglet window found")
    windows[0].push_handlers(keys)
    return keys


def main():
    global obs_scan
    env, obs, step_reward, done, info = env_init()
    obs_scan = obs["scans"][0]
    env.render()
    keys = kbd_init()

    # planner = planner()

    lap_time = 0.0
    while not done:
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        state = env.sim.agents[0].state

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

        # action = planner.plan(obs)

        ego_steer = 0
        ego_speed = 0

        psi_mid, e_y, _ = compute_midline(obs_scan)
        ego_speed = env.params["v_max"] / 4
        ego_steer = 1.6 * psi_mid - 0.2 * e_y

        if keys[pyg_window.key.Q]:
            breakpoint()
        if keys[pyg_window.key.W]:
            ego_speed = env.params["v_max"] / 8
        if keys[pyg_window.key.A]:
            ego_steer = env.params["s_max"] / 2
        if keys[pyg_window.key.D]:
            ego_steer = env.params["s_min"] / 2

        action = np.array(
            [
                [
                    np.clip(ego_steer, env.params["s_min"], env.params["s_max"]),
                    np.clip(ego_speed, env.params["v_min"], env.params["v_max"]),
                ]
            ]
        )

        obs, step_reward, done, info = env.step(action)
        obs_scan = obs["scans"][0]

        lap_time += step_reward
        env.render()


if __name__ == "__main__":
    main()
