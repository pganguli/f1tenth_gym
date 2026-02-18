"""
RANSAC Wall Fitting Visualization Render Callback
"""

import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer
from sklearn.linear_model import RANSACRegressor


# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_ransac_walls_renderer(
    agent_idx: int = 0,
    fov: float = 4.7,
    max_range: float = 10.0,
    ransac_thresh: float = 0.05,
    track_width: float = 0.31,
    horizon: float = 5.0,
):
    """
    Factory to create a RANSAC wall fitting visualization callback.

    Args:
        agent_idx: Index of the agent to visualize walls for
        fov: LiDAR field of view in radians
        max_range: Maximum valid LiDAR range
        ransac_thresh: RANSAC inlier distance threshold
        track_width: Expected track width for fallback
        horizon: Forward viewing distance in meters
    """
    attr_name = f"ransac_lines_{agent_idx}"

    def render_ransac_walls(env_renderer: EnvRenderer) -> None:
        if (
            env_renderer.scans is None
            or env_renderer.poses is None
            or env_renderer.poses.shape[0] <= agent_idx
            or not env_renderer.cars
        ):
            return

        # Clear previous lines
        if not hasattr(env_renderer, attr_name):
            setattr(env_renderer, attr_name, [])
        for line in getattr(env_renderer, attr_name):
            line.delete()
        getattr(env_renderer, attr_name).clear()

        # Compute fitted lines
        psi_mid, y_mid_f, walls = _compute_midline(
            env_renderer.scans[agent_idx], fov, max_range, ransac_thresh, track_width
        )

        # Pose and horizontal linspace
        car_pose = (
            np.mean(env_renderer.cars[agent_idx].vertices[::2]),
            np.mean(env_renderer.cars[agent_idx].vertices[1::2]),
            env_renderer.poses[agent_idx][2],
        )
        x_hor = np.linspace(0, horizon, 10)

        # Draw walls and midline
        for i, (a, b) in enumerate(walls + [(np.tan(psi_mid), y_mid_f)]):
            color = (255, 255, 0) if i == 2 else (0, 255, 255)
            _transform_and_draw(env_renderer, car_pose, a, b, color, x_hor, attr_name)

    return render_ransac_walls


def _compute_midline(ranges, fov, max_range, ransac_thresh, track_width):
    """Compute wall fitting and midline from LiDAR scan"""
    mask = (ranges > 0.01) & (ranges < max_range)
    if np.sum(mask) < 10:
        return 0.0, 0.0, [(0.0, 0.0), (0.0, 0.0)]

    angles = np.linspace(-fov / 2, fov / 2, len(ranges))
    x_pts, y_pts = (
        ranges[mask] * np.cos(angles[mask]),
        ranges[mask] * np.sin(angles[mask]),
    )

    # Fit first wall
    ransac1 = RANSACRegressor(residual_threshold=ransac_thresh, random_state=42)
    ransac1.fit(x_pts.reshape(-1, 1), y_pts)
    a1, b1 = ransac1.estimator_.coef_[0], ransac1.estimator_.intercept_

    # Fit second wall
    if np.sum(~ransac1.inlier_mask_) >= 2:
        ransac2 = RANSACRegressor(residual_threshold=ransac_thresh, random_state=42)
        ransac2.fit(
            x_pts[~ransac1.inlier_mask_].reshape(-1, 1), y_pts[~ransac1.inlier_mask_]
        )
        a2, b2 = ransac2.estimator_.coef_[0], ransac2.estimator_.intercept_
    else:
        a2, b2 = a1, b1 + track_width

    return 0.5 * (np.arctan(a1) + np.arctan(a2)), 0.5 * (b1 + b2), [(a1, b1), (a2, b2)]


def _transform_and_draw(env_renderer, car_pose, a, b, color, x_hor, attr_name):
    """Helper to transform and draw a line segment."""
    for i in range(len(x_hor) - 1):
        x_seg = np.array([x_hor[i], x_hor[i + 1]])
        y_seg = a * x_seg + b

        # Transform from vehicle frame to world frame
        # car_pose is (x_car, y_car, yaw)
        dx_val = np.cos(car_pose[2]) * x_seg - np.sin(car_pose[2]) * y_seg
        dy_val = np.sin(car_pose[2]) * x_seg + np.cos(car_pose[2]) * y_seg

        getattr(env_renderer, attr_name).append(
            pyglet.shapes.Line(
                car_pose[0] + dx_val[0] * 50.0,
                car_pose[1] + dy_val[0] * 50.0,
                car_pose[0] + dx_val[1] * 50.0,
                car_pose[1] + dy_val[1] * 50.0,
                thickness=2,
                color=color,
                batch=env_renderer.batch,
            )
        )
