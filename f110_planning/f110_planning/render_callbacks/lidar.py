import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer

from f110_planning.utils import get_side_distances


def render_lidar(env_renderer: EnvRenderer) -> None:
    """
    Draw LiDAR rays in Pyglet 2.x using pyglet.shapes.Line.
    """
    fov = 4.7
    e = env_renderer

    # Get latest scans from EnvRenderer
    if e.scans is None or e.poses is None:
        return

    scan = e.scans[0]  # Ego scan

    num_rays = len(scan)
    angles = np.linspace(-fov / 2, fov / 2, num_rays)
    yaw = e.poses[0][2]

    # Car center (approx) from vertices
    if not e.cars:
        return

    v = e.cars[0].vertices
    x_car = np.mean(v[::2])
    y_car = np.mean(v[1::2])

    # Draw every step-th ray for performance
    step = 30
    scan_sub = scan[::step]
    angles_sub = angles[::step]

    # Clear previous lines
    if not hasattr(e, "lidar_lines"):
        e.lidar_lines = []
    else:
        for line in e.lidar_lines:
            line.delete()
        e.lidar_lines = []

    # Scale for visualization
    scale = 50.0

    for r, theta in zip(scan_sub, angles_sub):
        if r <= 0.0:
            continue

        dx = r * np.cos(theta + yaw) * scale
        dy = r * np.sin(theta + yaw) * scale

        # Simple color coding: closer = red, farther = blue
        c = int(min(r / 10.0, 1.0) * 255)
        color = (255, 255 - c, 255 - c)

        line = pyglet.shapes.Line(
            x_car,
            y_car,
            x_car + dx,
            y_car + dy,
            thickness=1,
            color=color,
            batch=e.batch,
        )
        e.lidar_lines.append(line)


def render_side_distances(env_renderer: EnvRenderer) -> None:
    """
    Render the left and right side distances on the screen.
    """
    e = env_renderer
    if e.scans is None:
        return

    left_dist, right_dist = get_side_distances(e.scans[0])

    if not hasattr(e, "side_dist_labels"):
        e.side_dist_labels = {
            "left": pyglet.text.Label(
                "",
                font_size=16,
                x=e.width - 20,
                y=20,
                anchor_x="right",
                color=(255, 255, 255, 255),
                batch=e.ui_batch,
            ),
            "right": pyglet.text.Label(
                "",
                font_size=16,
                x=e.width - 20,
                y=50,
                anchor_x="right",
                color=(255, 255, 255, 255),
                batch=e.ui_batch,
            ),
        }

    e.side_dist_labels["left"].text = f"Left Distance: {left_dist:.2f} m"
    e.side_dist_labels["right"].text = f"Right Distance: {right_dist:.2f} m"

    # Update positions in case window was resized
    e.side_dist_labels["left"].x = e.width - 20
    e.side_dist_labels["left"].y = 20
    e.side_dist_labels["right"].x = e.width - 20
    e.side_dist_labels["right"].y = 50
