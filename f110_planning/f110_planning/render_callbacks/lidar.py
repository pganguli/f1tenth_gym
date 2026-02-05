import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer

def render_lidar(env_renderer: EnvRenderer) -> None:
    """
    Draw LiDAR rays in Pyglet 2.x using pyglet.shapes.Line.
    """
    fov = 4.7
    e = env_renderer

    # Get latest scans from EnvRenderer
    if e.scans is None or e.poses is None:
        return

    scan = e.scans[0] # Ego scan
        
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
            x_car, y_car, x_car + dx, y_car + dy,
            thickness=1,
            color=color,
            batch=e.batch
        )
        e.lidar_lines.append(line)
