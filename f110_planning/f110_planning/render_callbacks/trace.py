import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer


def create_trace_renderer(
    max_points: int = 1000, color: tuple[int, int, int] = (184, 134, 11)
):
    """
    Factory to create a trajectory trace rendering callback.

    Args:
        max_points: Maximum number of points to keep in the trace.
        color: RGB tuple for the trace points.
    """
    last_pos = None
    min_dist = 0.1  # Minimum distance between trace points in meters

    def render_trace(env_renderer: EnvRenderer) -> None:
        nonlocal last_pos
        e = env_renderer

        if not hasattr(e, "trace_shapes"):
            e.trace_shapes = []

        # Get current ego position
        if e.poses is not None:
            ego_pos = e.poses[e.ego_idx, 0:2]

            # Add point if moved enough or first point
            if last_pos is None or np.linalg.norm(ego_pos - last_pos) > min_dist:
                scaled_pos = 50.0 * ego_pos

                point = pyglet.shapes.Circle(
                    x=scaled_pos[0],
                    y=scaled_pos[1],
                    radius=1.5,
                    color=color,
                    batch=e.batch,
                )
                e.trace_shapes.append(point)
                last_pos = ego_pos.copy()

                # Maintain max points
                if len(e.trace_shapes) > max_points:
                    old_point = e.trace_shapes.pop(0)
                    old_point.delete()

    return render_trace
