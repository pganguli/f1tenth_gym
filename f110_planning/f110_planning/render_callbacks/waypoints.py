import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer

def create_waypoint_renderer(waypoints: np.ndarray):
    """
    Factory to create a waypoint rendering callback.
    
    Args:
        waypoints: np.ndarray of shape (N, 2+) where waypoints[:, 0:2] are x, y.
    """
    # Pre-scale waypoints for visualization
    scaled_wpts = 50.0 * waypoints[:, 0:2]
    
    def render_waypoints(env_renderer: EnvRenderer) -> None:
        e = env_renderer
        
        if not hasattr(e, "waypoint_shapes"):
            e.waypoint_shapes = []
            point_color = (183, 193, 222)
            for i in range(scaled_wpts.shape[0]):
                circle = pyglet.shapes.Circle(
                    x=scaled_wpts[i, 0],
                    y=scaled_wpts[i, 1],
                    radius=2.0,
                    color=point_color,
                    batch=e.batch
                )
                e.waypoint_shapes.append(circle)
        # If waypoints are static, we don't need to update every frame.
        # EnvRenderer.batch.draw() will handle it.
        
    return render_waypoints
