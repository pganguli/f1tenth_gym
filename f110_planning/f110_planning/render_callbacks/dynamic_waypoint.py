"""
Render callback for visualizing the dynamic waypoint
"""

import numpy as np
import pyglet

from ..utils import get_side_distances, get_heading_error


def create_dynamic_waypoint_renderer(waypoints, agent_idx=0, lookahead_distance=1.0, lateral_gain=1.0):
    """
    Factory function to create a render callback that displays the dynamic waypoint.
    
    Args:
        waypoints (np.ndarray): Reference waypoints for heading error computation
        agent_idx (int, default=0): Index of the agent to visualize
        lookahead_distance (float, default=1.0): Lookahead distance
        lateral_gain (float, default=1.0): Lateral correction gain
        
    Returns:
        callable: Render callback function
    """
    
    # Graphics objects (created once, updated each frame)
    waypoint_circle = None
    connection_line = None
    
    def render_callback(env_renderer):
        nonlocal waypoint_circle, connection_line
        
        e = env_renderer
        
        # Get observation data
        if e.scans is None or e.poses is None or e.poses.shape[0] <= agent_idx:
            return
            
        if not e.cars or len(e.cars) <= agent_idx:
            return
            
        scan = e.scans[agent_idx]
        car_position = np.array(e.cars[agent_idx].vertices[:2])  # Pixel coordinates
        car_theta = e.poses[agent_idx, 2]  # Theta in world frame
        world_position = np.array([e.poses[agent_idx, 0], e.poses[agent_idx, 1]])
        
        # Compute dynamic waypoint in world coordinates
        left_dist, right_dist = get_side_distances(scan)
        heading_error = get_heading_error(waypoints, world_position, car_theta)
        
        lateral_error = (left_dist - right_dist) / 2.0
        
        # Vehicle frame waypoint
        target_x_vehicle = lookahead_distance
        target_y_vehicle = lateral_gain * lateral_error + 0.5 * heading_error * lookahead_distance
        
        # World frame waypoint
        target_x_world = world_position[0] + target_x_vehicle * np.cos(car_theta) - target_y_vehicle * np.sin(car_theta)
        target_y_world = world_position[1] + target_x_vehicle * np.sin(car_theta) + target_y_vehicle * np.cos(car_theta)
        
        # Convert to pixel coordinates (scale of 50 used throughout the rendering system)
        scale = 50.0
        target_x_px = target_x_world * scale
        target_y_px = target_y_world * scale
        
        # Create or update circle
        if waypoint_circle is None:
            waypoint_circle = pyglet.shapes.Circle(
                x=target_x_px,
                y=target_y_px,
                radius=8,
                color=(255, 100, 255),  # Magenta
                batch=e.batch
            )
        else:
            waypoint_circle.x = target_x_px
            waypoint_circle.y = target_y_px
        
        # Create or update line from car to waypoint
        if connection_line is None:
            connection_line = pyglet.shapes.Line(
                x=car_position[0],
                y=car_position[1],
                x2=target_x_px,
                y2=target_y_px,
                thickness=2,
                color=(255, 100, 255, 150),  # Semi-transparent magenta
                batch=e.batch
            )
        else:
            connection_line.x = car_position[0]
            connection_line.y = car_position[1]
            connection_line.x2 = target_x_px
            connection_line.y2 = target_y_px
    
    return render_callback
