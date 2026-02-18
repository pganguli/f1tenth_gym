"""
Render callback for visualizing the dynamic waypoint
"""

import numpy as np
import pyglet

from ..utils import get_heading_error, get_side_distances


def create_dynamic_waypoint_renderer(planner, agent_idx=0):
    """
    Factory function to create a render callback that displays the dynamic waypoint.

    Args:
        planner (DynamicWaypointPlanner): The planner instance whose parameters to use
        agent_idx (int, default=0): Index of the agent to visualize

    Returns:
        callable: Render callback function
    """

    # Graphics objects (created once, updated each frame)
    waypoint_circle = None
    connection_line = None

    def render_callback(env_renderer):
        nonlocal waypoint_circle, connection_line

        # Get observation data
        if (
            env_renderer.scans is None
            or env_renderer.poses is None
            or env_renderer.poses.shape[0] <= agent_idx
        ):
            return

        if not env_renderer.cars or len(env_renderer.cars) <= agent_idx:
            return

        car_theta = env_renderer.poses[agent_idx, 2]
        world_position = env_renderer.poses[agent_idx, :2]
        car_center_px = 50.0 * world_position

        # Compute dynamic waypoint in world coordinates
        lateral_error = (
            get_side_distances(env_renderer.scans[agent_idx])[0]
            - get_side_distances(env_renderer.scans[agent_idx])[1]
        ) / 2.0
        heading_error = get_heading_error(planner.waypoints, world_position, car_theta)

        # Vehicle frame waypoint
        target_y_vehicle = (
            planner.lateral_gain * lateral_error
            + 0.5 * heading_error * planner.lookahead_distance
        )

        # World frame waypoint converted to pixels
        # Scale of 50 used throughout the rendering system
        target_px = 50.0 * np.array(
            [
                world_position[0]
                + planner.lookahead_distance * np.cos(car_theta)
                - target_y_vehicle * np.sin(car_theta),
                world_position[1]
                + planner.lookahead_distance * np.sin(car_theta)
                + target_y_vehicle * np.cos(car_theta),
            ]
        )

        # Create or update circle
        if waypoint_circle is None:
            waypoint_circle = pyglet.shapes.Circle(
                x=target_px[0],
                y=target_px[1],
                radius=8,
                color=(255, 100, 255),  # Magenta
                batch=env_renderer.batch,
            )
        else:
            waypoint_circle.x = target_px[0]
            waypoint_circle.y = target_px[1]

        # Create or update line from car to waypoint
        if connection_line is None:
            connection_line = pyglet.shapes.Line(
                x=car_center_px[0],
                y=car_center_px[1],
                x2=target_px[0],
                y2=target_px[1],
                thickness=2,
                color=(255, 100, 255, 150),  # Semi-transparent magenta
                batch=env_renderer.batch,
            )
        else:
            connection_line.x = car_center_px[0]
            connection_line.y = car_center_px[1]
            connection_line.x2 = target_px[0]
            connection_line.y2 = target_px[1]

    return render_callback
