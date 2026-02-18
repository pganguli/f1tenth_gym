"""
Pure Pursuit utilities
"""

from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def nearest_point(
    point: np.ndarray, trajectory: np.ndarray
) -> tuple[np.ndarray, float, float, int]:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique,
            a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector
            formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        float(dists[min_dist_segment]),
        float(t[min_dist_segment]),
        int(min_dist_segment),
    )


def intersect_point(
    point: np.ndarray,
    radius: float,
    trajectory: np.ndarray,
    t: float = 0.0,
    wrap: bool = False,
) -> tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """
    Starts at beginning of trajectory, and find the first point one radius away
    from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_idx = int(t)
    trajectory = np.ascontiguousarray(trajectory)

    for offset in range(len(trajectory) if wrap else len(trajectory) - 1 - start_idx):
        idx = (start_idx + offset) % len(trajectory)
        p1 = trajectory[idx]
        v_vec = trajectory[(idx + 1) % len(trajectory)] + 1e-6 - p1

        t1, t2 = _get_intersections(point, radius, p1, v_vec)
        if t1 is not None:
            # Check only intersections within the segment [0, 1]
            # If the first segment, only check after start_t
            lower = t % 1.0 if offset == 0 else 0.0
            for ti in (t1, t2):
                if ti is not None and lower <= ti <= 1.0:
                    return p1 + ti * v_vec, idx, ti

    return None, None, None


def _get_intersections(point, radius, p1, v_vec):
    """Calculate intersection times of a circle and a line segment."""
    a = np.dot(v_vec, v_vec)
    b = 2.0 * np.dot(v_vec, p1 - point)
    c = np.dot(p1, p1) + np.dot(point, point) - 2.0 * np.dot(p1, point) - radius**2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    disc = np.sqrt(disc)
    return (-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a)


@njit(cache=True)
def get_actuation(
    pose_theta: float,
    lookahead_point: np.ndarray,
    position: np.ndarray,
    lookahead_distance: float,
    wheelbase: float,
) -> tuple[float, float]:
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    speed = max(lookahead_point[2], 0.0) * (1.0 - np.abs(steering_angle) * 2.0 / np.pi)
    return speed, steering_angle
