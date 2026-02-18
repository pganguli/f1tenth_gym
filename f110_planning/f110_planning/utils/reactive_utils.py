"""
Utilities for reactive navigation, including coordinate conversions and LiDAR mappings.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def polar_to_rect(r: float, angle: float) -> np.ndarray:
    """
    Convert polar coordinates to rectangular coordinates.

    Args:
        r: Radius.
        angle: Angle in radians.

    Returns:
        np.ndarray: [x, y] coordinates.
    """
    return np.array([r * np.cos(angle), r * np.sin(angle)])


@njit(cache=True)
def circular_offset(x: float, angle: float, r: float) -> np.ndarray:
    """
    Compute a circular offset from a point.

    Args:
        x: Radial distance from center.
        angle: Angle in radians.
        r: Reference radius.

    Returns:
        np.ndarray: [dx, dy] offset.
    """
    offset_radius = r + x
    # angle = y / offset_radius
    return polar_to_rect(offset_radius, angle) - np.array([r, 0])


@njit(cache=True)
def index_to_angle(i: int) -> float:
    """
    Convert a LiDAR scan index to its corresponding angle.

    Args:
        i: Scan index (expected range 0 to 1079).

    Returns:
        float: Angle in radians.
    """
    max_index = 1080 - 1
    start_angle = -np.pi / 4
    end_angle = 5 * np.pi / 4
    angle_range = end_angle - start_angle
    return i / max_index * angle_range + start_angle


@njit(cache=True)
def get_point(obs: np.ndarray, i: int) -> np.ndarray:
    """
    Get the rectangular coordinates for a specific LiDAR observation.

    Args:
        obs: Array of LiDAR range measurements.
        i: Index of the measurement.

    Returns:
        np.ndarray: [x, y] point.
    """
    return polar_to_rect(obs[i], index_to_angle(i))
