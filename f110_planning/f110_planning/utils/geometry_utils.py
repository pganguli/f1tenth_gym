"""
Geometry utilities
"""

import numpy as np
from numba import njit


@njit(cache=True)
def quat_2_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """
    Converts a quaternion into euler angles (roll, pitch, yaw)

    Args:
        x, y, z, w (float): input quaternion

    Returns:
        r, p, y (float): roll, pitch yaw
    """
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.atan2(t3, t4)
    return roll, pitch, yaw


@njit(cache=True)
def get_rotation_matrix(theta: float) -> np.ndarray:
    """
    Get 2D rotation matrix for a given angle.

    Args:
        theta: Rotation angle in radians.

    Returns:
        np.ndarray: 2x2 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))


@njit(cache=True)
def pi_2_pi(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        float: Normalized angle.
    """
    if angle > np.pi:
        return angle - 2.0 * np.pi
    if angle < -np.pi:
        return angle + 2.0 * np.pi

    return angle
