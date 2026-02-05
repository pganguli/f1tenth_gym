import numpy as np
from numba import njit

@njit(cache=True)
def get_side_distances(scan: np.ndarray) -> tuple[float, float]:
    """
    Calculate the minimum distance to the left and right walls from a lidar scan.
    
    The lidar has a FOV of 4.7 radians (approx 270 degrees) and 1080 beams.
    Indices:
    - 0: -fov/2 (-2.35 rad, approx -135 degrees)
    - 540: 0 rad (forward)
    - 1079: fov/2 (2.35 rad, approx 135 degrees)
    
    Left side (90 degrees or pi/2 rad) is roughly at index:
    (pi/2 - (-fov/2)) / (fov / 1079)
    = (1.57 + 2.35) / (4.7 / 1079)
    = 3.92 / 0.004356 = 900
    
    Right side (-90 degrees or -pi/2 rad) is roughly at index:
    (-pi/2 - (-fov/2)) / (fov / 1079)
    = (-1.57 + 2.35) / (4.7 / 1079)
    = 0.78 / 0.004356 = 179
    
    We'll take a small window around these indices to find the minimum distance.
    """
    fov = 4.7
    num_beams = len(scan)
    angle_increment = fov / (num_beams - 1)
    
    # Left side (pi/2)
    left_angle = np.pi / 2
    left_idx = int((left_angle + fov / 2) / angle_increment)
    
    # Right side (-pi/2)
    right_angle = -np.pi / 2
    right_idx = int((right_angle + fov / 2) / angle_increment)
    
    # Window size (approx 10 degrees)
    window_angle = 10 * np.pi / 180
    window_size = int(window_angle / angle_increment)
    
    left_min = np.min(scan[max(0, left_idx - window_size):min(num_beams, left_idx + window_size)])
    right_min = np.min(scan[max(0, right_idx - window_size):min(num_beams, right_idx + window_size)])
    
    return left_min, right_min
