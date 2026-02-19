"""
Gap Follower Planner (FGM) for F1TENTH.

Logic: Contiguous Free Space.
Mechanism: 
1. Finds the closest obstacle and sets a 'bubble' of nearby LiDAR beams to zero.
2. Identifies the largest contiguous sequence (the 'gap') of non-zero LiDAR beams.
3. Finds the 'best point' (usually the midpoint or furthest point) within that 
   specific gap and steers towards it.
"""

from typing import Any

import numpy as np
from numba import njit

from ..base import Action, BasePlanner
from ..utils import (
    F110_MAX_STEER,
    LIDAR_FOV,
    LIDAR_MIN_ANGLE,
    index_to_angle,
)

# Constants
LIDAR_MIN_CONSIDERED_ANGLE = -np.pi / 2
LIDAR_MAX_CONSIDERED_ANGLE = np.pi / 2
MAX_LIDAR_DIST = 10.0  # meters


@njit(cache=True)
def find_max_gap_jit(free_space_ranges: np.ndarray) -> tuple[int, int]:
    """
    JIT-optimized version of find_max_gap.
    Finds the largest contiguous sequence of non-zero values.
    """
    max_start = 0
    max_end = 0
    max_len = 0

    current_start = -1
    for i, val in enumerate(free_space_ranges):
        if val != 0:
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                current_len = i - current_start
                if current_len > max_len:
                    max_len = current_len
                    max_start = current_start
                    max_end = i
                current_start = -1

    # Check last gap
    if current_start != -1:
        current_len = len(free_space_ranges) - current_start
        if current_len > max_len:
            max_start = current_start
            max_end = len(free_space_ranges)

    return max_start, max_end


@njit(cache=True)
def find_best_point_jit(
    start_i: int, end_i: int, ranges: np.ndarray, best_point_conv_size: int
) -> int:
    """
    JIT-optimized version of find_best_point.
    Does a sliding window average (convolution) and finds the argmax.
    """
    sub_ranges = ranges[start_i:end_i]
    if len(sub_ranges) == 0:
        return start_i

    # Manual convolution since np.convolve 'same' is not fully supported in all numba versions
    # or might be slower than a simple loop for this specific case.
    # Actually, we just need the argmax of the averaged window.
    n = len(sub_ranges)
    k = best_point_conv_size

    if n < k:
        # If the gap is smaller than the window, just find the max in the gap
        max_val = -1.0
        max_idx = 0
        for i in range(n):
            if sub_ranges[i] > max_val:
                max_val = sub_ranges[i]
                max_idx = i
        return max_idx + start_i

    # Sliding window sum
    current_sum = 0.0
    for i in range(k):
        current_sum += sub_ranges[i]

    max_sum = current_sum
    max_idx = k // 2  # Center of the first window

    for i in range(1, n - k + 1):
        current_sum = current_sum - sub_ranges[i - 1] + sub_ranges[i + k - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            max_idx = i + k // 2

    return max_idx + start_i


class GapFollowerPlanner(BasePlanner):  # pylint: disable=too-many-instance-attributes
    """
    Gap Follower Planner for F1TENTH
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        bubble_radius: int = 160,
        corners_speed: float = 4.0,
        straights_speed: float = 6.0,
        straights_steering_angle: float = np.pi / 18,  # 10 degrees
        preprocess_conv_size: int = 3,
        best_point_conv_size: int = 80,
        max_lidar_dist: float = MAX_LIDAR_DIST,
    ):
        # used when calculating the angles of the LiDAR data
        self.bubble_radius = bubble_radius
        self.corners_speed = corners_speed
        self.straights_speed = straights_speed
        self.straights_steering_angle = straights_steering_angle
        self.preprocess_conv_size = preprocess_conv_size
        self.best_point_conv_size = best_point_conv_size
        self.max_lidar_dist = max_lidar_dist

    def preprocess_lidar(self, ranges: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess the LiDAR scan array.
        1. Slices to front hemisphere (-90 to 90 degrees)
        2. Convolutional smoothing
        3. Clipping to max distance
        """
        num_beams = len(ranges)
        # Determine index range for front hemisphere
        idx_start = int(
            ((LIDAR_MIN_CONSIDERED_ANGLE - LIDAR_MIN_ANGLE) / LIDAR_FOV) * (num_beams - 1)
        )
        idx_end = int(
            ((LIDAR_MAX_CONSIDERED_ANGLE - LIDAR_MIN_ANGLE) / LIDAR_FOV) * (num_beams - 1)
        )

        proc_ranges = np.array(ranges[idx_start:idx_end])
        # sets each value to the mean over a given window
        proc_ranges = (
            np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), "same")
            / self.preprocess_conv_size
        )
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)
        return proc_ranges, idx_start, num_beams

    def plan(self, obs: dict[str, Any], ego_idx: int = 0) -> Action:  # pylint: disable=too-many-locals
        """
        Process each LiDAR scan as per the Follow Gap algorithm.
        """
        ranges = obs["scans"][ego_idx]
        proc_ranges, idx_offset, num_beams = self.preprocess_lidar(ranges)

        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = max(closest - self.bubble_radius, 0)
        max_index = min(closest + self.bubble_radius, len(proc_ranges) - 1)
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = find_max_gap_jit(proc_ranges)

        # Find the best point in the gap
        best = find_best_point_jit(
            gap_start, gap_end, proc_ranges, self.best_point_conv_size
        )

        # Get the actual index in the original scan
        final_idx = best + idx_offset
        steering_angle = index_to_angle(final_idx, num_beams)

        # Optimization: Clip to car limits
        steering_angle = np.clip(steering_angle, -F110_MAX_STEER, F110_MAX_STEER)

        if abs(steering_angle) > self.straights_steering_angle:
            speed = self.corners_speed
        else:
            speed = self.straights_speed
        return Action(steer=steering_angle, speed=speed)
