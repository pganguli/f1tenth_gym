"""
Utilities for reactive navigation, including coordinate conversions and LiDAR mappings.
"""


# F1TENTH Vehicle Constants
F110_WIDTH = 0.31  # meters
F110_LENGTH = 0.58  # meters
F110_MAX_STEER = 0.4189  # approx 24 degrees in radians
F110_WHEELBASE = 0.33  # meters

# LiDAR Constants (standard f110_gym config)
LIDAR_FOV = 4.7  # radians (approx 270 degrees)
LIDAR_MIN_ANGLE = -LIDAR_FOV / 2  # -2.35 rad
LIDAR_MAX_ANGLE = LIDAR_FOV / 2  # 2.35 rad
