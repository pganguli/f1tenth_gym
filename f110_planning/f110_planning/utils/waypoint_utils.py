"""
Utilities for loading and processing waypoints.
"""

import numpy as np
import pandas as pd


def load_waypoints(file_path, delimiter="\t"):
    """
    Loads waypoints from a CSV file and reorders columns to [x, y, v, th].

    Args:
        file_path (str): Path to the waypoint CSV file.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        np.ndarray: Waypoints as a numpy array with columns [x, y, v, th].
    """
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        waypoints = df[["x_m", "y_m"]].to_numpy()
    except (OSError, KeyError) as e:
        print(f"Could not load waypoints from {file_path}: {e}")
        waypoints = np.array([])
    return waypoints
