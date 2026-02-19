"""
Utilities for loading and processing waypoints.
"""

import numpy as np
import pandas as pd


def load_waypoints(file_path, delimiter="\t"):
    """
    Loads waypoints from a CSV file. Returns only x and y columns.

    Args:
        file_path (str): Path to the waypoint CSV file.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        np.ndarray: Waypoints as a numpy array with columns [x, y].
    """
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        if "x_m" in df.columns and "y_m" in df.columns:
            waypoints = df[["x_m", "y_m"]].to_numpy()
        else:
            # Fallback to first two columns
            waypoints = df.iloc[:, 0:2].to_numpy()
    except (OSError, KeyError, IndexError) as e:
        print(f"Could not load waypoints from {file_path}: {e}")
        waypoints = np.array([]).reshape(0, 2)
    return waypoints
