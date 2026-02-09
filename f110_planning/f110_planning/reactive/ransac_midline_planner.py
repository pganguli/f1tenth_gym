"""
RANSAC-based Midline Following Planner

This planner uses RANSAC to fit two walls from LiDAR data and follows
the midline between them using proportional control.
"""

import numpy as np
from typing import Any

from sklearn.linear_model import RANSACRegressor

from .. import Action, BasePlanner


class RansacMidlinePlanner(BasePlanner):
    """
    RANSAC-based midline following planner for F1TENTH
    
    This planner fits two walls using RANSAC linear regression on LiDAR data,
    computes the midline between them, and uses proportional control to follow it.
    
    Args:
        fov (float, default=4.7): LiDAR field of view in radians
        max_range (float, default=10.0): Maximum valid LiDAR range
        ransac_thresh (float, default=0.05): RANSAC inlier distance threshold
        max_speed (float, default=5.0): Maximum speed
        kp_heading (float, default=1.6): Proportional gain for heading control
        kp_lateral (float, default=0.2): Proportional gain for lateral error control
        track_width (float, default=0.31): Expected track width for fallback
    """
    
    def __init__(
        self, 
        fov: float = 4.7,
        max_range: float = 10.0,
        ransac_thresh: float = 0.05,
        max_speed: float = 5.0,
        kp_heading: float = 1.6,
        kp_lateral: float = 0.2,
        track_width: float = 0.31
    ):
        self.fov = fov
        self.max_range = max_range
        self.ransac_thresh = ransac_thresh
        self.max_speed = max_speed
        self.kp_heading = kp_heading
        self.kp_lateral = kp_lateral
        self.track_width = track_width
        
    def _compute_midline(self, ranges: np.ndarray) -> tuple[float, float, list]:
        """
        Compute midline heading and lateral error from LiDAR scan.
        
        Args:
            ranges (np.ndarray): LiDAR scan data [N,]
        
        Returns:
            tuple: (psi_mid, e_y, wall_lines)
                - psi_mid: midline heading in vehicle frame [rad]
                - e_y: lateral error [m], positive = car left of midline
                - wall_lines: list of (slope, intercept) tuples for walls
        """
        N = len(ranges)
        
        # Convert polar to Cartesian coordinates
        angles = np.linspace(-self.fov / 2, self.fov / 2, N)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Filter valid points
        mask = (ranges > 0.01) & (ranges < self.max_range)
        if np.sum(mask) < 10:
            # Insufficient valid points
            return 0.0, 0.0, [(0, 0), (0, 0)]
        
        x_valid = x[mask]
        y_valid = y[mask]
        
        # Fit first wall with RANSAC
        ransac1 = RANSACRegressor(residual_threshold=self.ransac_thresh, random_state=42)
        ransac1.fit(x_valid.reshape(-1, 1), y_valid)
        a1 = ransac1.estimator_.coef_[0]
        b1 = ransac1.estimator_.intercept_
        
        # Remove inliers to fit second wall
        inliers1 = ransac1.inlier_mask_
        remaining_x = x_valid[~inliers1]
        remaining_y = y_valid[~inliers1]
        
        if len(remaining_x) >= 2:
            ransac2 = RANSACRegressor(residual_threshold=self.ransac_thresh, random_state=42)
            ransac2.fit(remaining_x.reshape(-1, 1), remaining_y)
            a2 = ransac2.estimator_.coef_[0]
            b2 = ransac2.estimator_.intercept_
        else:
            # Second wall not visible, assume parallel offset
            a2 = a1
            b2 = b1 + self.track_width
        
        # Compute midline heading and lateral offset
        psi_mid = 0.5 * (np.arctan(a1) + np.arctan(a2))
        y_mid = 0.5 * (b1 + b2)
        e_y = -y_mid  # positive = car left of midline
        
        return psi_mid, e_y, [(a1, b1), (a2, b2)]
    
    def plan(self, obs: dict[str, Any], ego_idx: int) -> Action:
        """
        Plan action based on RANSAC midline following.
        
        Args:
            obs (dict): Observation dictionary containing 'scans'
            ego_idx (int): Index of ego vehicle
            
        Returns:
            Action: Steering angle and speed command
        """
        scan = obs["scans"][ego_idx]
        
        # Compute midline and errors
        psi_mid, e_y, _ = self._compute_midline(scan)
        
        # Proportional control
        steer = self.kp_heading * psi_mid - self.kp_lateral * e_y
        speed = self.max_speed
        
        return Action(steer=steer, speed=speed)