"""
RANSAC Wall Fitting Visualization Render Callback
"""

import numpy as np
import pyglet
from f110_gym.envs.rendering import EnvRenderer
from sklearn.linear_model import RANSACRegressor


def create_ransac_walls_renderer(
    agent_idx: int = 0,
    fov: float = 4.7,
    max_range: float = 10.0,
    ransac_thresh: float = 0.05,
    track_width: float = 0.31,
    horizon: float = 5.0
):
    """
    Factory to create a RANSAC wall fitting visualization callback.
    
    Args:
        agent_idx: Index of the agent to visualize walls for
        fov: LiDAR field of view in radians
        max_range: Maximum valid LiDAR range
        ransac_thresh: RANSAC inlier distance threshold
        track_width: Expected track width for fallback
        horizon: Forward viewing distance in meters
    """
    
    def compute_midline(ranges):
        """Compute wall fitting and midline from LiDAR scan"""
        N = len(ranges)
        
        # Convert polar to Cartesian coordinates
        angles = np.linspace(-fov / 2, fov / 2, N)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Filter valid points
        mask = (ranges > 0.01) & (ranges < max_range)
        if np.sum(mask) < 10:
            return 0.0, 0.0, [(0, 0), (0, 0)]
        
        x_valid = x[mask]
        y_valid = y[mask]
        
        # Fit first wall with RANSAC
        ransac1 = RANSACRegressor(residual_threshold=ransac_thresh, random_state=42)
        ransac1.fit(x_valid.reshape(-1, 1), y_valid)
        a1 = ransac1.estimator_.coef_[0]
        b1 = ransac1.estimator_.intercept_
        
        # Remove inliers to fit second wall
        inliers1 = ransac1.inlier_mask_
        remaining_x = x_valid[~inliers1]
        remaining_y = y_valid[~inliers1]
        
        if len(remaining_x) >= 2:
            ransac2 = RANSACRegressor(residual_threshold=ransac_thresh, random_state=42)
            ransac2.fit(remaining_x.reshape(-1, 1), remaining_y)
            a2 = ransac2.estimator_.coef_[0]
            b2 = ransac2.estimator_.intercept_
        else:
            # Second wall not visible, assume parallel offset
            a2 = a1
            b2 = b1 + track_width
        
        # Compute midline heading and lateral offset
        psi_mid = 0.5 * (np.arctan(a1) + np.arctan(a2))
        y_mid = 0.5 * (b1 + b2)
        e_y = -y_mid
        
        return psi_mid, e_y, [(a1, b1), (a2, b2)]

    def render_ransac_walls(env_renderer: EnvRenderer) -> None:
        e = env_renderer
        
        if e.scans is None or e.poses is None or e.poses.shape[0] <= agent_idx:
            return
        
        if not e.cars or len(e.cars) <= agent_idx:
            return
            
        # Get car pose from vertices (pixel coordinates)
        v = e.cars[agent_idx].vertices
        x_car = np.mean(v[::2])
        y_car = np.mean(v[1::2])
        yaw = e.poses[agent_idx][2]
        scan = e.scans[agent_idx]
        scale = 50  # Visualization scale
        
        # Compute fitted lines
        psi_mid, e_y, wall_lines = compute_midline(scan)
        
        # Clear previous lines
        if not hasattr(e, f"ransac_lines_{agent_idx}"):
            setattr(e, f"ransac_lines_{agent_idx}", [])
        
        ransac_lines = getattr(e, f"ransac_lines_{agent_idx}")
        for line in ransac_lines:
            line.delete()
        ransac_lines.clear()
        
        # Forward horizon in vehicle frame
        x_horizon = np.linspace(0, horizon, 10)
        
        # Draw walls (cyan) and midline (yellow)
        for idx, (a, b) in enumerate(wall_lines + [(0.0, 0.0)]):
            if idx == 2:  # midline
                a = np.tan(psi_mid)
                b = -e_y
                color = (255, 255, 0)  # yellow
            else:
                color = (0, 255, 255)  # cyan for walls
            
            # Draw line segments
            for i in range(len(x_horizon) - 1):
                x0, x1 = x_horizon[i], x_horizon[i + 1]
                y0, y1 = a * x0 + b, a * x1 + b
                
                # Transform from vehicle frame to world frame
                # Vehicle frame: x forward, y left
                # World frame: standard coordinates
                dx0 = np.cos(yaw) * x0 - np.sin(yaw) * y0
                dy0 = np.sin(yaw) * x0 + np.cos(yaw) * y0
                dx1 = np.cos(yaw) * x1 - np.sin(yaw) * y1
                dy1 = np.sin(yaw) * x1 + np.cos(yaw) * y1
                
                # Create line shape
                line = pyglet.shapes.Line(
                    x_car + dx0 * scale,
                    y_car + dy0 * scale,
                    x_car + dx1 * scale,
                    y_car + dy1 * scale,
                    thickness=2,
                    color=color,
                    batch=e.batch
                )
                ransac_lines.append(line)
    
    return render_ransac_walls