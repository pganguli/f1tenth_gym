"""
LQR waypoint tracker
Implementation inspired by https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/lqr_steer_control/lqr_steer_control.py
"""

from typing import Any

import numpy as np

from .. import Action, BasePlanner
from ..utils import nearest_point, pi_2_pi, solve_lqr, update_matrix


class LQRPlanner(BasePlanner):
    """
    Lateral Controller using LQR

    Args:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 5], optional, default=None): waypoints to track, columns are [x, y, velocity, heading, curvature]

    Attributes:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 5], optional, default=None): waypoints to track, columns are [x, y, velocity, heading, curvature]
        vehicle_control_e_cog (float): lateral error of cog to ref trajectory
        vehicle_control_theta_e (float): yaw error to ref trajectory
    """

    def __init__(
        self,
        wheelbase: float = 0.33,
        waypoints: np.ndarray = np.array([]),
        timestep: float = 0.01,
        matrix_q_1: float = 0.999,
        matrix_q_2: float = 0.0,
        matrix_q_3: float = 0.0066,
        matrix_q_4: float = 0.0,
        matrix_r: float = 0.75,
        iterations: int = 50,
        eps: float = 0.001,
    ):
        self.wheelbase = wheelbase
        self.waypoints = waypoints
        self.vehicle_control_e_cog = 0  # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0.0  # theta_e: yaw error to ref trajectory
        self.timestep = timestep
        self.matrix_q_1 = matrix_q_1
        self.matrix_q_2 = matrix_q_2
        self.matrix_q_3 = matrix_q_3
        self.matrix_q_4 = matrix_q_4
        self.matrix_r = matrix_r
        self.iterations = iterations
        self.eps = eps

    def calc_control_points(
        self, vehicle_state: np.ndarray, waypoints: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate the heading and cross-track errors and target velocity and curvature
        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track [x, y, velocity, heading, curvature]

        Returns:
            theta_e (float): heading error
            e_cog (float): lateral crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_veloctiy (float): target velocity
        """

        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.wheelbase * np.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * np.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        nearest_point_front, nearest_dist, t, target_index = nearest_point(
            position_front_axle, self.waypoints[:, 0:2]
        )
        vec_dist_nearest_point = position_front_axle - nearest_point_front

        # crosstrack error
        front_axle_vec_rot_90 = np.array(
            [
                [np.cos(vehicle_state[2] - np.pi / 2.0)],
                [np.sin(vehicle_state[2] - np.pi / 2.0)],
            ]
        )
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        # heading error
        # NOTE: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[target_index, 3]
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # target velocity
        goal_veloctiy = waypoints[target_index, 2]

        # reference curvature
        kappa_ref = self.waypoints[target_index, 4]

        # saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy

    def controller(
        self,
        vehicle_state: np.ndarray,
        waypoints: np.ndarray,
        ts: float,
        matrix_q: list[float],
        matrix_r: list[float],
        max_iteration: int,
        eps: float,
    ) -> tuple[float, float]:
        """
        Compute lateral control command.

        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track
            ts (float): discretization time step
            matrix_q ([float], len=4): weights on the states
            matrix_r ([float], len=1): weights on control input
            max_iteration (int): maximum iteration for solving
            eps (float): error tolerance for solving

        Returns:
            steer_angle (float): desired steering angle
            v_ref (float): desired velocity
        """

        # size of controlled states
        state_size = 4

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cg, yaw_ref, k_ref, v_ref = self.calc_control_points(
            vehicle_state, waypoints
        )

        # Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = update_matrix(
            vehicle_state, state_size, ts, self.wheelbase
        )

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = solve_lqr(
            matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration
        )

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cog_old) / ts
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts

        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        # Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = k_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, v_ref

    def plan(
        self,
        obs: dict[str, Any],
        ego_idx: int,
    ) -> Action:
        """
        Compute lateral control command.
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan()"
            )

        # Define LQR Matrix and Parameter
        matrix_q = [self.matrix_q_1, self.matrix_q_2, self.matrix_q_3, self.matrix_q_4]
        matrix_r = [self.matrix_r]

        # Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array(
            [
                obs["poses_x"][ego_idx],
                obs["poses_y"][ego_idx],
                obs["poses_theta"][ego_idx],
                obs["linear_vels_x"][ego_idx],
            ]
        )

        # Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(
            vehicle_state,
            self.waypoints,
            self.timestep,
            matrix_q,
            matrix_r,
            self.iterations,
            self.eps,
        )

        return Action(steer=steering_angle, speed=speed)
