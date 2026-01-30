"""
LQR utilities
"""

import numpy as np
from numba import njit


@njit(cache=True)
def solve_lqr(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K

    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration

    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = np.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = (
            AT @ P @ A
            - (AT @ P @ B + M) @ np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT)
            + Q
        )

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K


@njit(cache=True)
def update_matrix(vehicle_state, state_size, timestep, wheelbase):
    """
    calc A and b matrices of linearized, discrete system.

    Args:
        vehicle_state:
        state_size:
        timestep:
        wheelbase:

    Returns:
        A:
        b:
    """

    # Current vehicle velocity
    v = vehicle_state[3]

    # Initialization of the time discrete A matrix
    matrix_ad_ = np.zeros((state_size, state_size))

    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    # b = [0.0, 0.0, 0.0, v / L].T
    matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_
