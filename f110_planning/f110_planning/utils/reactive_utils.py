import numpy as np
from numba import njit


@njit(cache=True)
def polar2Rect(r: float, angle: float) -> np.ndarray:
    return np.array([r * np.cos(angle), r * np.sin(angle)])


@njit(cache=True)
def circularOffset(x: float, angle: float, r: float) -> np.ndarray:
    offsetRadius = r + x
    # angle = y / offsetRadius
    return polar2Rect(offsetRadius, angle) - np.array([r, 0])


@njit(cache=True)
def index2Angle(i: int) -> float:
    maxIndex = 1080 - 1
    startAngle = -np.pi / 4
    endAngle = 5 * np.pi / 4
    angleRange = endAngle - startAngle
    return i / maxIndex * angleRange + startAngle


@njit(cache=True)
def getPoint(obs: np.ndarray, i: int) -> np.ndarray:
    return polar2Rect(obs[i], index2Angle(i))
