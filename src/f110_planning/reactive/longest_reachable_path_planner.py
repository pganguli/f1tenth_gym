import numpy as np
from simple_pid import PID

from .. import Action, BasePlanner


class LongestReachablePathPlanner(BasePlanner):
    @staticmethod
    def getCoordinates(r, angle):
        return np.array([r * np.cos(angle), r * np.sin(angle)])

    @staticmethod
    def index2Angle(i, maxIndex, startAngle, endAngle):
        angleRange = endAngle - startAngle
        return i / maxIndex * angleRange + startAngle

    def __init__(
        self,
        v_max: float = 6,
        v_min: float = 4,
        car_width: float = 0.31 + 0.05,
        adjustment_step: int = 5,
    ):
        self.pid = PID(-1, 0, 0, setpoint=0)
        self.v_max = v_max
        self.v_min = v_min
        self.car_width = car_width
        self.adjustment_step = adjustment_step

    def plan(self, obs):
        obs = obs["scans"][0]
        octantN = len(obs) // 6

        frontIndexStart = 1 * octantN
        frontIndexEnd = 5 * octantN

        maxI = np.argmax(obs[frontIndexStart:frontIndexEnd]) + frontIndexStart
        originalI = maxI
        lidarOrigin = np.array([0, 0])
        carWidthOffset = np.array([self.car_width, 0])
        p0 = lidarOrigin - carWidthOffset
        p2 = lidarOrigin + carWidthOffset

        # iOffset = 0
        iDirection = (
            1
            if obs[originalI + self.adjustment_step]
            > obs[originalI - self.adjustment_step]
            else -1
        )
        switchedDirection = False
        while True:
            distantPoint = LongestReachablePathPlanner.getCoordinates(
                obs[maxI],
                LongestReachablePathPlanner.index2Angle(
                    maxI, len(obs) - 1, -np.pi / 4, 5 * np.pi / 4
                ),
            )
            p1 = distantPoint - carWidthOffset
            basis = np.linalg.inv(np.column_stack([p2 - p0, p1 - p0]))
            numObstacles = 0

            for i in range(0, len(obs) - 1):
                dist = obs[i]
                angle = LongestReachablePathPlanner.index2Angle(
                    i, len(obs) - 1, -np.pi / 4, 5 * np.pi / 4
                )
                point = LongestReachablePathPlanner.getCoordinates(dist, angle)
                changeOfBasis = basis.dot(point - p0)
                inRegion = 0 <= changeOfBasis[0] <= 1 and 0 <= changeOfBasis[1] <= 1
                if inRegion and changeOfBasis[1] < 0.8:
                    numObstacles += 1
                    if maxI != originalI:
                        break

            if numObstacles == 0:
                # we found an angle where there were no other points in the path
                break

            # if maxI > originalI:
            #     maxI = originalI - iOffset
            # else:
            #     iOffset += self.ADJUSTMENT_STEP
            #     maxI = originalI + iOffset
            maxI += self.adjustment_step * iDirection
            maxI = np.clip(maxI, 0, len(obs) - 1)
            if maxI == 0 or maxI == len(obs) - 1:
                if not switchedDirection:
                    iDirection *= -1
                    maxI = originalI + self.adjustment_step * iDirection
                else:
                    print("could not find adjustment")
                    break

        angle = (
            LongestReachablePathPlanner.index2Angle(
                maxI, len(obs) - 1, -np.pi / 4, 5 * np.pi / 4
            )
            - np.pi / 2
        )
        control = self.pid(float(angle))
        if control is None:
            control = angle
        speedInterpolation = 2 * abs(control) / np.pi
        return Action(
            steer=control,
            speed=self.v_max * (1 - speedInterpolation)
            + self.v_min * speedInterpolation,
        )
