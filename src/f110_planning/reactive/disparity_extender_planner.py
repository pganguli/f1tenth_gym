import numpy as np

from .. import Action, BasePlanner


class DisparityExtenderPlanner(BasePlanner):
    def __init__(self):
        self.car_width = 0.5
        self.disparity_threshold = 0.2
        self.scan_width = 270.0
        self.turn_clearance = 0.3
        self.max_turn_angle = 34.0
        self.min_speed = 4.0
        self.max_speed = 0.1  # .20
        self.absolute_max_speed = 4  # 0.3
        self.min_distance = 0.5  # changed from .35!
        self.max_distance = 3.0
        self.no_obstacles_distance = 6.0
        self.no_u_distance = 4.0
        self.min_considered_angle = -89.0
        self.max_considered_angle = 89.0
        self.should_stop = False
        self.total_packets = 0
        self.dropped_packets = 0
        self.lidar_distances: list[float] = []
        self.masked_disparities: np.ndarray
        self.possible_disparity_indices = None
        self.samples_per_degree = 0
        self.angle = 0
        self.velocity = 0

    def angle_from_index(self, i):
        """Returns the angle, in degrees, corresponding to index i in the
        LIDAR samples."""
        min_angle = -(self.scan_width / 2.0)
        return min_angle + (float(i) / self.samples_per_degree)

    def index_from_angle(self, i):
        center_index = self.scan_width * (self.samples_per_degree / 2)
        return int(center_index) + int(i * float(self.samples_per_degree))

    def find_disparities(self):
        """Scans each pair of subsequent values, and returns an array of indices
        where the difference between the two values is larger than the given
        threshold. The returned array contains only the index of the first value
        in pairs beyond the threshold."""
        to_return = []
        values = self.lidar_distances
        for i in range(len(values) - 1):
            if abs(values[i] - values[i + 1]) >= self.disparity_threshold:
                to_return.append(i)
        return to_return

    def lidar_callback(self, lidar_data):
        """This is asynchronously called every time we receive new LIDAR data."""
        self.total_packets += 1
        # If the lock is currently locked, then previous LIDAR data is still
        # being processed.
        distances = lidar_data
        self.lidar_distances = distances
        self.samples_per_degree = float(len(distances)) / self.scan_width
        target_distance, target_angle = self.find_new_angle()
        safe_distances = self.masked_disparities
        print(safe_distances)
        print(len(safe_distances))
        ind: int = len(safe_distances) // 2
        forward_distance = safe_distances[ind]

        # target_angle = self.adjust_angle_to_avoid_uturn(target_angle,
        #    forward_distance)
        target_angle = self.adjust_angle_for_car_side(target_angle)
        desired_speed = self.duty_cycle_from_distance(forward_distance)
        self.update_considered_angle(target_angle)
        # steering_percentage = self.degrees_to_steering_percentage(target_angle)
        self.angle = target_angle
        self.velocity = desired_speed

    def find_new_angle(self):
        """Returns the angle of the farthest possible distance that can be reached
        in a direct line without bumping into edges. Returns the distance in meters
        and the angle in degrees."""
        self.extend_disparities()
        limited_values = self.masked_disparities
        max_distance = -1.0e10
        angle = 0.0
        # Constrain the arc of possible angles we consider.
        min_sample_index = self.index_from_angle(self.min_considered_angle)
        max_sample_index = self.index_from_angle(self.max_considered_angle)
        print("Min sample index: " + str(min_sample_index))
        print("Max sample index: " + str(max_sample_index))
        limited_values = limited_values[min_sample_index:max_sample_index]
        distance = limited_values[0]
        for i in range(len(limited_values)):
            distance = limited_values[i]
            if distance > max_distance:
                angle = self.min_considered_angle + float(i) / self.samples_per_degree
                max_distance = distance
        return distance, angle

    def extend_disparities(self):
        """For each disparity in the list of distances, extends the nearest
        value by the car width in whichever direction covers up the more-
        distant points. Puts the resulting values in self.masked_disparities.
        """
        values = self.lidar_distances
        masked_disparities = np.copy(values)
        disparities = self.find_disparities()
        # Keep a list of disparity end points corresponding to safe driving
        # angles directly past a disparity. We will find the longest of these
        # constrained distances in situations where we need to turn towards a
        # disparity.
        self.possible_disparity_indices = []
        for d in disparities:
            a = values[d]
            b = values[d + 1]
            # If extend_positive is true, then extend the nearer value to
            # higher indices, otherwise extend it to lower indices.
            nearer_value = a
            nearer_index = d
            extend_positive = True
            if b < a:
                extend_positive = False
                nearer_value = b
                nearer_index = d + 1
            samples_to_extend = self.half_car_samples_at_distance(nearer_value)
            current_index = nearer_index
            for _ in range(samples_to_extend):
                # Stop trying to "extend" the disparity point if we reach the
                # end of the array.
                if current_index < 0:
                    current_index = 0
                    break
                if current_index >= len(masked_disparities):
                    current_index = len(masked_disparities) - 1
                    break
                # Don't overwrite values if we've already found a nearer point
                if masked_disparities[current_index] > nearer_value:
                    masked_disparities[current_index] = nearer_value
                # Finally, move left or right depending on the direction of the
                # disparity.
                if extend_positive:
                    current_index += 1
                else:
                    current_index -= 1
            self.possible_disparity_indices.append(current_index)
        self.masked_disparities = masked_disparities

    def adjust_angle_for_car_side(self, target_angle):
        """Takes the target steering angle, the distances from the LIDAR, and the
        angle covered by the LIDAR distances. Basically, this function attempts to
        keep the car from cutting corners too close to the wall. In short, it will
        make the car go straight if it's currently turning right and about to hit
        the right side of the car, or turning left or about to hit the left side
        f the car."""
        scan_width = self.scan_width
        car_tolerance = self.turn_clearance
        distances = self.lidar_distances
        turning_left = target_angle > 0.0
        # Get the portion of the LIDAR samples facing sideways and backwards on
        # the side of the car in the direction of the turn.
        samples_per_degree = float(len(distances)) / scan_width
        number_of_back_degrees = (scan_width / 2.0) - 90.0
        needed_sample_count = int(number_of_back_degrees * samples_per_degree)
        side_samples = []
        if turning_left:
            side_samples = distances[len(distances) - needed_sample_count :]
        else:
            side_samples = distances[:needed_sample_count]
        # Finally, just make sure no point in the backwards scan is too close.
        # This could definitely be more exact with some proper math.
        for v in side_samples:
            if v <= car_tolerance:
                return 0.0
        return target_angle

    def duty_cycle_from_distance(self, distance):
        """Takes a forward distance and returns a duty cycle value to set the
        car's velocity. Fairly unprincipled, basically just scales the speed
        directly based on distance, and stops if the car is blocked."""
        if distance <= self.min_distance:
            return 0.0
        if distance >= self.no_obstacles_distance:
            return self.absolute_max_speed
        if distance >= self.max_distance:
            return self.scale_speed_linearly(
                self.max_speed,
                self.absolute_max_speed,
                distance,
                self.max_distance,
                self.no_obstacles_distance,
            )
        return self.scale_speed_linearly(
            self.min_speed,
            self.max_speed,
            distance,
            self.min_distance,
            self.max_distance,
        )

    def update_considered_angle(self, steering_angle):
        actual_angle = steering_angle
        if actual_angle < -self.max_turn_angle:
            actual_angle = -self.max_turn_angle
        if actual_angle > self.max_turn_angle:
            actual_angle = self.max_turn_angle
        self.min_considered_angle = -89.0
        self.max_considered_angle = 89.0
        if actual_angle > 0:
            self.min_considered_angle -= actual_angle
        if actual_angle < 0:
            self.max_considered_angle += actual_angle

    def degrees_to_steering_percentage(self, degrees):
        """Returns a steering "percentage" value between 0.0 (left) and 1.0
        (right) that is as close as possible to the requested degrees. The car's
        wheels can't turn more than max_angle in either direction."""
        max_angle = self.max_turn_angle
        if degrees > max_angle:
            return 0.0
        if degrees < -max_angle:
            return 1.0
        # This maps degrees from -max_angle to +max_angle to values from 0 to 1.
        #   (degrees - min_angle) / (max_angle - min_angle)
        # = (degrees - (-max_angle)) / (max_angle - (-max_angle))
        # = (degrees + max_angle) / (max_angle * 2)
        return 1.0 - ((degrees + max_angle) / (2 * max_angle))

    def half_car_samples_at_distance(self, distance):
        """Returns the number of points in the LIDAR scan that will cover half of
        the width of the car along an arc at the given distance."""
        # This isn't exact, because it's really calculated based on the arc length
        # when it should be calculated based on the straight-line distance.
        # However, for simplicty we can just compensate for it by inflating the
        # "car width" slightly.
        distance_between_samples = np.pi * distance / (180.0 * self.samples_per_degree)
        return int(np.ceil(self.car_width / distance_between_samples))

    def scale_speed_linearly(
        self, speed_low, speed_high, distance, distance_low, distance_high
    ):
        """Scales the speed linearly in [speed_low, speed_high] based on the
        distance value, relative to the range [distance_low, distance_high]."""
        distance_range = distance_high - distance_low
        ratio = (distance - distance_low) / distance_range
        speed_range = speed_high - speed_low
        return speed_low + (speed_range * ratio)

    def plan(self, obs):
        self.lidar_distances = obs["scans"][0]
        self.lidar_callback(obs)
        print(self.angle)
        return Action(steer=self.angle, speed=self.velocity)
