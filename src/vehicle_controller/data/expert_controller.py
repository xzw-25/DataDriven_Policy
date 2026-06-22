"""Expert preview lateral and cascaded longitudinal PID controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vehicle_controller.features.error_calculator import nearest_trajectory_index
from vehicle_controller.types import NeuralPolicyOutput, TrajectoryPoint, VehicleState
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_limit: float
    integral: float = 0.0
    previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        derivative = (
            0.0
            if self.previous_error is None
            else (error - self.previous_error) / dt
        )
        candidate_integral = float(
            np.clip(
                self.integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        unclipped = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        output = float(np.clip(unclipped, self.output_min, self.output_max))
        if output == unclipped or np.sign(error) != np.sign(unclipped - output):
            self.integral = candidate_integral
        self.previous_error = error
        return output


class ExpertController:
    """Preview curvature feedforward plus feedback and position-speed PID."""

    def __init__(
        self,
        vehicle: VehicleParameters,
        actuator_limits: ActuatorLimits,
        preview_base_m: float = 5.0,
        preview_time_s: float = 0.6,
        heading_kp: float = 1.8,
        lateral_kp: float = 1.2,
        maximum_accel_mps2: float = 4.0,
        maximum_decel_mps2: float = 6.0,
        maximum_accel_rate_mps3: float = 8.0,
    ) -> None:
        self.vehicle = vehicle
        self.actuator_limits = actuator_limits
        self.preview_base_m = preview_base_m
        self.preview_time_s = preview_time_s
        self.heading_kp = heading_kp
        self.lateral_kp = lateral_kp
        self.maximum_accel_mps2 = maximum_accel_mps2
        self.maximum_accel_rate_mps3 = maximum_accel_rate_mps3
        self.position_pid = PIDController(0.35, 0.02, 0.05, -3.0, 3.0, 10.0)
        self.speed_pid = PIDController(
            1.2,
            0.15,
            0.03,
            -maximum_decel_mps2,
            maximum_accel_mps2,
            10.0,
        )
        self.previous_steering_rad = 0.0
        self.previous_acceleration_mps2 = 0.0

    def reset(self) -> None:
        self.position_pid.reset()
        self.speed_pid.reset()
        self.previous_steering_rad = 0.0
        self.previous_acceleration_mps2 = 0.0

    @staticmethod
    def _path_heading(points: tuple[TrajectoryPoint, ...], index: int) -> float:
        first = points[max(0, index - 1)]
        last = points[min(len(points) - 1, index + 1)]
        return math.atan2(last.y - first.y, last.x - first.x)

    @staticmethod
    def _preview_index(
        points: tuple[TrajectoryPoint, ...],
        nearest_index: int,
        preview_distance_m: float,
    ) -> int:
        target_s = points[nearest_index].s + preview_distance_m
        path_s = np.fromiter((point.s for point in points), dtype=np.float64)
        return min(int(np.searchsorted(path_s, target_s)), len(points) - 1)

    def compute(
        self,
        points: tuple[TrajectoryPoint, ...],
        state: VehicleState,
        reference_s_m: float,
        reference_speed_mps: float,
        reference_acceleration_mps2: float,
        dt: float,
    ) -> NeuralPolicyOutput:
        nearest_index = nearest_trajectory_index(points, state)
        preview_distance = self.preview_base_m + self.preview_time_s * max(state.vx, 0.0)
        preview_index = self._preview_index(points, nearest_index, preview_distance)
        preview_point = points[preview_index]
        nearest_point = points[nearest_index]
        path_heading = self._path_heading(points, nearest_index)

        lateral_error = (
            -math.sin(path_heading) * (state.pose.x - nearest_point.x)
            + math.cos(path_heading) * (state.pose.y - nearest_point.y)
        )
        heading_error = _wrap_angle(path_heading - state.pose.yaw)
        feedforward_front_angle = math.atan(
            self.vehicle.wheelbase_m * preview_point.kappa
        )
        feedback_front_angle = (
            self.heading_kp * heading_error
            - math.atan2(self.lateral_kp * lateral_error, max(state.vx, 1.0))
        )
        desired_steering = (
            feedforward_front_angle + feedback_front_angle
        ) * self.vehicle.steering_ratio
        maximum_steering_change = self.actuator_limits.steering_rate_max_radps * dt
        steering = float(
            np.clip(
                desired_steering,
                self.previous_steering_rad - maximum_steering_change,
                self.previous_steering_rad + maximum_steering_change,
            )
        )
        steering = float(
            np.clip(
                steering,
                self.actuator_limits.steering_min_rad,
                self.actuator_limits.steering_max_rad,
            )
        )

        position_error = reference_s_m - state.s
        speed_correction = self.position_pid.update(position_error, dt)
        target_speed = max(0.0, reference_speed_mps + speed_correction)
        acceleration = reference_acceleration_mps2 + self.speed_pid.update(
            target_speed - state.vx,
            dt,
        )
        maximum_accel_change = self.maximum_accel_rate_mps3 * dt
        acceleration = float(
            np.clip(
                acceleration,
                self.previous_acceleration_mps2 - maximum_accel_change,
                self.previous_acceleration_mps2 + maximum_accel_change,
            )
        )
        acceleration = float(
            np.clip(
                acceleration,
                -self.actuator_limits.brake_decel_max_mps2,
                self.maximum_accel_mps2,
            )
        )
        self.previous_steering_rad = steering
        self.previous_acceleration_mps2 = acceleration
        return NeuralPolicyOutput.from_rad(steering, acceleration)
