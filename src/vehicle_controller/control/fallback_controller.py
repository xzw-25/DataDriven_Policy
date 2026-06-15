"""Simple deterministic fallback controller."""

from __future__ import annotations

import math
from collections.abc import Sequence

from vehicle_controller.types import NeuralPolicyOutput, TrackingErrors, TrajectoryPoint, VehicleState
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


class FallbackController:
    def __init__(
        self,
        vehicle: VehicleParameters,
        speed_kp: float = 0.8,
        maximum_accel_mps2: float = 3.0,
        maximum_decel_mps2: float = 5.0,
    ) -> None:
        self.vehicle = vehicle
        self.speed_kp = speed_kp
        self.maximum_accel_mps2 = maximum_accel_mps2
        self.maximum_decel_mps2 = maximum_decel_mps2

    def compute(
        self,
        body_points: Sequence[TrajectoryPoint],
        errors: TrackingErrors,
        a_ref: float,
        state: VehicleState,
    ) -> NeuralPolicyOutput:
        target = body_points[min(2, len(body_points) - 1)]
        lookahead = max(math.hypot(target.x, target.y), 0.5)
        alpha = math.atan2(target.y, target.x)
        front_wheel_angle = math.atan2(
            2.0 * self.vehicle.wheelbase_m * math.sin(alpha),
            lookahead,
        )
        steering_wheel_angle = front_wheel_angle * self.vehicle.steering_ratio
        signed_accel = a_ref + self.speed_kp * errors.e_v
        signed_accel = min(max(signed_accel, -self.maximum_decel_mps2), self.maximum_accel_mps2)
        return NeuralPolicyOutput(steering_wheel_angle, signed_accel)
