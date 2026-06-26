"""Final safety decision layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vehicle_controller.types import (
    SafetyAction,
    SafetyDecision,
    TrackingErrors,
    VehicleCommand,
    VehicleState,
)


@dataclass(frozen=True)
class SafetyLimits:
    lateral_accel_max_mps2: float = 6.0
    yaw_rate_max_radps: float = 1.5
    lateral_error_max_m: float = 3.0
    speed_max_mps: float = 40.0


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def evaluate(
        self,
        candidate: VehicleCommand,
        fallback: VehicleCommand,
        state: VehicleState,
        errors: TrackingErrors,
    ) -> SafetyDecision:
        command_values = (
            candidate.steering_wheel_angle_deg,
            candidate.drive_wheel_torque_nm,
            candidate.brake_decel_mps2,
        )
        if not all(math.isfinite(value) for value in command_values):
            return SafetyDecision(SafetyAction.FALLBACK, "non_finite_command", fallback)
        if candidate.drive_valid and candidate.brake_valid:
            return SafetyDecision(SafetyAction.FALLBACK, "conflicting_longitudinal_command", fallback)
        if abs(errors.e_lat) > self.limits.lateral_error_max_m:
            return SafetyDecision(SafetyAction.FALLBACK, "lateral_error_limit", fallback)
        if abs(state.ay) > self.limits.lateral_accel_max_mps2:
            return SafetyDecision(SafetyAction.FALLBACK, "lateral_accel_limit", fallback)
        if abs(state.r) > self.limits.yaw_rate_max_radps:
            return SafetyDecision(SafetyAction.FALLBACK, "yaw_rate_limit", fallback)
        if state.vx > self.limits.speed_max_mps:
            return SafetyDecision(SafetyAction.FALLBACK, "speed_limit", fallback)
        return SafetyDecision(SafetyAction.PASS, "ok", candidate)
