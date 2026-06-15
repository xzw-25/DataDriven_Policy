"""Deterministic actuator magnitude and rate limits."""

from __future__ import annotations

import math

from vehicle_controller.types import CommandSource, VehicleCommand
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits


def _rate_limit(target: float, previous: float, rate: float, dt: float) -> float:
    maximum_change = rate * dt
    return min(max(target, previous - maximum_change), previous + maximum_change)


class CommandLimiter:
    def __init__(self, limits: ActuatorLimits) -> None:
        self.limits = limits

    def limit(
        self,
        command: VehicleCommand,
        previous: VehicleCommand,
        dt: float,
    ) -> VehicleCommand:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                command.steering_wheel_angle_rad,
                command.drive_torque_nm,
                command.brake_decel_mps2,
            )
        ):
            raise ValueError("Command contains non-finite values")

        steering = min(
            max(command.steering_wheel_angle_rad, self.limits.steering_min_rad),
            self.limits.steering_max_rad,
        )
        steering = _rate_limit(
            steering,
            previous.steering_wheel_angle_rad,
            self.limits.steering_rate_max_radps,
            dt,
        )
        drive_torque = min(max(command.drive_torque_nm, 0.0), self.limits.drive_torque_max_nm)
        drive_torque = _rate_limit(
            drive_torque,
            previous.drive_torque_nm,
            self.limits.drive_torque_rate_max_nmps,
            dt,
        )
        brake_decel = min(
            max(command.brake_decel_mps2, 0.0),
            self.limits.brake_decel_max_mps2,
        )
        brake_decel = _rate_limit(
            brake_decel,
            previous.brake_decel_mps2,
            self.limits.brake_jerk_max_mps3,
            dt,
        )

        if brake_decel > 0.0:
            drive_torque = 0.0
        changed = any(
            abs(first - second) > 1e-9
            for first, second in (
                (steering, command.steering_wheel_angle_rad),
                (drive_torque, command.drive_torque_nm),
                (brake_decel, command.brake_decel_mps2),
            )
        )
        return VehicleCommand(
            steering_wheel_angle_rad=steering,
            drive_torque_nm=drive_torque,
            brake_decel_mps2=brake_decel,
            source=CommandSource.LIMITED_NEURAL if changed else command.source,
            reason="command_limited" if changed else command.reason,
        )

