"""Map signed acceleration demand to mutually exclusive drive and brake commands."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


@dataclass(frozen=True)
class LongitudinalCommand:
    drive_torque_nm: float
    brake_decel_mps2: float


class LongitudinalAllocator:
    def __init__(
        self,
        vehicle: VehicleParameters,
        limits: ActuatorLimits,
        gravity_mps2: float = 9.81,
    ) -> None:
        self.vehicle = vehicle
        self.limits = limits
        self.gravity_mps2 = gravity_mps2

    def allocate(self, signed_accel_mps2: float, speed_mps: float) -> LongitudinalCommand:
        deadband = self.limits.longitudinal_deadband_mps2
        if signed_accel_mps2 < -deadband:
            return LongitudinalCommand(
                drive_torque_nm=0.0,
                brake_decel_mps2=min(-signed_accel_mps2, self.limits.brake_decel_max_mps2),
            )
        if signed_accel_mps2 <= deadband:
            return LongitudinalCommand(0.0, 0.0)

        rolling_accel = self.vehicle.rolling_resistance_coefficient * self.gravity_mps2
        drag_force = (
            0.5
            * self.vehicle.air_density_kgpm3
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * max(speed_mps, 0.0) ** 2
        )
        required_force = self.vehicle.mass_kg * (signed_accel_mps2 + rolling_accel) + drag_force
        wheel_torque = required_force * self.vehicle.wheel_radius_m
        drive_torque = wheel_torque / (
            self.vehicle.drivetrain_ratio * self.vehicle.drivetrain_efficiency
        )
        return LongitudinalCommand(
            drive_torque_nm=min(max(drive_torque, 0.0), self.limits.drive_torque_max_nm),
            brake_decel_mps2=0.0,
        )

