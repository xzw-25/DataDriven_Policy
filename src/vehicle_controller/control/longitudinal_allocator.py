"""Map signed acceleration demand to mutually exclusive drive and brake commands."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


@dataclass(frozen=True)
class LongitudinalCommand:
    drive_wheel_torque_nm: float
    drive_valid: bool
    brake_decel_mps2: float
    brake_valid: bool

    @property
    def drive_torque_nm(self) -> float:
        return self.drive_wheel_torque_nm


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
                drive_wheel_torque_nm=0.0,
                drive_valid=False,
                brake_decel_mps2=min(-signed_accel_mps2, self.limits.brake_decel_max_mps2),
                brake_valid=True,
            )
        if signed_accel_mps2 <= deadband:
            return LongitudinalCommand(0.0, True, 0.0, False)

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
        return LongitudinalCommand(
            drive_wheel_torque_nm=min(max(wheel_torque, 0.0), self.limits.drive_torque_max_nm),
            drive_valid=True,
            brake_decel_mps2=0.0,
            brake_valid=False,
        )
