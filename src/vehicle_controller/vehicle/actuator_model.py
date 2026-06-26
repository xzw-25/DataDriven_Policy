"""First-order actuator response model."""

from __future__ import annotations

from vehicle_controller.types import VehicleCommand


class FirstOrderActuatorModel:
    def __init__(self, time_constant_s: float = 0.1) -> None:
        if time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        self.time_constant_s = time_constant_s
        self.command = VehicleCommand()

    def step(self, target: VehicleCommand, dt: float) -> VehicleCommand:
        alpha = dt / (self.time_constant_s + dt)
        self.command = VehicleCommand(
            steering_wheel_angle_deg=self.command.steering_wheel_angle_deg
            + alpha * (target.steering_wheel_angle_deg - self.command.steering_wheel_angle_deg),
            drive_wheel_torque_nm=self.command.drive_wheel_torque_nm
            + alpha * (target.drive_wheel_torque_nm - self.command.drive_wheel_torque_nm),
            brake_decel_mps2=self.command.brake_decel_mps2
            + alpha * (target.brake_decel_mps2 - self.command.brake_decel_mps2),
            source=target.source,
            reason=target.reason,
        )
        return self.command
