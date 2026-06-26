import pytest

from vehicle_controller.control.command_limiter import CommandLimiter
from vehicle_controller.types import VehicleCommand
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits


def test_command_limiter_applies_rate_and_brake_priority() -> None:
    limiter = CommandLimiter(ActuatorLimits())
    limited = limiter.limit(
        VehicleCommand(
            steering_wheel_angle_rad=5.0,
            drive_torque_nm=1000.0,
            brake_decel_mps2=2.0,
        ),
        VehicleCommand(),
        0.1,
    )
    assert limited.steering_wheel_angle_rad == pytest.approx(0.6)
    assert limited.steering_wheel_angle_deg == pytest.approx(34.3774677)
    assert limited.drive_wheel_torque_nm == 0.0
    assert not limited.drive_valid
    assert limited.brake_decel_mps2 == pytest.approx(1.0)
    assert limited.brake_valid
