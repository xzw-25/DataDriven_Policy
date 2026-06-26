import pytest

from vehicle_controller.control.longitudinal_allocator import LongitudinalAllocator
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


def test_drive_and_brake_are_mutually_exclusive() -> None:
    vehicle = VehicleParameters()
    allocator = LongitudinalAllocator(vehicle, ActuatorLimits())
    drive = allocator.allocate(1.0, 5.0)
    brake = allocator.allocate(-1.0, 5.0)
    drag_force = (
        0.5
        * vehicle.air_density_kgpm3
        * vehicle.drag_coefficient
        * vehicle.frontal_area_m2
        * 5.0**2
    )
    expected_wheel_torque = (
        vehicle.mass_kg * (1.0 + vehicle.rolling_resistance_coefficient * 9.81) + drag_force
    ) * vehicle.wheel_radius_m
    assert drive.drive_wheel_torque_nm == pytest.approx(expected_wheel_torque)
    assert drive.drive_valid
    assert drive.brake_decel_mps2 == 0.0
    assert not drive.brake_valid
    assert brake.drive_wheel_torque_nm == 0.0
    assert not brake.drive_valid
    assert brake.brake_decel_mps2 == 1.0
    assert brake.brake_valid
