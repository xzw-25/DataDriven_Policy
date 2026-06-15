from vehicle_controller.control.longitudinal_allocator import LongitudinalAllocator
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


def test_drive_and_brake_are_mutually_exclusive() -> None:
    allocator = LongitudinalAllocator(VehicleParameters(), ActuatorLimits())
    drive = allocator.allocate(1.0, 5.0)
    brake = allocator.allocate(-1.0, 5.0)
    assert drive.drive_torque_nm > 0.0
    assert drive.brake_decel_mps2 == 0.0
    assert brake.drive_torque_nm == 0.0
    assert brake.brake_decel_mps2 == 1.0

