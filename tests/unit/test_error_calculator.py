import pytest

from vehicle_controller.features.error_calculator import calculate_tracking_errors
from vehicle_controller.types import Pose2D, TrajectoryPoint, VehicleState


def test_tracking_errors_use_lateral_position_and_longitudinal_speed() -> None:
    points = [TrajectoryPoint(0.0, 0.0), TrajectoryPoint(10.0, 0.0)]
    errors = calculate_tracking_errors(
        points,
        v_ref=6.0,
        s_ref=5.0,
        state=VehicleState(
            pose=Pose2D(0.0, 1.0, 0.0),
            vx=4.0,
            vy=0.5,
            ax=0.0,
            ay=0.0,
            r=0.0,
            s=2.0,
        ),
    )
    assert errors.e_lat == pytest.approx(-1.0)
    assert errors.e_v == pytest.approx(2.0)
    assert errors.e_s == pytest.approx(3.0)
