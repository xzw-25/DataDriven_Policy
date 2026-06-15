from vehicle_controller.geometry.curvature import three_point_curvature
from vehicle_controller.types import TrajectoryPoint


def test_left_turn_curvature_is_positive() -> None:
    kappa = three_point_curvature(
        TrajectoryPoint(0.0, 0.0),
        TrajectoryPoint(1.0, 0.0),
        TrajectoryPoint(2.0, 1.0),
    )
    assert kappa > 0.0

