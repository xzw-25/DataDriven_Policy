import math

import pytest

from vehicle_controller.geometry.coordinate_transform import global_point_to_body
from vehicle_controller.types import Pose2D, TrajectoryPoint


def test_global_point_to_body_with_quarter_turn() -> None:
    transformed = global_point_to_body(
        TrajectoryPoint(10.0, 1.0),
        Pose2D(10.0, 0.0, math.pi / 2.0),
    )
    assert transformed.x == pytest.approx(1.0)
    assert transformed.y == pytest.approx(0.0, abs=1e-7)

