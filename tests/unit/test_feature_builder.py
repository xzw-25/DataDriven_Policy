import pytest

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.features.feature_builder import FeatureBuilder
from vehicle_controller.types import (
    Pose2D,
    TrackingErrors,
    TrajectoryPoint,
    VehicleState,
)


def test_feature_builder_uses_fixed_22_value_order() -> None:
    points = [TrajectoryPoint(float(index), float(-index)) for index in range(1, 6)]
    state = VehicleState(Pose2D(0.0, 0.0, 0.0), 6.0, 0.2, 0.3, 0.4, 0.5)
    values = FeatureBuilder().build(
        points,
        0.02,
        TrackingErrors(e_lat=0.1, e_v=0.2, e_s=0.3),
        0.4,
        7.0,
        12.0,
        state,
    ).values
    assert values.shape == (FEATURE_COUNT,)
    assert values[:4].tolist() == [1.0, -1.0, 2.0, -2.0]
    assert values[10:].tolist() == pytest.approx([
        0.02,
        0.1,
        0.2,
        0.3,
        0.4,
        7.0,
        12.0,
        6.0,
        0.2,
        0.3,
        0.4,
        0.5,
    ])
