import pytest

from vehicle_controller.geometry.trajectory_sampler import sample_trajectory
from vehicle_controller.types import TrajectoryPoint


def test_sampler_interpolates_fixed_lookaheads() -> None:
    points = [TrajectoryPoint(float(index), 0.0) for index in range(21)]
    sampled = sample_trajectory(points, [2.0, 5.0, 10.0, 15.0, 20.0])
    assert [point.x for point in sampled] == pytest.approx([2.0, 5.0, 10.0, 15.0, 20.0])

