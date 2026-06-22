import pytest

from vehicle_controller.geometry.trajectory_sampler import (
    preview_distances_from_times,
    sample_trajectory,
    sample_trajectory_by_preview_time,
)
from vehicle_controller.types import TrajectoryPoint


def test_sampler_interpolates_fixed_lookaheads() -> None:
    points = [TrajectoryPoint(float(index), 0.0) for index in range(21)]
    sampled = sample_trajectory(points, [2.0, 5.0, 10.0, 15.0, 20.0])
    assert [point.x for point in sampled] == pytest.approx([2.0, 5.0, 10.0, 15.0, 20.0])


def test_preview_times_are_converted_to_distances() -> None:
    distances = preview_distances_from_times(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        speed_mps=10.0,
        acceleration_mps2=0.0,
    )

    assert distances == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_sampler_interpolates_preview_time_points() -> None:
    points = [TrajectoryPoint(float(index), 0.0) for index in range(21)]
    sampled = sample_trajectory_by_preview_time(
        points,
        [0.1, 0.2, 0.3, 0.4, 0.5],
        speed_mps=10.0,
        acceleration_mps2=0.0,
    )

    assert [point.x for point in sampled] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
