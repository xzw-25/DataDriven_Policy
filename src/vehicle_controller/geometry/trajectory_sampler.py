"""Arc-length trajectory interpolation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from vehicle_controller.types import TrajectoryPoint


DEFAULT_LOOKAHEAD_DISTANCES_M = (2.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_PREVIEW_TIMES_S = (0.1, 0.2, 0.3, 0.4, 0.5)


def preview_distances_from_times(
    preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
    speed_mps: float = 0.0,
    acceleration_mps2: float = 0.0,
) -> tuple[float, ...]:
    """Convert preview time horizons into non-negative arc-length offsets."""
    if len(preview_times_s) != 5:
        raise ValueError("Exactly five preview times are required")
    if any(time_s < 0.0 for time_s in preview_times_s):
        raise ValueError("Preview times must be non-negative")

    speed = max(float(speed_mps), 0.0)
    acceleration = float(acceleration_mps2)
    distances = np.asarray(
        [
            max(speed * float(time_s) + 0.5 * acceleration * float(time_s) ** 2, 0.0)
            for time_s in preview_times_s
        ],
        dtype=np.float64,
    )
    return tuple(float(value) for value in np.maximum.accumulate(distances))


def _arc_lengths(points: Sequence[TrajectoryPoint]) -> np.ndarray:
    distances = np.zeros(len(points), dtype=np.float64)
    for index in range(1, len(points)):
        distances[index] = distances[index - 1] + math.hypot(
            points[index].x - points[index - 1].x,
            points[index].y - points[index - 1].y,
        )
    return distances


def _interpolate(
    points: Sequence[TrajectoryPoint],
    arc_lengths: np.ndarray,
    target_s: float,
) -> TrajectoryPoint:
    target_s = float(np.clip(target_s, arc_lengths[0], arc_lengths[-1]))
    right = int(np.searchsorted(arc_lengths, target_s, side="right"))
    if right == 0:
        return points[0]
    if right >= len(points):
        return points[-1]

    left = right - 1
    segment_length = arc_lengths[right] - arc_lengths[left]
    ratio = 0.0 if segment_length <= 1e-9 else (target_s - arc_lengths[left]) / segment_length
    first = points[left]
    second = points[right]
    return TrajectoryPoint(
        x=first.x + ratio * (second.x - first.x),
        y=first.y + ratio * (second.y - first.y),
        s=first.s + ratio * (second.s - first.s),
        kappa=first.kappa + ratio * (second.kappa - first.kappa),
        v_ref=first.v_ref + ratio * (second.v_ref - first.v_ref),
        a_ref=first.a_ref + ratio * (second.a_ref - first.a_ref),
    )


def sample_trajectory(
    points: Sequence[TrajectoryPoint],
    lookahead_distances_m: Sequence[float] = DEFAULT_LOOKAHEAD_DISTANCES_M,
) -> list[TrajectoryPoint]:
    if len(points) < 2:
        raise ValueError("At least two trajectory points are required")
    if len(lookahead_distances_m) != 5:
        raise ValueError("Exactly five lookahead distances are required")
    if any(distance < 0.0 for distance in lookahead_distances_m):
        raise ValueError("Lookahead distances must be non-negative")

    arc_lengths = _arc_lengths(points)
    if arc_lengths[-1] <= 1e-9:
        raise ValueError("Trajectory length must be positive")
    return [_interpolate(points, arc_lengths, distance) for distance in lookahead_distances_m]


def sample_trajectory_by_preview_time(
    points: Sequence[TrajectoryPoint],
    preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
    speed_mps: float = 0.0,
    acceleration_mps2: float = 0.0,
) -> list[TrajectoryPoint]:
    return sample_trajectory(
        points,
        preview_distances_from_times(
            preview_times_s,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
        ),
    )
