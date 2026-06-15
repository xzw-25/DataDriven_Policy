"""Arc-length trajectory interpolation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from vehicle_controller.types import TrajectoryPoint


DEFAULT_LOOKAHEAD_DISTANCES_M = (2.0, 5.0, 10.0, 15.0, 20.0)


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
