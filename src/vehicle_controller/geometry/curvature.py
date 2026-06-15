"""Reference curvature calculation and filtering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vehicle_controller.types import TrajectoryPoint


def three_point_curvature(
    first: TrajectoryPoint,
    middle: TrajectoryPoint,
    last: TrajectoryPoint,
) -> float:
    a = np.array([middle.x - first.x, middle.y - first.y], dtype=np.float64)
    b = np.array([last.x - middle.x, last.y - middle.y], dtype=np.float64)
    c = np.array([last.x - first.x, last.y - first.y], dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
    if denominator <= 1e-9:
        return 0.0
    cross = a[0] * b[1] - a[1] * b[0]
    return float(2.0 * cross / denominator)


def weighted_mean_curvature(
    points: Sequence[TrajectoryPoint],
    weights: Sequence[float] | None = None,
) -> float:
    if not points:
        raise ValueError("Curvature requires at least one point")
    curvatures = np.asarray([point.kappa for point in points], dtype=np.float64)
    if np.allclose(curvatures, 0.0) and len(points) >= 3:
        estimated = [
            three_point_curvature(points[index - 1], points[index], points[index + 1])
            for index in range(1, len(points) - 1)
        ]
        curvatures = np.asarray([estimated[0], *estimated, estimated[-1]], dtype=np.float64)
    used_weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    if used_weights.shape != curvatures.shape:
        raise ValueError("Curvature weights must match the point count")
    if np.sum(used_weights) <= 0.0:
        raise ValueError("Curvature weights must have a positive sum")
    return float(np.average(curvatures, weights=used_weights))


def resolve_reference_curvature(
    points: Sequence[TrajectoryPoint],
    supplied_kappa: float | None,
    weights: Sequence[float] | None = None,
    limit: float = 0.5,
) -> float:
    kappa = weighted_mean_curvature(points, weights) if supplied_kappa is None else supplied_kappa
    return float(np.clip(kappa, -limit, limit))


class FirstOrderFilter:
    def __init__(self, time_constant_s: float, initial_value: float = 0.0) -> None:
        if time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        self.time_constant_s = time_constant_s
        self.value = initial_value

    def update(self, measurement: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        alpha = dt / (self.time_constant_s + dt)
        self.value += alpha * (measurement - self.value)
        return self.value

