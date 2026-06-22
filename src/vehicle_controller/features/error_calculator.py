"""Tracking error calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from vehicle_controller.types import TrackingErrors, TrajectoryPoint, VehicleState


def nearest_trajectory_index(
    points: Sequence[TrajectoryPoint],
    state: VehicleState,
) -> int:
    if not points:
        raise ValueError("At least one reference point is required")
    distances = [(point.x - state.pose.x) ** 2 + (point.y - state.pose.y) ** 2 for point in points]
    return int(np.argmin(distances))


def calculate_tracking_errors(
    points: Sequence[TrajectoryPoint],
    v_ref: float,
    s_ref: float,
    state: VehicleState,
) -> TrackingErrors:
    if len(points) < 2:
        raise ValueError("At least two reference points are required")
    index = nearest_trajectory_index(points, state)
    previous = points[max(0, index - 1)]
    following = points[min(len(points) - 1, index + 1)]
    path_yaw = math.atan2(following.y - previous.y, following.x - previous.x)
    nearest = points[index]
    dx = state.pose.x - nearest.x
    dy = state.pose.y - nearest.y
    vehicle_lateral_offset = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy
    return TrackingErrors(
        e_lat=-vehicle_lateral_offset,
        e_v=v_ref - state.vx,
        e_s=s_ref - state.s,
    )
