"""Coordinate transforms using x-forward, y-left vehicle coordinates."""

from __future__ import annotations

import math
from collections.abc import Iterable

from vehicle_controller.types import Pose2D, TrajectoryPoint


def global_point_to_body(point: TrajectoryPoint, pose: Pose2D) -> TrajectoryPoint:
    dx = point.x - pose.x
    dy = point.y - pose.y
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return TrajectoryPoint(
        x=cos_yaw * dx + sin_yaw * dy,
        y=-sin_yaw * dx + cos_yaw * dy,
        s=point.s,
        kappa=point.kappa,
        v_ref=point.v_ref,
        a_ref=point.a_ref,
    )


def global_to_body(
    points: Iterable[TrajectoryPoint],
    pose: Pose2D,
) -> list[TrajectoryPoint]:
    return [global_point_to_body(point, pose) for point in points]
