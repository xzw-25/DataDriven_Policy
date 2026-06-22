"""Typical path and speed profiles for simulation data generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_controller.geometry.curvature import three_point_curvature
from vehicle_controller.types import Pose2D, TrajectoryPoint, VehicleState


@dataclass(frozen=True)
class ReferenceProfile:
    name: str
    points: tuple[TrajectoryPoint, ...]
    time_s: np.ndarray
    reference_s_m: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])

    def sample(self, time_s: float) -> tuple[float, float, float]:
        time_s = float(np.clip(time_s, self.time_s[0], self.time_s[-1]))
        return (
            float(np.interp(time_s, self.time_s, self.reference_s_m)),
            float(np.interp(time_s, self.time_s, self.speed_mps)),
            float(np.interp(time_s, self.time_s, self.acceleration_mps2)),
        )


def initial_state_from_reference_profile(
    profile: ReferenceProfile,
    lateral_offset_m: float = 0.0,
    yaw_offset_rad: float = 0.0,
    speed_offset_mps: float = 0.0,
) -> VehicleState:
    """Build an initial state aligned with the first segment of a reference profile."""
    first = profile.points[0]
    second = profile.points[1]
    path_yaw = float(np.arctan2(second.y - first.y, second.x - first.x))
    x = first.x - np.sin(path_yaw) * lateral_offset_m
    y = first.y + np.cos(path_yaw) * lateral_offset_m
    return VehicleState(
        pose=Pose2D(float(x), float(y), float(path_yaw + yaw_offset_rad)),
        vx=max(0.0, float(profile.speed_mps[0] + speed_offset_mps)),
        vy=0.0,
        ax=0.0,
        ay=0.0,
        r=0.0,
    )


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _trajectory_points(x: np.ndarray, y: np.ndarray) -> tuple[TrajectoryPoint, ...]:
    s = _arc_length(x, y)
    base = [TrajectoryPoint(float(px), float(py), s=float(ps)) for px, py, ps in zip(x, y, s)]
    curvatures = np.zeros(len(base), dtype=np.float64)
    for index in range(1, len(base) - 1):
        curvatures[index] = three_point_curvature(
            base[index - 1],
            base[index],
            base[index + 1],
        )
    if len(base) > 2:
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
    return tuple(
        TrajectoryPoint(point.x, point.y, s=point.s, kappa=float(curvature))
        for point, curvature in zip(base, curvatures)
    )


def _integrated_speed_profile(
    duration_s: float,
    time_step_s: float,
    time_knots_s: tuple[float, ...],
    speed_knots_mps: tuple[float, ...],
    maximum_path_s_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.arange(0.0, duration_s + 0.5 * time_step_s, time_step_s)
    speed = np.interp(time_s, time_knots_s, speed_knots_mps)
    acceleration = np.gradient(speed, time_step_s)
    reference_s = np.zeros_like(time_s)
    reference_s[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * time_step_s)
    reference_s = np.minimum(reference_s, maximum_path_s_m)
    return time_s, reference_s, speed, acceleration


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _turn_path(direction: float) -> tuple[TrajectoryPoint, ...]:
    straight_in = np.linspace(0.0, 20.0, 81)
    radius = 25.0
    angle = np.linspace(0.0, np.pi / 2.0, 121)
    arc_x = 20.0 + radius * np.sin(angle)
    arc_y = direction * radius * (1.0 - np.cos(angle))
    straight_out = np.linspace(0.25, 50.0, 200)
    out_x = np.full_like(straight_out, arc_x[-1])
    out_y = arc_y[-1] + direction * straight_out
    x = np.concatenate((straight_in, arc_x[1:], out_x))
    y = np.concatenate((np.zeros_like(straight_in), arc_y[1:], out_y))
    return _trajectory_points(x, y)


def _lane_change_path(
    direction: float,
    return_to_center: bool = False,
) -> tuple[TrajectoryPoint, ...]:
    x = np.linspace(0.0, 140.0, 561)
    lane_width = 3.5
    first = _smoothstep((x - 20.0) / 30.0)
    y = direction * lane_width * first
    if return_to_center:
        second = _smoothstep((x - 60.0) / 25.0)
        y = direction * lane_width * (first - second)
    return _trajectory_points(x, y)


def _straight_path(length_m: float = 120.0) -> tuple[TrajectoryPoint, ...]:
    x = np.linspace(0.0, length_m, int(length_m * 4.0) + 1)
    return _trajectory_points(x, np.zeros_like(x))


def _profile(
    name: str,
    points: tuple[TrajectoryPoint, ...],
    time_step_s: float,
    time_knots_s: tuple[float, ...],
    speed_knots_mps: tuple[float, ...],
) -> ReferenceProfile:
    duration_s = time_knots_s[-1]
    time_s, reference_s, speed, acceleration = _integrated_speed_profile(
        duration_s,
        time_step_s,
        time_knots_s,
        speed_knots_mps,
        points[-1].s,
    )
    return ReferenceProfile(
        name=name,
        points=points,
        time_s=time_s,
        reference_s_m=reference_s,
        speed_mps=speed,
        acceleration_mps2=acceleration,
    )


def build_typical_scenarios(time_step_s: float = 0.02) -> tuple[ReferenceProfile, ...]:
    """Build left/right turns, stop-go, and lane-change scenarios."""
    cruise_time = (0.0, 2.0, 14.0, 18.0)
    cruise_speed = (2.0, 6.0, 6.0, 3.0)
    lane_time = (0.0, 2.0, 13.0, 17.0)
    lane_speed = (3.0, 7.0, 7.0, 4.0)
    return (
        _profile("left_turn", _turn_path(1.0), time_step_s, cruise_time, cruise_speed),
        _profile("right_turn", _turn_path(-1.0), time_step_s, cruise_time, cruise_speed),
        _profile(
            "stop_go",
            _straight_path(),
            time_step_s,
            (0.0, 2.0, 7.0, 10.0, 13.0, 16.0, 23.0, 27.0),
            (0.0, 5.0, 5.0, 0.0, 0.0, 4.0, 4.0, 0.0),
        ),
        _profile(
            "lane_change_left",
            _lane_change_path(1.0),
            time_step_s,
            lane_time,
            lane_speed,
        ),
        _profile(
            "lane_change_right",
            _lane_change_path(-1.0),
            time_step_s,
            lane_time,
            lane_speed,
        ),
        _profile(
            "double_lane_change",
            _lane_change_path(1.0, return_to_center=True),
            time_step_s,
            (0.0, 2.0, 15.0, 20.0),
            (3.0, 7.0, 7.0, 3.0),
        ),
    )
