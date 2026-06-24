"""Build controller features from extracted raw reference and pose signals."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vehicle_controller.constants import FEATURE_NAMES
from vehicle_controller.data.schema import TARGET_NAMES
from vehicle_controller.features.error_calculator import calculate_tracking_errors
from vehicle_controller.features.feature_builder import FeatureBuilder
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.geometry.coordinate_transform import global_to_body
from vehicle_controller.geometry.curvature import resolve_reference_curvature
from vehicle_controller.geometry.trajectory_sampler import (
    DEFAULT_PREVIEW_TIMES_S,
    preview_distances_from_times,
    sample_trajectory,
)
from vehicle_controller.types import Pose2D, TrajectoryPoint, VehicleState

CONTROL_TARGET_SIGNAL_NAMES = (
    "target_steering_wheel_angle",
    "target_longitudinal_acceleration",
)
ACCELERATION_REFERENCE_PREVIEW_TIME_S = 0.5


@dataclass(frozen=True)
class RawFrameFeature:
    values: np.ndarray
    control_target: np.ndarray
    target_valid: bool
    clip_id: str
    entry_index: int
    frame_index: int
    timestamp_s: float
    position_enu: np.ndarray
    heading_rad: float
    reference_point_count: int


@dataclass(frozen=True)
class RawFeatureDataset:
    features: np.ndarray
    raw_features: np.ndarray
    physical_targets: np.ndarray
    targets: np.ndarray
    target_valid_mask: np.ndarray
    clip_ids: np.ndarray
    entry_indices: np.ndarray
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    positions_enu: np.ndarray
    headings_rad: np.ndarray
    reference_point_counts: np.ndarray

    def save_npz(self, path: str | Path, metadata: Mapping[str, object]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            features=self.features,
            raw_features=self.raw_features,
            physical_targets=self.physical_targets,
            targets=self.targets,
            target_valid_mask=self.target_valid_mask,
            clip_ids=self.clip_ids,
            scenario_ids=self.clip_ids,
            entry_indices=self.entry_indices,
            frame_indices=self.frame_indices,
            timestamps_s=self.timestamps_s,
            positions_enu=self.positions_enu,
            headings_rad=self.headings_rad,
            pose_position_x_enu_m=self.positions_enu[:, 0],
            pose_position_y_enu_m=self.positions_enu[:, 1],
            pose_heading_rad=self.headings_rad,
            reference_point_counts=self.reference_point_counts,
            feature_names=np.asarray(FEATURE_NAMES),
            target_names=np.asarray(TARGET_NAMES),
            control_target_signal_names=np.asarray(CONTROL_TARGET_SIGNAL_NAMES),
            metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        )
        return output


def _array_at(mapping: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in mapping:
        raise KeyError(f"Missing signal: {key}")
    return np.asarray(mapping[key])


def _point_signal(points: Mapping[str, Any], name: str, frame_index: int, count: int) -> np.ndarray:
    if name not in points:
        return np.zeros(count, dtype=np.float64)
    return np.asarray(points[name][frame_index, :count], dtype=np.float64)


def reference_points_from_raw_frame(
    reference_traj: Mapping[str, Any],
    frame_index: int,
) -> list[TrajectoryPoint]:
    trajectory = reference_traj["trajectory"]
    points = trajectory["points"]
    valid_lengths = np.asarray(trajectory["valid_length"], dtype=np.int64)
    count = int(valid_lengths[frame_index])
    x_values = np.asarray(points["x"])
    if x_values.ndim != 2:
        raise ValueError(f"Expected reference x with shape (frames, points), got {x_values.shape}")
    count = max(0, min(count, x_values.shape[1]))
    if count < 2:
        raise ValueError(f"Frame {frame_index} has fewer than two valid reference points")

    x = _point_signal(points, "x", frame_index, count)
    y = _point_signal(points, "y", frame_index, count)
    s = _point_signal(points, "s", frame_index, count)
    kappa = _point_signal(points, "kappa", frame_index, count)
    v_ref = _point_signal(points, "v", frame_index, count)
    a_ref = _point_signal(points, "a", frame_index, count)

    return [
        TrajectoryPoint(
            float(x[index]),
            float(y[index]),
            s=float(s[index]),
            kappa=float(kappa[index]),
            v_ref=float(v_ref[index]),
            a_ref=float(a_ref[index]),
        )
        for index in range(count)
    ]


def reference_absolute_times_from_raw_frame(
    reference_traj: Mapping[str, Any],
    frame_index: int,
) -> np.ndarray:
    """Return point absolute times from reference timestamp and relative time."""
    trajectory = reference_traj["trajectory"]
    points = trajectory["points"]
    valid_lengths = np.asarray(trajectory["valid_length"], dtype=np.int64)
    count = int(valid_lengths[frame_index])
    relative_time_values = np.asarray(points["relative_time"])
    if relative_time_values.ndim != 2:
        raise ValueError(
            "Expected reference relative_time with shape "
            f"(frames, points), got {relative_time_values.shape}"
        )
    count = max(0, min(count, relative_time_values.shape[1]))
    if count < 2:
        raise ValueError(f"Frame {frame_index} has fewer than two valid reference times")

    base_timestamp = float(np.asarray(reference_traj["time"]["timestamp"])[frame_index])
    relative_times = np.asarray(
        relative_time_values[frame_index, :count],
        dtype=np.float64,
    )
    return base_timestamp + relative_times


def _interpolate_reference_signal(
    reference_traj: Mapping[str, Any],
    frame_index: int,
    absolute_times: np.ndarray,
    signal_name: str,
    target_timestamp_s: float,
) -> float:
    points = reference_traj["trajectory"]["points"]
    count = len(absolute_times)
    values = _point_signal(points, signal_name, frame_index, count)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Frame {frame_index} has non-finite reference {signal_name}")

    order = np.argsort(absolute_times, kind="stable")
    return float(
        np.interp(
            float(target_timestamp_s),
            absolute_times[order],
            values[order],
        )
    )


def longitudinal_reference_point_from_raw_frame(
    reference_traj: Mapping[str, Any],
    frame_index: int,
    pose_timestamp_s: float,
) -> TrajectoryPoint:
    absolute_times = reference_absolute_times_from_raw_frame(reference_traj, frame_index)
    if not np.all(np.isfinite(absolute_times)):
        raise ValueError(f"Frame {frame_index} has non-finite reference times")
    return TrajectoryPoint(
        x=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "x",
            pose_timestamp_s,
        ),
        y=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "y",
            pose_timestamp_s,
        ),
        s=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "s",
            pose_timestamp_s,
        ),
        kappa=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "kappa",
            pose_timestamp_s,
        ),
        v_ref=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "v",
            pose_timestamp_s,
        ),
        a_ref=_interpolate_reference_signal(
            reference_traj,
            frame_index,
            absolute_times,
            "a",
            pose_timestamp_s,
        ),
    )


def _enu_vector_to_body(vector_enu: np.ndarray, heading_rad: float) -> tuple[float, float]:
    east = float(vector_enu[0])
    north = float(vector_enu[1])
    cos_yaw = math.cos(heading_rad)
    sin_yaw = math.sin(heading_rad)
    return (
        cos_yaw * east + sin_yaw * north,
        -sin_yaw * east + cos_yaw * north,
    )


def vehicle_state_from_raw_pose(pose: Mapping[str, Any], frame_index: int) -> VehicleState:
    position = np.asarray(pose["position"]["position_enu"][frame_index], dtype=np.float64)
    heading = float(np.asarray(pose["orientation"]["heading"])[frame_index])
    linear_velocity_enu = np.asarray(
        pose["motion"]["linear_velocity_enu"][frame_index],
        dtype=np.float64,
    )
    linear_acceleration_flu = np.asarray(
        pose["motion"]["linear_acceleration_flu"][frame_index],
        dtype=np.float64,
    )
    angular_velocity_flu = np.asarray(
        pose["motion"]["angular_velocity_flu"][frame_index],
        dtype=np.float64,
    )
    timestamp = float(np.asarray(pose["time"]["timestamp"])[frame_index])
    vx, vy = _enu_vector_to_body(linear_velocity_enu, heading)
    return VehicleState(
        pose=Pose2D(float(position[0]), float(position[1]), heading),
        vx=vx,
        vy=vy,
        ax=float(linear_acceleration_flu[0]),
        ay=float(linear_acceleration_flu[1]),
        r=float(angular_velocity_flu[2]),
        timestamp_s=timestamp,
    )


def control_target_from_raw_frame(raw_data: Mapping[str, Any], frame_index: int) -> np.ndarray:
    command = raw_data["control_signal"]["command"]
    steering = float(np.asarray(command["target_steering_wheel_angle"])[frame_index])
    acceleration = float(np.asarray(command["target_longitudinal_acceleration"])[frame_index])
    return np.asarray([steering, acceleration], dtype=np.float32)


def _interpolate_trajectory_points(
    first: TrajectoryPoint,
    second: TrajectoryPoint,
    ratio: float,
) -> TrajectoryPoint:
    return TrajectoryPoint(
        x=first.x + ratio * (second.x - first.x),
        y=first.y + ratio * (second.y - first.y),
        s=first.s + ratio * (second.s - first.s),
        kappa=first.kappa + ratio * (second.kappa - first.kappa),
        v_ref=first.v_ref + ratio * (second.v_ref - first.v_ref),
        a_ref=first.a_ref + ratio * (second.a_ref - first.a_ref),
    )


def _project_state_to_forward_path(
    state: VehicleState,
    points: Sequence[TrajectoryPoint],
) -> tuple[TrajectoryPoint, list[TrajectoryPoint]]:
    if len(points) < 2:
        raise ValueError("At least two reference points are required")

    best_segment_index = 0
    best_ratio = 0.0
    best_distance_sq = math.inf
    for index in range(len(points) - 1):
        first = points[index]
        second = points[index + 1]
        segment_x = second.x - first.x
        segment_y = second.y - first.y
        segment_length_sq = segment_x * segment_x + segment_y * segment_y
        if segment_length_sq <= 1e-12:
            ratio = 0.0
        else:
            ratio = (
                (state.pose.x - first.x) * segment_x
                + (state.pose.y - first.y) * segment_y
            ) / segment_length_sq
            ratio = float(np.clip(ratio, 0.0, 1.0))

        projected_x = first.x + ratio * segment_x
        projected_y = first.y + ratio * segment_y
        distance_sq = (state.pose.x - projected_x) ** 2 + (state.pose.y - projected_y) ** 2
        if distance_sq < best_distance_sq:
            best_segment_index = index
            best_ratio = ratio
            best_distance_sq = distance_sq

    projected = _interpolate_trajectory_points(
        points[best_segment_index],
        points[best_segment_index + 1],
        best_ratio,
    )
    following_points = list(points[best_segment_index + 1 :])
    return projected, [projected, *following_points]


def build_raw_frame_feature(
    raw_data: Mapping[str, Any],
    frame_index: int,
    *,
    clip_id: str = "",
    entry_index: int = 0,
    preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
    lookahead_distances_m: Sequence[float] | None = None,
    curvature_weights: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    feature_builder: FeatureBuilder | None = None,
) -> RawFrameFeature:
    points = reference_points_from_raw_frame(raw_data["reference_traj"], frame_index)
    state = vehicle_state_from_raw_pose(raw_data["pose"], frame_index)
    longitudinal_reference_point = longitudinal_reference_point_from_raw_frame(
        raw_data["reference_traj"],
        frame_index,
        state.timestamp_s,
    )
    projected_point, forward_path_points = _project_state_to_forward_path(state, points)
    state = replace(state, s=projected_point.s)
    errors = calculate_tracking_errors(
        points,
        longitudinal_reference_point.v_ref,
        longitudinal_reference_point.s,
        state,
    )

    body_points = global_to_body(forward_path_points, state.pose)
    distances = lookahead_distances_m
    if distances is None:
        distances = preview_distances_from_times(
            preview_times_s,
            speed_mps=longitudinal_reference_point.v_ref,
            acceleration_mps2=longitudinal_reference_point.a_ref,
        )
    acceleration_preview_distance = preview_distances_from_times(
        (ACCELERATION_REFERENCE_PREVIEW_TIME_S,) * 5,
        speed_mps=longitudinal_reference_point.v_ref,
        acceleration_mps2=longitudinal_reference_point.a_ref,
    )[-1]
    acceleration_reference_point = sample_trajectory(
        body_points,
        (acceleration_preview_distance,) * 5,
    )[-1]
    sampled_points = sample_trajectory(body_points, distances)
    kappa = resolve_reference_curvature(sampled_points, None, curvature_weights)
    builder = feature_builder or FeatureBuilder()
    features = builder.build(
        sampled_points,
        kappa,
        errors,
        acceleration_reference_point.a_ref,
        longitudinal_reference_point.v_ref,
        longitudinal_reference_point.s,
        state,
    ).values
    control_target = control_target_from_raw_frame(raw_data, frame_index)
    position = np.asarray(raw_data["pose"]["position"]["position_enu"][frame_index], dtype=np.float64)
    return RawFrameFeature(
        values=features,
        control_target=control_target,
        target_valid=bool(np.all(np.isfinite(control_target))),
        clip_id=clip_id,
        entry_index=entry_index,
        frame_index=frame_index,
        timestamp_s=state.timestamp_s,
        position_enu=position,
        heading_rad=state.pose.yaw,
        reference_point_count=len(points),
    )


def build_raw_feature_dataset(
    entries: Sequence[Mapping[str, Any]],
    *,
    preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
    lookahead_distances_m: Sequence[float] | None = None,
    curvature_weights: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    normalizer: FeatureNormalizer | None = None,
    steering_scale_deg: float = 1.0,
    acceleration_scale_mps2: float = 1.0,
    target_clip: float = 1.0,
) -> RawFeatureDataset:
    if steering_scale_deg <= 0.0 or acceleration_scale_mps2 <= 0.0:
        raise ValueError("Target scales must be positive")
    if target_clip <= 0.0:
        raise ValueError("target_clip must be positive")

    frame_features: list[RawFrameFeature] = []
    builder = FeatureBuilder()
    for entry_index, entry in enumerate(entries):
        raw_data = entry["raw_data"]
        frame_count = int(raw_data["frame_count"])
        clip_id = str(entry.get("clip_id", f"entry_{entry_index:05d}"))
        for frame_index in range(frame_count):
            frame_features.append(
                build_raw_frame_feature(
                    raw_data,
                    frame_index,
                    clip_id=clip_id,
                    entry_index=entry_index,
                    preview_times_s=preview_times_s,
                    lookahead_distances_m=lookahead_distances_m,
                    curvature_weights=curvature_weights,
                    feature_builder=builder,
                )
            )

    if not frame_features:
        raise ValueError("Cannot build features from an empty entry list")

    raw_features = np.asarray([item.values for item in frame_features], dtype=np.float32)
    physical_targets = np.asarray([item.control_target for item in frame_features], dtype=np.float32)
    target_scales = np.asarray([steering_scale_deg, acceleration_scale_mps2], dtype=np.float32)
    targets = np.clip(physical_targets / target_scales, -target_clip, target_clip)
    features = (
        normalizer.normalize(raw_features)
        if normalizer is not None
        else raw_features.copy()
    ).astype(np.float32)
    return RawFeatureDataset(
        features=features,
        raw_features=raw_features,
        physical_targets=physical_targets,
        targets=targets.astype(np.float32),
        target_valid_mask=np.asarray([item.target_valid for item in frame_features], dtype=bool),
        clip_ids=np.asarray([item.clip_id for item in frame_features]),
        entry_indices=np.asarray([item.entry_index for item in frame_features], dtype=np.int32),
        frame_indices=np.asarray([item.frame_index for item in frame_features], dtype=np.int32),
        timestamps_s=np.asarray([item.timestamp_s for item in frame_features], dtype=np.float64),
        positions_enu=np.asarray([item.position_enu for item in frame_features], dtype=np.float64),
        headings_rad=np.asarray([item.heading_rad for item in frame_features], dtype=np.float64),
        reference_point_counts=np.asarray(
            [item.reference_point_count for item in frame_features],
            dtype=np.int32,
        ),
    )
