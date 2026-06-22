from __future__ import annotations

import numpy as np
import pytest

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.feature_builder import (
    build_raw_feature_dataset,
    build_raw_frame_feature,
    control_target_from_raw_frame,
    longitudinal_reference_point_from_raw_frame,
    reference_absolute_times_from_raw_frame,
    reference_points_from_raw_frame,
    vehicle_state_from_raw_pose,
)


def make_raw_data(frame_count: int = 2) -> dict:
    point_count = 21
    base_x = np.arange(point_count, dtype=np.float32)
    x = np.vstack([base_x + frame_index for frame_index in range(frame_count)])
    y = np.zeros((frame_count, point_count), dtype=np.float32)
    s = np.tile(base_x, (frame_count, 1)).astype(np.float32)
    relative_time = np.tile(base_x, (frame_count, 1)).astype(np.float32)
    return {
        "frame_count": frame_count,
        "reference_traj": {
            "time": {"timestamp": np.arange(frame_count, dtype=np.float64) + 100.0},
            "status": {"is_replan": np.zeros(frame_count, dtype=np.int16)},
            "trajectory": {
                "points": {
                    "x": x,
                    "y": y,
                    "theta": np.zeros_like(x),
                    "kappa": np.zeros_like(x),
                    "s": s,
                    "v": np.full_like(x, 6.0),
                    "a": np.full_like(x, 0.5),
                    "relative_time": relative_time,
                    "da": np.zeros_like(x),
                },
                "valid_length": np.full(frame_count, point_count, dtype=np.int16),
            },
        },
        "pose": {
            "time": {"timestamp": np.arange(frame_count, dtype=np.float64) + 100.0},
            "position": {
                "position_enu": np.asarray(
                    [[float(frame_index), 0.0, 0.0] for frame_index in range(frame_count)],
                    dtype=np.float64,
                )
            },
            "orientation": {
                "quaternion_xyzw": np.zeros((frame_count, 4), dtype=np.float64),
                "heading": np.zeros(frame_count, dtype=np.float64),
            },
            "motion": {
                "linear_velocity_enu": np.tile(np.array([[4.0, 0.0, 0.0]]), (frame_count, 1)),
                "linear_acceleration_flu": np.tile(
                    np.array([[0.5, 0.25, 0.0]]),
                    (frame_count, 1),
                ),
                "angular_velocity_flu": np.tile(np.array([[0.0, 0.0, 0.1]]), (frame_count, 1)),
            },
        },
        "control_signal": {
            "time": {"timestamp": np.arange(frame_count, dtype=np.float64) + 200.0},
            "command": {
                "target_steering_wheel_angle": np.asarray(
                    [1.5 + frame_index for frame_index in range(frame_count)],
                    dtype=np.float32,
                ),
                "target_longitudinal_acceleration": np.asarray(
                    [0.2 + frame_index for frame_index in range(frame_count)],
                    dtype=np.float32,
                ),
                "target_longitudinal_torque": np.zeros(frame_count, dtype=np.float32),
            },
        },
    }


def test_reference_points_from_raw_frame_uses_valid_xy_pairs():
    points = reference_points_from_raw_frame(make_raw_data()["reference_traj"], 0)

    assert len(points) == 21
    assert points[2].x == pytest.approx(2.0)
    assert points[2].y == pytest.approx(0.0)
    assert points[2].v_ref == pytest.approx(6.0)
    assert points[2].a_ref == pytest.approx(0.5)


def test_vehicle_state_from_raw_pose_uses_position_heading_and_body_velocity():
    raw_data = make_raw_data()

    state = vehicle_state_from_raw_pose(raw_data["pose"], 0)

    assert state.pose.x == pytest.approx(0.0)
    assert state.pose.y == pytest.approx(0.0)
    assert state.pose.yaw == pytest.approx(0.0)
    assert state.vx == pytest.approx(4.0)
    assert state.vy == pytest.approx(0.0)
    assert state.ax == pytest.approx(0.5)
    assert state.ay == pytest.approx(0.25)
    assert state.r == pytest.approx(0.1)


def test_build_raw_frame_feature_calls_controller_feature_builder_order():
    frame = build_raw_frame_feature(
        make_raw_data(frame_count=1),
        0,
        clip_id="clip",
        lookahead_distances_m=(2.0, 5.0, 10.0, 15.0, 20.0),
    )

    assert frame.values.shape == (FEATURE_COUNT,)
    np.testing.assert_array_equal(frame.control_target, np.array([1.5, 0.2], dtype=np.float32))
    assert frame.target_valid
    assert frame.values[:10].tolist() == pytest.approx(
        [2.0, 0.0, 5.0, 0.0, 10.0, 0.0, 15.0, 0.0, 20.0, 0.0]
    )
    assert frame.values[10:].tolist() == pytest.approx(
        [0.0, 0.0, 2.0, 0.0, 0.5, 6.0, 0.0, 4.0, 0.5, 0.25, 0.1]
    )


def test_build_raw_frame_feature_samples_forward_from_vehicle_projection():
    raw_data = make_raw_data(frame_count=1)
    raw_data["pose"]["position"]["position_enu"][0] = np.asarray([0.5, 0.2, 0.0])
    raw_data["reference_traj"]["trajectory"]["points"]["a"][0, :8] = np.arange(
        8,
        dtype=np.float32,
    )

    frame = build_raw_frame_feature(
        raw_data,
        0,
        lookahead_distances_m=(2.0, 5.0, 10.0, 15.0, 20.0),
    )

    assert frame.values[:10].tolist() == pytest.approx(
        [2.0, -0.2, 5.0, -0.2, 10.0, -0.2, 15.0, -0.2, 19.5, -0.2]
    )
    assert frame.values[13] == pytest.approx(-0.5)
    assert frame.values[14] == pytest.approx(3.5)


def test_reference_absolute_times_add_frame_timestamp_and_relative_time():
    raw_data = make_raw_data(frame_count=1)
    raw_data["reference_traj"]["time"]["timestamp"][0] = 100.0
    raw_data["reference_traj"]["trajectory"]["points"]["relative_time"][0, :3] = np.asarray(
        [0.0, 0.2, 0.5],
        dtype=np.float32,
    )

    absolute_times = reference_absolute_times_from_raw_frame(raw_data["reference_traj"], 0)

    assert absolute_times[:3].tolist() == pytest.approx([100.0, 100.2, 100.5])


def test_longitudinal_reference_point_interpolates_at_pose_timestamp():
    raw_data = make_raw_data(frame_count=1)
    raw_data["reference_traj"]["time"]["timestamp"][0] = 100.0
    raw_data["reference_traj"]["trajectory"]["points"]["relative_time"][0, :4] = np.asarray(
        [0.0, 0.5, 1.0, 1.5],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["s"][0, :4] = np.asarray(
        [0.0, 5.0, 10.0, 15.0],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["v"][0, :4] = np.asarray(
        [1.0, 2.0, 9.0, 11.0],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["a"][0, :4] = np.asarray(
        [0.1, 0.2, 0.9, 1.7],
        dtype=np.float32,
    )

    point = longitudinal_reference_point_from_raw_frame(
        raw_data["reference_traj"],
        0,
        pose_timestamp_s=100.75,
    )

    assert point.s == pytest.approx(7.5)
    assert point.v_ref == pytest.approx(5.5)
    assert point.a_ref == pytest.approx(0.55)


def test_build_raw_frame_feature_uses_temporal_reference_for_longitudinal_values():
    raw_data = make_raw_data(frame_count=1)
    point_count = raw_data["reference_traj"]["trajectory"]["valid_length"][0]
    raw_data["reference_traj"]["time"]["timestamp"][0] = 100.0
    raw_data["pose"]["time"]["timestamp"][0] = 100.75
    raw_data["reference_traj"]["trajectory"]["points"]["relative_time"][0, :point_count] = 10.0
    raw_data["reference_traj"]["trajectory"]["points"]["relative_time"][0, :4] = np.asarray(
        [0.0, 0.5, 1.0, 1.5],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["s"][0, :4] = np.asarray(
        [0.0, 5.0, 10.0, 15.0],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["v"][0, :4] = np.asarray(
        [1.0, 2.0, 9.0, 11.0],
        dtype=np.float32,
    )
    raw_data["reference_traj"]["trajectory"]["points"]["a"][0, :4] = np.asarray(
        [0.1, 0.2, 0.9, 1.7],
        dtype=np.float32,
    )

    frame = build_raw_frame_feature(
        raw_data,
        0,
        lookahead_distances_m=(2.0, 5.0, 10.0, 15.0, 20.0),
    )

    assert frame.values[12] == pytest.approx(1.5)
    assert frame.values[13] == pytest.approx(7.5)
    assert frame.values[14] == pytest.approx(1.555)
    assert frame.values[15] == pytest.approx(5.5)
    assert frame.values[16] == pytest.approx(7.5)


def test_control_target_from_raw_frame_extracts_steering_and_acceleration():
    raw_data = make_raw_data(frame_count=2)

    target = control_target_from_raw_frame(raw_data, 1)

    np.testing.assert_array_equal(target, np.array([2.5, 1.2], dtype=np.float32))


def test_build_raw_feature_dataset_preserves_entry_and_frame_order():
    dataset = build_raw_feature_dataset(
        [
            {"clip_id": "clip-a", "raw_data": make_raw_data(frame_count=2)},
            {"clip_id": "clip-b", "raw_data": make_raw_data(frame_count=1)},
        ],
        lookahead_distances_m=(2.0, 5.0, 10.0, 15.0, 20.0),
        steering_scale_deg=10.0,
        acceleration_scale_mps2=2.0,
    )

    assert dataset.raw_features.shape == (3, FEATURE_COUNT)
    assert dataset.features.shape == (3, FEATURE_COUNT)
    np.testing.assert_array_equal(dataset.physical_targets[0], np.array([1.5, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(dataset.targets[0], np.array([0.15, 0.1], dtype=np.float32))
    assert dataset.target_valid_mask.tolist() == [True, True, True]
    assert dataset.clip_ids.tolist() == ["clip-a", "clip-a", "clip-b"]
    assert dataset.entry_indices.tolist() == [0, 0, 1]
    assert dataset.frame_indices.tolist() == [0, 1, 0]
