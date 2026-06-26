from __future__ import annotations

import pickle

import numpy as np

from scripts.extract_task_raw_data import extract_task_raw_data


def dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def make_record(frame_count: int) -> dict:
    index = np.arange(frame_count)
    return {
        "frames": {
            "pose": {
                "time": {"timestamp": index.astype(np.float64)},
                "position": {"position_enu": np.stack([index, index + 1, index + 2], axis=1)},
                "orientation": {
                    "quaternion_xyzw": np.stack([index, index + 1, index + 2, index + 3], axis=1),
                    "heading": index.astype(np.float64) * 0.1,
                },
                "motion": {
                    "linear_velocity_enu": np.stack([index, index, index], axis=1),
                    "linear_acceleration_flu": np.stack([index + 1, index + 1, index + 1], axis=1),
                    "angular_velocity_flu": np.stack([index + 2, index + 2, index + 2], axis=1),
                },
            },
            "reference_traj": {
                "time": {"timestamp": index.astype(np.float64) + 100.0},
                "status": {"is_replan": np.zeros(frame_count, dtype=np.int16)},
                "trajectory": {
                    "points": {
                        name: np.stack([index, index + 10], axis=1)
                        for name in ("x", "y", "theta", "kappa", "s", "v", "a", "relative_time", "da")
                    },
                    "valid_length": np.full(frame_count, 2, dtype=np.int16),
                },
            },
            "control_signal": {
                "time": {"timestamp": index.astype(np.float64) + 200.0},
                "command": {
                    "target_steering_wheel_angle": index.astype(np.float32) * 0.1,
                    "target_longitudinal_acceleration": index.astype(np.float32) * 0.2,
                    "target_longitudinal_torque": index.astype(np.float32) * 0.3,
                },
                "longitudinal_status": {
                    "standstill_request": (index % 2 == 0),
                },
            },
        }
    }


def test_extract_task_raw_data_slices_and_concatenates_parts(tmp_path):
    task_manifest_path = tmp_path / "task.pkl"
    record_root = tmp_path / "record_pkl"
    output_path = tmp_path / "raw_data.pkl"

    dump_pickle(record_root / "vehicle" / "day" / "record-a.pkl", make_record(8))
    dump_pickle(record_root / "vehicle" / "day" / "record-b.pkl", make_record(10))
    dump_pickle(
        task_manifest_path,
        {
            "task_info": {"identity": {"task_manifest_id": "task"}},
            "entries": [
                {
                    "clip_id": "clip-0",
                    "frame_count": 5,
                    "parts": [
                        {
                            "clip_part_id": "part-0",
                            "record_pkl_id": "record-a",
                            "start_index_in_bag": 2,
                            "end_index_in_bag": 4,
                        },
                        {
                            "clip_part_id": "part-1",
                            "record_pkl_id": "record-b",
                            "start_index_in_bag": 7,
                            "end_index_in_bag": 8,
                        },
                    ],
                }
            ],
        },
    )

    result = extract_task_raw_data(task_manifest_path, record_root, output_path)

    assert result == output_path
    with output_path.open("rb") as file:
        dataset = pickle.load(file)

    assert dataset["schema"]["version"] == "ai_control_task_raw_data_v1.0"
    assert dataset["summary"] == {
        "entry_count": 1,
        "part_count": 2,
        "frame_count": 5,
        "record_pkl_count": 2,
    }
    entry = dataset["entries"][0]
    raw_data = entry["raw_data"]
    np.testing.assert_array_equal(
        raw_data["pose"]["time"]["timestamp"],
        np.array([2.0, 3.0, 4.0, 7.0, 8.0]),
    )
    np.testing.assert_array_equal(
        raw_data["reference_traj"]["trajectory"]["points"]["x"][:, 0],
        np.array([2, 3, 4, 7, 8]),
    )
    np.testing.assert_array_equal(
        raw_data["control_signal"]["command"]["target_longitudinal_acceleration"],
        np.array([0.4, 0.6, 0.8, 1.4, 1.6], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        raw_data["control_signal"]["longitudinal_status"]["standstill_request"],
        np.array([True, False, True, False, True]),
    )
    assert raw_data["parts"][0]["frame_count"] == 3
    assert raw_data["parts"][1]["frame_count"] == 2
