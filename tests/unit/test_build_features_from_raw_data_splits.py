from __future__ import annotations

import json

import numpy as np

from scripts.build_features_from_raw_data import write_train_val_test_splits
from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES


def test_write_train_val_test_splits_keeps_scenarios_disjoint(tmp_path) -> None:
    dataset_path = tmp_path / "features.npz"
    split_dir = tmp_path / "splits"
    scenario_ids = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    np.savez_compressed(
        dataset_path,
        features=np.arange(8 * FEATURE_COUNT, dtype=np.float32).reshape(8, FEATURE_COUNT),
        raw_features=np.arange(8 * FEATURE_COUNT, dtype=np.float32).reshape(8, FEATURE_COUNT),
        targets=np.arange(16, dtype=np.float32).reshape(8, 2),
        physical_targets=np.arange(16, dtype=np.float32).reshape(8, 2),
        target_valid_mask=np.ones(8, dtype=bool),
        scenario_ids=scenario_ids,
        clip_ids=scenario_ids,
        frame_indices=np.arange(8, dtype=np.int32),
        positions_enu=np.arange(24, dtype=np.float64).reshape(8, 3),
        headings_rad=np.linspace(0.0, 0.7, 8, dtype=np.float64),
        pose_position_x_enu_m=np.linspace(10.0, 17.0, 8, dtype=np.float64),
        pose_position_y_enu_m=np.linspace(20.0, 27.0, 8, dtype=np.float64),
        pose_heading_rad=np.linspace(0.0, 0.7, 8, dtype=np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(["steering", "accel"]),
        metadata_json=np.asarray(json.dumps({"frame_count": 8})),
    )

    split_paths = write_train_val_test_splits(
        dataset_path,
        split_dir,
        train_ratio=0.5,
        validation_ratio=0.25,
        seed=7,
        chunk_frame_count=2,
    )

    assert set(split_paths) == {"train", "val", "test"}
    split_scenarios: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for name, path in split_paths.items():
        assert path.is_file()
        with np.load(path, allow_pickle=False) as data:
            assert data["features"].shape[1] == FEATURE_COUNT
            assert data["targets"].shape == (data["features"].shape[0], 2)
            assert data["feature_names"].shape == (FEATURE_COUNT,)
            assert data["positions_enu"].shape == (data["features"].shape[0], 3)
            assert data["headings_rad"].shape == (data["features"].shape[0],)
            assert data["pose_position_x_enu_m"].shape == (data["features"].shape[0],)
            assert data["pose_position_y_enu_m"].shape == (data["features"].shape[0],)
            assert data["pose_heading_rad"].shape == (data["features"].shape[0],)
            metadata = json.loads(str(data["metadata_json"]))
            assert metadata["split"]["name"] == name
            assert metadata["split"]["sample_count"] == data["features"].shape[0]
            assert metadata["split"]["strategy"] == "shuffled_contiguous_frame_chunks"
            assert metadata["split"]["chunk_frame_count"] == 2
            split_scenarios[name] = set(str(value) for value in data["scenario_ids"])
            split_counts[name] = int(data["features"].shape[0])

    assert split_counts == {"train": 4, "val": 2, "test": 2}
    assert split_scenarios["train"].isdisjoint(split_scenarios["val"])
    assert split_scenarios["train"].isdisjoint(split_scenarios["test"])
    assert split_scenarios["val"].isdisjoint(split_scenarios["test"])


def test_write_train_val_test_splits_keeps_frame_chunks_together(tmp_path) -> None:
    dataset_path = tmp_path / "features.npz"
    split_dir = tmp_path / "splits"
    frame_count = 10
    np.savez_compressed(
        dataset_path,
        features=np.arange(frame_count * FEATURE_COUNT, dtype=np.float32).reshape(
            frame_count,
            FEATURE_COUNT,
        ),
        raw_features=np.arange(frame_count * FEATURE_COUNT, dtype=np.float32).reshape(
            frame_count,
            FEATURE_COUNT,
        ),
        targets=np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2),
        physical_targets=np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2),
        target_valid_mask=np.ones(frame_count, dtype=bool),
        scenario_ids=np.asarray(["same-scenario"] * frame_count),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(["steering", "accel"]),
        metadata_json=np.asarray(json.dumps({"frame_count": frame_count})),
    )

    split_paths = write_train_val_test_splits(
        dataset_path,
        split_dir,
        train_ratio=0.6,
        validation_ratio=0.2,
        seed=11,
        chunk_frame_count=2,
    )

    frame_to_split: dict[int, str] = {}
    for split_name, path in split_paths.items():
        with np.load(path, allow_pickle=False) as data:
            for frame_index in data["frame_indices"]:
                frame_to_split[int(frame_index)] = split_name

    assert set(frame_to_split) == set(range(frame_count))
    for chunk_start in range(0, frame_count, 2):
        assert frame_to_split[chunk_start] == frame_to_split[chunk_start + 1]
    assert {frame_to_split[index] for index in range(frame_count)} == {"train", "val", "test"}
