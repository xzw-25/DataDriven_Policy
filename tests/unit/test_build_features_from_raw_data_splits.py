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
            metadata = json.loads(str(data["metadata_json"]))
            assert metadata["split"]["name"] == name
            assert metadata["split"]["sample_count"] == data["features"].shape[0]
            split_scenarios[name] = set(str(value) for value in data["scenario_ids"])
            split_counts[name] = int(data["features"].shape[0])

    assert split_counts == {"train": 4, "val": 2, "test": 2}
    assert split_scenarios["train"].isdisjoint(split_scenarios["val"])
    assert split_scenarios["train"].isdisjoint(split_scenarios["test"])
    assert split_scenarios["val"].isdisjoint(split_scenarios["test"])
