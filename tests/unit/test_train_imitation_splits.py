from __future__ import annotations

from pathlib import Path

import yaml

from scripts.train_imitation import resolve_split_dataset_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return path


def test_resolve_split_dataset_paths_finds_configured_split_dir(tmp_path) -> None:
    dataset_path = _touch(tmp_path / "processed" / "features.npz")
    split_dir = tmp_path / "splits"
    train_path = _touch(split_dir / "features_train.npz")
    validation_path = _touch(split_dir / "features_val.npz")
    test_path = _touch(split_dir / "features_test.npz")
    data_config_path = tmp_path / "dataset.yaml"
    data_config_path.write_text(
        yaml.safe_dump({"split_dir": str(split_dir)}),
        encoding="utf-8",
    )
    main_config = {"data": {"config": str(data_config_path)}}

    resolved = resolve_split_dataset_paths(dataset_path, main_config)

    assert resolved == {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }


def test_resolve_split_dataset_paths_falls_back_to_dataset_when_train_split_missing(
    tmp_path,
) -> None:
    dataset_path = _touch(tmp_path / "processed" / "features.npz")
    data_config_path = tmp_path / "dataset.yaml"
    data_config_path.write_text(
        yaml.safe_dump({"split_dir": str(tmp_path / "splits")}),
        encoding="utf-8",
    )
    main_config = {"data": {"config": str(data_config_path)}}

    resolved = resolve_split_dataset_paths(dataset_path, main_config)

    assert resolved == {"train": dataset_path, "validation": None, "test": None}
