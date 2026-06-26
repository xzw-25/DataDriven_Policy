#!/usr/bin/env python3
"""Build per-frame controller features from extracted raw reference and pose data."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT

from vehicle_controller.data.feature_builder import (
    STANDSTILL_REQUEST_NPZ_KEY,
    STANDSTILL_REQUEST_SIGNAL_NAME,
    build_raw_feature_dataset,
)
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _float_tuple(values: Sequence[str] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)


def config_value(config: Mapping[str, object], section: str, key: str) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, Mapping) or key not in section_config:
        raise ValueError(f"Missing '{section}.{key}' in the main configuration")
    return project_path(str(section_config[key]))


def _ratio(config: Mapping[str, object], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _normalized_split_ratios(
    data_config: Mapping[str, object],
) -> tuple[float, float, float]:
    train_ratio = _ratio(data_config, "train_ratio", 0.7)
    validation_ratio = _ratio(data_config, "validation_ratio", 0.15)
    test_ratio = _ratio(data_config, "test_ratio", 0.15)
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive")
    return train_ratio / total, validation_ratio / total, test_ratio / total


def _split_chunk_frame_count(data_config: Mapping[str, object]) -> int:
    if "split_chunk_frame_count" in data_config:
        value = int(data_config["split_chunk_frame_count"])
    else:
        chunk_duration_s = float(data_config.get("split_chunk_duration_s", 5.0))
        if chunk_duration_s <= 0.0:
            raise ValueError("split_chunk_duration_s must be positive")
        sample_period_s = float(data_config.get("sample_period_s", 0.01))
        if sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        value = int(round(chunk_duration_s / sample_period_s))
    if value <= 0:
        raise ValueError("split_chunk_frame_count must be positive")
    return value


def _contiguous_ranges(values: np.ndarray | None, count: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise ValueError("Cannot split an empty dataset")
    if values is None:
        return [(0, count)]
    array = np.asarray(values).astype(str)
    if array.shape != (count,):
        raise ValueError("scenario_ids/clip_ids must have shape [N]")
    starts = [0]
    for index in range(1, count):
        if array[index] != array[index - 1]:
            starts.append(index)
    starts.append(count)
    return [(starts[index], starts[index + 1]) for index in range(len(starts) - 1)]


def _chunk_indices(
    count: int,
    *,
    chunk_frame_count: int,
    group_values: np.ndarray | None,
) -> list[np.ndarray]:
    if chunk_frame_count <= 0:
        raise ValueError("chunk_frame_count must be positive")
    chunks: list[np.ndarray] = []
    for start, stop in _contiguous_ranges(group_values, count):
        for chunk_start in range(start, stop, chunk_frame_count):
            chunk_stop = min(chunk_start + chunk_frame_count, stop)
            chunks.append(np.arange(chunk_start, chunk_stop, dtype=np.int64))
    if not chunks:
        raise ValueError("Cannot split an empty dataset")
    return chunks


def _split_indices_by_chunks(
    count: int,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    *,
    chunk_frame_count: int,
    group_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    chunks = _chunk_indices(
        count,
        chunk_frame_count=chunk_frame_count,
        group_values=group_values,
    )
    chunk_indices = np.arange(len(chunks), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(chunk_indices)
    train_end = int(len(chunk_indices) * train_ratio)
    validation_end = train_end + int(len(chunk_indices) * validation_ratio)
    split_chunk_indices = {
        "train": chunk_indices[:train_end],
        "val": chunk_indices[train_end:validation_end],
        "test": chunk_indices[validation_end:],
    }
    empty = np.asarray([], dtype=np.int64)
    return {
        name: (
            np.concatenate([chunks[index] for index in indices]).astype(np.int64)
            if len(indices)
            else empty.copy()
        )
        for name, indices in split_chunk_indices.items()
    }


def _split_indices_from_npz(
    npz: np.lib.npyio.NpzFile,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    *,
    chunk_frame_count: int,
) -> dict[str, np.ndarray]:
    count = int(npz["features"].shape[0])
    group_values = None
    if "scenario_ids" in npz:
        group_values = np.asarray(npz["scenario_ids"])
    elif "clip_ids" in npz:
        group_values = np.asarray(npz["clip_ids"])
    return _split_indices_by_chunks(
        count,
        train_ratio,
        validation_ratio,
        seed,
        chunk_frame_count=chunk_frame_count,
        group_values=group_values,
    )


def _metadata_with_split(
    metadata_value: np.ndarray,
    *,
    split_name: str,
    split_indices: np.ndarray,
    split_counts: Mapping[str, int],
    split_seed: int,
    chunk_frame_count: int,
) -> np.ndarray:
    metadata = dict(json.loads(str(metadata_value)))
    metadata["split"] = {
        "name": split_name,
        "sample_count": int(len(split_indices)),
        "indices": [int(value) for value in split_indices],
        "all_counts": {name: int(count) for name, count in split_counts.items()},
        "seed": int(split_seed),
        "strategy": "shuffled_contiguous_frame_chunks",
        "chunk_frame_count": int(chunk_frame_count),
    }
    return np.asarray(json.dumps(metadata, sort_keys=True))


def _split_npz_payload(
    npz: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    *,
    split_name: str,
    split_counts: Mapping[str, int],
    split_seed: int,
    chunk_frame_count: int,
) -> dict[str, np.ndarray]:
    frame_count = int(npz["features"].shape[0])
    payload: dict[str, np.ndarray] = {}
    for key in npz.files:
        value = npz[key]
        if key == "metadata_json":
            payload[key] = _metadata_with_split(
                value,
                split_name=split_name,
                split_indices=indices,
                split_counts=split_counts,
                split_seed=split_seed,
                chunk_frame_count=chunk_frame_count,
            )
        elif value.shape and value.shape[0] == frame_count:
            payload[key] = value[indices].copy()
        else:
            payload[key] = value.copy()
    return payload


def write_train_val_test_splits(
    dataset_path: str | Path,
    split_dir: str | Path,
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    chunk_frame_count: int = 500,
) -> dict[str, Path]:
    dataset = project_path(dataset_path)
    output_dir = project_path(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataset, allow_pickle=False) as npz:
        split_indices = _split_indices_from_npz(
            npz,
            train_ratio,
            validation_ratio,
            seed,
            chunk_frame_count=chunk_frame_count,
        )
        split_counts = {name: int(len(indices)) for name, indices in split_indices.items()}
        split_paths = {
            name: output_dir / f"{dataset.stem}_{name}.npz"
            for name in ("train", "val", "test")
        }
        for name, path in split_paths.items():
            payload = _split_npz_payload(
                npz,
                split_indices[name],
                split_name=name,
                split_counts=split_counts,
                split_seed=seed,
                chunk_frame_count=chunk_frame_count,
            )
            np.savez_compressed(path, **payload)
    return split_paths


def build_features_from_raw_data(
    raw_data_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = "configs/default.yaml",
    model_config_path: str | Path | None = None,
    normalization_config_path: str | Path | None = None,
    normalize_features: bool = True,
    preview_times_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    lookahead_distances_m: Sequence[float] | None = None,
    curvature_weights: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> Path:
    input_path = project_path(raw_data_path)
    output = project_path(output_path)
    main_config = load_yaml(project_path(config_path))
    data_config = load_yaml(config_value(main_config, "data", "config"))
    resolved_model_config_path = (
        project_path(model_config_path)
        if model_config_path is not None
        else config_value(main_config, "model", "config")
    )
    model_config: dict[str, Any] = dict(load_yaml(resolved_model_config_path))
    normalizer = None
    resolved_normalization_config_path = None
    if normalize_features:
        resolved_normalization_config_path = (
            project_path(normalization_config_path)
            if normalization_config_path is not None
            else config_value(main_config, "data", "normalization")
        )
        normalizer = FeatureNormalizer.from_yaml(str(resolved_normalization_config_path))

    with input_path.open("rb") as file:
        raw_dataset = pickle.load(file)
    entries = raw_dataset.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError(f"Raw data pickle has no entries sequence: {input_path}")

    dataset = build_raw_feature_dataset(
        entries,
        preview_times_s=preview_times_s,
        lookahead_distances_m=lookahead_distances_m,
        curvature_weights=curvature_weights,
        normalizer=normalizer,
        steering_scale_deg=steering_limit_deg_from_config(model_config),
        acceleration_scale_mps2=float(model_config["accel_limit_mps2"]),
    )
    split_chunk_duration_s = float(data_config.get("split_chunk_duration_s", 5.0))
    split_chunk_frame_count = _split_chunk_frame_count(data_config)
    metadata = {
        "source_raw_data_path": str(input_path),
        "source_summary": raw_dataset.get("summary", {}),
        "feature_builder": "vehicle_controller.data.feature_builder",
        "features_are_normalized": bool(normalize_features),
        "normalization_config_path": None
        if resolved_normalization_config_path is None
        else str(resolved_normalization_config_path),
        "model_config_path": str(resolved_model_config_path),
        "target_normalization": {
            "steering_scale_deg": steering_limit_deg_from_config(model_config),
            "accel_scale_mps2": float(model_config["accel_limit_mps2"]),
            "clip": 1.0,
        },
        "preview_times_s": list(preview_times_s),
        "lookahead_distances_m": None
        if lookahead_distances_m is None
        else list(lookahead_distances_m),
        "curvature_weights": list(curvature_weights),
        "standstill_request_signal_name": STANDSTILL_REQUEST_SIGNAL_NAME,
        "split_strategy": {
            "name": "shuffled_contiguous_frame_chunks",
            "chunk_duration_s": split_chunk_duration_s,
            "chunk_frame_count": int(split_chunk_frame_count),
        },
        "entry_count": int(len(entries)),
        "frame_count": int(dataset.raw_features.shape[0]),
    }
    saved_path = dataset.save_npz(output, metadata)
    train_ratio, validation_ratio, _ = _normalized_split_ratios(data_config)
    write_train_val_test_splits(
        saved_path,
        str(data_config.get("split_dir", saved_path.parent)),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=int(main_config.get("seed", 42)),
        chunk_frame_count=split_chunk_frame_count,
    )
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        default="data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl",
        help="Pickle generated by scripts/extract_task_raw_data.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/clean_ad_policy_sim_v1_aba9e399_features.npz",
        help="Output NPZ path.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--normalization-config")
    parser.add_argument(
        "--no-feature-normalization",
        action="store_true",
        help="Keep features equal to raw_features instead of applying FeatureNormalizer.",
    )
    parser.add_argument(
        "--preview-times-s",
        nargs=5,
        default=("0.1", "0.2", "0.3", "0.4", "0.5"),
        help="Five preview time horizons used when lookahead distances are not fixed.",
    )
    parser.add_argument(
        "--lookahead-distances-m",
        nargs=5,
        help="Optional five fixed lookahead distances. Overrides preview time conversion.",
    )
    parser.add_argument(
        "--curvature-weights",
        nargs=5,
        default=("1.0", "0.8", "0.6", "0.4", "0.2"),
        help="Five weights used to combine sampled trajectory curvature.",
    )
    args = parser.parse_args()

    output = build_features_from_raw_data(
        args.raw_data,
        args.output,
        config_path=args.config,
        model_config_path=args.model_config,
        normalization_config_path=args.normalization_config,
        normalize_features=not args.no_feature_normalization,
        preview_times_s=tuple(float(value) for value in args.preview_times_s),
        lookahead_distances_m=_float_tuple(args.lookahead_distances_m),
        curvature_weights=tuple(float(value) for value in args.curvature_weights),
    )
    data = dict()

    with np.load(output, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"]))
        data["features_shape"] = tuple(npz["features"].shape)
        data["raw_features_shape"] = tuple(npz["raw_features"].shape)
        data["targets_shape"] = tuple(npz["targets"].shape)
        data["valid_targets"] = int(npz["target_valid_mask"].sum())
        data["physical_targets_shape"] = tuple(npz["physical_targets"].shape)
        data["standstill_requests_shape"] = tuple(npz[STANDSTILL_REQUEST_NPZ_KEY].shape)
        data["frame_count"] = metadata["frame_count"]
        data["features_are_normalized"] = metadata["features_are_normalized"]
    print(f"output={output}")
    print(f"features_shape={data['features_shape']}")
    print(f"raw_features_shape={data['raw_features_shape']}")
    print(f"targets_shape={data['targets_shape']}")
    print(f"physical_targets_shape={data['physical_targets_shape']}")
    print(f"standstill_requests_shape={data['standstill_requests_shape']}")
    print(f"valid_targets={data['valid_targets']}")
    print(f"features_are_normalized={data['features_are_normalized']}")
    print(f"frames={data['frame_count']}")
    main_config = load_yaml(project_path(args.config))
    data_config = load_yaml(config_value(main_config, "data", "config"))
    split_dir = project_path(str(data_config.get("split_dir", output.parent)))
    for split_name in ("train", "val", "test"):
        split_path = split_dir / f"{output.stem}_{split_name}.npz"
        with np.load(split_path, allow_pickle=False) as npz:
            print(f"{split_name}_output={split_path}")
            print(f"{split_name}_features_shape={tuple(npz['features'].shape)}")


if __name__ == "__main__":
    main()
