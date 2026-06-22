#!/usr/bin/env python3
"""Extract raw signal slices for task manifest entries."""

from __future__ import annotations

import argparse
import copy
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT


SIGNAL_GROUPS = ("reference_traj", "pose", "control_signal")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def index_record_pickles(record_root: Path) -> dict[str, Path]:
    record_paths: dict[str, Path] = {}
    for path in record_root.rglob("*.pkl"):
        record_paths[path.stem] = path
    return record_paths


def slice_signal_tree(value: Any, start: int, stop: int) -> Any:
    """Slice all frame-aligned leaves in a nested signal tree."""
    if isinstance(value, Mapping):
        return {key: slice_signal_tree(item, start, stop) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.shape and value.shape[0] < stop:
            raise ValueError(f"Signal length {value.shape[0]} is shorter than requested stop {stop}")
        return value[start:stop].copy()
    if isinstance(value, list):
        if len(value) < stop:
            raise ValueError(f"Signal length {len(value)} is shorter than requested stop {stop}")
        return copy.deepcopy(value[start:stop])
    if isinstance(value, tuple):
        if len(value) < stop:
            raise ValueError(f"Signal length {len(value)} is shorter than requested stop {stop}")
        return tuple(copy.deepcopy(value[start:stop]))
    return copy.deepcopy(value)


def concat_signal_trees(parts: Sequence[Any]) -> Any:
    if not parts:
        raise ValueError("Cannot concatenate an empty signal part list")

    first = parts[0]
    if isinstance(first, Mapping):
        keys = first.keys()
        for part in parts[1:]:
            if not isinstance(part, Mapping) or part.keys() != keys:
                raise ValueError("Cannot concatenate signal trees with different mapping keys")
        return {key: concat_signal_trees([part[key] for part in parts]) for key in keys}

    if isinstance(first, np.ndarray):
        return np.concatenate(parts, axis=0)
    if isinstance(first, list):
        merged: list[Any] = []
        for part in parts:
            merged.extend(copy.deepcopy(part))
        return merged
    if isinstance(first, tuple):
        merged_tuple: tuple[Any, ...] = ()
        for part in parts:
            merged_tuple += copy.deepcopy(part)
        return merged_tuple

    if len(parts) == 1:
        return copy.deepcopy(first)
    raise ValueError(f"Cannot concatenate scalar signal values of type {type(first).__name__}")


def resolve_record_path(record_paths: Mapping[str, Path], record_pkl_id: str) -> Path:
    path = record_paths.get(record_pkl_id)
    if path is None:
        candidates = ", ".join(sorted(record_paths)[:10])
        raise FileNotFoundError(
            f"record_pkl_id not found under record root: {record_pkl_id}. "
            f"Known examples: {candidates}"
        )
    return path


def load_record(record_cache: dict[str, Mapping[str, Any]], record_path: Path) -> Mapping[str, Any]:
    cache_key = str(record_path)
    if cache_key not in record_cache:
        value = load_pickle(record_path)
        if not isinstance(value, Mapping):
            raise TypeError(f"Record pickle must contain a mapping: {record_path}")
        record_cache[cache_key] = value
    return record_cache[cache_key]


def extract_part_raw_data(
    part: Mapping[str, Any],
    record_paths: Mapping[str, Path],
    record_cache: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record_pkl_id = str(part["record_pkl_id"])
    start = int(part["start_index_in_bag"])
    end = int(part["end_index_in_bag"])
    if end < start:
        raise ValueError(f"Invalid part range for {part.get('clip_part_id')}: {start}..{end}")

    record_path = resolve_record_path(record_paths, record_pkl_id)
    target_bag_pkl = load_record(record_cache, record_path)
    frames = target_bag_pkl.get("frames")
    if not isinstance(frames, Mapping):
        raise KeyError(f"Record pickle has no frames mapping: {record_path}")

    stop = end + 1
    raw_data = {}
    for group_name in SIGNAL_GROUPS:
        if group_name not in frames:
            raise KeyError(f"Record pickle missing frames.{group_name}: {record_path}")
        raw_data[group_name] = slice_signal_tree(frames[group_name], start, stop)

    raw_data["source"] = {
        "record_pkl_id": record_pkl_id,
        "record_pkl_path": str(record_path),
        "start_index_in_bag": start,
        "end_index_in_bag": end,
        "frame_count": stop - start,
    }
    return raw_data


def extract_entry_raw_data(
    entry: Mapping[str, Any],
    record_paths: Mapping[str, Path],
    record_cache: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parts = entry.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise TypeError(f"Entry has no parts sequence: {entry.get('clip_id')}")

    part_raw_data = [
        extract_part_raw_data(part, record_paths, record_cache)
        for part in parts
        if isinstance(part, Mapping)
    ]
    if len(part_raw_data) != len(parts):
        raise TypeError(f"All parts must be mappings: {entry.get('clip_id')}")

    raw_data = {
        group_name: concat_signal_trees([part[group_name] for part in part_raw_data])
        for group_name in SIGNAL_GROUPS
    }
    raw_data["parts"] = [part["source"] for part in part_raw_data]
    raw_data["frame_count"] = sum(part["source"]["frame_count"] for part in part_raw_data)
    return raw_data


def extract_task_raw_data(
    task_manifest_path: str | Path,
    record_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    task_path = project_path(task_manifest_path)
    record_root_path = project_path(record_root)
    output = (
        project_path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "data" / "interim" / f"{task_path.stem}_raw_data.pkl"
    )

    task_manifest = load_pickle(task_path)
    if not isinstance(task_manifest, Mapping):
        raise TypeError(f"Task manifest must contain a mapping: {task_path}")
    entries = task_manifest.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError(f"Task manifest has no entries sequence: {task_path}")

    record_paths = index_record_pickles(record_root_path)
    if not record_paths:
        raise FileNotFoundError(f"No record pickle files found under: {record_root_path}")

    record_cache: dict[str, Mapping[str, Any]] = {}
    extracted_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("All task manifest entries must be mappings")
        extracted_entry = copy.deepcopy(dict(entry))
        extracted_entry["raw_data"] = extract_entry_raw_data(
            entry,
            record_paths,
            record_cache,
        )
        expected_count = int(entry.get("frame_count", extracted_entry["raw_data"]["frame_count"]))
        actual_count = int(extracted_entry["raw_data"]["frame_count"])
        if actual_count != expected_count:
            raise ValueError(
                f"Extracted frame count mismatch for {entry.get('clip_id')}: "
                f"expected {expected_count}, got {actual_count}"
            )
        extracted_entries.append(extracted_entry)

    dataset = {
        "schema": {"version": "ai_control_task_raw_data_v1.0"},
        "source": {
            "task_manifest_path": str(task_path),
            "record_pkl_root": str(record_root_path),
        },
        "task_info": copy.deepcopy(task_manifest.get("task_info", {})),
        "selection_rule": copy.deepcopy(task_manifest.get("selection_rule", {})),
        "split_rule": copy.deepcopy(task_manifest.get("split_rule", {})),
        "entries": extracted_entries,
        "summary": {
            "entry_count": len(extracted_entries),
            "part_count": sum(len(entry.get("parts", [])) for entry in extracted_entries),
            "frame_count": sum(entry["raw_data"]["frame_count"] for entry in extracted_entries),
            "record_pkl_count": len(record_cache),
        },
    }
    save_pickle(output, dataset)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-manifest",
        default="data/raw/ai_control_dataset/task_manifest/clean_ad_policy_sim_v1_aba9e399.pkl",
        help="Task manifest pickle containing entries.",
    )
    parser.add_argument(
        "--record-root",
        default="data/raw/ai_control_dataset/record_pkl",
        help="Directory containing record_pkl files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output pickle path. Defaults to data/interim/<task_manifest>_raw_data.pkl.",
    )
    args = parser.parse_args()

    output = extract_task_raw_data(args.task_manifest, args.record_root, args.output)
    with output.open("rb") as file:
        dataset = pickle.load(file)
    summary = dataset["summary"]
    print(f"output={output}")
    print(f"entries={summary['entry_count']}")
    print(f"parts={summary['part_count']}")
    print(f"frames={summary['frame_count']}")
    print(f"record_pkls={summary['record_pkl_count']}")


if __name__ == "__main__":
    main()
