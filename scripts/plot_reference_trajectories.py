#!/usr/bin/env python3
"""Plot reference trajectory curves from extracted task raw data."""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT

from vehicle_controller.plotting import load_pyplot


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def filename_prefix(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")


def selected_frame_indices(frame_count: int, maximum_frames: int) -> np.ndarray:
    if frame_count <= 0:
        return np.asarray([], dtype=np.int64)
    if maximum_frames <= 0 or frame_count <= maximum_frames:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(np.linspace(0, frame_count - 1, maximum_frames, dtype=np.int64))


def reference_points_for_frame(
    x_values: np.ndarray,
    y_values: np.ndarray,
    valid_lengths: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    if x_values.shape != y_values.shape:
        raise ValueError(f"x/y shape mismatch: {x_values.shape} != {y_values.shape}")
    if x_values.ndim != 2:
        raise ValueError(f"Expected x/y arrays with shape (frames, points), got {x_values.shape}")
    if len(valid_lengths) != x_values.shape[0]:
        raise ValueError(
            f"valid_length length {len(valid_lengths)} does not match frame count {x_values.shape[0]}"
        )
    point_count = int(valid_lengths[frame_index])
    point_count = max(0, min(point_count, x_values.shape[1]))
    return np.column_stack((x_values[frame_index, :point_count], y_values[frame_index, :point_count]))


def entry_reference_arrays(entry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_traj = entry["raw_data"]["reference_traj"]
    trajectory = reference_traj["trajectory"]
    points = trajectory["points"]
    x_values = np.asarray(points["x"], dtype=np.float64)
    y_values = np.asarray(points["y"], dtype=np.float64)
    valid_lengths = np.asarray(trajectory["valid_length"], dtype=np.int64)
    return x_values, y_values, valid_lengths


def plot_entry_reference_trajectories(
    entry: Mapping[str, Any],
    output_dir: str | Path,
    max_frames: int = 12,
    show_plots: bool = False,
) -> Path:
    x_values, y_values, valid_lengths = entry_reference_arrays(entry)
    frame_indices = selected_frame_indices(x_values.shape[0], max_frames)
    if len(frame_indices) == 0:
        raise ValueError(f"Entry has no frames: {entry.get('clip_id')}")

    plt = load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(1, 1, figsize=(8.8, 7.2))
    color_values = np.linspace(0.1, 0.95, len(frame_indices))
    colormap = plt.get_cmap("viridis")
    for color_value, frame_index in zip(color_values, frame_indices):
        points = reference_points_for_frame(x_values, y_values, valid_lengths, int(frame_index))
        if len(points) == 0:
            continue
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=colormap(color_value),
            linewidth=1.7,
            alpha=0.78,
            label=f"frame {int(frame_index)}",
        )
        axis.scatter(points[0, 0], points[0, 1], s=18, color=colormap(color_value), alpha=0.85)

    axis.set_title(str(entry.get("clip_id", "reference_trajectory")))
    axis.set_xlabel("Reference x [m]")
    axis.set_ylabel("Reference y [m]")
    axis.grid(True, alpha=0.28)
    axis.set_aspect("equal", adjustable="datalim")
    if len(frame_indices) <= 12:
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()

    clip_id = filename_prefix(str(entry.get("clip_id", "reference_trajectory")))
    plot_path = output_path / f"{clip_id}_reference_trajectory.png"
    figure.savefig(plot_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return plot_path


def plot_reference_trajectory_overview(
    entries: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    max_entries: int = 19,
    show_plots: bool = False,
) -> Path:
    if not entries:
        raise ValueError("Cannot plot an empty entry list")

    selected_indices = selected_frame_indices(len(entries), max_entries)
    plt = load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(1, 1, figsize=(10.5, 8.0))
    colormap = plt.get_cmap("tab20")
    for order, entry_index in enumerate(selected_indices):
        entry = entries[int(entry_index)]
        x_values, y_values, valid_lengths = entry_reference_arrays(entry)
        middle_frame = x_values.shape[0] // 2
        points = reference_points_for_frame(x_values, y_values, valid_lengths, middle_frame)
        if len(points) == 0:
            continue
        color = colormap(order % 20)
        axis.plot(points[:, 0], points[:, 1], linewidth=1.8, alpha=0.82, color=color)
        axis.scatter(points[0, 0], points[0, 1], s=18, color=color, alpha=0.9)

    axis.set_title("Reference Trajectory Overview")
    axis.set_xlabel("Reference x [m]")
    axis.set_ylabel("Reference y [m]")
    axis.grid(True, alpha=0.28)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()

    plot_path = output_path / "reference_trajectory_overview.png"
    figure.savefig(plot_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return plot_path


def plot_reference_trajectories(
    raw_data_path: str | Path,
    output_dir: str | Path,
    max_frames_per_entry: int = 12,
    max_overview_entries: int = 19,
    show_plots: bool = False,
) -> tuple[Path, ...]:
    input_path = project_path(raw_data_path)
    with input_path.open("rb") as file:
        dataset = pickle.load(file)
    entries = dataset.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError(f"Raw data pickle has no entries sequence: {input_path}")

    output_path = project_path(output_dir)
    plot_paths = [
        plot_reference_trajectory_overview(
            entries,
            output_path,
            max_entries=max_overview_entries,
            show_plots=show_plots,
        )
    ]
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("All raw data entries must be mappings")
        plot_paths.append(
            plot_entry_reference_trajectories(
                entry,
                output_path,
                max_frames=max_frames_per_entry,
                show_plots=show_plots,
            )
        )
    return tuple(plot_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        default="data/interim/clean_ad_policy_sim_v1_aba9e399_raw_data.pkl",
        help="Pickle generated by scripts/extract_task_raw_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/reference_trajectories",
        help="Directory for output trajectory plots.",
    )
    parser.add_argument("--max-frames-per-entry", type=int, default=12)
    parser.add_argument("--max-overview-entries", type=int, default=19)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    plot_paths = plot_reference_trajectories(
        args.raw_data,
        args.output_dir,
        max_frames_per_entry=args.max_frames_per_entry,
        max_overview_entries=args.max_overview_entries,
        show_plots=args.show_plots,
    )
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
