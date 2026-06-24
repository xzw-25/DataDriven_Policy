"""Offline validation plotting utilities."""

from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Sequence

import numpy as np

from vehicle_controller.plotting import load_pyplot


EXPERT_STEERING_LABEL = "Generated-data expert steering"
NEURAL_STEERING_LABEL = "Neural controller steering"
EXPERT_ACCELERATION_LABEL = "Generated-data expert signed acceleration"
NEURAL_ACCELERATION_LABEL = "Neural controller signed acceleration"


def _as_control_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _safe_file_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "validation"


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _downsample_indices(length: int, maximum_samples: int) -> np.ndarray:
    if maximum_samples <= 0:
        raise ValueError("maximum_samples_per_plot must be positive")
    if length <= maximum_samples:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum_samples).astype(np.int64))


def _plot_control_comparison(
    predicted_controls: np.ndarray,
    expert_controls: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
    show_plots: bool,
) -> Path:
    plt = load_pyplot(show_plots)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x_values, expert_controls[:, 0], label=EXPERT_STEERING_LABEL)
    axes[0].plot(
        x_values,
        predicted_controls[:, 0],
        "--",
        label=NEURAL_STEERING_LABEL,
        alpha=0.9,
    )
    axes[0].set_ylabel("Steering wheel angle [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(x_values, expert_controls[:, 1], label=EXPERT_ACCELERATION_LABEL)
    axes[1].plot(
        x_values,
        predicted_controls[:, 1],
        "--",
        label=NEURAL_ACCELERATION_LABEL,
        alpha=0.9,
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Signed acceleration [m/s2]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return output_path


def save_offline_control_comparison_plots(
    predicted_controls: np.ndarray,
    expert_controls: np.ndarray,
    output_dir: str | Path,
    timestamps_s: np.ndarray | None = None,
    scenario_ids: np.ndarray | None = None,
    dataset_label: str = "validation",
    max_scenarios: int | None = 8,
    maximum_samples_per_plot: int = 2000,
    show_plots: bool = False,
) -> tuple[Path, ...]:
    """Plot validation-set expert controls against current neural model predictions."""
    predicted = _as_control_array("predicted_controls", predicted_controls)
    expert = _as_control_array("expert_controls", expert_controls)
    if predicted.shape != expert.shape:
        raise ValueError("predicted_controls and expert_controls must have matching shapes")

    output_path = Path(output_dir)
    if timestamps_s is not None:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        if timestamps.shape != (len(predicted),):
            raise ValueError("timestamps_s must have shape [N]")
    else:
        timestamps = None

    if scenario_ids is not None:
        scenarios = np.asarray(scenario_ids).astype(str)
        if scenarios.shape != (len(predicted),):
            raise ValueError("scenario_ids must have shape [N]")
    else:
        scenarios = None

    plot_paths: list[Path] = []
    if scenarios is None:
        order = np.arange(len(predicted))
        selected = _downsample_indices(len(order), maximum_samples_per_plot)
        order = order[selected]
        x_values = timestamps[order] if timestamps is not None else order.astype(np.float64)
        x_label = "Time [s]" if timestamps is not None else "Sample index"
        plot_paths.append(
            _plot_control_comparison(
                predicted[order],
                expert[order],
                x_values,
                x_label,
                f"{dataset_label}: Expert and Neural Control Outputs",
                output_path / f"{_safe_file_stem(dataset_label)}_control_comparison.png",
                show_plots,
            )
        )
        return tuple(plot_paths)

    if max_scenarios is not None and max_scenarios <= 0:
        raise ValueError("max_scenarios must be positive")
    selected_scenarios = _ordered_unique(scenarios)
    if max_scenarios is not None:
        selected_scenarios = selected_scenarios[:max_scenarios]
    for scenario_id in selected_scenarios:
        scenario_mask = scenarios == scenario_id
        order = np.flatnonzero(scenario_mask)
        if timestamps is not None:
            order = order[np.argsort(timestamps[order], kind="stable")]
        selected = _downsample_indices(len(order), maximum_samples_per_plot)
        order = order[selected]
        x_values = timestamps[order] if timestamps is not None else np.arange(len(order))
        x_label = "Time [s]" if timestamps is not None else "Sample index"
        stem = _safe_file_stem(f"{dataset_label}_{scenario_id}")
        plot_paths.append(
            _plot_control_comparison(
                predicted[order],
                expert[order],
                x_values,
                x_label,
                f"{dataset_label}: {scenario_id} Expert and Neural Control Outputs",
                output_path / f"{stem}_control_comparison.png",
                show_plots,
            )
        )
    return tuple(plot_paths)
