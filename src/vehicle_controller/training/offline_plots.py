"""Offline validation plotting utilities."""

from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Sequence

import numpy as np

from vehicle_controller.constants import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    REFERENCE_ERROR_FEATURE_COUNT,
    STATE_FEATURE_COUNT,
    TRAJECTORY_FEATURE_COUNT,
)
from vehicle_controller.plotting import load_pyplot


EXPERT_STEERING_LABEL = "Generated-data expert steering"
NEURAL_STEERING_LABEL = "Neural controller steering"
EXPERT_ACCELERATION_LABEL = "Generated-data expert signed acceleration"
NEURAL_ACCELERATION_LABEL = "Neural controller signed acceleration"
FEATURE_UNITS = {
    "x": "m",
    "y": "m",
    "e_lat": "m",
    "e_s": "m",
    "v_ref": "m/s",
    "e_v": "m/s",
    "vx": "m/s",
    "a_ref": "m/s^2",
    "ax": "m/s^2",
    "ay": "m/s^2",
    "r": "deg/s",
    "kappa": "1/m",
    "s_ref": "m",
}


def _as_control_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_feature_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != FEATURE_COUNT:
        raise ValueError(f"{name} must have shape [N, {FEATURE_COUNT}]")
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


def _feature_axis_label(feature_name: str) -> str:
    normalized_name = str(feature_name)
    if re.fullmatch(r"[xy][1-5]", normalized_name):
        unit = FEATURE_UNITS.get(normalized_name[0])
    else:
        unit = FEATURE_UNITS.get(normalized_name)
    if unit is None:
        return normalized_name
    return f"{normalized_name} [{unit}]"


def _feature_plot_values(features: np.ndarray, feature_index: int, feature_name: str) -> np.ndarray:
    values = np.asarray(features[:, feature_index], dtype=np.float64)
    if str(feature_name) == "r":
        return np.rad2deg(values)
    return values


def _as_positions_enu_array(name: str, values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [N, 2+] or [N, 2]")
    if not np.all(np.isfinite(array[:, :2])):
        raise ValueError(f"{name} contains non-finite XY values")
    return array[:, :2]


def _as_heading_array(name: str, values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _preview_trajectory_points_enu(
    features: np.ndarray,
    positions_enu: np.ndarray,
    headings_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    preview_x_body = np.asarray(features[:, 0:TRAJECTORY_FEATURE_COUNT:2], dtype=np.float64)
    preview_y_body = np.asarray(features[:, 1:TRAJECTORY_FEATURE_COUNT:2], dtype=np.float64)
    cos_yaw = np.cos(headings_rad)[:, None]
    sin_yaw = np.sin(headings_rad)[:, None]
    preview_x_enu = positions_enu[:, 0:1] + cos_yaw * preview_x_body - sin_yaw * preview_y_body
    preview_y_enu = positions_enu[:, 1:2] + sin_yaw * preview_x_body + cos_yaw * preview_y_body
    return preview_x_enu, preview_y_enu


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


def _plot_feature_preview_xy(
    features: np.ndarray,
    title: str,
    output_path: Path,
    feature_names: Sequence[str],
    positions_enu: np.ndarray | None,
    headings_rad: np.ndarray | None,
    show_plots: bool,
) -> Path:
    plt = load_pyplot(show_plots)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    frame_indices = np.arange(len(features))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(frame_indices)))
    if positions_enu is not None and headings_rad is not None:
        preview_x, preview_y = _preview_trajectory_points_enu(features, positions_enu, headings_rad)
        x_label = "ENU x [m]"
        y_label = "ENU y [m]"
        title_suffix = "Preview trajectories in ENU across frames"
    else:
        preview_x = np.asarray(features[:, 0:TRAJECTORY_FEATURE_COUNT:2], dtype=np.float64)
        preview_y = np.asarray(features[:, 1:TRAJECTORY_FEATURE_COUNT:2], dtype=np.float64)
        x_label = _feature_axis_label(feature_names[0])
        y_label = _feature_axis_label(feature_names[1])
        title_suffix = "Preview XY trajectories across frames"
    for color, frame_index in zip(colors, frame_indices, strict=False):
        axis.plot(
            preview_x[frame_index],
            preview_y[frame_index],
            color=color,
            linewidth=0.9,
            alpha=0.45,
        )
    for point_index in range(preview_x.shape[1]):
        axis.scatter(
            preview_x[:, point_index],
            preview_y[:, point_index],
            s=10,
            alpha=0.4,
            label=f"Preview point {point_index + 1}",
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title_suffix, fontsize=11)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", ncol=2)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return output_path


def _plot_feature_reference_and_errors(
    features: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
    feature_names: Sequence[str],
    show_plots: bool,
) -> Path:
    plt = load_pyplot(show_plots)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = TRAJECTORY_FEATURE_COUNT
    stop = start + REFERENCE_ERROR_FEATURE_COUNT
    indices = tuple(range(start, stop))
    figure, axes = plt.subplots(REFERENCE_ERROR_FEATURE_COUNT, 1, figsize=(14, 16), sharex=True)
    for axis, index in zip(axes, indices):
        feature_name = str(feature_names[index])
        axis.plot(x_values, _feature_plot_values(features, index, feature_name), linewidth=1.2)
        axis.set_ylabel(_feature_axis_label(feature_name))
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Reference quantities and tracking errors", fontsize=11)
    axes[-1].set_xlabel(x_label)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return output_path


def _plot_feature_vehicle_state(
    features: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
    feature_names: Sequence[str],
    show_plots: bool,
) -> Path:
    plt = load_pyplot(show_plots)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = TRAJECTORY_FEATURE_COUNT + REFERENCE_ERROR_FEATURE_COUNT
    stop = start + STATE_FEATURE_COUNT
    indices = tuple(range(start, stop))
    figure, axes = plt.subplots(STATE_FEATURE_COUNT, 1, figsize=(14, 10), sharex=True)
    for axis, index in zip(axes, indices):
        feature_name = str(feature_names[index])
        axis.plot(x_values, _feature_plot_values(features, index, feature_name), linewidth=1.2)
        axis.set_ylabel(_feature_axis_label(feature_name))
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Vehicle state variations", fontsize=11)
    axes[-1].set_xlabel(x_label)
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


def save_feature_signal_plots(
    features: np.ndarray,
    output_dir: str | Path,
    timestamps_s: np.ndarray | None = None,
    scenario_ids: np.ndarray | None = None,
    positions_enu: np.ndarray | None = None,
    headings_rad: np.ndarray | None = None,
    dataset_label: str = "validation",
    feature_names: Sequence[str] | None = None,
    max_scenarios: int | None = None,
    maximum_samples_per_plot: int = 2000,
    show_plots: bool = False,
) -> tuple[Path, ...]:
    """Plot controller feature signals grouped into three per-scenario figures."""
    feature_array = _as_feature_array("features", features)
    resolved_feature_names = FEATURE_NAMES if feature_names is None else tuple(feature_names)
    if len(resolved_feature_names) != FEATURE_COUNT:
        raise ValueError(f"feature_names must contain {FEATURE_COUNT} values")

    output_path = Path(output_dir)
    if timestamps_s is not None:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        if timestamps.shape != (len(feature_array),):
            raise ValueError("timestamps_s must have shape [N]")
    else:
        timestamps = None

    if scenario_ids is not None:
        scenarios = np.asarray(scenario_ids).astype(str)
        if scenarios.shape != (len(feature_array),):
            raise ValueError("scenario_ids must have shape [N]")
    else:
        scenarios = None
    pose_positions = _as_positions_enu_array("positions_enu", positions_enu)
    pose_headings = _as_heading_array("headings_rad", headings_rad)
    if pose_positions is not None and pose_positions.shape[0] != len(feature_array):
        raise ValueError("positions_enu must align with features")
    if pose_headings is not None and pose_headings.shape[0] != len(feature_array):
        raise ValueError("headings_rad must align with features")
    if (pose_positions is None) != (pose_headings is None):
        raise ValueError("positions_enu and headings_rad must be provided together")

    plot_paths: list[Path] = []
    if scenarios is None:
        order = np.arange(len(feature_array))
        selected = _downsample_indices(len(order), maximum_samples_per_plot)
        order = order[selected]
        x_values = timestamps[order] if timestamps is not None else order.astype(np.float64)
        x_label = "Time [s]" if timestamps is not None else "Sample index"
        ordered_positions = None if pose_positions is None else pose_positions[order]
        ordered_headings = None if pose_headings is None else pose_headings[order]
        stem = _safe_file_stem(dataset_label)
        title_prefix = f"{dataset_label}: Controller Feature Signals"
        plot_paths.extend(
            (
                _plot_feature_preview_xy(
                    feature_array[order],
                    f"{title_prefix} - Preview XY",
                    output_path / f"{stem}_feature_preview_xy.png",
                    resolved_feature_names,
                    ordered_positions,
                    ordered_headings,
                    show_plots,
                ),
                _plot_feature_reference_and_errors(
                    feature_array[order],
                    x_values,
                    x_label,
                    f"{title_prefix} - Reference and Tracking Errors",
                    output_path / f"{stem}_feature_reference_errors.png",
                    resolved_feature_names,
                    show_plots,
                ),
                _plot_feature_vehicle_state(
                    feature_array[order],
                    x_values,
                    x_label,
                    f"{title_prefix} - Vehicle State",
                    output_path / f"{stem}_feature_vehicle_state.png",
                    resolved_feature_names,
                    show_plots,
                ),
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
        ordered_positions = None if pose_positions is None else pose_positions[order]
        ordered_headings = None if pose_headings is None else pose_headings[order]
        stem = _safe_file_stem(f"{dataset_label}_{scenario_id}")
        title_prefix = f"{dataset_label}: {scenario_id} Controller Feature Signals"
        plot_paths.extend(
            (
                _plot_feature_preview_xy(
                    feature_array[order],
                    f"{title_prefix} - Preview XY",
                    output_path / f"{stem}_feature_preview_xy.png",
                    resolved_feature_names,
                    ordered_positions,
                    ordered_headings,
                    show_plots,
                ),
                _plot_feature_reference_and_errors(
                    feature_array[order],
                    x_values,
                    x_label,
                    f"{title_prefix} - Reference and Tracking Errors",
                    output_path / f"{stem}_feature_reference_errors.png",
                    resolved_feature_names,
                    show_plots,
                ),
                _plot_feature_vehicle_state(
                    feature_array[order],
                    x_values,
                    x_label,
                    f"{title_prefix} - Vehicle State",
                    output_path / f"{stem}_feature_vehicle_state.png",
                    resolved_feature_names,
                    show_plots,
                ),
            )
        )
    return tuple(plot_paths)
