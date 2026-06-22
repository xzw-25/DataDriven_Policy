"""Closed-loop rollout plotting and post-training showcase utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from torch import nn

from vehicle_controller.data.expert_controller import ExpertController
from vehicle_controller.data.synthetic_scenarios import (
    ReferenceProfile,
    build_typical_scenarios,
    initial_state_from_reference_profile,
)
from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.plotting import load_pyplot
from vehicle_controller.simulation.rollout import rollout_reference_profile, summarize_rollout
from vehicle_controller.simulation.scenario import Scenario
from vehicle_controller.simulation.simulator import (
    SimulationSample,
    command_to_longitudinal_acceleration,
)
from vehicle_controller.units import rad_to_deg
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


@dataclass(frozen=True)
class ShowcaseScenarioResult:
    name: str
    summary: dict[str, float]
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ShowcaseRunResult:
    scenario_results: tuple[ShowcaseScenarioResult, ...]
    overview_plot: Path | None


def _load_pyplot(show_plots: bool) -> Any:
    return load_pyplot(show_plots)


def _diagnostic_series(
    samples: Sequence[SimulationSample],
    attribute: str,
    nested_attribute: str,
) -> np.ndarray:
    values = []
    for sample in samples:
        diagnostics = sample.diagnostics
        item = None if diagnostics is None else getattr(diagnostics, attribute)
        values.append(np.nan if item is None else getattr(item, nested_attribute))
    return np.asarray(values, dtype=np.float64)


def _filename_prefix(name: str) -> str:
    return name.replace(" ", "_")


def _reference_series_at_samples(
    sample_times_s: np.ndarray,
    reference_time_s: Sequence[float],
    reference_values: Sequence[float],
) -> np.ndarray:
    return np.interp(
        sample_times_s,
        np.asarray(reference_time_s, dtype=np.float64),
        np.asarray(reference_values, dtype=np.float64),
    )


def _expert_control_profile(
    profile: ReferenceProfile,
    samples: Sequence[SimulationSample],
    vehicle: VehicleParameters,
    actuator_limits: ActuatorLimits,
    expert_controller_config: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the data-generation expert controller on the rollout states."""
    if not samples:
        raise ValueError("Cannot compute expert controls for an empty rollout")
    expert = ExpertController(
        vehicle,
        actuator_limits,
        **dict(expert_controller_config or {}),
    )
    time_values: list[float] = []
    steering_values: list[float] = []
    acceleration_values: list[float] = []
    for index, sample in enumerate(samples):
        if index + 1 < len(samples):
            dt = float(samples[index + 1].time_s - sample.time_s)
        elif index > 0:
            dt = float(sample.time_s - samples[index - 1].time_s)
        else:
            dt = float(profile.time_s[1] - profile.time_s[0])
        if dt <= 0.0:
            raise ValueError("Sample times must be strictly increasing")
        time_s = float(sample.time_s)
        reference_s, reference_speed, reference_acceleration = profile.sample(float(time_s))
        expert_output = expert.compute(
            profile.points,
            sample.state,
            reference_s,
            reference_speed,
            reference_acceleration,
            dt,
        )
        time_values.append(time_s)
        steering_values.append(float(expert_output.steering_des_deg))
        acceleration_values.append(float(expert_output.signed_accel_des_mps2))
    return (
        np.asarray(time_values, dtype=np.float64),
        np.asarray(steering_values, dtype=np.float64),
        np.asarray(acceleration_values, dtype=np.float64),
    )


def save_closed_loop_plots(
    samples: Sequence[SimulationSample],
    reference_points: Sequence[Any],
    vehicle: VehicleParameters,
    output_dir: str | Path,
    scenario_name: str,
    show_plots: bool = False,
    reference_time_s: Sequence[float] | None = None,
    reference_speed_mps: Sequence[float] | None = None,
    expert_time_s: Sequence[float] | None = None,
    expert_steering_deg: Sequence[float] | None = None,
    expert_acceleration_mps2: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    if not samples:
        raise ValueError("Cannot plot an empty rollout")

    plt = _load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_prefix = _filename_prefix(scenario_name)

    time_s = np.asarray([sample.time_s for sample in samples], dtype=np.float64)
    actual_x = np.asarray([sample.state.pose.x for sample in samples], dtype=np.float64)
    actual_y = np.asarray([sample.state.pose.y for sample in samples], dtype=np.float64)
    actual_yaw = np.asarray([sample.state.pose.yaw for sample in samples], dtype=np.float64)
    reference_x = np.asarray([point.x for point in reference_points], dtype=np.float64)
    reference_y = np.asarray([point.y for point in reference_points], dtype=np.float64)

    reference_speed_series = None
    if reference_time_s is not None and reference_speed_mps is not None:
        reference_speed_series = _reference_series_at_samples(
            time_s,
            reference_time_s,
            reference_speed_mps,
        )
    expert_steering_series = None
    expert_accel_series = None
    if expert_time_s is not None and expert_steering_deg is not None:
        expert_steering_series = _reference_series_at_samples(
            time_s,
            expert_time_s,
            expert_steering_deg,
        )
    if expert_time_s is not None and expert_acceleration_mps2 is not None:
        expert_accel_series = _reference_series_at_samples(
            time_s,
            expert_time_s,
            expert_acceleration_mps2,
        )

    plot_paths: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis in axes:
        axis.plot(reference_x, reference_y, "--", linewidth=2.2, label="Reference path")
        axis.plot(actual_x, actual_y, linewidth=2.2, label="Vehicle trajectory")
        axis.scatter(actual_x[0], actual_y[0], s=70, marker="o", label="Start", zorder=3)
        axis.scatter(actual_x[-1], actual_y[-1], s=90, marker="*", label="End", zorder=3)
        axis.set_xlabel("Global x [m]")
        axis.set_ylabel("Global y [m]")
        axis.grid(True, alpha=0.3)
    axes[0].set_title(f"{scenario_name}: Full Reference Overview")
    overview_y = np.concatenate((reference_y, actual_y))
    overview_y_margin = max(0.5, 0.1 * max(np.ptp(overview_y), 0.1))
    axes[0].set_ylim(
        overview_y.min() - overview_y_margin,
        overview_y.max() + overview_y_margin,
    )
    axes[0].legend(loc="best")

    heading_stride = max(1, len(samples) // 20)
    axes[1].quiver(
        actual_x[::heading_stride],
        actual_y[::heading_stride],
        np.cos(actual_yaw[::heading_stride]),
        np.sin(actual_yaw[::heading_stride]),
        angles="xy",
        scale_units="xy",
        scale=1.5,
        width=0.003,
        alpha=0.55,
        label="Vehicle heading",
    )
    x_margin = max(1.0, 0.1 * np.ptp(actual_x))
    y_margin = max(0.5, 0.25 * max(np.ptp(actual_y), 0.1))
    axes[1].set_xlim(actual_x.min() - x_margin, actual_x.max() + x_margin)
    axes[1].set_ylim(actual_y.min() - y_margin, actual_y.max() + y_margin)
    axes[1].set_title(f"{scenario_name}: Local Tracking Detail")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(loc="best")
    figure.suptitle("Closed-loop Trajectory Comparison", fontsize=16)
    figure.tight_layout()
    trajectory_path = output_path / f"{file_prefix}_trajectory_comparison.png"
    figure.savefig(trajectory_path, dpi=180)
    plot_paths.append(trajectory_path)

    raw_steering = _diagnostic_series(
        samples,
        "neural_output",
        "steering_des_deg",
    )
    raw_acceleration = _diagnostic_series(
        samples,
        "neural_output",
        "signed_accel_des_mps2",
    )
    limited_steering = _diagnostic_series(
        samples,
        "limited_candidate",
        "steering_wheel_angle_rad",
    )
    limited_steering = np.asarray([rad_to_deg(value) for value in limited_steering])
    executed_steering = np.asarray(
        [rad_to_deg(sample.command.steering_wheel_angle_rad) for sample in samples],
        dtype=np.float64,
    )
    limited_acceleration = np.asarray(
        [
            np.nan
            if sample.diagnostics is None
            or sample.diagnostics.limited_candidate is None
            else command_to_longitudinal_acceleration(
                sample.diagnostics.limited_candidate,
                vehicle,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    executed_acceleration = np.asarray(
        [
            command_to_longitudinal_acceleration(sample.command, vehicle)
            for sample in samples
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, raw_steering, label="Raw neural steering demand", alpha=0.85)
    axes[0].plot(time_s, limited_steering, label="Limited neural steering command")
    axes[0].plot(time_s, executed_steering, "--", label="Executed steering command")
    if expert_steering_series is not None:
        axes[0].plot(
            time_s,
            expert_steering_series,
            ":",
            color="black",
            label="Generated-data expert steering",
        )
    axes[0].set_ylabel("Steering wheel angle [deg]")
    axes[0].set_title(f"{scenario_name}: Neural, Expert, and Executed Control")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(
        time_s,
        raw_acceleration,
        label="Raw neural acceleration demand",
        alpha=0.85,
    )
    axes[1].plot(
        time_s,
        limited_acceleration,
        label="Allocated/limited neural acceleration",
    )
    axes[1].plot(
        time_s,
        executed_acceleration,
        "--",
        label="Executed longitudinal acceleration",
    )
    if expert_accel_series is not None:
        axes[1].plot(
            time_s,
            expert_accel_series,
            ":",
            color="black",
            label="Generated-data expert signed acceleration",
        )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Signed acceleration [m/s2]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    figure.tight_layout()
    control_path = output_path / f"{file_prefix}_control_comparison.png"
    figure.savefig(control_path, dpi=180)
    plot_paths.append(control_path)

    lateral_error = _diagnostic_series(samples, "tracking_errors", "e_lat")
    speed_error = _diagnostic_series(samples, "tracking_errors", "e_v")
    longitudinal_error = _diagnostic_series(samples, "tracking_errors", "e_s")
    speed = np.asarray([sample.state.vx for sample in samples], dtype=np.float64)
    yaw_rate = np.asarray([sample.state.r for sample in samples], dtype=np.float64)
    lateral_acceleration = np.asarray([sample.state.ay for sample in samples], dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes[0, 0].plot(time_s, lateral_error, color="tab:red")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Lateral Tracking Error")
    axes[0, 0].set_ylabel("e_lat [m]")
    axes[0, 1].plot(time_s, speed, label="Vehicle speed")
    if reference_speed_series is not None:
        axes[0, 1].plot(
            time_s,
            reference_speed_series,
            color="black",
            linestyle="--",
            label="Reference speed",
        )
        axes[0, 1].plot(
            time_s,
            reference_speed_series - speed_error,
            ":",
            label="Speed reconstructed from e_v",
        )
    axes[0, 1].set_title("Speed Tracking")
    axes[0, 1].set_ylabel("Speed [m/s]")
    axes[0, 1].legend(loc="best")
    axes[1, 0].plot(time_s, longitudinal_error, color="tab:purple")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Longitudinal Tracking Error")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("e_s [m]")
    axes[1, 1].plot(time_s, yaw_rate, label="Yaw rate [rad/s]")
    axes[1, 1].plot(
        time_s,
        lateral_acceleration,
        label="Lateral acceleration [m/s2]",
    )
    axes[1, 1].set_title("Vehicle Stability")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].legend(loc="best")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    tracking_path = output_path / f"{file_prefix}_tracking_stability.png"
    figure.savefig(tracking_path, dpi=180)
    plot_paths.append(tracking_path)

    if show_plots:
        plt.show()
    plt.close("all")
    return tuple(plot_paths)


def save_scenario_plots(
    samples: Sequence[SimulationSample],
    scenario: Scenario,
    vehicle: VehicleParameters,
    output_dir: str | Path,
    show_plots: bool = False,
) -> tuple[Path, ...]:
    if not samples:
        raise ValueError("Cannot plot an empty rollout")
    end_time_s = max(float(scenario.duration_s), float(samples[-1].time_s))
    reference_time_s = (0.0, end_time_s)
    reference_speed_mps = (scenario.reference.v_ref, scenario.reference.v_ref)
    return save_closed_loop_plots(
        samples,
        scenario.reference.points,
        vehicle,
        output_dir,
        scenario.name,
        show_plots=show_plots,
        reference_time_s=reference_time_s,
        reference_speed_mps=reference_speed_mps,
    )


def save_reference_profile_plots(
    samples: Sequence[SimulationSample],
    profile: ReferenceProfile,
    vehicle: VehicleParameters,
    output_dir: str | Path,
    show_plots: bool = False,
    expert_time_s: Sequence[float] | None = None,
    expert_steering_deg: Sequence[float] | None = None,
    expert_acceleration_mps2: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    return save_closed_loop_plots(
        samples,
        profile.points,
        vehicle,
        output_dir,
        profile.name,
        show_plots=show_plots,
        reference_time_s=profile.time_s,
        reference_speed_mps=profile.speed_mps,
        expert_time_s=expert_time_s,
        expert_steering_deg=expert_steering_deg,
        expert_acceleration_mps2=expert_acceleration_mps2,
    )


def save_showcase_overview(
    rollouts: Sequence[tuple[ReferenceProfile, Sequence[SimulationSample], dict[str, float]]],
    output_dir: str | Path,
    show_plots: bool = False,
) -> Path:
    if not rollouts:
        raise ValueError("Cannot plot an empty showcase")

    plt = _load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    columns = min(3, len(rollouts))
    rows = math.ceil(len(rollouts) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.5 * rows))
    if not isinstance(axes, np.ndarray):
        axes_array = np.asarray([axes], dtype=object)
    else:
        axes_array = axes.reshape(-1)

    for axis, (profile, samples, summary) in zip(axes_array, rollouts):
        actual_x = np.asarray([sample.state.pose.x for sample in samples], dtype=np.float64)
        actual_y = np.asarray([sample.state.pose.y for sample in samples], dtype=np.float64)
        reference_x = np.asarray([point.x for point in profile.points], dtype=np.float64)
        reference_y = np.asarray([point.y for point in profile.points], dtype=np.float64)
        axis.plot(reference_x, reference_y, "--", linewidth=2.0, label="Reference path")
        axis.plot(actual_x, actual_y, linewidth=2.0, label="Vehicle trajectory")
        axis.scatter(actual_x[0], actual_y[0], s=45, marker="o", zorder=3)
        axis.scatter(actual_x[-1], actual_y[-1], s=60, marker="*", zorder=3)
        axis.set_title(
            f"{profile.name}\n"
            f"fallback={summary['fallback_fraction']:.2%}, "
            f"mean_v={summary['mean_speed_mps']:.2f} m/s"
        )
        axis.set_xlabel("Global x [m]")
        axis.set_ylabel("Global y [m]")
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    for axis in axes_array[len(rollouts) :]:
        axis.axis("off")

    axes_array[0].legend(loc="best")
    figure.suptitle("Post-training Closed-loop Tracking Showcase", fontsize=16)
    figure.tight_layout()
    overview_path = output_path / "training_showcase_overview.png"
    figure.savefig(overview_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close("all")
    return overview_path


def run_typical_reference_showcase(
    model: nn.Module,
    output_dir: str | Path,
    project_root: str | Path = ".",
    device: str = "cpu",
    model_config_path: str | Path = "configs/model/mlp_controller.yaml",
    vehicle_parameters_path: str | Path = "configs/vehicle/vehicle_params.yaml",
    actuator_limits_path: str | Path = "configs/vehicle/actuator_limits.yaml",
    normalization_path: str | Path = "configs/data/normalization.yaml",
    safety_limits_path: str | Path = "configs/deployment/safety_limits.yaml",
    dataset_config_path: str | Path = "configs/data/dataset.yaml",
    generation_config_path: str | Path = "configs/data/simulation_generation.yaml",
    generation_time_step_s: float = 0.02,
    show_plots: bool = False,
) -> ShowcaseRunResult:
    root = Path(project_root)

    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else root / path

    output_path = resolve(output_dir)
    vehicle = VehicleParameters.from_yaml(str(resolve(vehicle_parameters_path)))
    actuator_limits = ActuatorLimits.from_yaml(str(resolve(actuator_limits_path)))
    generation_config = load_yaml(resolve(generation_config_path))
    expert_controller_config = generation_config.get("expert_controller")
    if expert_controller_config is not None and not isinstance(expert_controller_config, dict):
        raise ValueError("generation expert_controller config must be a mapping")
    pipeline = build_baseline_pipeline(
        model,
        project_root=root,
        device=device,
        model_config_path=resolve(model_config_path),
        vehicle_parameters_path=resolve(vehicle_parameters_path),
        actuator_limits_path=resolve(actuator_limits_path),
        normalization_path=resolve(normalization_path),
        safety_limits_path=resolve(safety_limits_path),
        dataset_config_path=resolve(dataset_config_path),
    )
    profiles = build_typical_scenarios(float(generation_time_step_s))
    dynamics = KinematicBicycleModel(vehicle)

    scenario_results: list[ShowcaseScenarioResult] = []
    overview_rollouts: list[tuple[ReferenceProfile, Sequence[SimulationSample], dict[str, float]]] = []
    for profile in profiles:
        samples = rollout_reference_profile(
            pipeline,
            dynamics,
            profile,
            initial_state_from_reference_profile(profile),
        )
        expert_time_s, expert_steering_deg, expert_acceleration_mps2 = _expert_control_profile(
            profile,
            samples,
            vehicle,
            actuator_limits,
            expert_controller_config,
        )
        summary = summarize_rollout(samples)
        scenario_output_dir = output_path / profile.name
        plot_paths = save_reference_profile_plots(
            samples,
            profile,
            vehicle,
            scenario_output_dir,
            show_plots=show_plots,
            expert_time_s=expert_time_s,
            expert_steering_deg=expert_steering_deg,
            expert_acceleration_mps2=expert_acceleration_mps2,
        )
        scenario_results.append(
            ShowcaseScenarioResult(
                name=profile.name,
                summary=summary,
                plot_paths=plot_paths,
            )
        )
        overview_rollouts.append((profile, samples, summary))

    overview_plot = save_showcase_overview(
        overview_rollouts,
        output_path,
        show_plots=show_plots,
    )
    return ShowcaseRunResult(tuple(scenario_results), overview_plot)
