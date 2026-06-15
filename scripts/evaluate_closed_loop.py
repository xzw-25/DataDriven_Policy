#!/usr/bin/env python3
"""Run a straight-road software-in-the-loop smoke scenario."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.simulation.rollout import summarize_rollout
from vehicle_controller.simulation.scenario import Scenario
from vehicle_controller.simulation.simulator import (
    SimulationSample,
    Simulator,
    command_to_longitudinal_acceleration,
)
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.types import Pose2D, ReferenceTrajectory, TrajectoryPoint, VehicleState
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_value(
    config: dict[str, object],
    section: str,
    key: str,
) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or key not in section_config:
        raise ValueError(f"Missing '{section}.{key}' in the main configuration")
    return project_path(str(section_config[key]))


def _load_pyplot(show_plots: bool) -> Any:
    try:
        cache_dir = Path(tempfile.gettempdir()) / "vehicle_controller_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        if not show_plots:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install dependencies with "
            "'python3 -m pip install -e .'."
        ) from error
    return plt


def _diagnostic_series(
    samples: list[SimulationSample],
    attribute: str,
    nested_attribute: str,
) -> np.ndarray:
    values = []
    for sample in samples:
        diagnostics = sample.diagnostics
        item = None if diagnostics is None else getattr(diagnostics, attribute)
        values.append(np.nan if item is None else getattr(item, nested_attribute))
    return np.asarray(values, dtype=np.float64)


def save_evaluation_plots(
    samples: list[SimulationSample],
    scenario: Scenario,
    vehicle: VehicleParameters,
    output_dir: Path,
    show_plots: bool = False,
) -> list[Path]:
    if not samples:
        raise ValueError("Cannot plot an empty rollout")

    plt = _load_pyplot(show_plots)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray([sample.time_s for sample in samples])
    actual_x = np.asarray([sample.state.pose.x for sample in samples])
    actual_y = np.asarray([sample.state.pose.y for sample in samples])
    actual_yaw = np.asarray([sample.state.pose.yaw for sample in samples])
    reference_x = np.asarray([point.x for point in scenario.reference.points])
    reference_y = np.asarray([point.y for point in scenario.reference.points])

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
    axes[0].set_title("Full Reference Overview")
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
    axes[1].set_title("Local Tracking Detail")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(loc="best")
    figure.suptitle("Closed-loop Trajectory Comparison", fontsize=16)
    figure.tight_layout()
    trajectory_path = output_dir / "trajectory_comparison.png"
    figure.savefig(trajectory_path, dpi=180)
    plot_paths.append(trajectory_path)

    raw_steering = _diagnostic_series(
        samples,
        "neural_output",
        "steering_des_rad",
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
    executed_steering = np.asarray(
        [sample.command.steering_wheel_angle_rad for sample in samples]
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
        ]
    )
    executed_acceleration = np.asarray(
        [
            command_to_longitudinal_acceleration(sample.command, vehicle)
            for sample in samples
        ]
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, raw_steering, label="Raw neural demand", alpha=0.85)
    axes[0].plot(time_s, limited_steering, label="After rate/magnitude limit")
    axes[0].plot(time_s, executed_steering, "--", label="Executed command")
    axes[0].set_ylabel("Steering wheel angle [rad]")
    axes[0].set_title("Neural Output and Executed Control")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(time_s, raw_acceleration, label="Raw neural demand", alpha=0.85)
    axes[1].plot(time_s, limited_acceleration, label="After allocation/limit")
    axes[1].plot(time_s, executed_acceleration, "--", label="Executed acceleration")
    axes[1].axhline(
        scenario.reference.a_ref,
        color="black",
        linestyle=":",
        label="Reference acceleration",
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Signed acceleration [m/s2]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    figure.tight_layout()
    control_path = output_dir / "control_comparison.png"
    figure.savefig(control_path, dpi=180)
    plot_paths.append(control_path)

    lateral_error = _diagnostic_series(samples, "tracking_errors", "e_lat")
    speed_error = _diagnostic_series(samples, "tracking_errors", "e_v")
    longitudinal_error = _diagnostic_series(samples, "tracking_errors", "e_s")
    speed = np.asarray([sample.state.vx for sample in samples])
    yaw_rate = np.asarray([sample.state.r for sample in samples])
    lateral_acceleration = np.asarray([sample.state.ay for sample in samples])

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes[0, 0].plot(time_s, lateral_error, color="tab:red")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Lateral Tracking Error")
    axes[0, 0].set_ylabel("e_lat [m]")
    axes[0, 1].plot(time_s, speed, label="Vehicle speed")
    axes[0, 1].axhline(
        scenario.reference.v_ref,
        color="black",
        linestyle="--",
        label="Reference speed",
    )
    axes[0, 1].plot(
        time_s,
        scenario.reference.v_ref - speed_error,
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
    tracking_path = output_dir / "tracking_stability.png"
    figure.savefig(tracking_path, dpi=180)
    plot_paths.append(tracking_path)

    if show_plots:
        plt.show()
    plt.close("all")
    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--device")
    parser.add_argument("--model-config")
    parser.add_argument("--output-dir", default="artifacts/reports/closed_loop")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    main_config = load_yaml(project_path(args.config))
    model_config_path = (
        project_path(args.model_config)
        if args.model_config
        else config_value(main_config, "model", "config")
    )
    vehicle_parameters_path = config_value(main_config, "vehicle", "parameters")
    actuator_limits_path = config_value(main_config, "vehicle", "actuator_limits")
    normalization_path = config_value(main_config, "data", "normalization")
    dataset_config_path = config_value(main_config, "data", "config")
    safety_limits_path = config_value(main_config, "deployment", "safety_limits")
    device = args.device or str(main_config.get("device", "cpu"))

    model = build_model(model_config_path)
    if args.checkpoint:
        checkpoint_path = project_path(args.checkpoint)
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint not found: {checkpoint_path}")
        load_model_state(checkpoint_path, model, device=device)
    pipeline = build_baseline_pipeline(
        model,
        project_root=PROJECT_ROOT,
        device=device,
        model_config_path=model_config_path,
        vehicle_parameters_path=vehicle_parameters_path,
        actuator_limits_path=actuator_limits_path,
        normalization_path=normalization_path,
        safety_limits_path=safety_limits_path,
        dataset_config_path=dataset_config_path,
    )
    vehicle = VehicleParameters.from_yaml(str(vehicle_parameters_path))
    reference = ReferenceTrajectory(
        points=[TrajectoryPoint(float(x), 0.0, s=float(x), v_ref=5.0) for x in range(101)],
        v_ref=5.0,
        a_ref=0.0,
        s_ref=10.0,
        kappa=0.0,
    )
    scenario = Scenario(
        name="straight_smoke",
        reference=reference,
        initial_state=VehicleState(Pose2D(0.0, 0.2, 0.0), 3.0, 0.0, 0.0, 0.0, 0.0),
        duration_s=args.duration,
    )
    samples = Simulator(pipeline, KinematicBicycleModel(vehicle)).run(scenario)
    summary = summarize_rollout(samples)
    for name, value in summary.items():
        print(f"{name}: {value:.6f}")
    if not args.no_plots:
        plot_paths = save_evaluation_plots(
            samples,
            scenario,
            vehicle,
            project_path(args.output_dir),
            show_plots=args.show_plots,
        )
        for path in plot_paths:
            print(f"plot={path}")


if __name__ == "__main__":
    main()
