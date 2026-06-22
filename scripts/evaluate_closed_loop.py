#!/usr/bin/env python3
"""Run closed-loop neural-controller validation scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from vehicle_controller.data.synthetic_scenarios import (
    build_typical_scenarios,
    initial_state_from_reference_profile,
)
from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.simulation.rollout import rollout_reference_profile, summarize_rollout
from vehicle_controller.simulation.scenario import Scenario
from vehicle_controller.simulation.showcase import (
    run_typical_reference_showcase,
    save_scenario_plots,
)
from vehicle_controller.simulation.simulator import (
    Simulator,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--device")
    parser.add_argument("--model-config")
    parser.add_argument("--output-dir", default="artifacts/reports/closed_loop")
    parser.add_argument(
        "--training-scenario-output-dir",
        default="training_scenarios",
        help=(
            "Directory for validation scenarios matching training-data generation. "
            "Relative paths are placed under --output-dir."
        ),
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run the original straight-road smoke scenario.",
    )
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
    generation_config_path = config_value(main_config, "data", "generation")
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
        plot_paths = save_scenario_plots(
            samples,
            scenario,
            vehicle,
            project_path(args.output_dir),
            show_plots=args.show_plots,
        )
        for path in plot_paths:
            print(f"plot={path}")

    if args.smoke_only:
        return

    generation_config = load_yaml(generation_config_path)
    generation_time_step_s = float(generation_config["time_step_s"])
    if args.no_plots:
        profiles = build_typical_scenarios(generation_time_step_s)
        dynamics = KinematicBicycleModel(vehicle)
        for profile in profiles:
            profile_samples = rollout_reference_profile(
                pipeline,
                dynamics,
                profile,
                initial_state_from_reference_profile(profile),
            )
            profile_summary = summarize_rollout(profile_samples)
            summary_text = " ".join(
                f"{name}={value:.6f}" for name, value in sorted(profile_summary.items())
            )
            print(f"training_scenario={profile.name} {summary_text}")
    else:
        training_scenario_output_dir = Path(args.training_scenario_output_dir)
        if not training_scenario_output_dir.is_absolute():
            training_scenario_output_dir = (
                project_path(args.output_dir) / training_scenario_output_dir
            )
        showcase = run_typical_reference_showcase(
            model,
            output_dir=training_scenario_output_dir,
            project_root=PROJECT_ROOT,
            device=device,
            model_config_path=model_config_path,
            vehicle_parameters_path=vehicle_parameters_path,
            actuator_limits_path=actuator_limits_path,
            normalization_path=normalization_path,
            safety_limits_path=safety_limits_path,
            dataset_config_path=dataset_config_path,
            generation_config_path=generation_config_path,
            generation_time_step_s=generation_time_step_s,
            show_plots=args.show_plots,
        )
        if showcase.overview_plot is not None:
            print(f"training_scenarios_overview={showcase.overview_plot}")
        for result in showcase.scenario_results:
            summary_text = " ".join(
                f"{name}={value:.6f}" for name, value in sorted(result.summary.items())
            )
            print(f"training_scenario={result.name} {summary_text}")
            for path in result.plot_paths:
                print(f"training_scenario_plot={path}")


if __name__ == "__main__":
    main()
