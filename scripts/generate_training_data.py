#!/usr/bin/env python3
"""Generate imitation-learning data with an expert closed-loop controller."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from vehicle_controller.data.simulation_generator import (
    SimulationDataGenerator,
    generation_metadata,
)
from vehicle_controller.data.synthetic_scenarios import build_typical_scenarios
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


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
    parser.add_argument("--generation-config")
    parser.add_argument("--output")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-randomization", action="store_true")
    args = parser.parse_args()

    main_config = load_yaml(project_path(args.config))
    generation_config_path = (
        project_path(args.generation_config)
        if args.generation_config
        else config_value(main_config, "data", "generation")
    )
    generation_config = load_yaml(generation_config_path)
    dataset_config = load_yaml(config_value(main_config, "data", "config"))
    model_config = load_yaml(config_value(main_config, "model", "config"))
    vehicle = VehicleParameters.from_yaml(
        str(config_value(main_config, "vehicle", "parameters"))
    )
    actuator_limits = ActuatorLimits.from_yaml(
        str(config_value(main_config, "vehicle", "actuator_limits"))
    )
    normalizer = FeatureNormalizer.from_yaml(
        str(config_value(main_config, "data", "normalization"))
    )

    time_step_s = float(generation_config["time_step_s"])
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(generation_config["repetitions"])
    )
    seed = args.seed if args.seed is not None else int(generation_config["seed"])
    randomize = (
        False
        if args.no_randomization
        else bool(generation_config["randomize_initial_state"])
    )
    profiles = build_typical_scenarios(time_step_s)
    generator = SimulationDataGenerator(
        vehicle=vehicle,
        actuator_limits=actuator_limits,
        preview_times_s=dataset_config.get("preview_times_s", (0.1, 0.2, 0.3, 0.4, 0.5)),
        lookahead_distances_m=dataset_config.get("lookahead_distances_m"),
        curvature_weights=dataset_config["curvature_weights"],
        steering_scale_deg=steering_limit_deg_from_config(model_config),
        acceleration_scale_mps2=float(model_config["accel_limit_mps2"]),
        time_step_s=time_step_s,
        normalizer=normalizer,
        expert_controller_config=generation_config.get("expert_controller"),
        seed=seed,
    )
    dataset = generator.generate(
        profiles,
        repetitions=repetitions,
        randomize_initial_state=randomize,
        lateral_offset_samples_m=generation_config.get("lateral_offset_samples_m"),
        maximum_lateral_offset_m=(
            None
            if "maximum_lateral_offset_m" not in generation_config
            else float(generation_config["maximum_lateral_offset_m"])
        ),
    )
    output = project_path(args.output or str(generation_config["output_path"]))
    metadata = generation_metadata(
        generation_config,
        [profile.name for profile in profiles],
        len(dataset.features),
    )
    dataset.save_npz(output, metadata)

    unique_scenarios = np.unique(dataset.scenario_ids)
    print(f"output={output}")
    print(f"samples={len(dataset.features)}")
    print(f"scenario_rollouts={len(unique_scenarios)}")
    print(f"feature_shape={dataset.features.shape}")
    print(f"target_shape={dataset.targets.shape}")
    print(
        "target_range="
        f"[{dataset.targets.min():.4f}, {dataset.targets.max():.4f}]"
    )
    for scenario_id in unique_scenarios:
        mask = dataset.scenario_ids == scenario_id
        lateral_rmse = float(np.sqrt(np.mean(dataset.raw_features[mask, 11] ** 2)))
        speed_rmse = float(np.sqrt(np.mean(dataset.raw_features[mask, 12] ** 2)))
        print(
            f"scenario={scenario_id} samples={int(mask.sum())} "
            f"lateral_rmse_m={lateral_rmse:.4f} "
            f"speed_rmse_mps={speed_rmse:.4f}"
        )


if __name__ == "__main__":
    main()
