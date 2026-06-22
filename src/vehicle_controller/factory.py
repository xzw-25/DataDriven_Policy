"""Convenience construction of the baseline controller pipeline."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from vehicle_controller.control.command_limiter import CommandLimiter
from vehicle_controller.control.controller_pipeline import ControllerPipeline
from vehicle_controller.control.fallback_controller import FallbackController
from vehicle_controller.control.longitudinal_allocator import LongitudinalAllocator
from vehicle_controller.control.neural_policy import NeuralPolicy
from vehicle_controller.control.safety_supervisor import SafetyLimits, SafetySupervisor
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.features.validator import FeatureValidator
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


def build_baseline_pipeline(
    model: nn.Module,
    project_root: str | Path = ".",
    device: str = "cpu",
    model_config_path: str | Path = "configs/model/mlp_controller.yaml",
    vehicle_parameters_path: str | Path = "configs/vehicle/vehicle_params.yaml",
    actuator_limits_path: str | Path = "configs/vehicle/actuator_limits.yaml",
    normalization_path: str | Path = "configs/data/normalization.yaml",
    safety_limits_path: str | Path = "configs/deployment/safety_limits.yaml",
    dataset_config_path: str | Path = "configs/data/dataset.yaml",
) -> ControllerPipeline:
    root = Path(project_root)

    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else root / path

    vehicle = VehicleParameters.from_yaml(str(resolve(vehicle_parameters_path)))
    actuator_limits = ActuatorLimits.from_yaml(str(resolve(actuator_limits_path)))
    normalizer = FeatureNormalizer.from_yaml(str(resolve(normalization_path)))
    model_config = load_yaml(resolve(model_config_path))
    safety_config = load_yaml(resolve(safety_limits_path))
    dataset_config = load_yaml(resolve(dataset_config_path))
    safety_limits = SafetyLimits(
        lateral_accel_max_mps2=float(safety_config["lateral_accel_max_mps2"]),
        yaw_rate_max_radps=float(safety_config["yaw_rate_max_radps"]),
        lateral_error_max_m=float(safety_config["lateral_error_max_m"]),
        speed_max_mps=float(safety_config["speed_max_mps"]),
    )
    return ControllerPipeline(
        neural_policy=NeuralPolicy(
            model=model,
            normalizer=normalizer,
            steering_limit_deg=steering_limit_deg_from_config(model_config),
            accel_limit_mps2=float(model_config["accel_limit_mps2"]),
            device=device,
        ),
        allocator=LongitudinalAllocator(vehicle, actuator_limits),
        limiter=CommandLimiter(actuator_limits),
        fallback_controller=FallbackController(vehicle),
        safety_supervisor=SafetySupervisor(safety_limits),
        feature_validator=FeatureValidator(
            float(safety_config["input_abs_normalized_max"])
        ),
        preview_times_s=dataset_config.get("preview_times_s", (0.1, 0.2, 0.3, 0.4, 0.5)),
        lookahead_distances_m=dataset_config.get("lookahead_distances_m"),
        curvature_weights=dataset_config["curvature_weights"],
    )
