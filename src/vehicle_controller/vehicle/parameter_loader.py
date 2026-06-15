"""Vehicle and actuator parameter data classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vehicle_controller.utils.config import load_yaml


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float = 1800.0
    wheelbase_m: float = 2.8
    wheel_radius_m: float = 0.33
    steering_ratio: float = 15.0
    drivetrain_ratio: float = 9.0
    drivetrain_efficiency: float = 0.9
    rolling_resistance_coefficient: float = 0.012
    drag_coefficient: float = 0.30
    frontal_area_m2: float = 2.3
    air_density_kgpm3: float = 1.225

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VehicleParameters":
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "VehicleParameters":
        return cls.from_mapping(load_yaml(path))


@dataclass(frozen=True)
class ActuatorLimits:
    steering_min_rad: float = -8.0
    steering_max_rad: float = 8.0
    steering_rate_max_radps: float = 6.0
    drive_torque_max_nm: float = 3000.0
    drive_torque_rate_max_nmps: float = 6000.0
    brake_decel_max_mps2: float = 8.0
    brake_jerk_max_mps3: float = 10.0
    longitudinal_deadband_mps2: float = 0.05

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ActuatorLimits":
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "ActuatorLimits":
        return cls.from_mapping(load_yaml(path))

