"""Unit conversion helpers for controller inputs and outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping


RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


def rad_to_deg(value: float) -> float:
    return float(value) * RAD_TO_DEG


def deg_to_rad(value: float) -> float:
    return float(value) * DEG_TO_RAD


def steering_limit_deg_from_config(config: Mapping[str, object]) -> float:
    if "steering_limit_deg" in config:
        return float(config["steering_limit_deg"])
    if "steering_limit_rad" in config:
        return rad_to_deg(float(config["steering_limit_rad"]))
    raise ValueError("Model config must define steering_limit_deg")


def steering_limit_rad_from_config(config: Mapping[str, object]) -> float:
    if "steering_limit_rad" in config:
        return float(config["steering_limit_rad"])
    if "steering_limit_deg" in config:
        return deg_to_rad(float(config["steering_limit_deg"]))
    raise ValueError("Model config must define steering_limit_deg")
