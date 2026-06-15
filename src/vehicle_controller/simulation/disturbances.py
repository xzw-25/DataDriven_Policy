"""Deterministic and random simulation disturbances."""

import numpy as np


def add_measurement_noise(
    value: float,
    standard_deviation: float,
    generator: np.random.Generator,
) -> float:
    return float(value + generator.normal(0.0, standard_deviation))

