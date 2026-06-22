"""Physically consistent feature augmentation."""

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT


SIGNED_LATERAL_FEATURE_INDICES = (1, 3, 5, 7, 9, 10, 11, 19, 20)


def mirror_left_right(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirrored_features = np.asarray(features, dtype=np.float32).copy()
    mirrored_targets = np.asarray(targets, dtype=np.float32).copy()
    mirrored_features[..., list(SIGNED_LATERAL_FEATURE_INDICES)] *= -1.0
    mirrored_targets[..., 0] *= -1.0
    return mirrored_features, mirrored_targets


def add_gaussian_noise(
    features: np.ndarray,
    standard_deviation: np.ndarray,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    generator = generator or np.random.default_rng()
    standard_deviation = np.asarray(standard_deviation, dtype=np.float32)
    if standard_deviation.shape != (FEATURE_COUNT,):
        raise ValueError(f"standard_deviation must contain {FEATURE_COUNT} values")
    return np.asarray(features, dtype=np.float32) + generator.normal(
        0.0,
        standard_deviation,
        size=np.asarray(features).shape,
    ).astype(np.float32)
