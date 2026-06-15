"""Input validation before neural inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = "ok"


class FeatureValidator:
    def __init__(self, normalized_abs_limit: float = 5.0) -> None:
        self.normalized_abs_limit = normalized_abs_limit

    def validate_raw(self, values: np.ndarray) -> ValidationResult:
        values = np.asarray(values)
        if values.shape != (FEATURE_COUNT,):
            return ValidationResult(False, "invalid_feature_shape")
        if not np.all(np.isfinite(values)):
            return ValidationResult(False, "non_finite_feature")
        if values[17] < -0.1:
            return ValidationResult(False, "negative_longitudinal_speed")
        return ValidationResult(True)

    def validate_normalized(self, values: np.ndarray) -> ValidationResult:
        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            return ValidationResult(False, "non_finite_normalized_feature")
        if np.max(np.abs(values)) > self.normalized_abs_limit + 1e-6:
            return ValidationResult(False, "normalized_feature_out_of_range")
        return ValidationResult(True)
