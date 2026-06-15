"""Feature normalization shared by training and deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES
from vehicle_controller.types import ControllerFeatures
from vehicle_controller.utils.config import load_yaml


class FeatureNormalizer:
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        clip: float = 5.0,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.clip = float(clip)
        if self.mean.shape != (FEATURE_COUNT,) or self.std.shape != (FEATURE_COUNT,):
            raise ValueError(f"mean and std must each contain {FEATURE_COUNT} values")
        if np.any(self.std <= 0.0):
            raise ValueError("All standard deviations must be positive")
        if self.clip <= 0.0:
            raise ValueError("clip must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "FeatureNormalizer":
        names = tuple(str(name) for name in config["feature_names"])  # type: ignore[index]
        if names != FEATURE_NAMES:
            raise ValueError("Normalization feature order does not match FEATURE_NAMES")
        return cls(
            mean=config["mean"],  # type: ignore[arg-type]
            std=config["std"],  # type: ignore[arg-type]
            clip=float(config.get("clip", 5.0)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "FeatureNormalizer":
        return cls.from_mapping(load_yaml(path))

    def normalize(self, features: ControllerFeatures | np.ndarray) -> np.ndarray:
        values = features.values if isinstance(features, ControllerFeatures) else np.asarray(features)
        if values.shape[-1] != FEATURE_COUNT:
            raise ValueError(f"Last feature dimension must be {FEATURE_COUNT}")
        normalized = (values.astype(np.float32) - self.mean) / self.std
        return np.clip(normalized, -self.clip, self.clip)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32)
        if values.shape[-1] != FEATURE_COUNT:
            raise ValueError(f"Last feature dimension must be {FEATURE_COUNT}")
        return values * self.std + self.mean

