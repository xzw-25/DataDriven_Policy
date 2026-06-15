"""Base neural controller interface."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from vehicle_controller.constants import FEATURE_COUNT


class BaseControllerModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Return normalized steering and signed acceleration."""
        raise NotImplementedError


def require_feature_shape(features: Tensor, feature_count: int = FEATURE_COUNT) -> None:
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return
    if features.ndim != 2 or features.shape[-1] != feature_count:
        raise ValueError(f"Expected [batch, {feature_count}], got {tuple(features.shape)}")
