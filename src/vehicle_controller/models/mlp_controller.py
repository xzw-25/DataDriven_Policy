"""Three-branch MLP controller for the 22-dimensional feature contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vehicle_controller.constants import (
    REFERENCE_ERROR_FEATURE_COUNT,
    TRAJECTORY_FEATURE_COUNT,
)
from vehicle_controller.models.base import BaseControllerModel, require_feature_shape
from vehicle_controller.models.heads import BoundedOutputHead


def _branch(input_size: int, hidden_sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for index, hidden_size in enumerate(hidden_sizes):
        layers.append(nn.Linear(current_size, hidden_size))
        if index == 0:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.SiLU())
        current_size = hidden_size
    return nn.Sequential(*layers)


class MLPController(BaseControllerModel):
    def __init__(
        self,
        trajectory_hidden: list[int] | None = None,
        error_hidden: list[int] | None = None,
        state_hidden: list[int] | None = None,
        shared_hidden: list[int] | None = None,
        head_hidden: int = 32,
    ) -> None:
        super().__init__()
        trajectory_hidden = trajectory_hidden or [64, 64] 
        error_hidden = error_hidden or [32]
        state_hidden = state_hidden or [32]
        shared_hidden = shared_hidden or [128, 128, 64]
        self.trajectory_encoder = _branch(10, trajectory_hidden)
        self.error_encoder = _branch(7, error_hidden)
        self.state_encoder = _branch(5, state_hidden)
        encoded_size = trajectory_hidden[-1] + error_hidden[-1] + state_hidden[-1]
        self.shared = _branch(encoded_size, shared_hidden)
        self.steering_head = BoundedOutputHead(shared_hidden[-1], head_hidden)
        self.acceleration_head = BoundedOutputHead(shared_hidden[-1], head_hidden)

    def forward(self, features: Tensor) -> Tensor:
        require_feature_shape(features)
        trajectory_end = TRAJECTORY_FEATURE_COUNT
        reference_end = trajectory_end + REFERENCE_ERROR_FEATURE_COUNT
        trajectory = self.trajectory_encoder(features[:, :trajectory_end])
        reference_and_errors = self.error_encoder(features[:, trajectory_end:reference_end])
        state = self.state_encoder(features[:, reference_end:])
        shared = self.shared(torch.cat((trajectory, reference_and_errors, state), dim=-1))
        return torch.cat(
            (self.steering_head(shared), self.acceleration_head(shared)),
            dim=-1,
        )
