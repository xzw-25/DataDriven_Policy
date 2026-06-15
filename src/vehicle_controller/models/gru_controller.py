"""Optional recurrent controller for future temporal experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.heads import BoundedOutputHead


class GRUController(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, head_hidden: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(FEATURE_COUNT, hidden_size, num_layers=num_layers, batch_first=True)
        self.steering_head = BoundedOutputHead(hidden_size, head_hidden)
        self.acceleration_head = BoundedOutputHead(hidden_size, head_hidden)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != FEATURE_COUNT:
            raise ValueError(
                f"Expected [batch, sequence, {FEATURE_COUNT}], got {tuple(features.shape)}"
            )
        output, _ = self.gru(features)
        hidden = output[:, -1, :]
        return torch.cat(
            (self.steering_head(hidden), self.acceleration_head(hidden)),
            dim=-1,
        )
