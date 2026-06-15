"""Direct MLP controller without input branch encoders."""

from __future__ import annotations

from torch import Tensor, nn

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.base import BaseControllerModel, require_feature_shape


class DirectMLPController(BaseControllerModel):
    """Map the normalized 22-D feature vector directly to two bounded controls."""

    def __init__(self, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 64]

        layers: list[nn.Module] = []
        input_size = FEATURE_COUNT
        for index, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_size, hidden_size))
            if index == 0:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            input_size = hidden_size
        layers.extend((nn.Linear(input_size, 2), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        require_feature_shape(features)
        return self.network(features)
