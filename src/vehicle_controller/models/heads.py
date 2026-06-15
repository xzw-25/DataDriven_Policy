"""Reusable bounded controller output heads."""

import torch
from torch import Tensor, nn


class BoundedOutputHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.tanh(self.network(features))

