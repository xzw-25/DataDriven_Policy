"""Sliding-window sequence dataset."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from vehicle_controller.constants import FEATURE_COUNT


class SequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, features: Tensor, targets: Tensor, sequence_length: int) -> None:
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError(f"features must have shape [N, {FEATURE_COUNT}]")
        if targets.shape != (features.shape[0], 2):
            raise ValueError("targets must have shape [N, 2]")
        if sequence_length <= 0 or sequence_length > len(features):
            raise ValueError("Invalid sequence_length")
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        stop = index + self.sequence_length
        return self.features[index:stop], self.targets[stop - 1]
