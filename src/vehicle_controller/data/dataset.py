"""Array-backed PyTorch dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from vehicle_controller.constants import FEATURE_COUNT


class ControllerDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError(f"features must have shape [N, {FEATURE_COUNT}]")
        if targets.shape != (features.shape[0], 2):
            raise ValueError("targets must have shape [N, 2]")
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    @classmethod
    def from_npz(cls, path: str | Path) -> "ControllerDataset":
        data = np.load(path)
        return cls(data["features"], data["targets"])

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]

