"""Compact supervised training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    loss: float
    sample_count: int


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
        device: str = "cpu",
        gradient_clip_norm: float = 5.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.gradient_clip_norm = gradient_clip_norm

    def train_epoch(self, loader: DataLoader) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        sample_count = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(features)
            loss = self.loss_function(prediction, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            batch_size = features.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
        return EpochResult(total_loss / max(sample_count, 1), sample_count)

