"""Model evaluation loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from vehicle_controller.training.metrics import controller_metrics


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    predictions = []
    targets = []
    with torch.inference_mode():
        for features, batch_targets in loader:
            predictions.append(model(features.to(device)).cpu())
            targets.append(batch_targets)
    if not predictions:
        raise ValueError("Cannot evaluate an empty data loader")
    return torch.cat(predictions), torch.cat(targets)


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict[str, float]:
    predictions, targets = predict(model, loader, device=device)
    return controller_metrics(predictions, targets)
