"""Model evaluation loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from vehicle_controller.training.metrics import controller_metrics


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for features, batch_targets in loader:
            predictions.append(model(features.to(device)).cpu())
            targets.append(batch_targets)
    if not predictions:
        raise ValueError("Cannot evaluate an empty data loader")
    return controller_metrics(torch.cat(predictions), torch.cat(targets))

