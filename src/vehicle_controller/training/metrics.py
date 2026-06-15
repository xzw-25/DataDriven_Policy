"""Offline controller metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def controller_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    error = prediction - target
    return {
        "steering_mae": float(torch.mean(torch.abs(error[:, 0])).item()),
        "acceleration_mae": float(torch.mean(torch.abs(error[:, 1])).item()),
        "steering_rmse": float(torch.sqrt(torch.mean(error[:, 0] ** 2)).item()),
        "acceleration_rmse": float(torch.sqrt(torch.mean(error[:, 1] ** 2)).item()),
        "steering_direction_error_rate": float(
            torch.mean((torch.sign(prediction[:, 0]) != torch.sign(target[:, 0])).float()).item()
        ),
    }

