"""Closed-loop loss helpers for later differentiable dynamics integration."""

import torch
from torch import Tensor

from vehicle_controller.training.losses import closed_loop_tracking_loss


def tracking_rollout_loss(
    lateral_error: Tensor,
    speed_error: Tensor,
    longitudinal_error: Tensor,
    control_delta: Tensor,
    weights: tuple[float, float, float, float] = (10.0, 2.0, 1.0, 0.1),
) -> Tensor:
    return closed_loop_tracking_loss(
        lateral_error,
        speed_error,
        longitudinal_error,
        lateral_weight=weights[0],
        speed_weight=weights[1],
        longitudinal_weight=weights[2],
    ) + weights[3] * torch.mean(control_delta**2)
