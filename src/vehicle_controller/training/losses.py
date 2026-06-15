"""Supervised and closed-loop controller losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_control_pair(prediction: Tensor, target: Tensor) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim < 1
        or prediction.shape[-1] != 2
    ):
        raise ValueError("prediction and target must have matching [..., 2] shapes")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty")


def _validate_control_sequence(outputs: Tensor) -> None:
    if outputs.ndim != 3 or outputs.shape[-1] != 2:
        raise ValueError("outputs must have shape [batch, time, 2]")
    if outputs.shape[0] == 0:
        raise ValueError("outputs must contain at least one batch item")


def _validate_matching_non_empty(name: str, *values: Tensor) -> None:
    if not values or values[0].numel() == 0:
        raise ValueError(f"{name} tensors must not be empty")
    expected_shape = values[0].shape
    if any(value.shape != expected_shape for value in values[1:]):
        raise ValueError(f"{name} tensors must have matching shapes")


def _validate_time_step(time_step_s: float) -> None:
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")


def steering_huber_loss(
    prediction: Tensor,
    target: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Return Huber loss for the steering channel at index zero."""
    _validate_control_pair(prediction, target)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return F.huber_loss(prediction[..., 0], target[..., 0], delta=delta)


def longitudinal_huber_loss(
    prediction: Tensor,
    target: Tensor,
    delta: float = 1.0,
) -> Tensor:
    """Return Huber loss for the signed longitudinal acceleration channel."""
    _validate_control_pair(prediction, target)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return F.huber_loss(prediction[..., 1], target[..., 1], delta=delta)


class ControllerLoss(nn.Module):
    """Weighted steering and longitudinal Huber supervision."""

    def __init__(
        self,
        steering_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        steering_delta: float = 1.0,
        acceleration_delta: float = 1.0,
    ) -> None:
        super().__init__()
        _validate_non_negative("steering_weight", steering_weight)
        _validate_non_negative("acceleration_weight", acceleration_weight)
        if steering_delta <= 0.0 or acceleration_delta <= 0.0:
            raise ValueError("Huber deltas must be positive")
        self.steering_weight = steering_weight
        self.acceleration_weight = acceleration_weight
        self.steering_delta = steering_delta
        self.acceleration_delta = acceleration_delta

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        steering_loss = steering_huber_loss(
            prediction,
            target,
            delta=self.steering_delta,
        )
        acceleration_loss = longitudinal_huber_loss(
            prediction,
            target,
            delta=self.acceleration_delta,
        )
        return (
            self.steering_weight * steering_loss
            + self.acceleration_weight * acceleration_loss
        )


def _control_difference(
    outputs: Tensor,
    order: int,
    time_step_s: float,
) -> Tensor | None:
    _validate_control_sequence(outputs)
    _validate_time_step(time_step_s)
    if outputs.shape[1] <= order:
        return None

    difference = outputs
    for _ in range(order):
        difference = difference[:, 1:] - difference[:, :-1]
    return difference / (time_step_s**order)


def _weighted_control_mean_square(
    difference: Tensor | None,
    outputs: Tensor,
    steering_weight: float,
    longitudinal_weight: float,
) -> Tensor:
    _validate_non_negative("steering_weight", steering_weight)
    _validate_non_negative("longitudinal_weight", longitudinal_weight)
    if difference is None:
        return outputs.sum() * 0.0
    return (
        steering_weight * torch.mean(difference[..., 0] ** 2)
        + longitudinal_weight * torch.mean(difference[..., 1] ** 2)
    )


def first_order_smoothness_loss(
    outputs: Tensor,
    time_step_s: float = 1.0,
    steering_weight: float = 1.0,
    longitudinal_weight: float = 1.0,
) -> Tensor:
    """Penalize steering rate and longitudinal acceleration rate (jerk)."""
    difference = _control_difference(outputs, order=1, time_step_s=time_step_s)
    return _weighted_control_mean_square(
        difference,
        outputs,
        steering_weight,
        longitudinal_weight,
    )


def second_order_smoothness_loss(
    outputs: Tensor,
    time_step_s: float = 1.0,
    steering_weight: float = 1.0,
    longitudinal_weight: float = 1.0,
) -> Tensor:
    """Penalize second temporal differences of both controller outputs."""
    difference = _control_difference(outputs, order=2, time_step_s=time_step_s)
    return _weighted_control_mean_square(
        difference,
        outputs,
        steering_weight,
        longitudinal_weight,
    )


def temporal_smoothness_loss(outputs: Tensor) -> Tensor:
    """Backward-compatible unscaled first-order smoothness loss."""
    return first_order_smoothness_loss(outputs)


def closed_loop_tracking_loss(
    lateral_error: Tensor,
    speed_error: Tensor,
    longitudinal_error: Tensor,
    lateral_weight: float = 10.0,
    speed_weight: float = 2.0,
    longitudinal_weight: float = 1.0,
) -> Tensor:
    """Penalize lateral, speed, and longitudinal rollout tracking errors."""
    _validate_matching_non_empty(
        "tracking error",
        lateral_error,
        speed_error,
        longitudinal_error,
    )
    _validate_non_negative("lateral_weight", lateral_weight)
    _validate_non_negative("speed_weight", speed_weight)
    _validate_non_negative("longitudinal_weight", longitudinal_weight)
    return (
        lateral_weight * torch.mean(lateral_error**2)
        + speed_weight * torch.mean(speed_error**2)
        + longitudinal_weight * torch.mean(longitudinal_error**2)
    )


def closed_loop_stability_loss(
    yaw_rate: Tensor,
    lateral_acceleration: Tensor,
    yaw_rate_weight: float = 0.2,
    lateral_acceleration_weight: float = 0.2,
) -> Tensor:
    """Penalize excessive yaw rate and lateral acceleration."""
    _validate_matching_non_empty("stability", yaw_rate, lateral_acceleration)
    _validate_non_negative("yaw_rate_weight", yaw_rate_weight)
    _validate_non_negative(
        "lateral_acceleration_weight",
        lateral_acceleration_weight,
    )
    return (
        yaw_rate_weight * torch.mean(yaw_rate**2)
        + lateral_acceleration_weight * torch.mean(lateral_acceleration**2)
    )


def closed_loop_comfort_loss(
    outputs: Tensor,
    time_step_s: float,
    steering_rate_weight: float = 0.1,
    longitudinal_jerk_weight: float = 0.1,
    steering_acceleration_weight: float = 0.01,
    longitudinal_snap_weight: float = 0.01,
) -> Tensor:
    """Penalize control rate and acceleration for passenger comfort."""
    first_order = first_order_smoothness_loss(
        outputs,
        time_step_s=time_step_s,
        steering_weight=steering_rate_weight,
        longitudinal_weight=longitudinal_jerk_weight,
    )
    second_order = second_order_smoothness_loss(
        outputs,
        time_step_s=time_step_s,
        steering_weight=steering_acceleration_weight,
        longitudinal_weight=longitudinal_snap_weight,
    )
    return first_order + second_order


class ClosedLoopLoss(nn.Module):
    """Combine rollout tracking, stability, and comfort objectives."""

    def __init__(
        self,
        lateral_error_weight: float = 10.0,
        speed_error_weight: float = 2.0,
        longitudinal_error_weight: float = 1.0,
        yaw_rate_weight: float = 0.2,
        lateral_acceleration_weight: float = 0.2,
        steering_rate_weight: float = 0.1,
        longitudinal_jerk_weight: float = 0.1,
        steering_acceleration_weight: float = 0.01,
        longitudinal_snap_weight: float = 0.01,
    ) -> None:
        super().__init__()
        weights = {
            "lateral_error_weight": lateral_error_weight,
            "speed_error_weight": speed_error_weight,
            "longitudinal_error_weight": longitudinal_error_weight,
            "yaw_rate_weight": yaw_rate_weight,
            "lateral_acceleration_weight": lateral_acceleration_weight,
            "steering_rate_weight": steering_rate_weight,
            "longitudinal_jerk_weight": longitudinal_jerk_weight,
            "steering_acceleration_weight": steering_acceleration_weight,
            "longitudinal_snap_weight": longitudinal_snap_weight,
        }
        for name, value in weights.items():
            _validate_non_negative(name, value)
            setattr(self, name, value)

    def forward(
        self,
        lateral_error: Tensor,
        speed_error: Tensor,
        longitudinal_error: Tensor,
        yaw_rate: Tensor,
        lateral_acceleration: Tensor,
        outputs: Tensor,
        time_step_s: float,
    ) -> Tensor:
        tracking = closed_loop_tracking_loss(
            lateral_error,
            speed_error,
            longitudinal_error,
            lateral_weight=self.lateral_error_weight,
            speed_weight=self.speed_error_weight,
            longitudinal_weight=self.longitudinal_error_weight,
        )
        stability = closed_loop_stability_loss(
            yaw_rate,
            lateral_acceleration,
            yaw_rate_weight=self.yaw_rate_weight,
            lateral_acceleration_weight=self.lateral_acceleration_weight,
        )
        comfort = closed_loop_comfort_loss(
            outputs,
            time_step_s=time_step_s,
            steering_rate_weight=self.steering_rate_weight,
            longitudinal_jerk_weight=self.longitudinal_jerk_weight,
            steering_acceleration_weight=self.steering_acceleration_weight,
            longitudinal_snap_weight=self.longitudinal_snap_weight,
        )
        return tracking + stability + comfort
