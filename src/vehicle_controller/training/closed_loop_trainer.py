"""Differentiable closed-loop rollout training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from vehicle_controller.data.synthetic_scenarios import (
    ReferenceProfile,
    initial_state_from_reference_profile,
)
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.geometry.trajectory_sampler import DEFAULT_PREVIEW_TIMES_S
from vehicle_controller.units import DEG_TO_RAD
from vehicle_controller.training.losses import (
    ClosedLoopLoss,
    closed_loop_comfort_loss,
    closed_loop_stability_loss,
    closed_loop_tracking_loss,
)
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


@dataclass(frozen=True)
class ClosedLoopScales:
    steering_limit_deg: float
    accel_limit_mps2: float


@dataclass(frozen=True)
class ClosedLoopRolloutResult:
    loss: Tensor
    tracking_loss: Tensor
    stability_loss: Tensor
    comfort_loss: Tensor
    outputs: Tensor
    lateral_error: Tensor
    speed_error: Tensor
    longitudinal_error: Tensor


@dataclass(frozen=True)
class ReferenceBatch:
    lookahead_x: Tensor
    lookahead_y: Tensor
    lookahead_kappa: Tensor
    reference_x: Tensor
    reference_y: Tensor
    reference_yaw: Tensor
    reference_s: Tensor
    reference_speed: Tensor
    reference_acceleration: Tensor
    initial_x: Tensor
    initial_y: Tensor
    initial_yaw: Tensor
    initial_speed: Tensor
    time_step_s: float

    @property
    def batch_size(self) -> int:
        return int(self.reference_s.shape[0])

    @property
    def horizon_steps(self) -> int:
        return int(self.reference_s.shape[1])


def _profile_arrays(profile: ReferenceProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_s = np.asarray([point.s for point in profile.points], dtype=np.float64)
    path_x = np.asarray([point.x for point in profile.points], dtype=np.float64)
    path_y = np.asarray([point.y for point in profile.points], dtype=np.float64)
    return path_s, path_x, path_y


def _path_yaw(path_s: np.ndarray, path_x: np.ndarray, path_y: np.ndarray) -> np.ndarray:
    dx_ds = np.gradient(path_x, path_s, edge_order=1)
    dy_ds = np.gradient(path_y, path_s, edge_order=1)
    return np.unwrap(np.arctan2(dy_ds, dx_ds))


def _interp_path(
    target_s: np.ndarray,
    path_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(np.clip(target_s, path_s[0], path_s[-1]), path_s, values)


def build_reference_batch(
    profiles: Sequence[ReferenceProfile],
    preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
    curvature_weights: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    horizon_steps: int = 500,
    device: str | torch.device = "cpu",
    lookahead_distances_m: Sequence[float] | None = None,
) -> ReferenceBatch:
    """Precompute reference tensors for differentiable closed-loop rollout."""
    if not profiles:
        raise ValueError("At least one reference profile is required")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if lookahead_distances_m is None and len(preview_times_s) != 5:
        raise ValueError("Exactly five preview times are required")
    if lookahead_distances_m is not None and len(lookahead_distances_m) != 5:
        raise ValueError("Exactly five lookahead distances are required")
    if len(curvature_weights) != 5:
        raise ValueError("Exactly five curvature weights are required")

    available_steps = min(len(profile.time_s) - 1 for profile in profiles)
    steps = min(int(horizon_steps), int(available_steps))
    if steps <= 0:
        raise ValueError("Reference profiles must contain at least two time samples")

    time_step_s = float(profiles[0].time_s[1] - profiles[0].time_s[0])
    if any(
        not np.isclose(profile.time_s[1] - profile.time_s[0], time_step_s)
        for profile in profiles
    ):
        raise ValueError("All profiles in a batch must use the same time step")

    preview_times = np.asarray(preview_times_s, dtype=np.float64)
    if np.any(preview_times < 0.0):
        raise ValueError("Preview times must be non-negative")
    fixed_lookahead = (
        None
        if lookahead_distances_m is None
        else np.asarray(lookahead_distances_m, dtype=np.float64)
    )
    if fixed_lookahead is not None and np.any(fixed_lookahead < 0.0):
        raise ValueError("Lookahead distances must be non-negative")
    weights = np.asarray(curvature_weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    lookahead_x: list[np.ndarray] = []
    lookahead_y: list[np.ndarray] = []
    lookahead_kappa: list[np.ndarray] = []
    reference_x: list[np.ndarray] = []
    reference_y: list[np.ndarray] = []
    reference_yaw: list[np.ndarray] = []
    reference_s: list[np.ndarray] = []
    reference_speed: list[np.ndarray] = []
    reference_acceleration: list[np.ndarray] = []
    initial_x: list[float] = []
    initial_y: list[float] = []
    initial_yaw: list[float] = []
    initial_speed: list[float] = []

    for profile in profiles:
        path_s, path_x, path_y = _profile_arrays(profile)
        path_kappa = np.asarray([point.kappa for point in profile.points], dtype=np.float64)
        path_heading = _path_yaw(path_s, path_x, path_y)

        ref_s = np.asarray(profile.reference_s_m[:steps], dtype=np.float64)
        reference_speed_values = np.asarray(profile.speed_mps[:steps], dtype=np.float64)
        reference_acceleration_values = np.asarray(
            profile.acceleration_mps2[:steps],
            dtype=np.float64,
        )
        if fixed_lookahead is None:
            lookahead = np.maximum(
                reference_speed_values[:, None] * preview_times[None, :]
                + 0.5 * reference_acceleration_values[:, None] * preview_times[None, :] ** 2,
                0.0,
            )
            lookahead = np.maximum.accumulate(lookahead, axis=1)
        else:
            lookahead = fixed_lookahead[None, :]
        sampled_s = ref_s[:, None] + lookahead
        sampled_x = _interp_path(sampled_s.reshape(-1), path_s, path_x).reshape(steps, 5)
        sampled_y = _interp_path(sampled_s.reshape(-1), path_s, path_y).reshape(steps, 5)
        sampled_kappa = _interp_path(sampled_s.reshape(-1), path_s, path_kappa).reshape(steps, 5)

        lookahead_x.append(sampled_x)
        lookahead_y.append(sampled_y)
        lookahead_kappa.append(sampled_kappa)
        reference_x.append(_interp_path(ref_s, path_s, path_x))
        reference_y.append(_interp_path(ref_s, path_s, path_y))
        reference_yaw.append(_interp_path(ref_s, path_s, path_heading))
        reference_s.append(ref_s)
        reference_speed.append(reference_speed_values)
        reference_acceleration.append(reference_acceleration_values)

        state = initial_state_from_reference_profile(profile)
        initial_x.append(float(state.pose.x))
        initial_y.append(float(state.pose.y))
        initial_yaw.append(float(state.pose.yaw))
        initial_speed.append(float(state.vx))

    torch_device = torch.device(device)

    def tensor(values: object) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=torch_device)

    return ReferenceBatch(
        lookahead_x=tensor(np.stack(lookahead_x)),
        lookahead_y=tensor(np.stack(lookahead_y)),
        lookahead_kappa=tensor(np.stack(lookahead_kappa)),
        reference_x=tensor(np.stack(reference_x)),
        reference_y=tensor(np.stack(reference_y)),
        reference_yaw=tensor(np.stack(reference_yaw)),
        reference_s=tensor(np.stack(reference_s)),
        reference_speed=tensor(np.stack(reference_speed)),
        reference_acceleration=tensor(np.stack(reference_acceleration)),
        initial_x=tensor(initial_x),
        initial_y=tensor(initial_y),
        initial_yaw=tensor(initial_yaw),
        initial_speed=tensor(initial_speed),
        time_step_s=time_step_s,
    )


def _normalizer_tensors(
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> tuple[Tensor, Tensor, float]:
    mean = torch.as_tensor(normalizer.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(normalizer.std, dtype=torch.float32, device=device)
    return mean, std, normalizer.clip


def _controller_features(
    reference: ReferenceBatch,
    step: int,
    x: Tensor,
    y: Tensor,
    yaw: Tensor,
    vx: Tensor,
    ax: Tensor,
    ay: Tensor,
    yaw_rate: Tensor,
    path_s: Tensor,
    curvature_weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dx = reference.lookahead_x[:, step, :] - x[:, None]
    dy = reference.lookahead_y[:, step, :] - y[:, None]
    cos_yaw = torch.cos(yaw)[:, None]
    sin_yaw = torch.sin(yaw)[:, None]
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    trajectory = torch.stack((body_x, body_y), dim=-1).reshape(reference.batch_size, 10)
    kappa = torch.sum(reference.lookahead_kappa[:, step, :] * curvature_weights[None, :], dim=-1)

    ref_yaw = reference.reference_yaw[:, step]
    ref_dx = x - reference.reference_x[:, step]
    ref_dy = y - reference.reference_y[:, step]
    vehicle_lateral_offset = -torch.sin(ref_yaw) * ref_dx + torch.cos(ref_yaw) * ref_dy
    lateral_error = -vehicle_lateral_offset
    speed_error = reference.reference_speed[:, step] - vx
    longitudinal_error = reference.reference_s[:, step] - path_s

    feature_values = torch.cat(
        (
            trajectory,
            kappa[:, None],
            lateral_error[:, None],
            speed_error[:, None],
            longitudinal_error[:, None],
            reference.reference_acceleration[:, step, None],
            reference.reference_speed[:, step, None],
            reference.reference_s[:, step, None],
            vx[:, None],
            torch.zeros_like(vx)[:, None],
            ax[:, None],
            ay[:, None],
            yaw_rate[:, None],
        ),
        dim=-1,
    )
    return feature_values, lateral_error, speed_error, longitudinal_error


def differentiable_closed_loop_rollout(
    model: nn.Module,
    reference: ReferenceBatch,
    normalizer: FeatureNormalizer,
    vehicle: VehicleParameters,
    scales: ClosedLoopScales,
    loss_function: ClosedLoopLoss,
    curvature_weights: Sequence[float],
) -> ClosedLoopRolloutResult:
    """Roll out a neural controller through a differentiable bicycle model."""
    if reference.horizon_steps <= 0:
        raise ValueError("reference horizon must be positive")
    device = reference.reference_s.device
    mean, std, clip = _normalizer_tensors(normalizer, device)
    used_curvature_weights = torch.as_tensor(
        curvature_weights,
        dtype=torch.float32,
        device=device,
    )
    used_curvature_weights = used_curvature_weights / torch.sum(used_curvature_weights)

    x = reference.initial_x
    y = reference.initial_y
    yaw = reference.initial_yaw
    vx = reference.initial_speed
    ax = torch.zeros_like(vx)
    ay = torch.zeros_like(vx)
    yaw_rate = torch.zeros_like(vx)
    path_s = torch.zeros_like(vx)

    outputs: list[Tensor] = []
    lateral_errors: list[Tensor] = []
    speed_errors: list[Tensor] = []
    longitudinal_errors: list[Tensor] = []
    yaw_rates: list[Tensor] = []
    lateral_accelerations: list[Tensor] = []

    dt = reference.time_step_s
    for step in range(reference.horizon_steps):
        features, lateral_error, speed_error, longitudinal_error = _controller_features(
            reference,
            step,
            x,
            y,
            yaw,
            vx,
            ax,
            ay,
            yaw_rate,
            path_s,
            used_curvature_weights,
        )
        normalized = torch.clamp((features - mean) / std, -clip, clip)
        output = model(normalized)
        if output.ndim != 2 or output.shape != (reference.batch_size, 2):
            raise ValueError(
                "Differentiable closed-loop training currently expects a feed-forward "
                f"model output with shape [{reference.batch_size}, 2], got {tuple(output.shape)}"
            )

        steering_deg = output[:, 0] * scales.steering_limit_deg
        steering_rad = steering_deg * DEG_TO_RAD
        acceleration = output[:, 1] * scales.accel_limit_mps2
        control = torch.stack((steering_deg, acceleration), dim=-1)

        front_wheel_angle = steering_rad / float(vehicle.steering_ratio)
        current_yaw_rate = vx * torch.tan(front_wheel_angle) / float(vehicle.wheelbase_m)
        next_vx = torch.clamp(vx + acceleration * dt, min=0.0)
        average_vx = 0.5 * (vx + next_vx)
        next_yaw = yaw + current_yaw_rate * dt
        next_x = x + average_vx * torch.cos(yaw) * dt
        next_y = y + average_vx * torch.sin(yaw) * dt
        current_lateral_accel = average_vx * current_yaw_rate

        outputs.append(control)
        lateral_errors.append(lateral_error)
        speed_errors.append(speed_error)
        longitudinal_errors.append(longitudinal_error)
        yaw_rates.append(current_yaw_rate)
        lateral_accelerations.append(current_lateral_accel)

        x = next_x
        y = next_y
        yaw = next_yaw
        vx = next_vx
        ax = acceleration
        ay = current_lateral_accel
        yaw_rate = current_yaw_rate
        path_s = path_s + average_vx * dt

    output_tensor = torch.stack(outputs, dim=1)
    lateral_error_tensor = torch.stack(lateral_errors, dim=1)
    speed_error_tensor = torch.stack(speed_errors, dim=1)
    longitudinal_error_tensor = torch.stack(longitudinal_errors, dim=1)
    yaw_rate_tensor = torch.stack(yaw_rates, dim=1)
    lateral_acceleration_tensor = torch.stack(lateral_accelerations, dim=1)

    tracking = closed_loop_tracking_loss(
        lateral_error_tensor,
        speed_error_tensor,
        longitudinal_error_tensor,
        lateral_weight=loss_function.lateral_error_weight,
        speed_weight=loss_function.speed_error_weight,
        longitudinal_weight=loss_function.longitudinal_error_weight,
    )
    stability = closed_loop_stability_loss(
        yaw_rate_tensor,
        lateral_acceleration_tensor,
        yaw_rate_weight=loss_function.yaw_rate_weight,
        lateral_acceleration_weight=loss_function.lateral_acceleration_weight,
    )
    comfort = closed_loop_comfort_loss(
        output_tensor,
        time_step_s=dt,
        steering_rate_weight=loss_function.steering_rate_weight,
        longitudinal_jerk_weight=loss_function.longitudinal_jerk_weight,
        steering_acceleration_weight=loss_function.steering_acceleration_weight,
        longitudinal_snap_weight=loss_function.longitudinal_snap_weight,
    )
    loss = tracking + stability + comfort
    return ClosedLoopRolloutResult(
        loss=loss,
        tracking_loss=tracking,
        stability_loss=stability,
        comfort_loss=comfort,
        outputs=output_tensor,
        lateral_error=lateral_error_tensor,
        speed_error=speed_error_tensor,
        longitudinal_error=longitudinal_error_tensor,
    )


def closed_loop_loss_from_config(config: Mapping[str, object]) -> ClosedLoopLoss:
    return ClosedLoopLoss(
        lateral_error_weight=float(config.get("lateral_error_weight", 10.0)),
        speed_error_weight=float(config.get("speed_error_weight", 2.0)),
        longitudinal_error_weight=float(config.get("longitudinal_error_weight", 1.0)),
        yaw_rate_weight=float(config.get("yaw_rate_weight", 0.2)),
        lateral_acceleration_weight=float(config.get("lateral_acceleration_weight", 0.2)),
        steering_rate_weight=float(config.get("steering_rate_weight", 0.1)),
        longitudinal_jerk_weight=float(config.get("longitudinal_jerk_weight", 0.1)),
        steering_acceleration_weight=float(config.get("steering_acceleration_weight", 0.01)),
        longitudinal_snap_weight=float(config.get("longitudinal_snap_weight", 0.01)),
    )
