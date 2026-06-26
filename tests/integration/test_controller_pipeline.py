import math

import torch
from torch import Tensor, nn

from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.types import (
    CommandSource,
    Pose2D,
    ReferenceTrajectory,
    TrackingErrors,
    TrajectoryPoint,
    VehicleState,
)


class ZeroModel(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return torch.zeros((features.shape[0], 2), dtype=features.dtype, device=features.device)


def test_pipeline_produces_finite_mutually_exclusive_command() -> None:
    pipeline = build_baseline_pipeline(ZeroModel())
    reference = ReferenceTrajectory(
        points=[TrajectoryPoint(float(index), 0.0, s=float(index)) for index in range(101)],
        v_ref=5.0,
        a_ref=0.0,
        s_ref=5.0,
        kappa=0.0,
    )
    command = pipeline.step(
        reference,
        VehicleState(Pose2D(0.0, 0.0, 0.0), 5.0, 0.0, 0.0, 0.0, 0.0),
        0.01,
    )
    assert command.source in (CommandSource.NEURAL, CommandSource.LIMITED_NEURAL)
    assert math.isfinite(command.steering_wheel_angle_deg)
    assert math.isfinite(command.drive_wheel_torque_nm)
    assert math.isfinite(command.brake_decel_mps2)
    assert not (command.drive_valid and command.brake_valid)
    assert command.drive_wheel_torque_nm * command.brake_decel_mps2 == 0.0
    assert pipeline.last_diagnostics is not None
    assert pipeline.last_diagnostics.neural_output is not None
    assert pipeline.last_diagnostics.final_command == command


def test_pipeline_falls_back_for_excessive_lateral_error() -> None:
    pipeline = build_baseline_pipeline(ZeroModel())
    reference = ReferenceTrajectory(
        points=[TrajectoryPoint(float(index), 0.0, s=float(index)) for index in range(101)],
        v_ref=5.0,
        a_ref=0.0,
        s_ref=0.0,
        kappa=0.0,
    )
    command = pipeline.step(
        reference,
        VehicleState(Pose2D(0.0, 4.0, 0.0), 5.0, 0.0, 0.0, 0.0, 0.0),
        0.01,
    )
    assert command.source == CommandSource.FALLBACK
    assert command.reason == "lateral_error_limit"
    assert pipeline.last_diagnostics is not None
    assert pipeline.last_diagnostics.final_command == command


def test_pipeline_accepts_upstream_tracking_errors() -> None:
    pipeline = build_baseline_pipeline(ZeroModel())
    reference = ReferenceTrajectory(
        points=[TrajectoryPoint(float(index), 0.0, s=float(index)) for index in range(101)],
        v_ref=5.0,
        s_ref=5.0,
        kappa=0.0,
    )
    command = pipeline.step(
        reference,
        VehicleState(Pose2D(0.0, 0.0, 0.0), 5.0, 0.0, 0.0, 0.0, 0.0),
        0.01,
        tracking_errors=TrackingErrors(e_lat=0.2, e_v=0.0, e_s=1.0),
    )
    assert command.source in (CommandSource.NEURAL, CommandSource.LIMITED_NEURAL)
