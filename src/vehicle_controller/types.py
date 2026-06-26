"""Shared strongly typed controller data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.units import deg_to_rad, rad_to_deg


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float
    s: float = 0.0
    kappa: float = 0.0
    v_ref: float = 0.0
    a_ref: float = 0.0


@dataclass(frozen=True)
class ReferenceTrajectory:
    points: Sequence[TrajectoryPoint]
    v_ref: float
    a_ref: float = 0.0
    s_ref: float = 0.0
    kappa: float | None = None


@dataclass(frozen=True)
class TrackingErrors:
    e_lat: float
    e_v: float
    e_s: float


@dataclass(frozen=True)
class VehicleState:
    pose: Pose2D
    vx: float
    vy: float
    ax: float
    ay: float
    r: float
    s: float = 0.0
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class ControllerFeatures:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (FEATURE_COUNT,):
            raise ValueError(f"Expected {FEATURE_COUNT} features, got shape {values.shape}")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class NeuralPolicyOutput:
    steering_des_deg: float
    signed_accel_des_mps2: float

    @classmethod
    def from_rad(
        cls,
        steering_des_rad: float,
        signed_accel_des_mps2: float,
    ) -> NeuralPolicyOutput:
        return cls(rad_to_deg(steering_des_rad), signed_accel_des_mps2)

    @property
    def steering_des_rad(self) -> float:
        return deg_to_rad(self.steering_des_deg)


@dataclass(frozen=True)
class ControllerStepDiagnostics:
    tracking_errors: TrackingErrors
    neural_output: NeuralPolicyOutput | None
    neural_candidate: VehicleCommand | None
    limited_candidate: VehicleCommand | None
    final_command: VehicleCommand


class CommandSource(str, Enum):
    NEURAL = "neural"
    LIMITED_NEURAL = "limited_neural"
    FALLBACK = "fallback"


@dataclass(frozen=True, init=False)
class VehicleCommand:
    steering_wheel_angle_deg: float
    drive_wheel_torque_nm: float
    drive_valid: bool
    brake_decel_mps2: float
    brake_valid: bool
    source: CommandSource
    reason: str

    def __init__(
        self,
        steering_wheel_angle_deg: float | None = None,
        drive_wheel_torque_nm: float | None = None,
        drive_valid: bool | None = None,
        brake_decel_mps2: float = 0.0,
        brake_valid: bool | None = None,
        source: CommandSource = CommandSource.NEURAL,
        reason: str = "ok",
        *,
        steering_wheel_angle_rad: float | None = None,
        drive_torque_nm: float | None = None,
    ) -> None:
        if steering_wheel_angle_deg is None:
            steering_wheel_angle_deg = (
                0.0 if steering_wheel_angle_rad is None else rad_to_deg(steering_wheel_angle_rad)
            )
        elif steering_wheel_angle_rad is not None:
            raise ValueError("Specify either steering_wheel_angle_deg or steering_wheel_angle_rad")

        if drive_wheel_torque_nm is None:
            drive_wheel_torque_nm = 0.0 if drive_torque_nm is None else drive_torque_nm
        elif drive_torque_nm is not None:
            raise ValueError("Specify either drive_wheel_torque_nm or drive_torque_nm")

        if drive_valid is None:
            drive_valid = drive_wheel_torque_nm > 0.0
        if brake_valid is None:
            brake_valid = brake_decel_mps2 > 0.0

        object.__setattr__(self, "steering_wheel_angle_deg", float(steering_wheel_angle_deg))
        object.__setattr__(self, "drive_wheel_torque_nm", float(drive_wheel_torque_nm))
        object.__setattr__(self, "drive_valid", bool(drive_valid))
        object.__setattr__(self, "brake_decel_mps2", float(brake_decel_mps2))
        object.__setattr__(self, "brake_valid", bool(brake_valid))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reason", reason)

    @property
    def steering_wheel_angle_rad(self) -> float:
        return deg_to_rad(self.steering_wheel_angle_deg)

    @property
    def drive_torque_nm(self) -> float:
        return self.drive_wheel_torque_nm


class SafetyAction(str, Enum):
    PASS = "pass"
    LIMIT = "limit"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    reason: str
    command: VehicleCommand
    diagnostics: dict[str, float | str] = field(default_factory=dict)
