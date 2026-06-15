"""Simple kinematic bicycle model for software-in-the-loop tests."""

from __future__ import annotations

import math

from vehicle_controller.types import Pose2D, VehicleState
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


class KinematicBicycleModel:
    def __init__(self, parameters: VehicleParameters) -> None:
        self.parameters = parameters

    def step(
        self,
        state: VehicleState,
        steering_wheel_angle_rad: float,
        longitudinal_accel_mps2: float,
        dt: float,
    ) -> VehicleState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        front_wheel_angle = steering_wheel_angle_rad / self.parameters.steering_ratio
        yaw_rate = state.vx * math.tan(front_wheel_angle) / self.parameters.wheelbase_m
        next_vx = max(0.0, state.vx + longitudinal_accel_mps2 * dt)
        average_vx = 0.5 * (state.vx + next_vx)
        next_yaw = state.pose.yaw + yaw_rate * dt
        next_x = state.pose.x + average_vx * math.cos(state.pose.yaw) * dt
        next_y = state.pose.y + average_vx * math.sin(state.pose.yaw) * dt
        lateral_accel = average_vx * yaw_rate
        return VehicleState(
            pose=Pose2D(next_x, next_y, next_yaw),
            vx=next_vx,
            vy=0.0,
            ax=longitudinal_accel_mps2,
            ay=lateral_accel,
            r=yaw_rate,
            s=state.s + average_vx * dt,
            timestamp_s=state.timestamp_s + dt,
        )

