"""Simulation metric extraction."""

from __future__ import annotations

import numpy as np

from vehicle_controller.control.controller_pipeline import ControllerPipeline
from vehicle_controller.data.synthetic_scenarios import ReferenceProfile
from vehicle_controller.simulation.simulator import (
    SimulationSample,
    command_to_longitudinal_acceleration,
)
from vehicle_controller.types import ReferenceTrajectory, VehicleState
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel


def rollout_reference_profile(
    controller: ControllerPipeline,
    vehicle_model: KinematicBicycleModel,
    profile: ReferenceProfile,
    initial_state: VehicleState,
) -> list[SimulationSample]:
    """Run a closed-loop rollout against a time-varying synthetic reference profile."""
    if len(profile.time_s) < 2:
        raise ValueError("Reference profile must contain at least two time samples")

    controller.reset()
    state = initial_state
    samples: list[SimulationSample] = []
    time_pairs = zip(profile.time_s[:-1], profile.time_s[1:])
    for time_s, next_time_s in time_pairs:
        dt = float(next_time_s - time_s)
        if dt <= 0.0:
            raise ValueError("Reference profile time steps must be strictly increasing")
        reference_s, reference_speed, reference_acceleration = profile.sample(float(time_s))
        reference = ReferenceTrajectory(
            points=profile.points,
            v_ref=reference_speed,
            a_ref=reference_acceleration,
            s_ref=reference_s,
        )
        command = controller.step(reference, state, dt)
        longitudinal_accel = command_to_longitudinal_acceleration(
            command,
            vehicle_model.parameters,
        )
        samples.append(
            SimulationSample(
                state.timestamp_s,
                state,
                command,
                controller.last_diagnostics,
            )
        )
        state = vehicle_model.step(
            state,
            command.steering_wheel_angle_rad,
            longitudinal_accel,
            dt,
        )
    return samples


def summarize_rollout(samples: list[SimulationSample]) -> dict[str, float]:
    if not samples:
        raise ValueError("No simulation samples")
    return {
        "duration_s": samples[-1].time_s - samples[0].time_s,
        "maximum_abs_lateral_accel_mps2": float(max(abs(item.state.ay) for item in samples)),
        "maximum_abs_yaw_rate_radps": float(max(abs(item.state.r) for item in samples)),
        "mean_speed_mps": float(np.mean([item.state.vx for item in samples])),
        "fallback_fraction": float(
            np.mean([item.command.source.value == "fallback" for item in samples])
        ),
    }
