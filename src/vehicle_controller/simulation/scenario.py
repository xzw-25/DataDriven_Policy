"""Simulation scenario definitions."""

from dataclasses import dataclass

from vehicle_controller.types import ReferenceTrajectory, VehicleState


@dataclass(frozen=True)
class Scenario:
    name: str
    reference: ReferenceTrajectory
    initial_state: VehicleState
    duration_s: float
    time_step_s: float = 0.01

