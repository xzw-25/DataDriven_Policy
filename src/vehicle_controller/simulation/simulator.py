"""Closed-loop simulator."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_controller.control.controller_pipeline import ControllerPipeline
from vehicle_controller.simulation.scenario import Scenario
from vehicle_controller.types import ControllerStepDiagnostics, VehicleCommand, VehicleState
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


@dataclass(frozen=True)
class SimulationSample:
    time_s: float
    state: VehicleState
    command: VehicleCommand
    diagnostics: ControllerStepDiagnostics | None = None


def command_to_longitudinal_acceleration(
    command: VehicleCommand,
    parameters: VehicleParameters,
) -> float:
    acceleration = -command.brake_decel_mps2
    if command.drive_torque_nm > 0.0:
        acceleration += (
            command.drive_torque_nm
            * parameters.drivetrain_ratio
            * parameters.drivetrain_efficiency
            / (parameters.mass_kg * parameters.wheel_radius_m)
        )
    return acceleration


class Simulator:
    def __init__(
        self,
        controller: ControllerPipeline,
        vehicle_model: KinematicBicycleModel,
    ) -> None:
        self.controller = controller
        self.vehicle_model = vehicle_model

    def run(self, scenario: Scenario) -> list[SimulationSample]:
        state = scenario.initial_state
        samples: list[SimulationSample] = []
        steps = int(scenario.duration_s / scenario.time_step_s)
        for _ in range(steps):
            command = self.controller.step(scenario.reference, state, scenario.time_step_s)
            longitudinal_accel = command_to_longitudinal_acceleration(
                command,
                self.vehicle_model.parameters,
            )
            samples.append(
                SimulationSample(
                    state.timestamp_s,
                    state,
                    command,
                    self.controller.last_diagnostics,
                )
            )
            state = self.vehicle_model.step(
                state,
                command.steering_wheel_angle_rad,
                longitudinal_accel,
                scenario.time_step_s,
            )
        return samples
