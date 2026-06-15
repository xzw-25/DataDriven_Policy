"""Deadline-aware wrapper around the controller pipeline."""

from time import perf_counter

from vehicle_controller.control.controller_pipeline import ControllerPipeline
from vehicle_controller.deployment.health_monitor import HealthMonitor
from vehicle_controller.types import ReferenceTrajectory, VehicleCommand, VehicleState


class RealtimeController:
    def __init__(self, pipeline: ControllerPipeline, health_monitor: HealthMonitor) -> None:
        self.pipeline = pipeline
        self.health_monitor = health_monitor

    def step(
        self,
        reference: ReferenceTrajectory,
        state: VehicleState,
        dt: float,
    ) -> VehicleCommand:
        start = perf_counter()
        command = self.pipeline.step(reference, state, dt)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.health_monitor.observe_inference(elapsed_ms)
        return command

