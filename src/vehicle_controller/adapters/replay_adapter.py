"""In-memory adapter for log replay and integration tests."""

from collections.abc import Iterable, Iterator

from vehicle_controller.adapters.base import ControllerAdapter
from vehicle_controller.types import ReferenceTrajectory, VehicleCommand, VehicleState


class ReplayAdapter(ControllerAdapter):
    def __init__(
        self,
        frames: Iterable[tuple[ReferenceTrajectory, VehicleState]],
    ) -> None:
        self.frames: Iterator[tuple[ReferenceTrajectory, VehicleState]] = iter(frames)
        self.commands: list[VehicleCommand] = []

    def read(self) -> tuple[ReferenceTrajectory, VehicleState]:
        return next(self.frames)

    def write(self, command: VehicleCommand) -> None:
        self.commands.append(command)

