"""Abstract platform adapter."""

from abc import ABC, abstractmethod

from vehicle_controller.types import ReferenceTrajectory, VehicleCommand, VehicleState


class ControllerAdapter(ABC):
    @abstractmethod
    def read(self) -> tuple[ReferenceTrajectory, VehicleState]:
        raise NotImplementedError

    @abstractmethod
    def write(self, command: VehicleCommand) -> None:
        raise NotImplementedError

