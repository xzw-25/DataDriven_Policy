"""ROS 2 integration boundary.

Message types are intentionally left to the target vehicle project.
"""

from vehicle_controller.adapters.base import ControllerAdapter


class Ros2Adapter(ControllerAdapter):
    def read(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Bind project-specific ROS 2 messages here")

    def write(self, command):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Bind project-specific ROS 2 messages here")

