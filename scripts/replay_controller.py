#!/usr/bin/env python3
"""Run the controller repeatedly on a deterministic reference frame."""

from _bootstrap import PROJECT_ROOT
from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.models.mlp_controller import MLPController
from vehicle_controller.types import Pose2D, ReferenceTrajectory, TrajectoryPoint, VehicleState


def main() -> None:
    pipeline = build_baseline_pipeline(MLPController(), project_root=PROJECT_ROOT)
    reference = ReferenceTrajectory(
        [TrajectoryPoint(float(x), 0.0, s=float(x)) for x in range(101)],
        v_ref=5.0,
        a_ref=0.0,
        s_ref=10.0,
        kappa=0.0,
    )
    state = VehicleState(Pose2D(0.0, 0.0, 0.0), 5.0, 0.0, 0.0, 0.0, 0.0)
    for index in range(10):
        print(index, pipeline.step(reference, state, 0.01))


if __name__ == "__main__":
    main()
