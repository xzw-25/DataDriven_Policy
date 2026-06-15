"""Build the stable 22-dimensional controller input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vehicle_controller.types import (
    ControllerFeatures,
    TrackingErrors,
    TrajectoryPoint,
    VehicleState,
)


class FeatureBuilder:
    def build(
        self,
        sampled_body_points: Sequence[TrajectoryPoint],
        kappa: float,
        errors: TrackingErrors,
        a_ref: float,
        v_ref: float,
        s_ref: float,
        state: VehicleState,
    ) -> ControllerFeatures:
        if len(sampled_body_points) != 5:
            raise ValueError("Exactly five sampled trajectory points are required")
        trajectory_values = [
            coordinate
            for point in sampled_body_points
            for coordinate in (point.x, point.y)
        ]
        values = np.asarray(
            [
                *trajectory_values,
                kappa,
                errors.e_lat,
                errors.e_v,
                errors.e_s,
                a_ref,
                v_ref,
                s_ref,
                state.vx,
                state.vy,
                state.ax,
                state.ay,
                state.r,
            ],
            dtype=np.float32,
        )
        return ControllerFeatures(values)
