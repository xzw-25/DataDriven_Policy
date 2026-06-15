"""Simulation metric extraction."""

from __future__ import annotations

import numpy as np

from vehicle_controller.simulation.simulator import SimulationSample


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

