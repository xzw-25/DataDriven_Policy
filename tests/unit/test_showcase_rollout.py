import numpy as np
import pytest
import torch
from torch import Tensor, nn

from vehicle_controller.data.synthetic_scenarios import (
    build_typical_scenarios,
    initial_state_from_reference_profile,
)
from vehicle_controller.factory import build_baseline_pipeline
from vehicle_controller.simulation import showcase
from vehicle_controller.simulation.rollout import rollout_reference_profile, summarize_rollout
from vehicle_controller.simulation.scenario import Scenario
from vehicle_controller.simulation.simulator import SimulationSample
from vehicle_controller.types import ReferenceTrajectory, VehicleCommand
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


class ZeroModel(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return torch.zeros((features.shape[0], 2), dtype=features.dtype, device=features.device)


def test_initial_state_from_reference_profile_aligns_with_path_start() -> None:
    profile = build_typical_scenarios(0.1)[0]

    state = initial_state_from_reference_profile(profile)

    assert np.isclose(state.pose.x, profile.points[0].x)
    assert np.isclose(state.pose.y, profile.points[0].y)
    assert np.isclose(state.vx, profile.speed_mps[0])


def test_rollout_reference_profile_runs_with_time_varying_reference() -> None:
    profile = build_typical_scenarios(0.1)[0]
    vehicle = VehicleParameters()
    pipeline = build_baseline_pipeline(ZeroModel())

    samples = rollout_reference_profile(
        pipeline,
        KinematicBicycleModel(vehicle),
        profile,
        initial_state_from_reference_profile(profile),
    )

    assert len(samples) == len(profile.time_s) - 1
    assert pipeline.last_diagnostics is not None

    summary = summarize_rollout(samples)
    assert all(np.isfinite(value) for value in summary.values())
    assert summary["mean_speed_mps"] >= 0.0


def test_reference_profile_plot_does_not_pass_speed_plan_acceleration_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_save_closed_loop_plots(*args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(showcase, "save_closed_loop_plots", fake_save_closed_loop_plots)
    profile = build_typical_scenarios(0.1)[0]

    showcase.save_reference_profile_plots(
        [],
        profile,
        VehicleParameters(),
        "unused",
    )

    assert "reference_acceleration_mps2" not in captured


def test_reference_profile_plot_passes_expert_control_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_save_closed_loop_plots(*args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(showcase, "save_closed_loop_plots", fake_save_closed_loop_plots)
    profile = build_typical_scenarios(0.1)[0]
    expert_time_s = np.asarray([0.0, 0.1])
    expert_steering_deg = np.asarray([12.0, 18.0])
    expert_acceleration_mps2 = np.asarray([1.0, 1.1])

    showcase.save_reference_profile_plots(
        [],
        profile,
        VehicleParameters(),
        "unused",
        expert_time_s=expert_time_s,
        expert_steering_deg=expert_steering_deg,
        expert_acceleration_mps2=expert_acceleration_mps2,
    )

    assert captured["expert_time_s"] is expert_time_s
    assert captured["expert_steering_deg"] is expert_steering_deg
    assert captured["expert_acceleration_mps2"] is expert_acceleration_mps2


def test_expert_control_profile_matches_reference_timeline() -> None:
    profile = build_typical_scenarios(0.1)[0]
    state = initial_state_from_reference_profile(profile)
    samples = [
        SimulationSample(float(time_s), state, VehicleCommand())
        for time_s in profile.time_s[:3]
    ]

    time_s, steering, acceleration = showcase._expert_control_profile(
        profile,
        samples,
        VehicleParameters(),
        showcase.ActuatorLimits(),
    )

    assert len(time_s) == len(samples)
    assert steering.shape == time_s.shape
    assert acceleration.shape == time_s.shape
    assert np.all(np.isfinite(steering))
    assert np.all(np.isfinite(acceleration))


def test_fixed_scenario_plot_does_not_pass_reference_acceleration_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_save_closed_loop_plots(*args: object, **kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(showcase, "save_closed_loop_plots", fake_save_closed_loop_plots)
    profile = build_typical_scenarios(0.1)[0]
    state = initial_state_from_reference_profile(profile)
    sample = SimulationSample(0.0, state, VehicleCommand())
    scenario = Scenario(
        name="fixed_reference",
        reference=ReferenceTrajectory(
            points=profile.points,
            v_ref=5.0,
            a_ref=0.2,
        ),
        initial_state=state,
        duration_s=1.0,
    )

    showcase.save_scenario_plots(
        [sample],
        scenario,
        VehicleParameters(),
        "unused",
    )

    assert "reference_acceleration_mps2" not in captured
