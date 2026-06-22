import numpy as np
import torch

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.synthetic_scenarios import build_typical_scenarios
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.models.direct_mlp_controller import DirectMLPController
from vehicle_controller.training.closed_loop_trainer import (
    ClosedLoopScales,
    build_reference_batch,
    closed_loop_loss_from_config,
    differentiable_closed_loop_rollout,
)
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


def test_reference_batch_uses_requested_horizon_and_profiles() -> None:
    profiles = build_typical_scenarios(0.1)[:2]

    reference = build_reference_batch(
        profiles,
        preview_times_s=(0.1, 0.2, 0.3, 0.4, 0.5),
        curvature_weights=(1.0, 0.8, 0.6, 0.4, 0.2),
        horizon_steps=4,
    )

    assert reference.batch_size == 2
    assert reference.horizon_steps == 4
    assert reference.lookahead_x.shape == (2, 4, 5)
    assert reference.reference_speed.shape == (2, 4)


def test_differentiable_closed_loop_rollout_backpropagates_to_model() -> None:
    profiles = build_typical_scenarios(0.1)[:2]
    reference = build_reference_batch(
        profiles,
        preview_times_s=(0.1, 0.2, 0.3, 0.4, 0.5),
        curvature_weights=(1.0, 0.8, 0.6, 0.4, 0.2),
        horizon_steps=3,
    )
    model = DirectMLPController(hidden_sizes=[16])
    normalizer = FeatureNormalizer(
        mean=np.zeros(FEATURE_COUNT, dtype=np.float32),
        std=np.ones(FEATURE_COUNT, dtype=np.float32) * 10.0,
    )
    loss_function = closed_loop_loss_from_config({})

    result = differentiable_closed_loop_rollout(
        model,
        reference,
        normalizer,
        VehicleParameters(),
        ClosedLoopScales(steering_limit_deg=458.3662361046586, accel_limit_mps2=6.0),
        loss_function,
        curvature_weights=(1.0, 0.8, 0.6, 0.4, 0.2),
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.outputs.shape == (2, 3, 2)
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0.0)
        for parameter in model.parameters()
    )
