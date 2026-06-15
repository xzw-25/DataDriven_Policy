import pytest
import torch

from vehicle_controller.training.losses import (
    ClosedLoopLoss,
    ControllerLoss,
    closed_loop_comfort_loss,
    closed_loop_stability_loss,
    closed_loop_tracking_loss,
    first_order_smoothness_loss,
    longitudinal_huber_loss,
    second_order_smoothness_loss,
    steering_huber_loss,
    temporal_smoothness_loss,
)


def test_control_huber_losses_use_separate_channels() -> None:
    prediction = torch.tensor([[2.0, 0.5], [0.0, -2.0]])
    target = torch.zeros_like(prediction)

    assert steering_huber_loss(prediction, target).item() == pytest.approx(0.75)
    assert longitudinal_huber_loss(prediction, target).item() == pytest.approx(0.8125)
    assert ControllerLoss(2.0, 3.0)(prediction, target).item() == pytest.approx(
        2.0 * 0.75 + 3.0 * 0.8125
    )


def test_controller_loss_supports_sequence_outputs() -> None:
    prediction = torch.ones(2, 3, 2, requires_grad=True)
    target = torch.zeros_like(prediction)

    loss = ControllerLoss()(prediction, target)
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert prediction.grad is not None


def test_first_order_smoothness_scales_by_time_step() -> None:
    outputs = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]])

    loss = first_order_smoothness_loss(outputs, time_step_s=0.5)

    assert loss.item() == pytest.approx(20.0)
    assert temporal_smoothness_loss(outputs).item() == pytest.approx(5.0)


def test_second_order_smoothness_detects_control_curvature() -> None:
    linear = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]])
    curved = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]]])

    assert second_order_smoothness_loss(linear).item() == pytest.approx(0.0)
    assert second_order_smoothness_loss(curved).item() == pytest.approx(5.0)


def test_short_sequences_return_differentiable_zero() -> None:
    outputs = torch.ones(2, 1, 2, requires_grad=True)

    loss = first_order_smoothness_loss(outputs) + second_order_smoothness_loss(outputs)
    loss.backward()

    assert loss.item() == 0.0
    assert outputs.grad is not None


def test_closed_loop_tracking_loss_uses_configurable_weights() -> None:
    lateral = torch.tensor([1.0, 2.0])
    speed = torch.tensor([2.0, 0.0])
    longitudinal = torch.tensor([1.0, 1.0])

    loss = closed_loop_tracking_loss(
        lateral,
        speed,
        longitudinal,
        lateral_weight=2.0,
        speed_weight=3.0,
        longitudinal_weight=4.0,
    )

    assert loss.item() == pytest.approx(15.0)


def test_closed_loop_stability_loss_penalizes_yaw_and_lateral_acceleration() -> None:
    yaw_rate = torch.tensor([1.0, 1.0])
    lateral_acceleration = torch.tensor([2.0, 2.0])

    loss = closed_loop_stability_loss(
        yaw_rate,
        lateral_acceleration,
        yaw_rate_weight=2.0,
        lateral_acceleration_weight=3.0,
    )

    assert loss.item() == pytest.approx(14.0)


def test_comfort_loss_combines_first_and_second_order_terms() -> None:
    outputs = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]]])

    loss = closed_loop_comfort_loss(
        outputs,
        time_step_s=1.0,
        steering_rate_weight=1.0,
        longitudinal_jerk_weight=1.0,
        steering_acceleration_weight=1.0,
        longitudinal_snap_weight=1.0,
    )

    assert loss.item() == pytest.approx(12.5)


def test_closed_loop_loss_combines_all_objectives_and_backpropagates() -> None:
    errors = torch.ones(1, 3)
    outputs = torch.tensor(
        [[[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]]],
        requires_grad=True,
    )
    loss_function = ClosedLoopLoss(
        lateral_error_weight=1.0,
        speed_error_weight=1.0,
        longitudinal_error_weight=1.0,
        yaw_rate_weight=1.0,
        lateral_acceleration_weight=1.0,
        steering_rate_weight=1.0,
        longitudinal_jerk_weight=1.0,
        steering_acceleration_weight=1.0,
        longitudinal_snap_weight=1.0,
    )

    loss = loss_function(
        errors,
        errors,
        errors,
        errors,
        errors,
        outputs,
        time_step_s=1.0,
    )
    loss.backward()

    assert loss.item() == pytest.approx(6.25)
    assert outputs.grad is not None


@pytest.mark.parametrize(
    "call",
    [
        lambda: ControllerLoss(-1.0),
        lambda: first_order_smoothness_loss(torch.zeros(2, 2), 1.0),
        lambda: second_order_smoothness_loss(torch.zeros(1, 3, 2), 0.0),
        lambda: closed_loop_tracking_loss(
            torch.zeros(2),
            torch.zeros(3),
            torch.zeros(2),
        ),
    ],
)
def test_losses_reject_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
